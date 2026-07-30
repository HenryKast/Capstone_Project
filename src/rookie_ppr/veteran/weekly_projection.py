"""
Week-by-week projections built bottom-up from component stat models.

Each player's projected season stat totals are turned into a per-game rate, laid
out across their team's real schedule (byes excluded), and scored to points. The
season sum is then compared against the top-down PPR projection on the board.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rookie_ppr.ingest_nfl import load_schedules
from rookie_ppr.utils import normalize_team_abbr
from rookie_ppr.veteran.defense_adjust import (
    CATEGORY_STATS,
    load_opponent_sensitivity,
    matchup_label,
    matchup_multipliers,
    upcoming_defense_z,
)
from rookie_ppr.veteran.config import (
    CSV_OUTPUT_DIR,
    VET_FEATURES_CSV,
    VET_WEEKLY_COMPARE_CSV,
    VET_WEEKLY_CSV,
)
from rookie_ppr.veteran.draft_board import (
    REG_SEASON_GAMES,
    VET_DRAFT_BOARD_CSV,
    _expected_games,
    expected_games_table,
)
from rookie_ppr.veteran.stat_models import load_stat_models, predict_stats

# 0.1 / yard rushing-receiving, 6 per rushing-receiving TD, 1 point per 25 passing
# yards, 4 per passing TD, -2 per fumble lost, -2 per interception, 1 per reception.
# This matches nflverse fantasy_points_ppr, the label every model is trained on.
WEEKLY_SCORING: dict[str, float] = {
    "rushing_yards": 0.1,
    "receiving_yards": 0.1,
    "rushing_tds": 6.0,
    "receiving_tds": 6.0,
    "passing_yards": 0.04,
    "passing_tds": 4.0,
    "fumbles_lost": -2.0,
    "interceptions": -2.0,
    "receptions": 1.0,
}

# Fixed reference scoring, held equal to the model's label. Identical to
# WEEKLY_SCORING unless scoring is overridden (e.g. --receptions-pt 0), which
# keeps the season comparison against the PPR projection apples-to-apples.
PPR_LABEL_SCORING: dict[str, float] = dict(WEEKLY_SCORING)

MAX_WEEK = 18

# Which defensive category scales each stat; turnovers stay unadjusted
STAT_CATEGORY = {stat: category for category, stats in CATEGORY_STATS.items() for stat in stats}

STAT_OUTPUT_NAMES = {
    "passing_yards": "proj_pass_yds",
    "passing_tds": "proj_pass_td",
    "interceptions": "proj_int",
    "rushing_yards": "proj_rush_yds",
    "rushing_tds": "proj_rush_td",
    "receptions": "proj_rec",
    "receiving_yards": "proj_rec_yds",
    "receiving_tds": "proj_rec_td",
    "fumbles_lost": "proj_fumbles_lost",
}


def team_week_slots(season: int) -> dict[str, dict[int, str]]:
    """{team: {week: opponent}} for the regular season, byes simply absent."""
    sched = load_schedules([int(season)])
    if sched.empty:
        return {}
    if "game_type" in sched.columns:
        sched = sched[sched["game_type"].astype(str).str.upper().isin(["REG", ""])]
    slots: dict[str, dict[int, str]] = {}
    for row in sched.itertuples(index=False):
        week = pd.to_numeric(getattr(row, "week", None), errors="coerce")
        if pd.isna(week):
            continue
        home = normalize_team_abbr(getattr(row, "home_team", None))
        away = normalize_team_abbr(getattr(row, "away_team", None))
        if not home or not away:
            continue
        slots.setdefault(home, {})[int(week)] = away
        slots.setdefault(away, {})[int(week)] = home
    return slots


def _score(stats: dict[str, float], scoring: dict[str, float]) -> float:
    return float(sum(stats.get(k, 0.0) * w for k, w in scoring.items()))


def _normalized_multipliers(
    game_weeks: list[int],
    per_week: dict[int, dict[str, float]],
    *,
    normalize: bool,
) -> dict[int, dict[str, float]]:
    """
    Rescale each category's multipliers to average 1.0 over the player's games.

    Season strength of schedule is already a feature of the season model
    (next_opp_def_*_faced), so by default the matchup adjustment only moves
    production between weeks instead of changing the season total.
    """
    if not normalize or not game_weeks:
        return per_week
    out = {w: dict(m) for w, m in per_week.items()}
    for category in ("rush", "pass"):
        vals = [per_week[w].get(category, 1.0) for w in game_weeks]
        mean = float(np.mean(vals)) if vals else 1.0
        if mean <= 0:
            continue
        for w in game_weeks:
            out[w][category] = per_week[w].get(category, 1.0) / mean
    return out


def build_weekly_projections(
    veteran_features: pd.DataFrame | None = None,
    *,
    scoring: dict[str, float] | None = None,
    adjust_for_defense: bool = True,
    normalize_to_season: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (weekly rows, per-player comparison vs the season projection)."""
    scoring = scoring or WEEKLY_SCORING

    if veteran_features is None:
        path = CSV_OUTPUT_DIR / VET_FEATURES_CSV
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Run the veteran compile first.")
        veteran_features = pd.read_csv(path, low_memory=False)

    feats = veteran_features.copy()
    feats["season"] = pd.to_numeric(feats.get("season"), errors="coerce")
    feats["games"] = pd.to_numeric(feats.get("games"), errors="coerce")
    latest_season = int(feats["season"].max())
    target_season = latest_season + 1

    rows = feats[(feats["season"] == latest_season) & (feats["games"].fillna(0) > 0)].copy()
    rows = rows.sort_values("games", ascending=False).drop_duplicates("gsis_id", keep="first")
    if rows.empty:
        return pd.DataFrame(), pd.DataFrame()

    artifact = load_stat_models()
    scored = predict_stats(rows, artifact)

    games_table = expected_games_table(feats)
    slots = team_week_slots(target_season)

    slopes = load_opponent_sensitivity() if adjust_for_defense else {}
    defense_z = upcoming_defense_z(target_season) if slopes else {}

    weekly_records: list[dict[str, Any]] = []
    compare_records: list[dict[str, Any]] = []

    for row in scored.itertuples(index=False):
        position = str(getattr(row, "position", "") or "")
        team = normalize_team_abbr(getattr(row, "team", None)) or ""
        exp_games = _expected_games(position, games_table)

        season_stats: dict[str, float] = {}
        for stat in STAT_OUTPUT_NAMES:
            val = getattr(row, f"pred_{stat}_next", np.nan)
            season_stats[stat] = float(val) if pd.notna(val) else 0.0
        per_game = {k: v / exp_games for k, v in season_stats.items()}

        pred_games = getattr(row, "pred_games_next", np.nan)
        pred_games = float(pred_games) if pd.notna(pred_games) else float("nan")

        team_slots = slots.get(team, {})
        game_weeks = sorted(team_slots) if team_slots else list(range(1, REG_SEASON_GAMES + 1))

        raw_mults = {
            week: matchup_multipliers(position, defense_z.get(team_slots.get(week, "")), slopes)
            for week in game_weeks
        }
        mults = _normalized_multipliers(game_weeks, raw_mults, normalize=normalize_to_season)

        week_points_total = 0.0
        week_points_ppr_total = 0.0
        for week in range(1, MAX_WEEK + 1):
            playing = week in game_weeks
            opponent = team_slots.get(week, "") if playing else ""
            week_mult = mults.get(week, {"rush": 1.0, "pass": 1.0})
            opp_z = defense_z.get(opponent) or {}

            week_stats = {
                stat: value * week_mult.get(STAT_CATEGORY.get(stat, ""), 1.0)
                for stat, value in per_game.items()
            }

            record: dict[str, Any] = {
                "gsis_id": getattr(row, "gsis_id", None),
                "player_name": getattr(row, "player_name", ""),
                "position": position,
                "team": team or "FA",
                "season": target_season,
                "week": week,
                "opponent": opponent if playing else "BYE",
                "is_bye": not playing,
            }
            for stat, out_name in STAT_OUTPUT_NAMES.items():
                record[out_name] = round(week_stats[stat], 2) if playing else 0.0

            pts = _score(week_stats, scoring) if playing else 0.0
            pts_ppr = _score(week_stats, PPR_LABEL_SCORING) if playing else 0.0
            record["week_points"] = round(pts, 2)
            record["week_points_ppr_basis"] = round(pts_ppr, 2)
            record["rush_multiplier"] = round(week_mult.get("rush", 1.0), 3) if playing else np.nan
            record["pass_multiplier"] = round(week_mult.get("pass", 1.0), 3) if playing else np.nan
            record["opp_def_rush_z"] = round(float(opp_z.get("rush")), 3) if opp_z.get("rush") is not None else np.nan
            record["opp_def_pass_z"] = round(float(opp_z.get("pass")), 3) if opp_z.get("pass") is not None else np.nan
            if playing:
                driver = "rush" if position == "RB" else "pass"
                record["matchup"] = matchup_label(week_mult.get(driver, 1.0))
            else:
                record["matchup"] = "bye"

            week_points_total += pts
            week_points_ppr_total += pts_ppr
            weekly_records.append(record)

        n_weeks = len(game_weeks)
        compare_records.append(
            {
                "gsis_id": getattr(row, "gsis_id", None),
                "player_name": getattr(row, "player_name", ""),
                "position": position,
                "team": team or "FA",
                "season": target_season,
                "game_weeks": n_weeks,
                "expected_games": round(exp_games, 2),
                "projected_games_model": round(pred_games, 2) if np.isfinite(pred_games) else np.nan,
                "weekly_sum_points": round(week_points_total, 2),
                "weekly_sum_points_ppr_basis": round(week_points_ppr_total, 2),
                "best_matchup_week": (
                    max(game_weeks, key=lambda w: mults[w].get("rush" if position == "RB" else "pass", 1.0))
                    if game_weeks
                    else np.nan
                ),
                "worst_matchup_week": (
                    min(game_weeks, key=lambda w: mults[w].get("rush" if position == "RB" else "pass", 1.0))
                    if game_weeks
                    else np.nan
                ),
                "weekly_sum_availability_adj": (
                    round(week_points_ppr_total / n_weeks * pred_games, 2)
                    if n_weeks and np.isfinite(pred_games)
                    else np.nan
                ),
                **{f"proj_{k}": round(v, 1) for k, v in season_stats.items()},
            }
        )

    weekly = pd.DataFrame(weekly_records)
    compare = pd.DataFrame(compare_records)

    board_path = CSV_OUTPUT_DIR / VET_DRAFT_BOARD_CSV
    if board_path.exists() and not compare.empty:
        board = pd.read_csv(board_path, low_memory=False)
        keep = [
            c
            for c in (
                "gsis_id",
                "synthetic_adp_rank",
                "projected_ppr_17",
                "projected_season_ppr",
            )
            if c in board.columns
        ]
        compare = compare.merge(board[keep], how="left", on="gsis_id")
        compare["diff_vs_pace"] = (
            compare["weekly_sum_points_ppr_basis"] - compare["projected_ppr_17"]
        ).round(2)
        compare["ratio_vs_pace"] = (
            compare["weekly_sum_points_ppr_basis"] / compare["projected_ppr_17"]
        ).round(3)

    if not compare.empty:
        compare = compare.sort_values("weekly_sum_points_ppr_basis", ascending=False).reset_index(
            drop=True
        )
    return weekly, compare


def comparison_summary(compare: pd.DataFrame) -> dict[str, Any]:
    """How the bottom-up season total lines up with the top-down PPR pace."""
    if compare.empty or "projected_ppr_17" not in compare.columns:
        return {}
    pair = compare[["position", "weekly_sum_points_ppr_basis", "projected_ppr_17"]].dropna()
    if len(pair) < 5:
        return {"n": int(len(pair))}

    def block(g: pd.DataFrame) -> dict[str, Any]:
        a = g["weekly_sum_points_ppr_basis"].to_numpy(dtype=float)
        b = g["projected_ppr_17"].to_numpy(dtype=float)
        r = float(np.corrcoef(a, b)[0, 1]) if len(g) > 2 else float("nan")
        return {
            "n": int(len(g)),
            "pearson_r": round(r, 4) if np.isfinite(r) else None,
            "mean_bottom_up": round(float(a.mean()), 2),
            "mean_top_down": round(float(b.mean()), 2),
            "mean_diff": round(float((a - b).mean()), 2),
            "mean_abs_diff": round(float(np.abs(a - b).mean()), 2),
        }

    out = {"overall": block(pair), "by_position": {}}
    for pos, g in pair.groupby("position"):
        if len(g) >= 5:
            out["by_position"][str(pos)] = block(g)
    return out


def save_weekly_projections(
    weekly: pd.DataFrame, compare: pd.DataFrame
) -> tuple[Path, Path]:
    CSV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    wp = CSV_OUTPUT_DIR / VET_WEEKLY_CSV
    cp = CSV_OUTPUT_DIR / VET_WEEKLY_COMPARE_CSV
    weekly.to_csv(wp, index=False)
    compare.to_csv(cp, index=False)
    return wp, cp


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Build week-by-week stat projections and compare to the season projection"
    )
    parser.add_argument(
        "--receptions-pt",
        type=float,
        default=WEEKLY_SCORING["receptions"],
        help="Points per reception (1.0 = PPR, 0 = standard)",
    )
    parser.add_argument(
        "--no-defense-adjust",
        action="store_true",
        help="Project every week at the same rate, ignoring the opponent",
    )
    parser.add_argument(
        "--absolute-sos",
        action="store_true",
        help="Let matchups move the season total instead of only shifting between weeks",
    )
    args = parser.parse_args()

    scoring = {**WEEKLY_SCORING, "receptions": args.receptions_pt}
    weekly, compare = build_weekly_projections(
        scoring=scoring,
        adjust_for_defense=not args.no_defense_adjust,
        normalize_to_season=not args.absolute_sos,
    )
    if weekly.empty:
        raise SystemExit("No weekly rows — run: python -m rookie_ppr.veteran.compile")
    wp, cp = save_weekly_projections(weekly, compare)
    print(f"Wrote {len(weekly)} weekly rows -> {wp.name}")
    print(f"Wrote {len(compare)} player comparisons -> {cp.name}")

    summary = comparison_summary(compare)
    overall = summary.get("overall") or {}
    if overall:
        print(
            f"  bottom-up vs top-down: r={overall.get('pearson_r')} "
            f"mean {overall.get('mean_bottom_up')} vs {overall.get('mean_top_down')} "
            f"(diff {overall.get('mean_diff')}, abs {overall.get('mean_abs_diff')}) n={overall.get('n')}"
        )
    for pos, block in (summary.get("by_position") or {}).items():
        print(
            f"  {pos}: r={block.get('pearson_r')} bottom-up {block.get('mean_bottom_up')} "
            f"vs top-down {block.get('mean_top_down')} diff {block.get('mean_diff')}"
        )


if __name__ == "__main__":
    main()
