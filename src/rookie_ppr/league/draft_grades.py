"""Draft grades: what you took against what the board said was there.

Every team gets two grades, kept apart on purpose because they answer different
questions and regularly disagree:

  * **value** - did you extract more than your pick slots were worth? Each skill
    pick is compared to the player the ADP board had sitting at that point in
    the draft, in value-over-replacement terms so a quarterback and a running
    back can be compared honestly. Summed over a team this is exactly zero-sum
    across the league: one manager's gain is another's loss.
  * **roster** - how strong is the team you walked away with? This is projected
    points per week from the same engine the ForeKast uses, so it accounts for
    lineup slots, positional scarcity and NFL bye weeks.

Both are schedule-free, so they can be published the moment a draft ends. The
playoff and title odds carried alongside them are schedule-dependent and stay
null until the season's odds have been published.

Letters are not curved. ``build_calibration`` reports what each letter has
actually been worth across 2018-2025, and that table ships with the grades so
the reader can judge how much to trust them.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from rookie_ppr.league.blend import (
    DEFAULT_BLEND_MODEL_WEIGHT,
    blend_weight_map,
    fit_blend_weights_walkforward,
)
from rookie_ppr.league.config import (
    CSV_OUTPUT_DIR,
    LEAGUE_BACKTEST_ROSTERS_CSV,
    LEAGUE_DRAFT_CSV,
    LEAGUE_DRAFT_GRADE_CALIBRATION_CSV,
    LEAGUE_DRAFT_GRADES_CSV,
    LEAGUE_ALL_SEASONS,
    LEAGUE_ROSTERS_CSV,
    LEAGUE_SEASONS,
    LEAGUE_SETTINGS_CSV,
    LEAGUE_TEAMS_CSV,
    LEAGUE_WEEKLY_ODDS_CSV,
    MODELED_POSITIONS,
)
from rookie_ppr.league.draft_reaches import _ascii, load_manager_map
from rookie_ppr.league.simulate_season import (
    N_FLEX,
    PLAYOFF_ROUNDS,
    PLAYOFF_TEAMS,
    SLOT_PLAN,
    _projection_column,
    calibrate_weekly_noise,
    slot_average_stats,
    team_games_and_byes,
    weekly_team_moments,
)

# Letter thresholds on the within-season z-score. Fixed rather than curved so a
# season where everybody drafted well does not manufacture a D.
GRADE_BANDS: tuple[tuple[str, float], ...] = (
    ("A", 0.85),
    ("A-", 0.35),
    ("B", -0.15),
    ("C", -0.85),
    ("D", float("-inf")),
)
GRADE_ORDER = [letter for letter, _ in GRADE_BANDS]

GRADE_COLS = [
    "season",
    "team_id",
    "manager_name",
    "team_name",
    "value_points",
    "value_z",
    "value_grade",
    "roster_points_per_week",
    "roster_z",
    "roster_grade",
    "playoff_odds",
    "title_odds",
    "n_skill_picks",
    "first_pick",
    "best_pick_player",
    "best_pick_position",
    "best_pick_overall",
    "best_pick_adp",
    "best_pick_value",
    "reach_player",
    "reach_position",
    "reach_overall",
    "reach_adp",
    "reach_value",
]

CALIBRATION_COLS = [
    "grade_kind",
    "grade",
    "n_team_seasons",
    "avg_wins",
    "avg_final_rank",
    "playoff_rate",
    "title_rate",
]


def letter_for(z: float) -> str:
    """Map a within-season z-score onto a letter grade."""
    if not np.isfinite(z):
        return "-"
    for letter, floor in GRADE_BANDS:
        if z >= floor:
            return letter
    return GRADE_ORDER[-1]


def replacement_levels(pool: pd.DataFrame, n_teams: int) -> dict[str, float]:
    """Projection of the last startable player at each position.

    Flex slots are shared out across RB/WR/TE in proportion to their base
    starter counts, so the flex does not silently inflate one position.
    """
    base = dict(SLOT_PLAN)
    flexable = {p: c for p, c in base.items() if p in ("RB", "WR", "TE")}
    total_flexable = sum(flexable.values()) or 1
    levels: dict[str, float] = {}
    for position, starters in base.items():
        share = flexable.get(position, 0) / total_flexable
        demand = n_teams * starters + round(n_teams * N_FLEX * share)
        values = (
            pool[pool["position"] == position]["board_value"]
            .dropna()
            .sort_values(ascending=False)
            .to_numpy()
        )
        if len(values) == 0:
            levels[position] = 0.0
        else:
            idx = min(int(demand), len(values)) - 1
            levels[position] = float(values[max(idx, 0)])
    return levels


def _season_picks(rosters: pd.DataFrame, season: int, blend_weight: float) -> pd.DataFrame:
    """Skill picks for one draft, valued over replacement and over the board."""
    picks = rosters[
        (rosters["season"] == season) & rosters["position"].isin(MODELED_POSITIONS)
    ].copy()
    if picks.empty:
        return picks
    picks["board_value"] = _projection_column(
        picks, "blend", blend_weight=blend_weight
    )
    picks = picks[picks["board_value"].notna()].copy()
    if picks.empty:
        return picks

    n_teams = int(picks["team_id"].nunique()) or 1
    levels = replacement_levels(picks, n_teams)
    picks["vorp"] = picks["board_value"] - picks["position"].map(levels).astype(float)

    # The board: every skill player taken, ordered as the market ranked them.
    # Unranked players sit at the back. The kth pick of the draft is measured
    # against the kth best player the board had available.
    picks["adp_sort"] = pd.to_numeric(picks["adp_rank"], errors="coerce").fillna(np.inf)
    board = picks.sort_values(["adp_sort", "vorp"], ascending=[True, False])
    expected = board["vorp"].to_numpy()

    picks = picks.sort_values("overall_pick").reset_index(drop=True)
    picks["board_expected_vorp"] = expected[: len(picks)]
    picks["pick_value"] = picks["vorp"] - picks["board_expected_vorp"]
    return picks


def _roster_strength(
    rosters: pd.DataFrame,
    season: int,
    *,
    reg_weeks: int,
    blend_weight: float,
    league_rosters: pd.DataFrame,
    pro_teams: pd.DataFrame,
) -> dict[int, float]:
    """Projected points per week per team, schedule-free."""
    frame = rosters[rosters["season"] == season]
    if frame.empty:
        return {}
    if "pro_team" not in frame.columns:
        frame = frame.merge(
            pro_teams[pro_teams["season"] == season][["overall_pick", "pro_team"]],
            on="overall_pick",
            how="left",
        )
    games, byes = team_games_and_byes(season)
    noise = calibrate_weekly_noise(league_rosters, before_season=season) or {}
    slots = slot_average_stats(league_rosters, before_season=season) or {}
    weeks = list(range(1, reg_weeks + PLAYOFF_ROUNDS + 1))
    default_games = 17 if season >= 2021 else 16
    out: dict[int, float] = {}
    for team_id, group in frame.groupby("team_id"):
        means, _sds = weekly_team_moments(
            group,
            "blend",
            weeks=weeks,
            games=games,
            byes=byes,
            noise=noise,
            slot_stats=slots,
            default_games=default_games,
            blend_weight=blend_weight,
        )
        out[int(team_id)] = float(np.mean(means[:reg_weeks]))
    return out


def _post_draft_odds(season: int) -> dict[int, tuple[float, float]]:
    """Week-0 playoff and title odds, when the season's odds have been published."""
    path = CSV_OUTPUT_DIR / LEAGUE_WEEKLY_ODDS_CSV
    if not path.exists():
        return {}
    odds = pd.read_csv(path)
    sdf = odds[(odds["season"] == season) & (odds["as_of_week"] == 0)]
    return {
        int(r.team_id): (float(r.playoff_odds), float(r.title_odds))
        for r in sdf.itertuples(index=False)
    }


def _zscore(values: pd.Series) -> pd.Series:
    sd = values.std()
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(0.0, index=values.index)
    return (values - values.mean()) / sd


def build_draft_grades(
    seasons: list[int] | None = None,
    *,
    rosters: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Value and roster grades per team-season.

    ``rosters`` lets a caller grade a draft that is not on disk yet, which is how
    an upcoming season can be graded before its artifacts are committed.
    """
    seasons = list(seasons or LEAGUE_ALL_SEASONS)
    if rosters is None:
        rosters = pd.read_csv(CSV_OUTPUT_DIR / LEAGUE_BACKTEST_ROSTERS_CSV)
    league_rosters = pd.read_csv(CSV_OUTPUT_DIR / LEAGUE_ROSTERS_CSV)
    pro_teams = pd.read_csv(CSV_OUTPUT_DIR / LEAGUE_DRAFT_CSV)[
        ["season", "overall_pick", "pro_team"]
    ]
    settings = pd.read_csv(CSV_OUTPUT_DIR / LEAGUE_SETTINGS_CSV)
    reg_weeks_by_season = dict(zip(settings["season"], settings["reg_weeks"]))
    teams = pd.read_csv(CSV_OUTPUT_DIR / LEAGUE_TEAMS_CSV)[
        ["season", "team_id", "team_name"]
    ]
    weights = blend_weight_map(fit_blend_weights_walkforward(rosters, seasons))

    blocks: list[pd.DataFrame] = []
    for season in seasons:
        blend_weight = float(weights.get(season, DEFAULT_BLEND_MODEL_WEIGHT))
        picks = _season_picks(rosters, season, blend_weight)
        if picks.empty:
            continue
        reg_weeks = int(reg_weeks_by_season.get(season, 14))
        strength = _roster_strength(
            rosters,
            season,
            reg_weeks=reg_weeks,
            blend_weight=blend_weight,
            league_rosters=league_rosters,
            pro_teams=pro_teams,
        )
        odds = _post_draft_odds(season)

        rows: list[dict] = []
        for team_id, group in picks.groupby("team_id"):
            best = group.loc[group["pick_value"].idxmax()]
            worst = group.loc[group["pick_value"].idxmin()]
            playoff, title = odds.get(int(team_id), (np.nan, np.nan))
            rows.append(
                {
                    "season": int(season),
                    "team_id": int(team_id),
                    "value_points": float(group["pick_value"].sum()),
                    "roster_points_per_week": strength.get(int(team_id), np.nan),
                    "playoff_odds": playoff,
                    "title_odds": title,
                    "n_skill_picks": int(len(group)),
                    "first_pick": int(group["overall_pick"].min()),
                    "best_pick_player": _ascii(best["player_name"]),
                    "best_pick_position": best["position"],
                    "best_pick_overall": int(best["overall_pick"]),
                    "best_pick_adp": float(best["adp_sort"])
                    if np.isfinite(best["adp_sort"])
                    else np.nan,
                    "best_pick_value": float(best["pick_value"]),
                    "reach_player": _ascii(worst["player_name"]),
                    "reach_position": worst["position"],
                    "reach_overall": int(worst["overall_pick"]),
                    "reach_adp": float(worst["adp_sort"])
                    if np.isfinite(worst["adp_sort"])
                    else np.nan,
                    "reach_value": float(worst["pick_value"]),
                }
            )
        block = pd.DataFrame(rows)
        block["value_z"] = _zscore(block["value_points"])
        block["roster_z"] = _zscore(block["roster_points_per_week"])
        block["value_grade"] = block["value_z"].map(letter_for)
        block["roster_grade"] = block["roster_z"].map(letter_for)
        blocks.append(block)

    if not blocks:
        return pd.DataFrame(columns=GRADE_COLS)

    out = pd.concat(blocks, ignore_index=True)
    managers = load_manager_map(seasons)
    if not managers.empty:
        out = out.merge(managers, on=["season", "team_id"], how="left")
    else:
        out["manager_name"] = pd.NA
    out = out.merge(teams, on=["season", "team_id"], how="left")
    out["manager_name"] = out["manager_name"].fillna(
        "Team " + out["team_id"].astype(str)
    )
    out["team_name"] = out["team_name"].map(lambda v: _ascii(v) if pd.notna(v) else "")
    out = out.sort_values(["season", "value_points"], ascending=[True, False])
    return out[GRADE_COLS].reset_index(drop=True)


def build_calibration(grades: pd.DataFrame) -> pd.DataFrame:
    """What each letter has actually been worth, across seasons that finished.

    This is the honesty check that ships with the grade: it is what lets a
    reader see that the letters separate the bottom of the league from the top
    but cannot finely rank the teams inside the top.
    """
    teams = pd.read_csv(CSV_OUTPUT_DIR / LEAGUE_TEAMS_CSV)[
        ["season", "team_id", "wins", "final_rank", "playoff_seed"]
    ]
    frame = grades.merge(teams, on=["season", "team_id"], how="inner")
    # A season under way reports zero wins and no final rank, which would read as
    # a genuinely terrible outcome, so calibrate only on seasons that finished.
    frame = frame[frame["season"].isin(LEAGUE_SEASONS)]
    frame = frame.dropna(subset=["wins", "final_rank"])
    frame = frame[frame["final_rank"] >= 1]
    if frame.empty:
        return pd.DataFrame(columns=CALIBRATION_COLS)
    frame["made_playoffs"] = (frame["playoff_seed"] <= PLAYOFF_TEAMS).astype(float)
    frame["was_champion"] = (frame["final_rank"] == 1).astype(float)

    rows: list[dict] = []
    for kind, column in (("value", "value_grade"), ("roster", "roster_grade")):
        for grade, group in frame.groupby(column):
            rows.append(
                {
                    "grade_kind": kind,
                    "grade": grade,
                    "n_team_seasons": int(len(group)),
                    "avg_wins": round(float(group["wins"].mean()), 2),
                    "avg_final_rank": round(float(group["final_rank"].mean()), 2),
                    "playoff_rate": round(float(group["made_playoffs"].mean()), 3),
                    "title_rate": round(float(group["was_champion"].mean()), 3),
                }
            )
    out = pd.DataFrame(rows)
    out["_order"] = out["grade"].map({g: i for i, g in enumerate(GRADE_ORDER)})
    out = out.sort_values(["grade_kind", "_order"]).drop(columns="_order")
    return out[CALIBRATION_COLS].reset_index(drop=True)


def refresh_draft_grades(
    seasons: list[int] | None = None,
    *,
    rosters: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Grade ``seasons`` and rewrite both artifacts, preserving other seasons.

    Returns the freshly graded rows and the calibration table, which always
    spans every graded season on disk rather than just this run.
    """
    seasons = list(seasons or LEAGUE_ALL_SEASONS)
    grades = build_draft_grades(seasons, rosters=rosters)
    if grades.empty:
        return grades, pd.DataFrame(columns=CALIBRATION_COLS)

    path = CSV_OUTPUT_DIR / LEAGUE_DRAFT_GRADES_CSV
    merged = grades
    if path.exists():
        prior = pd.read_csv(path)
        if not prior.empty and "season" in prior.columns:
            kept = prior[~prior["season"].isin(seasons)]
            merged = pd.concat([kept, grades], ignore_index=True)
    merged = merged.sort_values(["season", "value_points"], ascending=[True, False])
    CSV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(path, index=False)

    calibration = build_calibration(merged)
    calibration.to_csv(CSV_OUTPUT_DIR / LEAGUE_DRAFT_GRADE_CALIBRATION_CSV, index=False)
    return grades, calibration


def main() -> None:
    parser = argparse.ArgumentParser(description="Draft value and roster grades")
    parser.add_argument("--seasons", type=int, nargs="+", default=list(LEAGUE_ALL_SEASONS))
    args = parser.parse_args()

    grades, calibration = refresh_draft_grades(args.seasons)
    if grades.empty:
        raise SystemExit("no drafts to grade for those seasons")

    print(
        f"graded {len(grades)} team-seasons "
        f"({grades['season'].nunique()} drafts) -> {LEAGUE_DRAFT_GRADES_CSV}"
    )
    print(
        grades[
            [
                "season",
                "manager_name",
                "value_points",
                "value_grade",
                "roster_points_per_week",
                "roster_grade",
            ]
        ]
        .tail(10)
        .round(1)
        .to_string(index=False)
    )
    print(f"\ncalibration -> {LEAGUE_DRAFT_GRADE_CALIBRATION_CSV}")
    print(calibration.to_string(index=False))


if __name__ == "__main__":
    main()
