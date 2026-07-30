"""Head-to-head: our season-pace weekly projections vs ESPN's stored weekly projections.

For every skill-position starter week in the league history we have three numbers:
the player's actual fantasy points, ESPN's projection for that week, and our own
season projection converted to a weekly pace (season_proj / games_in_season).

ESPN's number is week-specific. Ours is a flat season pace with no matchup
adjustment here, so this is a conservative test of the season model rather than
a claim that our weekly module is ready. If the flat pace still competes, the
season signal is real; if it loses badly, ESPN's week-level information is
doing work we do not yet match.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from rookie_ppr.league.backtest_rosters import LEAGUE_BACKTEST_ROSTERS_CSV
from rookie_ppr.league.config import (
    CSV_OUTPUT_DIR,
    LEAGUE_ROSTERS_CSV,
    LEAGUE_SEASONS,
    LEAGUE_WF_PROJECTIONS_CSV,
    MODELED_POSITIONS,
)

LEAGUE_ESPN_H2H_CSV = "league_espn_weekly_h2h.csv"
LEAGUE_ESPN_H2H_GRADES_CSV = "league_espn_weekly_h2h_grades.csv"


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 5 or np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).mean())


def _load_season_proj(filename: str, col_name: str) -> pd.DataFrame:
    path = CSV_OUTPUT_DIR / filename
    if not path.exists():
        return pd.DataFrame(columns=["season", "gsis_id", col_name])
    frame = pd.read_csv(path)[["target_season", "gsis_id", "predicted_ppr"]].rename(
        columns={"target_season": "season", "predicted_ppr": col_name}
    )
    return frame.dropna(subset=["gsis_id", col_name]).drop_duplicates(
        subset=["season", "gsis_id"]
    )


def build_weekly_h2h(
    seasons: list[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    seasons = list(seasons or LEAGUE_SEASONS)
    rosters = pd.read_csv(CSV_OUTPUT_DIR / LEAGUE_ROSTERS_CSV)
    drafted = pd.read_csv(CSV_OUTPUT_DIR / LEAGUE_BACKTEST_ROSTERS_CSV)

    skill = rosters[
        rosters["season"].isin(seasons)
        & rosters["is_starter"]
        & rosters["position"].isin(MODELED_POSITIONS)
        & rosters["actual_points"].notna()
        & rosters["espn_projected_points"].notna()
        & rosters["gsis_id"].notna()
    ].copy()

    market = drafted[
        drafted["position"].isin(MODELED_POSITIONS) & drafted["adp_curve_ppr"].notna()
    ][["season", "gsis_id", "adp_curve_ppr"]].drop_duplicates(subset=["season", "gsis_id"])

    base = _load_season_proj(LEAGUE_WF_PROJECTIONS_CSV, "base_season_ppr")
    adp_opp = _load_season_proj("league_wf_projections_adp_opp.csv", "adp_opp_season_ppr")

    merged = skill.merge(base, on=["season", "gsis_id"], how="inner")
    merged = merged.merge(adp_opp, on=["season", "gsis_id"], how="left")
    merged = merged.merge(market, on=["season", "gsis_id"], how="left")

    merged["team_games"] = np.where(merged["season"] >= 2021, 17, 16)
    merged["espn"] = merged["espn_projected_points"]
    merged["our_base_pace"] = merged["base_season_ppr"] / merged["team_games"]
    merged["our_adp_opp_pace"] = merged["adp_opp_season_ppr"] / merged["team_games"]
    merged["market_pace"] = merged["adp_curve_ppr"] / merged["team_games"]
    has_market = merged["adp_curve_ppr"].notna()
    blended = 0.35 * merged["base_season_ppr"] + 0.65 * merged["adp_curve_ppr"]
    merged["blend_pace"] = np.where(
        has_market, blended / merged["team_games"], merged["our_base_pace"]
    )

    h2h = merged[
        [
            "season",
            "week",
            "team_id",
            "gsis_id",
            "player_name",
            "position",
            "lineup_slot",
            "actual_points",
            "espn",
            "our_base_pace",
            "our_adp_opp_pace",
            "market_pace",
            "blend_pace",
        ]
    ].copy()

    sources = (
        ("espn", "espn"),
        ("our_base_pace", "our_base_pace"),
        ("our_adp_opp_pace", "our_adp_opp_pace"),
        ("market_pace", "market_pace"),
        ("blend_pace", "blend_pace"),
    )

    grade_rows: list[dict] = []
    for label, col in sources:
        g = h2h.dropna(subset=[col, "actual_points"])
        if g.empty:
            continue
        y = g["actual_points"].to_numpy(dtype=float)
        p = g[col].to_numpy(dtype=float)
        grade_rows.append(
            {
                "scope": "overall",
                "position": "",
                "source": label,
                "n": len(g),
                "pearson_r": round(_pearson(p, y), 4),
                "mae": round(_mae(p, y), 3),
                "bias": round(float((p - y).mean()), 3),
            }
        )
        for pos, grp in g.groupby("position"):
            yy = grp["actual_points"].to_numpy(dtype=float)
            pp = grp[col].to_numpy(dtype=float)
            grade_rows.append(
                {
                    "scope": "by_position",
                    "position": str(pos),
                    "source": label,
                    "n": len(grp),
                    "pearson_r": round(_pearson(pp, yy), 4),
                    "mae": round(_mae(pp, yy), 3),
                    "bias": round(float((pp - yy).mean()), 3),
                }
            )

    for season, grp in h2h.groupby("season"):
        for label, col in (
            ("espn", "espn"),
            ("blend_pace", "blend_pace"),
            ("our_adp_opp_pace", "our_adp_opp_pace"),
        ):
            g = grp.dropna(subset=[col, "actual_points"])
            if len(g) < 20:
                continue
            y = g["actual_points"].to_numpy(dtype=float)
            p = g[col].to_numpy(dtype=float)
            grade_rows.append(
                {
                    "scope": f"season_{int(season)}",
                    "position": "",
                    "source": label,
                    "n": len(g),
                    "pearson_r": round(_pearson(p, y), 4),
                    "mae": round(_mae(p, y), 3),
                    "bias": round(float((p - y).mean()), 3),
                }
            )

    return h2h, pd.DataFrame(grade_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare our weekly pace projections to ESPN's stored weekly projections"
    )
    parser.add_argument("--seasons", type=int, nargs="+", default=list(LEAGUE_SEASONS))
    args = parser.parse_args()

    h2h, grades = build_weekly_h2h(args.seasons)
    CSV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    h2h.to_csv(CSV_OUTPUT_DIR / LEAGUE_ESPN_H2H_CSV, index=False)
    grades.to_csv(CSV_OUTPUT_DIR / LEAGUE_ESPN_H2H_GRADES_CSV, index=False)

    print(f"player-weeks={len(h2h)} -> {CSV_OUTPUT_DIR / LEAGUE_ESPN_H2H_CSV}")
    print("\noverall:")
    print(grades[grades["scope"] == "overall"].to_string(index=False))
    print("\nby position:")
    print(grades[grades["scope"] == "by_position"].to_string(index=False))
    print("\nby season:")
    print(grades[grades["scope"].str.startswith("season_")].to_string(index=False))


if __name__ == "__main__":
    main()
