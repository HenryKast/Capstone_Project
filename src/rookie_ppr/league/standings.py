"""Standings and playoff seeding for this league.

The league settings advertise ``playoffSeedingRule=TOTAL_POINTS_SCORED``, which
is misleading: seeds follow neither pure points nor pure record. The rule that
actually reproduces ESPN's seeds in every season 2018-2025 is division winners
first, then everyone else, each group ordered by wins with points scored as the
tiebreak.
"""
from __future__ import annotations

import pandas as pd

SEED_SORT_COLS = ["is_division_winner", "wins", "points_for"]


def _division_winners(teams: pd.DataFrame) -> pd.Series:
    """True for the best record (points tiebreak) in each division."""
    ranked = teams.sort_values(["division_id", "wins", "points_for"], ascending=[True, False, False])
    winner_ids = ranked.groupby("division_id", sort=True)["team_id"].first()
    return teams["team_id"].isin(set(winner_ids))


def assign_playoff_seeds(teams: pd.DataFrame) -> pd.DataFrame:
    """Add ``is_division_winner`` and ``seed`` for one season of teams.

    Expects ``team_id``, ``division_id``, ``wins``, ``points_for``.
    """
    out = teams.copy()
    out["is_division_winner"] = _division_winners(out)
    out = out.sort_values(SEED_SORT_COLS, ascending=[False, False, False]).reset_index(drop=True)
    out["seed"] = range(1, len(out) + 1)
    return out


def validate_seeding_rule(teams: pd.DataFrame) -> pd.DataFrame:
    """Compare derived seeds to ESPN's own, one row per season.

    Guards the assumption above: if a future season changes divisions or
    tiebreaks, ``exact_match`` goes False and the simulator needs revisiting.
    """
    rows: list[dict] = []
    for season, group in teams.groupby("season", sort=True):
        actual = group.dropna(subset=["playoff_seed"])
        if actual.empty:
            rows.append({"season": int(season), "exact_match": None, "top6_match": None})
            continue
        derived = assign_playoff_seeds(group)
        actual_order = actual.sort_values("playoff_seed")["team_id"].tolist()
        derived_order = derived.sort_values("seed")["team_id"].tolist()
        playoff_n = int(actual["playoff_seed"].notna().sum())
        cutoff = min(6, playoff_n)
        rows.append(
            {
                "season": int(season),
                "exact_match": derived_order == actual_order,
                "top6_match": set(derived_order[:cutoff]) == set(actual_order[:cutoff]),
            }
        )
    return pd.DataFrame(rows)


def regular_season_standings(matchups: pd.DataFrame, reg_weeks: dict[int, int]) -> pd.DataFrame:
    """Rebuild wins/losses/ties and points from regular-season matchup rows.

    Lets the simulator score a hypothetical season with the same code path that
    reproduces real history.
    """
    df = matchups.copy()
    df["reg_weeks"] = df["season"].map(reg_weeks)
    df = df[df["week"] <= df["reg_weeks"]]

    df["win"] = (df["points"] > df["opp_points"]).astype(int)
    df["loss"] = (df["points"] < df["opp_points"]).astype(int)
    df["tie"] = (df["points"] == df["opp_points"]).astype(int)

    grouped = (
        df.groupby(["season", "team_id"], as_index=False)
        .agg(
            wins=("win", "sum"),
            losses=("loss", "sum"),
            ties=("tie", "sum"),
            points_for=("points", "sum"),
            points_against=("opp_points", "sum"),
        )
        .sort_values(["season", "wins", "points_for"], ascending=[True, False, False])
    )
    return grouped.reset_index(drop=True)


def reconcile_starter_points(
    rosters: pd.DataFrame, matchups: pd.DataFrame, *, tolerance: float = 0.05
) -> pd.DataFrame:
    """Per season, do summed starter points equal ESPN's team score?

    The score ESPN publishes includes any commissioner ``adjustment``, which no
    player earned, so that comes back off before comparing. A season below 100%
    means the roster parse is dropping or misassigning scoring players.
    """
    if rosters.empty:
        return pd.DataFrame(columns=["season", "team_weeks", "pct_match", "max_abs_diff"])

    summed = (
        rosters[rosters["is_starter"]]
        .groupby(["season", "week", "team_id"], as_index=False)["actual_points"]
        .sum()
        .rename(columns={"actual_points": "starter_points"})
    )
    cols = ["season", "week", "team_id", "points"]
    if "adjustment" in matchups.columns:
        cols.append("adjustment")
    merged = summed.merge(matchups[cols], on=["season", "week", "team_id"], how="inner")
    merged["expected"] = merged["points"] - merged.get("adjustment", 0.0).fillna(0.0)
    merged["diff"] = merged["starter_points"] - merged["expected"]
    merged["ok"] = merged["diff"].abs() <= tolerance

    return (
        merged.groupby("season", as_index=False)
        .agg(
            team_weeks=("ok", "size"),
            pct_match=("ok", "mean"),
            max_abs_diff=("diff", lambda s: round(s.abs().max(), 2)),
        )
        .sort_values("season")
        .reset_index(drop=True)
    )


__all__ = [
    "assign_playoff_seeds",
    "reconcile_starter_points",
    "regular_season_standings",
    "validate_seeding_rule",
]
