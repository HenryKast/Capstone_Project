"""Post-draft rosters per season, with a season projection attached to every pick.

The honest pre-season view of a team is the roster it drafted, so that is what
gets projected here. Waivers and trades are how the season actually unfolded and
would leak outcome information into a forecast.

Most picks get their number from the walk-forward veteran model. The ones that
can't are drafted rookies and players who missed the prior season entirely, who
have no prior-season features to model from. Those fall back to a market curve:
expected season points as a function of FantasyPros ADP rank, fit only on
seasons strictly before the one being projected. ADP is pre-season information,
so the fallback stays leakage-free, and it also doubles as a "what did the market
alone predict" baseline to grade the model against.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from rookie_ppr.ingest_fantasypros_adp import load_fantasypros_adp
from rookie_ppr.league.config import (
    CSV_OUTPUT_DIR,
    LEAGUE_BACKTEST_ROSTERS_CSV,
    LEAGUE_DRAFT_CSV,
    LEAGUE_SEASONS,
    LEAGUE_WF_PROJECTIONS_CSV,
    MODELED_POSITIONS,
    UNMODELED_SLOTS,
)
from rookie_ppr.utils import normalize_name
from rookie_ppr.veteran.config import VET_FEATURES_CSV

ROSTER_COLS = [
    "season",
    "team_id",
    "overall_pick",
    "round",
    "gsis_id",
    "player_name",
    "position",
    "adp_rank",
    "proj_source",
    "proj_season_ppr",
    "adp_curve_ppr",
    "actual_season_ppr",
]

# An unranked pick is deeper than any ranked player; treat it as this rank so the
# curve still returns a sane replacement-level number instead of extrapolating.
UNRANKED_ADP_RANK = 250.0
MIN_CURVE_ROWS = 30


def _actual_season_ppr() -> pd.DataFrame:
    """(gsis_id, season) -> actual PPR. nflverse PPR matches this league's scoring."""
    path = CSV_OUTPUT_DIR / VET_FEATURES_CSV
    frame = pd.read_csv(
        path, low_memory=False, usecols=["gsis_id", "player_name", "position", "season", "ppr"]
    )
    frame["season"] = pd.to_numeric(frame["season"], errors="coerce")
    frame["ppr"] = pd.to_numeric(frame["ppr"], errors="coerce")
    out = frame.dropna(subset=["gsis_id", "season"]).drop_duplicates(
        subset=["gsis_id", "season"]
    )
    out["name_norm"] = out["player_name"].map(normalize_name)
    return out


def _adp_with_actuals(actuals: pd.DataFrame) -> pd.DataFrame:
    """ADP rows joined to the season each player actually produced.

    A ranked player with no stat row that season really did score nothing, so
    those become zeros rather than being dropped, which is what keeps the curve
    honest about bust risk. Names that never appear anywhere in the panel are
    match failures, not zeros, and are discarded.
    """
    adp = load_fantasypros_adp()
    if adp.empty:
        return pd.DataFrame(columns=["season", "position", "adp_rank", "actual_ppr"])

    adp = adp.dropna(subset=["overall_rank"]).copy()
    adp["adp_rank"] = pd.to_numeric(adp["overall_rank"], errors="coerce")

    known_names = set(actuals["name_norm"])
    by_name_season = actuals.set_index(["name_norm", "season"])["ppr"].to_dict()

    rows: list[dict] = []
    for row in adp.itertuples(index=False):
        name = getattr(row, "player_name_norm", "")
        if name not in known_names:
            continue
        season = int(row.season)
        actual = by_name_season.get((name, season))
        rows.append(
            {
                "season": season,
                "position": row.position,
                "adp_rank": float(row.adp_rank),
                "actual_ppr": 0.0 if actual is None or pd.isna(actual) else float(actual),
            }
        )
    return pd.DataFrame(rows)


def fit_adp_curves(adp_actuals: pd.DataFrame, before_season: int) -> dict[str, np.ndarray]:
    """Per position, expected PPR as a linear function of log(ADP rank).

    Fit on seasons strictly before ``before_season``. Log rank because value
    falls off steeply at the top of a draft and flattens deep in it.
    """
    pool = adp_actuals[adp_actuals["season"] < before_season]
    curves: dict[str, np.ndarray] = {}
    for position in MODELED_POSITIONS:
        pos = pool[(pool["position"] == position) & (pool["adp_rank"] > 0)]
        if len(pos) < MIN_CURVE_ROWS:
            continue
        x = np.log(pos["adp_rank"].to_numpy(dtype=float))
        y = pos["actual_ppr"].to_numpy(dtype=float)
        curves[position] = np.polyfit(x, y, 1)
    return curves


def adp_curve_value(curves: dict[str, np.ndarray], position: str, adp_rank: float) -> float:
    coeffs = curves.get(position)
    if coeffs is None or not np.isfinite(adp_rank) or adp_rank <= 0:
        return float("nan")
    return float(max(0.0, np.polyval(coeffs, np.log(adp_rank))))


def _attach_adp_rank(draft: pd.DataFrame) -> pd.DataFrame:
    """Best FantasyPros overall rank for each pick, matched on normalized name."""
    adp = load_fantasypros_adp()
    out = draft.copy()
    out["name_norm"] = out["player_name"].map(normalize_name)
    out["adp_rank"] = np.nan
    if adp.empty:
        return out

    ranked = adp.dropna(subset=["overall_rank"]).copy()
    lookup = (
        ranked.sort_values("overall_rank")
        .drop_duplicates(subset=["season", "player_name_norm"])
        .set_index(["season", "player_name_norm"])["overall_rank"]
        .to_dict()
    )
    out["adp_rank"] = [
        lookup.get((int(season), name), np.nan)
        for season, name in zip(out["season"], out["name_norm"])
    ]
    return out


def build_backtest_rosters(
    seasons: list[int] | None = None,
    *,
    projections_csv: str | None = None,
) -> pd.DataFrame:
    seasons = list(seasons or LEAGUE_SEASONS)

    draft = pd.read_csv(CSV_OUTPUT_DIR / LEAGUE_DRAFT_CSV)
    draft = draft[draft["season"].isin(seasons)].copy()
    proj_path = CSV_OUTPUT_DIR / (projections_csv or LEAGUE_WF_PROJECTIONS_CSV)
    projections = pd.read_csv(proj_path)
    actuals = _actual_season_ppr()
    adp_actuals = _adp_with_actuals(actuals)

    draft = _attach_adp_rank(draft)

    model_lookup = {
        (int(s), str(g)): float(p)
        for s, g, p in zip(
            projections["target_season"], projections["gsis_id"], projections["predicted_ppr"]
        )
        if pd.notna(g) and pd.notna(p)
    }
    actual_lookup = {
        (str(g), int(s)): float(v)
        for g, s, v in zip(actuals["gsis_id"], actuals["season"], actuals["ppr"])
        if pd.notna(v)
    }
    # A season with no stat rows at all has not been played. Its picks get NaN
    # rather than 0.0, so an upcoming season never looks like a league-wide bust
    # to the blend fit or the grader.
    played_seasons = {int(s) for s in actuals["season"].dropna().unique()}

    rows: list[dict] = []
    for season in seasons:
        curves = fit_adp_curves(adp_actuals, before_season=season)
        for pick in draft[draft["season"] == season].itertuples(index=False):
            position = pick.position
            gsis = pick.gsis_id if pd.notna(pick.gsis_id) else None
            adp_rank = float(pick.adp_rank) if pd.notna(pick.adp_rank) else np.nan

            if position in UNMODELED_SLOTS:
                # Kickers and defenses are simulated from league averages, so
                # they intentionally carry no player-level projection.
                proj, source, curve_val = np.nan, "slot_average", np.nan
            else:
                effective_rank = adp_rank if np.isfinite(adp_rank) else UNRANKED_ADP_RANK
                curve_val = adp_curve_value(curves, position, effective_rank)
                model_proj = model_lookup.get((season, str(gsis))) if gsis else None
                if model_proj is not None:
                    proj, source = model_proj, "model"
                else:
                    proj, source = curve_val, "adp_curve"

            rows.append(
                {
                    "season": season,
                    "team_id": pick.team_id,
                    "overall_pick": pick.overall_pick,
                    "round": getattr(pick, "round"),
                    "gsis_id": gsis,
                    "player_name": pick.player_name,
                    "position": position,
                    "adp_rank": adp_rank,
                    "proj_source": source,
                    "proj_season_ppr": proj,
                    "adp_curve_ppr": curve_val,
                    "actual_season_ppr": actual_lookup.get((str(gsis), season), 0.0)
                    if gsis
                    and position in MODELED_POSITIONS
                    and season in played_seasons
                    else np.nan,
                }
            )

    return pd.DataFrame(rows, columns=ROSTER_COLS)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build post-draft rosters with walk-forward season projections"
    )
    parser.add_argument("--seasons", type=int, nargs="+", default=list(LEAGUE_SEASONS))
    parser.add_argument(
        "--projections",
        default=LEAGUE_WF_PROJECTIONS_CSV,
        help="Walk-forward projections CSV under data/output/csv",
    )
    args = parser.parse_args()

    rosters = build_backtest_rosters(args.seasons, projections_csv=args.projections)
    CSV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = CSV_OUTPUT_DIR / LEAGUE_BACKTEST_ROSTERS_CSV
    rosters.to_csv(path, index=False)
    print(f"rows={len(rosters)} -> {path}")

    skill = rosters[rosters["position"].isin(MODELED_POSITIONS)]
    print("\nprojection source by season:")
    print(
        skill.pivot_table(
            index="season", columns="proj_source", values="overall_pick", aggfunc="count"
        )
        .fillna(0)
        .astype(int)
        .to_string()
    )

    print("\nADP rank matched on skill picks: "
          f"{skill['adp_rank'].notna().mean():.1%}")

    graded = skill.dropna(subset=["proj_season_ppr", "actual_season_ppr"])
    print(f"\ncorrelation of pick-level projection vs actual (n={len(graded)}):")
    for source, group in graded.groupby("proj_source"):
        if len(group) < 5:
            continue
        r = np.corrcoef(group["proj_season_ppr"], group["actual_season_ppr"])[0, 1]
        print(f"  {source:<10} n={len(group):<4} r={r:.4f}")
    r_all = np.corrcoef(graded["proj_season_ppr"], graded["actual_season_ppr"])[0, 1]
    curve_only = graded.dropna(subset=["adp_curve_ppr"])
    r_curve = np.corrcoef(curve_only["adp_curve_ppr"], curve_only["actual_season_ppr"])[0, 1]
    print(f"  {'combined':<10} n={len(graded):<4} r={r_all:.4f}")
    print(f"  {'adp-only':<10} n={len(curve_only):<4} r={r_curve:.4f}  (market baseline)")


if __name__ == "__main__":
    main()
