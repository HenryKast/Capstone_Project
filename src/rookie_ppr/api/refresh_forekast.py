"""Regenerate weekly odds, injury events, trade events, and draft grades.

Requires ESPN cookies in ``.env`` only when refreshing live roster weeks.
Historical CSVs already ship under ``data/output/csv/``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from rookie_ppr.api.csv_store import clear_cache
from rookie_ppr.league.config import (
    CSV_OUTPUT_DIR,
    LEAGUE_ALL_SEASONS,
    LEAGUE_BACKTEST_ROSTERS_CSV,
    LEAGUE_DRAFT_CSV,
    LEAGUE_DRAFT_GRADE_CALIBRATION_CSV,
    LEAGUE_DRAFT_GRADES_CSV,
    LEAGUE_INJURY_EVENTS_CSV,
    LEAGUE_SETTINGS_CSV,
    LEAGUE_TRADE_EVENTS_CSV,
    LEAGUE_WEEKLY_ODDS_CSV,
)
from rookie_ppr.league.draft_grades import refresh_draft_grades
from rookie_ppr.league.draft_reaches import export_league_managers
from rookie_ppr.league.injury_events import detect_injury_events
from rookie_ppr.league.target_season import build_target_rosters
from rookie_ppr.league.trade_events import detect_trade_events
from rookie_ppr.league.weekly_odds import run_weekly_odds


def _write_merged(
    path: Path, fresh: pd.DataFrame, seasons: list[int], sort_by: list[str]
) -> pd.DataFrame:
    """Replace only ``seasons`` in ``path``, keeping every other season intact.

    A scoped refresh (e.g. just the upcoming season) must not drop the history
    the API serves for earlier seasons.
    """
    CSV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = fresh
    if path.exists():
        prior = pd.read_csv(path)
        if not prior.empty and "season" in prior.columns:
            kept = prior[~prior["season"].isin(seasons)]
            out = pd.concat([kept, fresh], ignore_index=True) if not fresh.empty else kept
    if not out.empty:
        cols = [c for c in sort_by if c in out.columns]
        if cols:
            out = out.sort_values(cols).reset_index(drop=True)
    out.to_csv(path, index=False)
    return out


def _prepare_seasons(seasons: list[int]) -> list[int]:
    """Drop seasons that were never ingested, and build any missing rosters.

    The in-season refresh runs unattended, so it should ingest-check itself
    rather than fail deep inside the simulation. Drafted rosters are the one
    upstream artifact the odds need that no refresh step rebuilds, and they are
    schedule-free, so building them here is safe at any point in the season.
    """
    settings = pd.read_csv(CSV_OUTPUT_DIR / LEAGUE_SETTINGS_CSV)
    ingested = set(settings["season"].astype(int))
    ready = [s for s in seasons if s in ingested]
    for missing in sorted(set(seasons) - ingested):
        print(
            f"  skipping {missing}: not in {LEAGUE_SETTINGS_CSV} "
            f"(run: python -m rookie_ppr.league.ingest --seasons {missing})"
        )

    draft = pd.read_csv(CSV_OUTPUT_DIR / LEAGUE_DRAFT_CSV)
    backtest_path = CSV_OUTPUT_DIR / LEAGUE_BACKTEST_ROSTERS_CSV
    have = (
        set(pd.read_csv(backtest_path, usecols=["season"])["season"].astype(int))
        if backtest_path.exists()
        else set()
    )
    for season in ready:
        if season in have or not (draft["season"] == season).any():
            continue
        print(f"  building {season} drafted rosters (missing from {LEAGUE_BACKTEST_ROSTERS_CSV}) ...")
        build_target_rosters(season)
    return ready


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Refresh ForeKast weekly odds, injury/trade event CSVs, and draft grades"
        )
    )
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        default=None,
        help="Seasons to rebuild (default: every season, including the one under way)",
    )
    parser.add_argument(
        "--skip-odds",
        action="store_true",
        help="Only rebuild injury/trade event tables",
    )
    parser.add_argument(
        "--skip-events",
        action="store_true",
        help="Only rebuild weekly odds",
    )
    parser.add_argument(
        "--skip-managers",
        action="store_true",
        help="Do not rebuild league_managers.csv",
    )
    parser.add_argument(
        "--skip-grades",
        action="store_true",
        help="Do not rebuild draft grades",
    )
    args = parser.parse_args()
    requested = list(args.seasons or LEAGUE_ALL_SEASONS)
    seasons = _prepare_seasons(requested)
    if not seasons:
        raise SystemExit(f"none of {requested} are ingested; nothing to refresh")

    if not args.skip_managers:
        print(f"league managers seasons={seasons} ...")
        mgr = export_league_managers(seasons)
        print(f"  -> rows={len(mgr)}")

    if not args.skip_odds:
        print(f"weekly odds seasons={seasons} ...")
        odds = run_weekly_odds(seasons=seasons)
        path = CSV_OUTPUT_DIR / LEAGUE_WEEKLY_ODDS_CSV
        merged = _write_merged(
            path, odds, seasons, ["season", "as_of_week", "team_id"]
        )
        print(f"  -> {path} rows={len(merged)} (refreshed {len(odds)})")
        if not merged.empty:
            latest = (
                merged.sort_values(["season", "as_of_week"])
                .groupby("season")["as_of_week"]
                .max()
            )
            print("  latest as_of_week by season:")
            for season, week in latest.items():
                print(f"    {int(season)}: W{int(week)}")

    if not args.skip_events:
        print(f"injury events seasons={seasons} ...")
        inj = detect_injury_events(seasons)
        ipath = CSV_OUTPUT_DIR / LEAGUE_INJURY_EVENTS_CSV
        imerged = _write_merged(ipath, inj, seasons, ["season", "as_of_week"])
        print(f"  -> {ipath} rows={len(imerged)} (refreshed {len(inj)})")

        print(f"trade events seasons={seasons} ...")
        trades = detect_trade_events(seasons)
        tpath = CSV_OUTPUT_DIR / LEAGUE_TRADE_EVENTS_CSV
        tmerged = _write_merged(tpath, trades, seasons, ["season", "as_of_week"])
        print(f"  -> {tpath} rows={len(tmerged)} (refreshed {len(trades)})")

    if not args.skip_grades:
        # After odds, so a freshly published week 0 lands on the grade rows.
        print(f"draft grades seasons={seasons} ...")
        grades, calibration = refresh_draft_grades(seasons)
        if grades.empty:
            print("  -> no drafted rosters for those seasons; skipped")
        else:
            print(f"  -> {LEAGUE_DRAFT_GRADES_CSV} refreshed {len(grades)} team-seasons")
            print(f"  -> {LEAGUE_DRAFT_GRADE_CALIBRATION_CSV} rows={len(calibration)}")

    clear_cache()
    print("CSV cache cleared (API will reload on next request).")


if __name__ == "__main__":
    main()
