"""Regenerate weekly odds, injury events, and trade events for ForeKast.

Requires ESPN cookies in ``.env`` only when refreshing live roster weeks.
Historical CSVs already ship under ``data/output/csv/``.
"""
from __future__ import annotations

import argparse

from rookie_ppr.api.csv_store import clear_cache
from rookie_ppr.league.config import LEAGUE_SEASONS
from rookie_ppr.league.injury_events import detect_injury_events
from rookie_ppr.league.trade_events import detect_trade_events
from rookie_ppr.league.weekly_odds import run_weekly_odds
from rookie_ppr.league.config import (
    CSV_OUTPUT_DIR,
    LEAGUE_INJURY_EVENTS_CSV,
    LEAGUE_TRADE_EVENTS_CSV,
    LEAGUE_WEEKLY_ODDS_CSV,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refresh ForeKast weekly odds + injury/trade event CSVs"
    )
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        default=None,
        help="Seasons to rebuild (default: all LEAGUE_SEASONS)",
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
    args = parser.parse_args()
    seasons = list(args.seasons or LEAGUE_SEASONS)

    if not args.skip_odds:
        print(f"weekly odds seasons={seasons} ...")
        odds = run_weekly_odds(seasons=seasons)
        path = CSV_OUTPUT_DIR / LEAGUE_WEEKLY_ODDS_CSV
        CSV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        odds.to_csv(path, index=False)
        print(f"  -> {path} rows={len(odds)}")
        if not odds.empty:
            latest = (
                odds.sort_values(["season", "as_of_week"])
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
        inj.to_csv(ipath, index=False)
        print(f"  -> {ipath} rows={len(inj)}")

        print(f"trade events seasons={seasons} ...")
        trades = detect_trade_events(seasons)
        tpath = CSV_OUTPUT_DIR / LEAGUE_TRADE_EVENTS_CSV
        trades.to_csv(tpath, index=False)
        print(f"  -> {tpath} rows={len(trades)}")

    clear_cache()
    print("CSV cache cleared (API will reload on next request).")


if __name__ == "__main__":
    main()
