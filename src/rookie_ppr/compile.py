from __future__ import annotations

import sys
from pathlib import Path

# Allow `python -m rookie_ppr.compile` from repo without editable install
_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rookie_ppr.config import DRAFT_YEAR_MAX, DRAFT_YEAR_MIN, ensure_directories
from rookie_ppr.export_outputs import export_workbook_and_csvs
from rookie_ppr.features_sos import compute_team_sos
from rookie_ppr.ingest_college import load_college_production
from rookie_ppr.ingest_fantasypros_adp import load_fantasypros_adp
from rookie_ppr.ingest_nfl import (
    build_rookie_fantasy,
    load_combine,
    load_draft_picks,
    load_player_season_stats,
    load_rosters,
    load_schedules,
    load_team_season_stats,
)
from rookie_ppr.ingest_opportunity import build_landing_opportunity, build_team_offensive_environment
from rookie_ppr.ingest_recruiting import load_recruiting, write_recruiting_template
from rookie_ppr.join_players import build_tables


def main(fetch_on3: bool = False) -> int:
    ensure_directories()
    write_recruiting_template()

    if fetch_on3:
        from rookie_ppr.fetch_on3_recruiting import main as fetch_main

        print("Fetching On3 recruiting rankings ...")
        fetch_main()

    seasons = list(range(DRAFT_YEAR_MIN, DRAFT_YEAR_MAX + 1))
    print(f"Loading draft picks {DRAFT_YEAR_MIN}-{DRAFT_YEAR_MAX} ...")
    draft = load_draft_picks()
    print(f"  draft skill players: {len(draft)}")

    print("Loading rosters ...")
    rosters = load_rosters(seasons)
    print(f"  roster rows: {len(rosters)}")

    print("Loading player season stats ...")
    stats = load_player_season_stats(seasons)
    print(f"  stat rows: {len(stats)}")

    print("Building rookie fantasy targets ...")
    fantasy = build_rookie_fantasy(draft, stats, rosters)
    print(f"  fantasy rows: {len(fantasy)}")

    print("Loading recruiting CSVs (if present) ...")
    recruiting = load_recruiting()
    print(f"  recruiting rows: {len(recruiting)}")

    print("Loading combine ...")
    combine = load_combine()
    print(f"  combine rows: {len(combine)}")

    print("Loading schedules / SOS ...")
    schedules = load_schedules(seasons)
    sos = compute_team_sos(schedules)
    print(f"  sos rows: {len(sos)}")

    print("Loading team offensive environment ...")
    team_stats = load_team_season_stats(seasons)
    offense_env = build_team_offensive_environment(team_stats)
    print(f"  offense env rows: {len(offense_env)}")

    print("Building landing-spot opportunity ...")
    opportunity = build_landing_opportunity(stats, draft)
    print(f"  opportunity rows: {len(opportunity)}")

    print("Loading college production (optional CFBD) ...")
    college = load_college_production(draft)
    print(f"  college rows: {len(college)}")

    print("Loading FantasyPros ADP from data/manual (rookie season only) ...")
    ff_rankings = load_fantasypros_adp()
    print(f"  fantasypros adp rows: {len(ff_rankings)}")

    print("Joining tables ...")
    tables = build_tables(
        fantasy=fantasy,
        recruiting=recruiting,
        combine=combine,
        sos=sos,
        offense_env=offense_env,
        opportunity=opportunity,
        college=college,
        ff_rankings=ff_rankings,
    )
    master_n = len(tables.get("players_master", []))
    print(f"  players_master rows: {master_n}")

    print("Exporting workbook + CSVs ...")
    xlsx_path = export_workbook_and_csvs(tables)
    print(f"Wrote {xlsx_path}")
    print("Wrote individual CSVs under data/output/csv/")
    print("Done (compile + export only; analysis/ML scorer not run).")
    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compile rookie PPR dataset workbook/CSVs")
    parser.add_argument(
        "--fetch-on3",
        action="store_true",
        help="Download On3 Industry Comparison + Industry Player rankings before compile",
    )
    args = parser.parse_args()
    raise SystemExit(main(fetch_on3=args.fetch_on3))
