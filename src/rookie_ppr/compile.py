from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Allow `python -m rookie_ppr.compile` from repo without editable install
_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rookie_ppr.analyze_correlation import run_correlation_analysis
from rookie_ppr.analyze_groups import build_group_averages
from rookie_ppr.config import DRAFT_YEAR_MAX, DRAFT_YEAR_MIN, INCOMING_DRAFT_YEAR, ensure_directories
from rookie_ppr.export_outputs import export_workbook_and_csvs
from rookie_ppr.features_sos import compute_team_sos
from rookie_ppr.ingest_college import load_college_production
from rookie_ppr.ingest_fantasypros_adp import load_fantasypros_adp
from rookie_ppr.ingest_nfl import (
    build_dynasty_fantasy,
    build_rookie_fantasy,
    load_combine,
    load_draft_picks,
    load_player_season_stats,
    load_rosters,
    load_schedules,
    load_team_season_stats,
)
from rookie_ppr.ingest_opportunity import (
    build_incumbent_competition,
    build_landing_opportunity,
    build_team_offensive_environment,
)
from rookie_ppr.ingest_recruiting import load_recruiting, write_recruiting_template
from rookie_ppr.incoming_rookies import build_incoming_rookies_sheet
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

    # nflverse often lags one season; fill gaps from NFL.com leaderboards (e.g. 2025 for 2026 drafts)
    from rookie_ppr.ingest_nfl_com_stats import fill_missing_season_stats

    prior_for_incoming = INCOMING_DRAFT_YEAR - 1
    stats = fill_missing_season_stats(stats, rosters, seasons=[prior_for_incoming])
    print(f"  stat rows after NFL.com fill: {len(stats)}")
    if not stats.empty and "season" in stats.columns:
        by_season = (
            pd.to_numeric(stats["season"], errors="coerce")
            .dropna()
            .astype(int)
            .value_counts()
            .sort_index()
        )
        print(f"  seasons present: {dict(by_season.tail(6))}")

    print("Building rookie fantasy targets ...")
    fantasy = build_rookie_fantasy(draft, stats, rosters)
    print(f"  fantasy rows: {len(fantasy)}")

    print("Building dynasty fantasy targets (Y1â€“Y3, parallel to redraft) ...")
    from rookie_ppr.config import DYNASTY_YEARS

    dynasty = build_dynasty_fantasy(fantasy, stats, years=DYNASTY_YEARS)
    n_complete = int(pd.to_numeric(dynasty.get("dynasty_seasons_complete"), errors="coerce").fillna(0).sum()) if not dynasty.empty else 0
    print(f"  dynasty rows: {len(dynasty)}, complete Y1â€“Y3: {n_complete}")

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
    print(
        f"  opportunity fill {INCOMING_DRAFT_YEAR}: "
        f"{opportunity.loc[opportunity['draft_year'] == INCOMING_DRAFT_YEAR, 'team_opportunity_ppr'].notna().mean():.0%}"
    )

    print("Building incumbent competition ...")
    incumbent = build_incumbent_competition(stats, draft, rosters)
    print(f"  incumbent rows: {len(incumbent)}")
    print(
        f"  incumbent fill {INCOMING_DRAFT_YEAR}: "
        f"{incumbent.loc[incumbent['draft_year'] == INCOMING_DRAFT_YEAR, 'incumbent_pos_ppr'].notna().mean():.0%}"
    )

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
        incumbent=incumbent,
        dynasty=dynasty,
    )
    master_n = len(tables.get("players_master", []))
    print(f"  players_master rows: {master_n}")

    print("Running redraft correlation analysis ...")
    master = tables.get("players_master", pd.DataFrame())
    if not master.empty:
        from rookie_ppr.analyze_trajectory import build_dynasty_trajectory

        trajectory = build_dynasty_trajectory(master)
        tables["dynasty_trajectory"] = trajectory
        n_late = int(trajectory["late_bloomer_y3"].sum()) if not trajectory.empty and "late_bloomer_y3" in trajectory.columns else 0
        print(f"  dynasty_trajectory rows: {len(trajectory)}, late_bloomer_y3: {n_late}")

        analysis = run_correlation_analysis(master)
        tables.update(analysis)
        tables["group_averages"] = build_group_averages(master)
        # Extend data dictionary
        dd = tables.get("data_dictionary", pd.DataFrame())
        extra = pd.DataFrame(
            [
                {"sheet": "feature_correlation", "column": "*", "description": "Per-feature correlation vs rookie PPR", "source": "analyze_correlation"},
                {"sheet": "strongest_factors", "column": "*", "description": "Factors ranked by impact on rookie PPR success", "source": "analyze_correlation"},
                {"sheet": "dataset_correlation", "column": "*", "description": "Dataset-block correlation scores vs rookie PPR", "source": "analyze_correlation"},
                {"sheet": "group_averages", "column": "*", "description": "Mean rookie PPR by position, round, recruiting band, etc.", "source": "analyze_groups"},
                {"sheet": "fantasy_dynasty", "column": "*", "description": "Y1â€“Y3 PPR outcomes for dynasty mode (aggregates require complete windows)", "source": "ingest_nfl.build_dynasty_fantasy"},
                {"sheet": "dynasty_trajectory", "column": "*", "description": "Within-class year ranks and late-bloomer flags (dynasty exploration)", "source": "analyze_trajectory"},
            ]
        )
        tables["data_dictionary"] = pd.concat([dd, extra], ignore_index=True)
        print(f"  strongest_factors rows: {len(tables.get('strongest_factors', []))}")

        print("Training ML success scorer (composite features) ...")
        from rookie_ppr.model_score import train_and_score

        feature_corr = tables.get("feature_correlation", pd.DataFrame())
        ml_features, composite_corr, metrics = train_and_score(master, feature_corr)
        tables["ml_features"] = ml_features
        tables["composite_correlation"] = composite_corr
        if metrics:
            holdout_r = metrics.get("holdout_pearson_r", "n/a")
            print(
                f"  ML train rows: {metrics.get('train_rows')}, "
                f"holdout r: {holdout_r}, holdout MAE: {metrics.get('holdout_mae', 'n/a')}"
            )
            cov = metrics.get("holdout_interval_coverage")
            if cov is not None:
                print(
                    f"  CQR {100 * (1 - metrics.get('cqr_alpha', 0.2)):.0f}% interval "
                    f"holdout coverage: {cov}, "
                    f"mean width: {metrics.get('holdout_interval_width_mean', 'n/a')}"
                )
            adp_base = ((metrics.get("baselines") or {}).get("adp_only") or {})
            cmp = adp_base.get("comparable") or {}
            if cmp:
                print(
                    f"  ADP-only baseline (comparable n={cmp.get('n')}): "
                    f"r={cmp.get('adp_r')} MAE={cmp.get('adp_mae')} | "
                    f"full r={cmp.get('full_r')} MAE={cmp.get('full_mae')} | "
                    f"lift r={cmp.get('lift_r')} MAE={cmp.get('lift_mae')}"
                )
            elif adp_base.get("holdout_pearson_r") is not None:
                print(
                    f"  ADP-only baseline holdout r: {adp_base.get('holdout_pearson_r')}, "
                    f"MAE: {adp_base.get('holdout_mae')}"
                )
        dd = tables.get("data_dictionary", pd.DataFrame())
        ml_dd = pd.DataFrame(
            [
                {"sheet": "ml_features", "column": "*", "description": "Correlation-weighted composite scores + predicted PPR + success score", "source": "model_score"},
                {"sheet": "composite_correlation", "column": "*", "description": "Composite score correlation vs rookie PPR", "source": "feature_composites"},
            ]
        )
        tables["data_dictionary"] = pd.concat([dd, ml_dd], ignore_index=True)

        print("Training dynasty ML scorer (Y1â€“Y3 total) ...")
        from rookie_ppr.model_score import train_and_score_dynasty

        ml_features_dynasty, composite_corr_dynasty, dynasty_metrics = train_and_score_dynasty(master)
        tables["ml_features_dynasty"] = ml_features_dynasty
        tables["composite_correlation_dynasty"] = composite_corr_dynasty
        if dynasty_metrics:
            holdout_r = dynasty_metrics.get("holdout_pearson_r", "n/a")
            print(
                f"  Dynasty train rows: {dynasty_metrics.get('train_rows')}, "
                f"holdout r: {holdout_r}, holdout MAE: {dynasty_metrics.get('holdout_mae', 'n/a')}"
            )
            by_pos = (dynasty_metrics.get("holdout") or {}).get("by_position") or dynasty_metrics.get(
                "holdout_by_position"
            ) or {}
            if by_pos:
                parts = [
                    f"{pos} r={pos_stats.get('pearson_r')} MAE={pos_stats.get('mae')} n={pos_stats.get('n')}"
                    for pos, pos_stats in sorted(by_pos.items())
                ]
                print(f"  Dynasty holdout by position: {'; '.join(parts)}")
            for yt, block in (dynasty_metrics.get("year_targets") or {}).items():
                print(
                    f"  Dynasty {yt}: r={block.get('pearson_r')} MAE={block.get('mae')} n={block.get('n')}"
                )
        dd = tables.get("data_dictionary", pd.DataFrame())
        dynasty_dd = pd.DataFrame(
            [
                {
                    "sheet": "ml_features_dynasty",
                    "column": "*",
                    "description": "Dynasty composite scores + predicted Y1â€“Y3 PPR total + success score",
                    "source": "model_score.train_and_score_dynasty",
                },
                {
                    "sheet": "composite_correlation_dynasty",
                    "column": "*",
                    "description": "Composite score correlation vs dynasty Y1â€“Y3 PPR total",
                    "source": "model_score.train_and_score_dynasty",
                },
            ]
        )
        tables["data_dictionary"] = pd.concat([dd, dynasty_dd], ignore_index=True)

        incoming = build_incoming_rookies_sheet(master, ml_features)
        tables[f"incoming_rookies_{INCOMING_DRAFT_YEAR}"] = incoming
        print(f"  incoming_rookies_{INCOMING_DRAFT_YEAR} rows: {len(incoming)}")
        dd = tables.get("data_dictionary", pd.DataFrame())
        incoming_dd = pd.DataFrame(
            [
                {
                    "sheet": f"incoming_rookies_{INCOMING_DRAFT_YEAR}",
                    "column": "*",
                    "description": (
                        f"Pre-rookie inputs + ML predictions for {INCOMING_DRAFT_YEAR} draft class "
                        f"({INCOMING_DRAFT_YEAR}-{INCOMING_DRAFT_YEAR + 1} NFL season); rookie PPR blank until season ends"
                    ),
                    "source": "compile + model_score",
                },
            ]
        )
        tables["data_dictionary"] = pd.concat([dd, incoming_dd], ignore_index=True)

    print("Exporting workbook + CSVs ...")
    xlsx_path = export_workbook_and_csvs(tables)
    if xlsx_path is not None:
        print(f"Wrote {xlsx_path}")
    else:
        print("CSVs updated; xlsx skipped (likely locked).")
    print("Wrote individual CSVs under data/output/csv/")
    print("Done (compile + correlation + ML scorer).")
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
