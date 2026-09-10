"""Production league pipeline: two-track projections, blend sim, player board.

Settled defaults from the 2018-2025 backtest:

1. **Team track** — walk-forward BASE model + ADP-curve blend (weight fit on
   earlier drafted picks only). Best at predicting wins / playoff sets because
   the base model stays partly independent of the market.
2. **Player track** — walk-forward ADP + opportunity-share model. Best standalone
   player forecast on drafted skill players; used for the ranking board.

Run::

    python -m rookie_ppr.league.compile
    python -m rookie_ppr.league.compile --skip-projections   # reuse cached CSVs
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from rookie_ppr.config import MODELS_DIR, ensure_directories
from rookie_ppr.league.backtest_projections import (
    save_projections,
    walk_forward_projections,
)
from rookie_ppr.league.backtest_rosters import build_backtest_rosters
from rookie_ppr.league.config import (
    CSV_OUTPUT_DIR,
    LEAGUE_BACKTEST_ROSTERS_CSV,
    LEAGUE_BLEND_WEIGHTS_CSV,
    LEAGUE_PLAYER_BOARD_CSV,
    LEAGUE_PLAYER_METRICS_FILE,
    LEAGUE_PLAYER_PROJECTIONS_CSV,
    LEAGUE_SEASONS,
    LEAGUE_SIM_GRADES_CSV,
    LEAGUE_SIM_TEAMS_CSV,
    LEAGUE_TARGET_SEASON,
    LEAGUE_WF_METRICS_FILE,
    LEAGUE_WF_PROJECTIONS_CSV,
)
from rookie_ppr.league.player_board import build_player_board, save_player_board
from rookie_ppr.league.simulate_season import run_backtest


def _copy(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def compile_league(
    seasons: list[int] | None = None,
    *,
    skip_projections: bool = False,
    n_sims: int = 3000,
    include_target_season: bool = True,
) -> dict[str, str]:
    """Run the production two-track pipeline. Returns artifact paths."""
    ensure_directories()
    seasons = list(seasons or LEAGUE_SEASONS)
    board_seasons = list(seasons)
    if include_target_season and LEAGUE_TARGET_SEASON not in board_seasons:
        board_seasons.append(LEAGUE_TARGET_SEASON)

    artifacts: dict[str, str] = {}

    # ---- 1. Team-track projections (base model) ----
    base_path = CSV_OUTPUT_DIR / LEAGUE_WF_PROJECTIONS_CSV
    if skip_projections and base_path.exists():
        print(f"reusing team-track projections: {base_path}")
    else:
        print("1/5 team-track walk-forward (base model, no ADP/opportunity features):")
        proj_seasons = list(seasons)
        if include_target_season:
            proj_seasons = sorted(set(proj_seasons) | {LEAGUE_TARGET_SEASON})
        projections, metrics = walk_forward_projections(
            proj_seasons, use_adp=False, use_opportunity=False, clamp=True
        )
        save_projections(projections, metrics, tag="")
        # save_projections with empty tag writes the default filenames.
        print(f"  -> {base_path}")
    artifacts["team_projections"] = str(base_path)
    artifacts["team_metrics"] = str(MODELS_DIR / LEAGUE_WF_METRICS_FILE)

    # ---- 2. Player-track projections (ADP + opportunity) ----
    player_path = CSV_OUTPUT_DIR / LEAGUE_PLAYER_PROJECTIONS_CSV
    if skip_projections and player_path.exists():
        print(f"reusing player-track projections: {player_path}")
    else:
        print("2/5 player-track walk-forward (ADP + opportunity features):")
        proj_seasons = list(seasons)
        if include_target_season:
            proj_seasons = sorted(set(proj_seasons) | {LEAGUE_TARGET_SEASON})
        projections, metrics = walk_forward_projections(
            proj_seasons, use_adp=True, use_opportunity=True, clamp=True
        )
        # Write under the tagged name, then promote to the production filename.
        tagged = save_projections(projections, metrics, tag="adp_opp")
        _copy(Path(tagged), player_path)
        tagged_metrics = MODELS_DIR / LEAGUE_WF_METRICS_FILE.replace(
            ".json", "_adp_opp.json"
        )
        if tagged_metrics.exists():
            _copy(tagged_metrics, MODELS_DIR / LEAGUE_PLAYER_METRICS_FILE)
        print(f"  -> {player_path}")
    artifacts["player_projections"] = str(player_path)
    artifacts["player_metrics"] = str(MODELS_DIR / LEAGUE_PLAYER_METRICS_FILE)

    # ---- 3. Drafted rosters for team sim (BASE projections only) ----
    print("3/5 building drafted rosters from team-track (base) projections:")
    # The target season is included when it has been drafted, so the preseason
    # ForeKast can run off the same table. Grading below stays on completed
    # seasons only.
    roster_seasons = list(seasons)
    if include_target_season and LEAGUE_TARGET_SEASON not in roster_seasons:
        roster_seasons.append(LEAGUE_TARGET_SEASON)
    rosters = build_backtest_rosters(
        roster_seasons, projections_csv=LEAGUE_WF_PROJECTIONS_CSV
    )
    roster_path = CSV_OUTPUT_DIR / LEAGUE_BACKTEST_ROSTERS_CSV
    rosters.to_csv(roster_path, index=False)
    print(f"  rows={len(rosters)} -> {roster_path}")
    artifacts["backtest_rosters"] = str(roster_path)

    # ---- 4. Walk-forward blend + season simulation ----
    print(f"4/5 simulating {n_sims} seasons per league-year (walk-forward blend):")
    sim, grades, blend_weights = run_backtest(seasons, n_sims=n_sims)
    sim_path = CSV_OUTPUT_DIR / LEAGUE_SIM_TEAMS_CSV
    grades_path = CSV_OUTPUT_DIR / LEAGUE_SIM_GRADES_CSV
    weights_path = CSV_OUTPUT_DIR / LEAGUE_BLEND_WEIGHTS_CSV
    sim.to_csv(sim_path, index=False)
    grades.to_csv(grades_path, index=False)
    blend_weights.to_csv(weights_path, index=False)
    print(f"  team-seasons={len(sim)} -> {sim_path}")
    print("  grades:")
    print(grades.to_string(index=False))
    artifacts["sim_teams"] = str(sim_path)
    artifacts["sim_grades"] = str(grades_path)
    artifacts["blend_weights"] = str(weights_path)

    # ---- 5. Player ranking board from ADP+opportunity ----
    print("5/5 building production player ranking board:")
    board = build_player_board(seasons=board_seasons)
    board_path = save_player_board(board)
    print(f"  rows={len(board)} -> {board_path}")
    artifacts["player_board"] = str(board_path)

    latest = int(board["target_season"].max()) if len(board) else None
    if latest is not None:
        top = board[board["target_season"] == latest].nsmallest(12, "vorp_rank")
        print(f"\n  top 12 by VORP ({latest}):")
        print(
            top[
                ["vorp_rank", "player_name", "position", "projected_ppr_17", "vorp", "adp_rank"]
            ]
            .round(1)
            .to_string(index=False)
        )

    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile the production league pipeline (two-track projections + sim)"
    )
    parser.add_argument("--seasons", type=int, nargs="+", default=list(LEAGUE_SEASONS))
    parser.add_argument(
        "--skip-projections",
        action="store_true",
        help="Reuse existing walk-forward projection CSVs",
    )
    parser.add_argument("--sims", type=int, default=3000)
    parser.add_argument(
        "--no-target-season",
        action="store_true",
        help=f"Skip projecting {LEAGUE_TARGET_SEASON} for the player board",
    )
    args = parser.parse_args()

    print("=== league production compile ===")
    print("team track:  base model + walk-forward ADP blend")
    print("player track: ADP + opportunity model ranking board")
    print()
    artifacts = compile_league(
        args.seasons,
        skip_projections=args.skip_projections,
        n_sims=args.sims,
        include_target_season=not args.no_target_season,
    )
    print("\n=== artifacts ===")
    for key, path in artifacts.items():
        print(f"  {key:<22} {path}")


if __name__ == "__main__":
    main()
