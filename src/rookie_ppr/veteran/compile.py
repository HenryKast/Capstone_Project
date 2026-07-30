"""
Compile + train Phase-1 veteran next-season PPR models.

Parallel to redraft/dynasty — does not modify those pipelines or artifacts.
"""
from __future__ import annotations

import argparse

from rookie_ppr.config import ensure_directories
from rookie_ppr.veteran.config import (
    CSV_OUTPUT_DIR,
    VET_HOLDOUT_TARGET_SEASONS,
    VET_PANEL_CSV,
    VET_SEASON_MAX,
    VET_SEASON_MIN,
    VET_TUNING_TARGET_SEASON,
)
from rookie_ppr.veteran.features import build_feature_frame
from rookie_ppr.veteran.panel import build_veteran_panel
from rookie_ppr.veteran.defense_adjust import calibrate_opponent_sensitivity
from rookie_ppr.veteran.draft_board import build_draft_board, save_draft_board
from rookie_ppr.veteran.stat_models import train_stat_models
from rookie_ppr.veteran.train import train_veteran_models
from rookie_ppr.veteran.weekly_projection import (
    build_weekly_projections,
    comparison_summary,
    save_weekly_projections,
)


def run(*, season_min: int = VET_SEASON_MIN, season_max: int = VET_SEASON_MAX) -> None:
    ensure_directories()
    print(f"Veteran Phase 3 compile ({season_min}–{season_max})")
    print(f"  tune target season: {VET_TUNING_TARGET_SEASON}")
    print(f"  holdout target seasons: {VET_HOLDOUT_TARGET_SEASONS}")

    panel = build_veteran_panel(season_min=season_min, season_max=season_max)
    if panel.empty:
        raise SystemExit("Veteran panel empty — check nflverse player_stats downloads.")

    features = build_feature_frame(panel)
    labeled_n = features["ppr_next"].notna().sum() if "ppr_next" in features.columns else 0
    print(f"  feature rows: {len(features)} (labeled ppr_next: {labeled_n})")

    CSV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    features.to_csv(CSV_OUTPUT_DIR / VET_PANEL_CSV, index=False)
    print(f"  wrote {VET_PANEL_CSV}")

    print("Training veteran models ...")
    scored, metrics = train_veteran_models(features)
    print(
        f"  holdout r={metrics.get('holdout_pearson_r')} "
        f"MAE={metrics.get('holdout_mae')} n={metrics.get('n_holdout')}"
    )
    baselines = metrics.get("baselines") or {}
    last_b = baselines.get("last_season_ppr") or {}
    trail_b = baselines.get("trail3_mean_ppr") or {}
    print(
        f"  vs last-season baseline: model r={last_b.get('full_r')} "
        f"baseline r={last_b.get('baseline_r')} lift={last_b.get('lift_r')}"
    )
    print(
        f"  vs trail-3 baseline: model r={trail_b.get('full_r')} "
        f"baseline r={trail_b.get('baseline_r')} lift={trail_b.get('lift_r')}"
    )
    by_pos = (metrics.get("holdout") or {}).get("by_position") or {}
    for pos in ("QB", "RB", "WR", "TE"):
        row = by_pos.get(pos) or {}
        if row:
            print(f"  {pos}: r={row.get('pearson_r')} n={row.get('n')} MAE={row.get('mae')}")
    tier = metrics.get("holdout_tiered_by_prior_ppr") or {}
    pooled = tier.get("overall_pooled") or {}
    for label in ("top_20", "top_50", "top_100"):
        block = pooled.get(label) or {}
        if block.get("n"):
            print(f"  holdout {label} (prior PPR): r={block.get('pearson_r')} n={block.get('n')}")
    star = metrics.get("holdout_weighted_star_r") or {}
    if star.get("weighted_pearson_r") is not None:
        print(f"  holdout weighted star r={star.get('weighted_pearson_r')} n={star.get('n')}")

    board = build_draft_board(scored)
    if not board.empty:
        path = save_draft_board(board)
        target = int(board["target_season"].iloc[0])
        print(f"  unified draft board: {len(board)} players, target {target} -> {path.name}")

    print("Training component stat models ...")
    train_stat_models(features)

    print("Calibrating opponent-defense sensitivity ...")
    try:
        payload = calibrate_opponent_sensitivity()
        for pos, block in (payload.get("diagnostics") or {}).items():
            parts = [
                f"{cat} {info.get('pct_swing_per_sd')}%/SD" for cat, info in block.items()
            ]
            print(f"  {pos}: {', '.join(parts)}")
    except RuntimeError as exc:
        print(f"  skipped: {exc}")

    print("Building week-by-week projections ...")
    weekly, compare = build_weekly_projections(features)
    if not weekly.empty:
        wp, cp = save_weekly_projections(weekly, compare)
        print(f"  {len(weekly)} weekly rows -> {wp.name}; {len(compare)} players -> {cp.name}")
        overall = (comparison_summary(compare) or {}).get("overall") or {}
        if overall:
            print(
                f"  bottom-up vs top-down season total: r={overall.get('pearson_r')} "
                f"mean {overall.get('mean_bottom_up')} vs {overall.get('mean_top_down')} "
                f"(mean diff {overall.get('mean_diff')})"
            )
    print(f"Done. Scored rows: {len(scored)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile/train veteran next-season PPR (Phase 3)")
    parser.add_argument("--season-min", type=int, default=VET_SEASON_MIN)
    parser.add_argument("--season-max", type=int, default=VET_SEASON_MAX)
    args = parser.parse_args()
    run(season_min=args.season_min, season_max=args.season_max)


if __name__ == "__main__":
    main()
