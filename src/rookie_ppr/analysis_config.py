from __future__ import annotations

from dataclasses import dataclass, field

from rookie_ppr.config import (
    HOLDOUT_DYNASTY_FIRST_STAT_SEASONS,
    HOLDOUT_ROOKIE_SEASONS,
    TUNING_DYNASTY_FIRST_STAT_SEASON,
    TUNING_ROOKIE_SEASON,
)


@dataclass(frozen=True)
class AnalysisMode:
    """Redraft and dynasty are separate use cases with parallel artifacts."""

    name: str
    primary_target: str
    predicted_col: str
    success_score_col: str
    model_file: str
    metrics_file: str
    composite_weights_file: str
    percentile_file: str
    holdout_seasons: tuple[int, ...]
    tuning_season: int
    extra_targets: tuple[str, ...] = field(default_factory=tuple)
    # Per-season dynasty heads (Y1/Y2/Y3 PPR) with full redraft-style uncertainty
    year_targets: tuple[str, ...] = field(default_factory=tuple)
    # Dynasty labels require complete Y1–Y3; redraft uses any non-null target
    require_dynasty_complete: bool = False


REDRAFT_MODE = AnalysisMode(
    name="redraft",
    primary_target="rookie_ppr",
    predicted_col="predicted_rookie_ppr",
    success_score_col="success_score_0_100",
    model_file="hgb_rookie_ppr.joblib",
    metrics_file="model_metrics.json",
    composite_weights_file="composite_weights.json",
    percentile_file="percentile_lookup.json",
    holdout_seasons=HOLDOUT_ROOKIE_SEASONS,
    tuning_season=TUNING_ROOKIE_SEASON,
)

DYNASTY_MODE = AnalysisMode(
    name="dynasty",
    primary_target="dynasty_ppr_y1_y3_total",
    predicted_col="predicted_dynasty_ppr_y1_y3_total",
    success_score_col="success_score_dynasty_0_100",
    model_file="hgb_dynasty_ppr.joblib",
    metrics_file="model_metrics_dynasty.json",
    composite_weights_file="composite_weights_dynasty.json",
    percentile_file="percentile_lookup_dynasty.json",
    holdout_seasons=HOLDOUT_DYNASTY_FIRST_STAT_SEASONS,
    tuning_season=TUNING_DYNASTY_FIRST_STAT_SEASON,
    year_targets=("ppr_y1", "ppr_y2", "ppr_y3"),
    require_dynasty_complete=True,
)


def get_mode(name: str) -> AnalysisMode:
    key = (name or "redraft").strip().lower()
    if key == "dynasty":
        return DYNASTY_MODE
    return REDRAFT_MODE
