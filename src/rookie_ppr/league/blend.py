"""Walk-forward blend of the model and the market ADP curve.

A fixed 0.3/0.7 weight was the best of a coarse sweep on all eight seasons at
once, which mildly peeks at the seasons being graded. Here the weight for season
Y is chosen only from drafted skill picks in seasons strictly before Y, so the
blend used in the backtest is honest.

When no prior league seasons exist (2018), the fallback is the global default
found on that earlier sweep. That single season is the only one that still uses
a non-walk-forward weight.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from rookie_ppr.league.backtest_rosters import LEAGUE_BACKTEST_ROSTERS_CSV
from rookie_ppr.league.config import (
    CSV_OUTPUT_DIR,
    LEAGUE_SEASONS,
    MODELED_POSITIONS,
    PRODUCTION_BLEND_DEFAULT,
)

# Default when there is nothing earlier to fit on (first league season).
DEFAULT_BLEND_MODEL_WEIGHT = PRODUCTION_BLEND_DEFAULT
WEIGHT_GRID = np.round(np.arange(0.0, 1.01, 0.05), 2)
MIN_FIT_ROWS = 40


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 5:
        return float("nan")
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def best_blend_weight(
    model: np.ndarray, market: np.ndarray, actual: np.ndarray
) -> tuple[float, float]:
    """Return (weight_on_model, pearson_r) maximizing correlation with actual."""
    best_w, best_r = DEFAULT_BLEND_MODEL_WEIGHT, float("-inf")
    for w in WEIGHT_GRID:
        blended = w * model + (1.0 - w) * market
        r = _pearson(blended, actual)
        if np.isfinite(r) and r > best_r:
            best_w, best_r = float(w), r
    return best_w, best_r


def fit_blend_weights_walkforward(
    rosters: pd.DataFrame | None = None,
    seasons: list[int] | None = None,
) -> pd.DataFrame:
    """One row per season: weight fit only on earlier drafted skill picks."""
    seasons = list(seasons or LEAGUE_SEASONS)
    if rosters is None:
        rosters = pd.read_csv(CSV_OUTPUT_DIR / LEAGUE_BACKTEST_ROSTERS_CSV)

    skill = rosters[
        rosters["position"].isin(MODELED_POSITIONS)
        & (rosters["proj_source"] == "model")
    ].dropna(subset=["proj_season_ppr", "adp_curve_ppr", "actual_season_ppr"])

    rows: list[dict] = []
    for season in seasons:
        prior = skill[skill["season"] < season]
        if len(prior) < MIN_FIT_ROWS:
            rows.append(
                {
                    "season": season,
                    "blend_model_weight": DEFAULT_BLEND_MODEL_WEIGHT,
                    "fit_n": int(len(prior)),
                    "fit_r": None,
                    "source": "default",
                }
            )
            continue
        w, r = best_blend_weight(
            prior["proj_season_ppr"].to_numpy(dtype=float),
            prior["adp_curve_ppr"].to_numpy(dtype=float),
            prior["actual_season_ppr"].to_numpy(dtype=float),
        )
        rows.append(
            {
                "season": season,
                "blend_model_weight": w,
                "fit_n": int(len(prior)),
                "fit_r": round(r, 4),
                "source": "walkforward",
            }
        )
    return pd.DataFrame(rows)


def blend_weight_map(weights: pd.DataFrame) -> dict[int, float]:
    return {
        int(row.season): float(row.blend_model_weight)
        for row in weights.itertuples(index=False)
    }


__all__ = [
    "DEFAULT_BLEND_MODEL_WEIGHT",
    "best_blend_weight",
    "blend_weight_map",
    "fit_blend_weights_walkforward",
]
