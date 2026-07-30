"""
Per-stat next-season models: one model per position × component stat.

These feed the week-by-week projection, which is built bottom-up from stat
totals instead of decomposing the single PPR prediction. Same features, same
train/tune/holdout split as the PPR model, so the two are comparable.
"""
from __future__ import annotations

import json
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from rookie_ppr.model_score import _feature_matrix, _pearson_r
from rookie_ppr.veteran.config import (
    MODELS_DIR,
    POSITION_FEATURE_COLS,
    SKILL_POSITIONS,
    STAT_TARGETS,
    VET_HOLDOUT_TARGET_SEASONS,
    VET_METRICS_FILE,
    VET_STAT_METRICS_FILE,
    VET_STAT_MODEL_FILE,
)
from rookie_ppr.veteran.train import _usable_feature_cols

# Counting stats cannot go negative
NON_NEGATIVE = True
MIN_TRAIN_ROWS = 40
# A stat a position barely records (WR passing yards, TE interceptions) gets the
# training mean instead of a model, which otherwise invents small nonzero values.
MIN_NONZERO_SHARE = 0.05
DEFAULT_PARAMS = {
    "learning_rate": 0.06,
    "max_iter": 300,
    "max_depth": 3,
    "min_samples_leaf": 20,
    "l2_regularization": 1.0,
}


def _position_params() -> dict[str, dict[str, Any]]:
    """Reuse the PPR model's tuned params per position when available."""
    path = MODELS_DIR / VET_METRICS_FILE
    if not path.exists():
        return {}
    try:
        metrics = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return metrics.get("best_params_by_position") or {}


def _fit(frame: pd.DataFrame, cols: list[str], target: str, params: dict[str, Any]):
    X = _feature_matrix(frame, cols)
    y = pd.to_numeric(frame[target], errors="coerce").to_numpy()
    model = HistGradientBoostingRegressor(random_state=42, **params)
    model.fit(X, y)
    return model


def train_stat_models(feature_frame: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit every position × stat model; return (artifact, metrics)."""
    frame = feature_frame.copy()
    frame["target_season"] = pd.to_numeric(frame.get("target_season"), errors="coerce")
    holdout_mask = frame["target_season"].isin(VET_HOLDOUT_TARGET_SEASONS)

    tuned = _position_params()
    models: dict[str, dict[str, Any]] = {}
    constants: dict[str, dict[str, float]] = {}
    feature_cols_by_pos: dict[str, list[str]] = {}
    metrics_by_pos: dict[str, dict[str, Any]] = {}

    targets = [t for t in STAT_TARGETS if t in frame.columns]

    for position in SKILL_POSITIONS:
        pos_mask = frame["position"] == position
        fit_pool = frame.loc[pos_mask & ~holdout_mask]
        hold_pool = frame.loc[pos_mask & holdout_mask]
        if len(fit_pool) < MIN_TRAIN_ROWS:
            continue

        params = {**DEFAULT_PARAMS, **(tuned.get(position) or {})}
        params.pop("quantile", None)
        params.pop("loss", None)

        cols = _usable_feature_cols(fit_pool, list(POSITION_FEATURE_COLS.get(position, [])))
        if len(cols) < 3:
            continue
        feature_cols_by_pos[position] = cols

        pos_models: dict[str, Any] = {}
        pos_metrics: dict[str, Any] = {}
        pos_constants: dict[str, float] = {}
        for target in targets:
            fit_rows = fit_pool[pd.to_numeric(fit_pool[target], errors="coerce").notna()]
            if len(fit_rows) < MIN_TRAIN_ROWS:
                continue
            y_fit = pd.to_numeric(fit_rows[target], errors="coerce")
            nonzero_share = float((y_fit > 0).mean())
            if y_fit.nunique() < 3 or nonzero_share < MIN_NONZERO_SHARE:
                constant = max(float(y_fit.mean()), 0.0)
                pos_constants[target] = constant
                pos_metrics[target] = {
                    "skipped": "rare for position",
                    "nonzero_share": round(nonzero_share, 4),
                    "constant": round(constant, 3),
                }
                continue

            model = _fit(fit_rows, cols, target, params)
            pos_models[target] = model

            hold_rows = hold_pool[pd.to_numeric(hold_pool[target], errors="coerce").notna()]
            if len(hold_rows) >= 20:
                y = pd.to_numeric(hold_rows[target], errors="coerce")
                pred = model.predict(_feature_matrix(hold_rows, cols))
                if NON_NEGATIVE:
                    pred = np.clip(pred, 0, None)
                pos_metrics[target] = {
                    "n_train": int(len(fit_rows)),
                    "n_holdout": int(len(hold_rows)),
                    "holdout_pearson_r": round(_pearson_r(y, pred), 4),
                    "holdout_mae": round(float(mean_absolute_error(y, pred)), 3),
                    "holdout_mean_actual": round(float(y.mean()), 2),
                    "holdout_mean_pred": round(float(np.mean(pred)), 2),
                }
            else:
                pos_metrics[target] = {"n_train": int(len(fit_rows)), "n_holdout": int(len(hold_rows))}

        models[position] = pos_models
        constants[position] = pos_constants
        metrics_by_pos[position] = pos_metrics
        print(
            f"  {position}: {len(pos_models)} stat models "
            f"({len(pos_constants)} constant), features={len(cols)}, train={len(fit_pool)}"
        )

    artifact = {
        "models_by_position": models,
        "constants_by_position": constants,
        "feature_cols_by_position": feature_cols_by_pos,
        "targets": targets,
        "non_negative": NON_NEGATIVE,
    }
    metrics = {
        "model_type": "veteran_component_stats",
        "targets": targets,
        "holdout_target_seasons": list(VET_HOLDOUT_TARGET_SEASONS),
        "by_position": metrics_by_pos,
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODELS_DIR / VET_STAT_MODEL_FILE)
    (MODELS_DIR / VET_STAT_METRICS_FILE).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return artifact, metrics


def load_stat_models() -> dict[str, Any]:
    path = MODELS_DIR / VET_STAT_MODEL_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path.name}. Run: python -m rookie_ppr.veteran.compile"
        )
    return joblib.load(path)


def predict_stats(rows: pd.DataFrame, artifact: dict[str, Any] | None = None) -> pd.DataFrame:
    """Season-total predictions for each component stat, one column per stat."""
    artifact = artifact or load_stat_models()
    models = artifact["models_by_position"]
    cols_by_pos = artifact["feature_cols_by_position"]
    targets: list[str] = artifact["targets"]

    out = rows.copy()
    for target in targets:
        out[f"pred_{target}"] = np.nan

    constants = artifact.get("constants_by_position") or {}
    for position, pos_models in models.items():
        mask = out["position"] == position
        if not mask.any():
            continue
        cols = cols_by_pos.get(position) or []
        X = _feature_matrix(out.loc[mask], cols)
        for target, model in pos_models.items():
            pred = model.predict(X)
            if artifact.get("non_negative", True):
                pred = np.clip(pred, 0, None)
            out.loc[mask, f"pred_{target}"] = pred
        for target, value in (constants.get(position) or {}).items():
            out.loc[mask, f"pred_{target}"] = value

    return out
