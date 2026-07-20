from __future__ import annotations

import json
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from rookie_ppr.config import HOLDOUT_ROOKIE_SEASONS, MODELS_DIR, SKILL_POSITIONS, TUNING_ROOKIE_SEASON
from rookie_ppr.feature_composites import (
    ID_COLUMNS,
    SCORE_COLUMNS,
    apply_composites,
    build_composite_artifacts,
    build_composite_correlation,
    save_composite_artifacts,
)

TARGET = "rookie_ppr"
MODEL_FILE = "hgb_rookie_ppr.joblib"
COMPOSITE_FILE = "composite_weights.json"
PERCENTILE_FILE = "percentile_lookup.json"
METRICS_FILE = "model_metrics.json"

# Position-specific composite subsets (drop low-signal or irrelevant blocks per role)
POSITION_FEATURE_COLS: dict[str, list[str]] = {
    "QB": [
        "score_draft_capital",
        "score_pre_draft_fantasy",
        "score_recruiting",
        "score_timing",
        "score_cfb_passing",
        "score_combine_size",
        "score_combine_speed",
        "score_team_context",
    ],
    "RB": [
        "score_draft_capital",
        "score_pre_draft_fantasy",
        "score_recruiting",
        "score_timing",
        "score_cfb_rushing",
        "score_cfb_receiving",
        "score_combine_speed",
        "score_combine_explosion",
        "score_combine_size",
        "score_team_context",
    ],
    "WR": [
        "score_draft_capital",
        "score_pre_draft_fantasy",
        "score_recruiting",
        "score_timing",
        "score_cfb_receiving",
        "score_combine_speed",
        "score_combine_explosion",
        "score_combine_size",
        "score_team_context",
    ],
    "TE": [
        "score_draft_capital",
        "score_pre_draft_fantasy",
        "score_recruiting",
        "score_timing",
        "score_cfb_receiving",
        "score_combine_size",
        "score_combine_explosion",
        "score_team_context",
    ],
}

PARAM_GRID: list[dict[str, Any]] = [
    {"max_depth": 2, "learning_rate": 0.03, "max_iter": 200, "min_samples_leaf": 25, "l2_regularization": 2.0},
    {"max_depth": 3, "learning_rate": 0.05, "max_iter": 250, "min_samples_leaf": 20, "l2_regularization": 1.0},
    {"max_depth": 3, "learning_rate": 0.08, "max_iter": 300, "min_samples_leaf": 15, "l2_regularization": 0.5},
    {"max_depth": 4, "learning_rate": 0.05, "max_iter": 300, "min_samples_leaf": 10, "l2_regularization": 0.25},
    {"max_depth": 5, "learning_rate": 0.05, "max_iter": 350, "min_samples_leaf": 8, "l2_regularization": 0.1},
    {"max_depth": 6, "learning_rate": 0.05, "max_iter": 300, "min_samples_leaf": 5, "l2_regularization": 0.0},
]

# WR holdout favored pre-draft fantasy; allow a lighter fantasy-heavy grid slice
WR_PARAM_GRID: list[dict[str, Any]] = PARAM_GRID + [
    {"max_depth": 2, "learning_rate": 0.1, "max_iter": 150, "min_samples_leaf": 30, "l2_regularization": 3.0},
    {"max_depth": 3, "learning_rate": 0.1, "max_iter": 200, "min_samples_leaf": 25, "l2_regularization": 2.0},
]


def _percentile_lookup(train_df: pd.DataFrame) -> dict[str, list[float]]:
    lookup: dict[str, list[float]] = {}
    if TARGET not in train_df.columns or "position" not in train_df.columns:
        return lookup
    for pos, grp in train_df.groupby("position"):
        vals = pd.to_numeric(grp[TARGET], errors="coerce").dropna().sort_values().tolist()
        lookup[str(pos)] = vals
    return lookup


def _to_success_score(predicted: float, position: str, lookup: dict[str, list[float]]) -> float:
    hist = lookup.get(str(position), [])
    if not hist or pd.isna(predicted):
        return float("nan")
    arr = np.asarray(hist, dtype=float)
    return float((arr <= predicted).mean() * 100)


def _feature_matrix(frame: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    x = pd.DataFrame(index=frame.index)
    for col in feature_cols:
        if col in frame.columns:
            x[col] = pd.to_numeric(frame[col], errors="coerce")
        else:
            x[col] = np.nan
    return x


def _pearson_r(y_true: pd.Series, y_pred: np.ndarray) -> float:
    pair = pd.DataFrame({"y": pd.to_numeric(y_true, errors="coerce"), "p": y_pred}).dropna()
    if len(pair) < 5:
        return float("nan")
    r, _ = stats.pearsonr(pair["y"], pair["p"])
    return float(r)


def _position_feature_cols(position: str) -> list[str]:
    cols = POSITION_FEATURE_COLS.get(position, SCORE_COLUMNS)
    return [c for c in cols if c in SCORE_COLUMNS]


def _param_grid_for_position(position: str) -> list[dict[str, Any]]:
    return WR_PARAM_GRID if position == "WR" else PARAM_GRID


def _tune_position_model(
    position: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[HistGradientBoostingRegressor, dict[str, Any], float]:
    """Pick hyperparameters maximizing Pearson r on the validation fold."""
    best_r = float("-inf")
    best_params: dict[str, Any] = PARAM_GRID[2]
    best_model: HistGradientBoostingRegressor | None = None

    X_val = _feature_matrix(val_df, feature_cols)
    y_val = pd.to_numeric(val_df[TARGET], errors="coerce")

    for params in _param_grid_for_position(position):
        model = HistGradientBoostingRegressor(random_state=42, **params)
        X_train = _feature_matrix(train_df, feature_cols)
        y_train = pd.to_numeric(train_df[TARGET], errors="coerce")
        if len(X_train) < 15:
            continue
        model.fit(X_train, y_train)
        pred_val = model.predict(X_val) if len(X_val) >= 5 else model.predict(X_train)
        y_eval = y_val if len(X_val) >= 5 else y_train
        r = _pearson_r(y_eval, pred_val)
        if pd.notna(r) and r > best_r:
            best_r = r
            best_params = params
            best_model = model

    if best_model is None:
        best_params = {"max_depth": 3, "learning_rate": 0.05, "max_iter": 250, "min_samples_leaf": 15, "l2_regularization": 1.0}
        best_model = HistGradientBoostingRegressor(random_state=42, **best_params)
        X_train = _feature_matrix(train_df, feature_cols)
        y_train = pd.to_numeric(train_df[TARGET], errors="coerce")
        best_model.fit(X_train, y_train)
        best_r = _pearson_r(y_train, best_model.predict(X_train))

    return best_model, best_params, best_r


def _rookie_season(frame: pd.DataFrame) -> pd.Series:
    """Rookie fantasy season: first_stat_season when present, else draft_year."""
    stat = pd.to_numeric(frame["first_stat_season"], errors="coerce") if "first_stat_season" in frame.columns else pd.Series(np.nan, index=frame.index)
    draft = pd.to_numeric(frame["draft_year"], errors="coerce") if "draft_year" in frame.columns else pd.Series(np.nan, index=frame.index)
    return stat.where(stat.notna(), draft)


def _holdout_mask(frame: pd.DataFrame) -> pd.Series:
    season = _rookie_season(frame)
    return season.isin(HOLDOUT_ROOKIE_SEASONS)


def _split_train_val(position_df: pd.DataFrame, val_season: int | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if val_season is None or position_df.empty:
        return position_df, position_df.iloc[0:0]

    season = _rookie_season(position_df)
    val = position_df[season == val_season]
    train = position_df[season != val_season]
    if len(val) < 5 or len(train) < 15:
        work = position_df.copy()
        work["_rookie_season"] = season
        ordered = work.sort_values("_rookie_season")
        cut = max(int(len(ordered) * 0.8), 15)
        return ordered.iloc[:cut].drop(columns=["_rookie_season"]), ordered.iloc[cut:].drop(columns=["_rookie_season"])
    return train, val


def train_and_score(
    master: pd.DataFrame,
    feature_corr: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Train one tuned HistGradientBoosting model per position on composite features.
    Hyperparameters are selected on TUNING_ROOKIE_SEASON; metrics reported on HOLDOUT_ROOKIE_SEASONS.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if master.empty or TARGET not in master.columns:
        return pd.DataFrame(), pd.DataFrame(), {}

    labeled = master[master[TARGET].notna()].copy()
    if labeled.empty:
        return pd.DataFrame(), pd.DataFrame(), {}

    holdout_mask = _holdout_mask(labeled)
    train_labeled = labeled[~holdout_mask].copy()
    val_season = TUNING_ROOKIE_SEASON

    artifacts = build_composite_artifacts(train_labeled, feature_corr)
    save_composite_artifacts(artifacts, MODELS_DIR / COMPOSITE_FILE)

    composites_all = apply_composites(master, artifacts, feature_corr)
    composite_corr = build_composite_correlation(composites_all, master)

    id_cols = [c for c in ID_COLUMNS if c in master.columns]
    labeled_extra = [TARGET]
    if "first_stat_season" in labeled.columns:
        labeled_extra.append("first_stat_season")
    if "draft_year" in labeled.columns and "draft_year" not in composites_all.columns:
        labeled_extra.append("draft_year")
    labeled_composites = composites_all.merge(labeled[id_cols + labeled_extra], on=id_cols, how="inner")
    if "first_stat_season" not in labeled_composites.columns and "first_stat_season" in master.columns:
        labeled_composites = labeled_composites.merge(
            master[id_cols + ["first_stat_season"]], on=id_cols, how="left"
        )

    models_by_position: dict[str, HistGradientBoostingRegressor] = {}
    feature_cols_by_position: dict[str, list[str]] = {}
    best_params_by_position: dict[str, dict[str, Any]] = {}
    tuning_r_by_position: dict[str, float] = {}

    for position in SKILL_POSITIONS:
        pos_df = labeled_composites[labeled_composites["position"] == position].copy()
        if pos_df.empty:
            continue

        feature_cols = _position_feature_cols(position)
        feature_cols_by_position[position] = feature_cols

        pre_holdout = pos_df[~_holdout_mask(pos_df)]
        tune_train, tune_val = _split_train_val(pre_holdout, val_season)

        _, best_params, tune_r = _tune_position_model(position, tune_train, tune_val, feature_cols)
        best_params_by_position[position] = best_params
        tuning_r_by_position[position] = round(tune_r, 4) if pd.notna(tune_r) else None

        final_model = HistGradientBoostingRegressor(random_state=42, **best_params)
        X_fit = _feature_matrix(pre_holdout, feature_cols)
        y_fit = pd.to_numeric(pre_holdout[TARGET], errors="coerce")
        final_model.fit(X_fit, y_fit)
        models_by_position[position] = final_model

    bundle = {
        "model_type": "position_specific",
        "models_by_position": models_by_position,
        "feature_cols_by_position": feature_cols_by_position,
        "best_params_by_position": best_params_by_position,
        "holdout_rookie_seasons": list(HOLDOUT_ROOKIE_SEASONS),
    }
    joblib.dump(bundle, MODELS_DIR / MODEL_FILE)

    percentile_lookup = _percentile_lookup(train_labeled)
    (MODELS_DIR / PERCENTILE_FILE).write_text(json.dumps(percentile_lookup), encoding="utf-8")

    ml_features = composites_all.copy()
    preds = np.full(len(ml_features), np.nan, dtype=float)
    for position, model in models_by_position.items():
        mask = ml_features["position"] == position
        if not mask.any():
            continue
        cols = feature_cols_by_position[position]
        preds[mask.to_numpy()] = model.predict(_feature_matrix(ml_features.loc[mask], cols))
    ml_features["predicted_rookie_ppr"] = preds
    ml_features["success_score_0_100"] = [
        _to_success_score(pred, pos, percentile_lookup)
        for pred, pos in zip(ml_features["predicted_rookie_ppr"], ml_features["position"], strict=False)
    ]

    master_extra = [c for c in [TARGET, "first_stat_season", "draft_year"] if c in master.columns and c not in ml_features.columns]
    if master_extra:
        ml_features = ml_features.merge(master[id_cols + master_extra], on=id_cols, how="left")

    metrics: dict[str, Any] = {
        "model_type": "position_specific",
        "train_rows": int(len(train_labeled)),
        "holdout_rookie_seasons": list(HOLDOUT_ROOKIE_SEASONS),
        "tuning_validation_season": val_season,
        "tuning_r_by_position": tuning_r_by_position,
        "best_params_by_position": best_params_by_position,
        "feature_cols_by_position": feature_cols_by_position,
    }

    if TARGET in ml_features.columns:
        holdout_df = ml_features[_holdout_mask(ml_features)].dropna(subset=["predicted_rookie_ppr", TARGET])
        if not holdout_df.empty:
            y_true = pd.to_numeric(holdout_df[TARGET], errors="coerce")
            y_pred = holdout_df["predicted_rookie_ppr"].to_numpy()
            metrics["holdout_rows"] = int(len(holdout_df))
            metrics["holdout_mae"] = round(float(mean_absolute_error(y_true, y_pred)), 3)
            metrics["holdout_r2"] = round(float(r2_score(y_true, y_pred)), 3)
            metrics["holdout_pearson_r"] = round(_pearson_r(y_true, y_pred), 4)

            by_pos = {}
            for position, grp in holdout_df.groupby("position"):
                yt = pd.to_numeric(grp[TARGET], errors="coerce")
                yp = grp["predicted_rookie_ppr"].to_numpy()
                by_pos[position] = {
                    "n": int(len(grp)),
                    "pearson_r": round(_pearson_r(yt, yp), 4),
                    "mae": round(float(mean_absolute_error(yt, yp)), 3),
                }
            metrics["holdout_by_position"] = by_pos

    (MODELS_DIR / METRICS_FILE).write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    return ml_features, composite_corr, metrics


def load_model_bundle() -> dict:
    path = MODELS_DIR / MODEL_FILE
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}. Run python -m rookie_ppr.compile first.")
    return joblib.load(path)


def predict_from_composites(
    composites_row: pd.DataFrame,
    position: str,
    bundle: dict | None = None,
    percentile_lookup: dict[str, list[float]] | None = None,
) -> pd.DataFrame:
    bundle = bundle or load_model_bundle()
    if percentile_lookup is None:
        percentile_lookup = json.loads((MODELS_DIR / PERCENTILE_FILE).read_text(encoding="utf-8"))

    row = composites_row.copy()
    models = bundle.get("models_by_position", {})
    feature_cols_map = bundle.get("feature_cols_by_position", {})

    if position not in models:
        raise ValueError(f"No trained model for position {position}. Available: {list(models)}")

    feature_cols = feature_cols_map.get(position, _position_feature_cols(position))
    model = models[position]
    X = _feature_matrix(row, feature_cols)

    pred = float(model.predict(X)[0])
    score = _to_success_score(pred, position, percentile_lookup)
    row["predicted_rookie_ppr"] = pred
    row["success_score_0_100"] = score
    return row
