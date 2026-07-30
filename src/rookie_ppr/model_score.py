from __future__ import annotations

import json
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from rookie_ppr.analysis_config import (
    DYNASTY_MODE,
    REDRAFT_MODE,
    AnalysisMode,
    get_mode,
)
from rookie_ppr.analyze_correlation import NUMERIC_FEATURES, _add_derived_columns
from rookie_ppr.config import (
    CSV_OUTPUT_DIR,
    HOLDOUT_ROOKIE_SEASONS,
    MODELS_DIR,
    SKILL_POSITIONS,
    TUNING_ROOKIE_SEASON,
)
from rookie_ppr.feature_composites import (
    ID_COLUMNS,
    SCORE_COLUMNS,
    apply_composites,
    build_composite_artifacts,
    build_composite_correlation,
    feature_corr_for_weights,
    save_composite_artifacts,
)

TARGET = "rookie_ppr"
MODEL_FILE = "hgb_rookie_ppr.joblib"
COMPOSITE_FILE = "composite_weights.json"
PERCENTILE_FILE = "percentile_lookup.json"
METRICS_FILE = "model_metrics.json"
ADP_BASELINE_CSV = "model_vs_adp_baseline.csv"
ML_FEATURES_DYNASTY_CSV = "ml_features_dynasty.csv"
COMPOSITE_CORR_DYNASTY_CSV = "composite_correlation_dynasty.csv"
DYNASTY_ABS_CORR_COL = "abs_corr_dynasty_ppr"

# True ADP-only baseline (not the full pre-draft fantasy composite)
ADP_FEATURE_COLS = ["ff_adp", "ff_adp_rank"]

# Conformalized quantile regression (CQR): nominal 80% intervals
CQR_ALPHA = 0.2
CQR_Q_LO = 0.1
CQR_Q_HI = 0.9
# Predictive IQR for boom/bust gauge (25% chance below Q1 / above Q3 of error model)
PRED_Q25 = 0.25
PRED_Q50 = 0.5
PRED_Q75 = 0.75
CQR_MIN_CAL = 8


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
    # TE: lean on draft capital + ADP + receiving/production context; drop noisy combine blocks
    "TE": [
        "score_draft_capital",
        "score_pre_draft_fantasy",
        "score_recruiting",
        "score_timing",
        "score_cfb_receiving",
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

# RB: exclude deep / low-leaf / no-L2 configs that memorize rare late boom paths
RB_PARAM_GRID: list[dict[str, Any]] = [
    {"max_depth": 2, "learning_rate": 0.03, "max_iter": 200, "min_samples_leaf": 25, "l2_regularization": 2.0},
    {"max_depth": 2, "learning_rate": 0.05, "max_iter": 250, "min_samples_leaf": 20, "l2_regularization": 2.0},
    {"max_depth": 3, "learning_rate": 0.05, "max_iter": 250, "min_samples_leaf": 20, "l2_regularization": 1.5},
    {"max_depth": 3, "learning_rate": 0.05, "max_iter": 300, "min_samples_leaf": 15, "l2_regularization": 1.0},
]

# When ADP is a feature, allow a bit more capacity so early-ADP / high-volume RB
# leaves are not forced into the same shallow mean-reverting surface.
RB_PARAM_GRID_WITH_MARKET: list[dict[str, Any]] = RB_PARAM_GRID + [
    {"max_depth": 4, "learning_rate": 0.05, "max_iter": 300, "min_samples_leaf": 12, "l2_regularization": 0.5},
    {"max_depth": 4, "learning_rate": 0.05, "max_iter": 350, "min_samples_leaf": 10, "l2_regularization": 0.25},
    {"max_depth": 5, "learning_rate": 0.05, "max_iter": 300, "min_samples_leaf": 8, "l2_regularization": 0.25},
]

# TE: small-n friendly — shallow, high L2, prioritize stable ADP/draft signal
TE_PARAM_GRID: list[dict[str, Any]] = [
    {"max_depth": 2, "learning_rate": 0.03, "max_iter": 200, "min_samples_leaf": 20, "l2_regularization": 3.0},
    {"max_depth": 2, "learning_rate": 0.05, "max_iter": 250, "min_samples_leaf": 15, "l2_regularization": 2.0},
    {"max_depth": 3, "learning_rate": 0.05, "max_iter": 250, "min_samples_leaf": 15, "l2_regularization": 2.0},
    {"max_depth": 3, "learning_rate": 0.05, "max_iter": 300, "min_samples_leaf": 12, "l2_regularization": 1.0},
]

# WR holdout favored pre-draft fantasy; allow a lighter fantasy-heavy grid slice
WR_PARAM_GRID: list[dict[str, Any]] = PARAM_GRID + [
    {"max_depth": 2, "learning_rate": 0.1, "max_iter": 150, "min_samples_leaf": 30, "l2_regularization": 3.0},
    {"max_depth": 3, "learning_rate": 0.1, "max_iter": 200, "min_samples_leaf": 25, "l2_regularization": 2.0},
]

# Dynasty residual path: exclude score_pre_draft_fantasy (ADP handled via residual baseline)
POSITION_FEATURE_COLS_DYNASTY: dict[str, list[str]] = {
    "QB": [
        "score_draft_capital",
        "score_recruiting",
        "score_timing",
        "score_cfb_passing",
        "score_team_context",
    ],
    "RB": [
        "score_draft_capital",
        "score_recruiting",
        "score_timing",
        "score_cfb_rushing",
        "score_cfb_receiving",
        "score_combine_speed",
        "score_team_context",
    ],
    # WR: drop combine + pre-draft fantasy (ADP via residual path)
    "WR": [
        "score_draft_capital",
        "score_recruiting",
        "score_timing",
        "score_cfb_receiving",
        "score_team_context",
    ],
    "TE": [
        "score_draft_capital",
        "score_recruiting",
        "score_timing",
        "score_cfb_receiving",
        "score_team_context",
    ],
}

CFB_COMPOSITE_COLS = ["score_cfb_rushing", "score_cfb_receiving", "score_cfb_passing"]
CFB_GATE_DRAFT_OVERALL = 50
CFB_GATE_FF_ADP = 40

# Kept for loading older ADP-residual dynasty joblibs (current trainer is position_specific)
ADP_EXPECTED_COL = "adp_expected_dynasty_total"
DYNASTY_ECR_COLS = ["dynasty_ecr", "dynasty_ecr_rank"]

# Composites are position z-score averages; 0 = typical player on that axis
TYPICAL_COMPOSITE_BASELINE = 0.0


def _percentile_lookup(train_df: pd.DataFrame, target: str = TARGET) -> dict[str, list[float]]:
    lookup: dict[str, list[float]] = {}
    if target not in train_df.columns or "position" not in train_df.columns:
        return lookup
    for pos, grp in train_df.groupby("position"):
        vals = pd.to_numeric(grp[target], errors="coerce").dropna().sort_values().tolist()
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


def _attach_adp_features(frame: pd.DataFrame, master: pd.DataFrame, id_cols: list[str]) -> pd.DataFrame:
    """Merge raw ADP columns from master when missing on the scoring frame."""
    adp_cols = [c for c in ADP_FEATURE_COLS if c in master.columns]
    if not adp_cols or not id_cols:
        return frame
    out = frame.copy()
    missing = [c for c in adp_cols if c not in out.columns]
    if not missing:
        return out
    keys = [c for c in id_cols if c in out.columns and c in master.columns]
    if not keys:
        return out
    return out.merge(master[keys + missing].drop_duplicates(subset=keys), on=keys, how="left")


def _holdout_metric_block(y_true: pd.Series, y_pred: np.ndarray, positions: pd.Series) -> dict[str, Any]:
    """Overall + by-position Pearson r / MAE for a prediction vector."""
    yt = pd.to_numeric(y_true, errors="coerce")
    block: dict[str, Any] = {
        "n": int(len(yt)),
        "pearson_r": round(_pearson_r(yt, y_pred), 4) if len(yt) >= 5 else None,
        "mae": round(float(mean_absolute_error(yt, y_pred)), 3) if len(yt) else None,
    }
    by_pos: dict[str, Any] = {}
    work = pd.DataFrame({"y": yt, "p": y_pred, "position": positions})
    for position, grp in work.groupby("position"):
        n = int(len(grp))
        by_pos[str(position)] = {
            "n": n,
            "pearson_r": round(_pearson_r(grp["y"], grp["p"].to_numpy()), 4) if n >= 5 else None,
            "mae": round(float(mean_absolute_error(grp["y"], grp["p"])), 3) if n else None,
        }
    block["by_position"] = by_pos
    return block


def _export_adp_baseline_csv(comparable: dict[str, Any], filename: str = ADP_BASELINE_CSV) -> None:
    """Write overall + by-position full vs ADP-only comparison CSV."""
    CSV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = [
        {
            "scope": "overall",
            "position": "",
            "n": comparable.get("n"),
            "full_r": comparable.get("full_r"),
            "adp_r": comparable.get("adp_r"),
            "lift_r": comparable.get("lift_r"),
            "full_mae": comparable.get("full_mae"),
            "adp_mae": comparable.get("adp_mae"),
            "lift_mae": comparable.get("lift_mae"),
        }
    ]
    for position, stats_row in (comparable.get("by_position") or {}).items():
        rows.append(
            {
                "scope": "by_position",
                "position": position,
                "n": stats_row.get("n"),
                "full_r": stats_row.get("full_r"),
                "adp_r": stats_row.get("adp_r"),
                "lift_r": stats_row.get("lift_r"),
                "full_mae": stats_row.get("full_mae"),
                "adp_mae": stats_row.get("adp_mae"),
                "lift_mae": stats_row.get("lift_mae"),
            }
        )
    pd.DataFrame(rows).to_csv(CSV_OUTPUT_DIR / filename, index=False)


def _position_feature_cols(position: str) -> list[str]:
    cols = POSITION_FEATURE_COLS.get(position, SCORE_COLUMNS)
    return [c for c in cols if c in SCORE_COLUMNS]


def _position_feature_cols_dynasty(position: str, available_cols: list[str] | set[str] | None = None) -> list[str]:
    """Dynasty residual features: position slim set, never score_pre_draft_fantasy.

    Market ECR columns are intentionally excluded here (reported as baselines only)
    to avoid current-market leakage into historical holdouts.
    """
    cols = list(POSITION_FEATURE_COLS_DYNASTY.get(position, POSITION_FEATURE_COLS.get(position, SCORE_COLUMNS)))
    cols = [c for c in cols if c != "score_pre_draft_fantasy" and c in SCORE_COLUMNS]
    if available_cols is not None:
        avail = set(available_cols)
        cols = [c for c in cols if c in avail]
    return cols


def _param_grid_for_position(position: str) -> list[dict[str, Any]]:
    if position == "WR":
        return WR_PARAM_GRID
    if position == "RB":
        return RB_PARAM_GRID
    if position == "TE":
        return TE_PARAM_GRID
    return PARAM_GRID


def _apply_cfb_gate(frame: pd.DataFrame) -> pd.DataFrame:
    """Zero/NaN CFB composites for early-draft or strong-ADP elites (legacy residual path)."""
    if frame.empty:
        return frame
    out = frame.copy()
    draft = (
        pd.to_numeric(out["draft_overall"], errors="coerce")
        if "draft_overall" in out.columns
        else pd.Series(np.nan, index=out.index)
    )
    adp = (
        pd.to_numeric(out["ff_adp"], errors="coerce")
        if "ff_adp" in out.columns
        else pd.Series(np.nan, index=out.index)
    )
    gate = (draft.notna() & (draft <= CFB_GATE_DRAFT_OVERALL)) | (adp.notna() & (adp <= CFB_GATE_FF_ADP))
    if not gate.any():
        return out
    for col in CFB_COMPOSITE_COLS:
        if col in out.columns:
            out.loc[gate, col] = np.nan
    return out


def _fit_point_model(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    params: dict[str, Any],
    target: str = TARGET,
    *,
    monotonic_cst: list[int] | None = None,
) -> HistGradientBoostingRegressor:
    """Median (q50) point model — robust to right-skewed rookie outcomes."""
    X = _feature_matrix(train_df, feature_cols)
    y = pd.to_numeric(train_df[target], errors="coerce")
    fit_kwargs: dict[str, Any] = {
        "loss": "quantile",
        "quantile": PRED_Q50,
        "random_state": 42,
        **params,
    }
    if monotonic_cst is not None:
        fit_kwargs["monotonic_cst"] = monotonic_cst
    model = HistGradientBoostingRegressor(**fit_kwargs)
    try:
        model.fit(X, y)
    except ValueError:
        # Rare binning failure with mono constraints on sparse/gated columns
        if monotonic_cst is not None:
            fit_kwargs.pop("monotonic_cst", None)
            model = HistGradientBoostingRegressor(**fit_kwargs)
            model.fit(X, y)
        else:
            raise
    return model


# How hard the point estimate is pulled into the predictive IQR.
# 1.0 = hard clip (old behavior); 0.0 = leave the median prediction alone.
# The IQR is fit on realized outcomes, so a hard clip bakes injury/missed-game
# mean-reversion into every printed number. Soft strength lets boom/recovery
# medians sit outside that band while still damping extreme outliers.
DEFAULT_IQR_CLAMP_STRENGTH = 0.4

# Prior seasons shorter than this are treated as injury-truncated for IQR relief.
# Pull the median toward q75 (less injury-pessimistic edge) before soft-clamping.
INJURY_IQR_GAMES_CUTOFF = 8.0
DEFAULT_INJURY_IQR_RELIEF = 0.75


def _clamp_pred_to_iqr(
    pred: float,
    q25: float,
    q75: float,
    *,
    strength: float = DEFAULT_IQR_CLAMP_STRENGTH,
    prior_games: float | None = None,
    injury_relief: float = DEFAULT_INJURY_IQR_RELIEF,
    injury_games_cutoff: float = INJURY_IQR_GAMES_CUTOFF,
) -> float:
    """Soft-pull the point estimate toward the predictive IQR.

    ``strength=1`` hard-clips into ``[q25, q75]``. ``strength=0`` returns ``pred``
    unchanged (after optional injury relief). Values in between blend the raw
    median with the clipped value so injury-aware IQR bounds do not fully
    override the point model.

    When ``prior_games`` is below ``injury_games_cutoff``, blend toward ``q75``
    first — the upper quartile is the less injury-pessimistic edge of the band,
    so short prior seasons are not clamped as hard into a truncated-year median.
    """
    if not np.isfinite(pred):
        return pred
    lo, hi = float(q25), float(q75)
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return float(pred)
    if lo > hi:
        lo, hi = hi, lo

    adjusted = float(pred)
    if (
        prior_games is not None
        and np.isfinite(prior_games)
        and injury_games_cutoff > 0
        and float(prior_games) < float(injury_games_cutoff)
        and np.isfinite(hi)
    ):
        w = float(injury_relief) * (
            1.0 - float(prior_games) / float(injury_games_cutoff)
        )
        w = float(np.clip(w, 0.0, 1.0))
        if w > 0.0:
            # Never pull *down* for injury relief; only lift toward / past q75.
            adjusted = (1.0 - w) * adjusted + w * max(hi, adjusted)

    hard = float(np.clip(adjusted, lo, hi))
    s = float(np.clip(strength, 0.0, 1.0))
    if s >= 1.0:
        return hard
    if s <= 0.0:
        return float(adjusted)
    return float((1.0 - s) * float(adjusted) + s * hard)


def _tune_position_model(
    position: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    target: str = TARGET,
    param_grid: list[dict[str, Any]] | None = None,
    monotonic_cst: list[int] | None = None,
) -> tuple[HistGradientBoostingRegressor, dict[str, Any], float]:
    """Pick hyperparameters maximizing Pearson r on the validation fold (median model)."""
    grid = param_grid if param_grid is not None else _param_grid_for_position(position)
    best_r = float("-inf")
    best_params: dict[str, Any] = grid[0]
    best_model: HistGradientBoostingRegressor | None = None

    X_val = _feature_matrix(val_df, feature_cols)
    y_val = pd.to_numeric(val_df[target], errors="coerce")

    for params in grid:
        if len(train_df) < 15:
            continue
        model = _fit_point_model(
            train_df, feature_cols, params, target=target, monotonic_cst=monotonic_cst
        )
        pred_val = model.predict(X_val) if len(X_val) >= 5 else model.predict(_feature_matrix(train_df, feature_cols))
        y_eval = y_val if len(X_val) >= 5 else pd.to_numeric(train_df[target], errors="coerce")
        r = _pearson_r(y_eval, pred_val)
        if pd.notna(r) and r > best_r:
            best_r = r
            best_params = params
            best_model = model

    if best_model is None:
        best_params = {"max_depth": 3, "learning_rate": 0.05, "max_iter": 250, "min_samples_leaf": 15, "l2_regularization": 1.0}
        best_model = _fit_point_model(
            train_df, feature_cols, best_params, target=target, monotonic_cst=monotonic_cst
        )
        best_r = _pearson_r(
            pd.to_numeric(train_df[target], errors="coerce"),
            best_model.predict(_feature_matrix(train_df, feature_cols)),
        )

    return best_model, best_params, best_r


def _rookie_season(frame: pd.DataFrame) -> pd.Series:
    """Rookie fantasy season: first_stat_season when present, else draft_year."""
    stat = pd.to_numeric(frame["first_stat_season"], errors="coerce") if "first_stat_season" in frame.columns else pd.Series(np.nan, index=frame.index)
    draft = pd.to_numeric(frame["draft_year"], errors="coerce") if "draft_year" in frame.columns else pd.Series(np.nan, index=frame.index)
    return stat.where(stat.notna(), draft)


def _holdout_mask(frame: pd.DataFrame, holdout_seasons: tuple[int, ...] = HOLDOUT_ROOKIE_SEASONS) -> pd.Series:
    season = _rookie_season(frame)
    return season.isin(holdout_seasons)


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


def _split_fit_cal(position_df: pd.DataFrame, cal_season: int | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit vs calibration split for inductive CQR (prefer cal_season; fallback last ~20%)."""
    if position_df.empty:
        return position_df, position_df.iloc[0:0]

    season = _rookie_season(position_df)
    if cal_season is not None:
        cal = position_df[season == cal_season]
        fit = position_df[season != cal_season]
        if len(cal) >= CQR_MIN_CAL and len(fit) >= 15:
            return fit, cal

    work = position_df.copy()
    work["_rookie_season"] = season
    ordered = work.sort_values("_rookie_season")
    cut = max(int(len(ordered) * 0.8), 15)
    if len(ordered) - cut < CQR_MIN_CAL:
        cut = max(len(ordered) - CQR_MIN_CAL, 15)
    if cut >= len(ordered):
        return position_df, position_df.iloc[0:0]
    return ordered.iloc[:cut].drop(columns=["_rookie_season"]), ordered.iloc[cut:].drop(columns=["_rookie_season"])


def _conformal_q_hat(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample conformal quantile of conformity scores."""
    s = np.sort(np.asarray(scores, dtype=float))
    s = s[np.isfinite(s)]
    if s.size == 0:
        return 0.0
    n = s.size
    # level = ceil((n+1)(1-alpha)) / n  → 0-based index
    k = int(np.ceil((n + 1) * (1.0 - alpha))) - 1
    k = min(max(k, 0), n - 1)
    return float(s[k])


def _fit_quantile_model(
    fit_df: pd.DataFrame,
    feature_cols: list[str],
    params: dict[str, Any],
    quantile: float,
    *,
    target: str = TARGET,
    monotonic_cst: list[int] | None = None,
) -> HistGradientBoostingRegressor:
    X = _feature_matrix(fit_df, feature_cols)
    y = pd.to_numeric(fit_df[target], errors="coerce")
    fit_kwargs: dict[str, Any] = {
        "loss": "quantile",
        "quantile": quantile,
        "random_state": 42,
        **params,
    }
    if monotonic_cst is not None:
        fit_kwargs["monotonic_cst"] = monotonic_cst
    model = HistGradientBoostingRegressor(**fit_kwargs)
    try:
        model.fit(X, y)
    except ValueError:
        if monotonic_cst is not None:
            fit_kwargs.pop("monotonic_cst", None)
            model = HistGradientBoostingRegressor(**fit_kwargs)
            model.fit(X, y)
        else:
            raise
    return model


def _fit_quantile_pair(
    fit_df: pd.DataFrame,
    feature_cols: list[str],
    params: dict[str, Any],
    *,
    target: str = TARGET,
    monotonic_cst: list[int] | None = None,
) -> tuple[HistGradientBoostingRegressor, HistGradientBoostingRegressor]:
    return (
        _fit_quantile_model(
            fit_df, feature_cols, params, CQR_Q_LO, target=target, monotonic_cst=monotonic_cst
        ),
        _fit_quantile_model(
            fit_df, feature_cols, params, CQR_Q_HI, target=target, monotonic_cst=monotonic_cst
        ),
    )


def _fit_predictive_quartile_pair(
    fit_df: pd.DataFrame,
    feature_cols: list[str],
    params: dict[str, Any],
    *,
    target: str = TARGET,
    monotonic_cst: list[int] | None = None,
) -> tuple[HistGradientBoostingRegressor, HistGradientBoostingRegressor]:
    """25th / 75th percentile models: P(Y < q25)≈25%, P(Y > q75)≈25%."""
    return (
        _fit_quantile_model(
            fit_df, feature_cols, params, PRED_Q25, target=target, monotonic_cst=monotonic_cst
        ),
        _fit_quantile_model(
            fit_df, feature_cols, params, PRED_Q75, target=target, monotonic_cst=monotonic_cst
        ),
    )


def _build_dynasty_feature_corr(master: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Feature abs-corr vs dynasty target on complete-label rows.
    Emits abs_corr_dynasty_ppr; use feature_corr_for_weights() for composite weights.
    """
    if master.empty or target not in master.columns:
        return pd.DataFrame()
    work = _add_derived_columns(master)
    if "dynasty_seasons_complete" in work.columns:
        work = work[pd.to_numeric(work["dynasty_seasons_complete"], errors="coerce") == 1].copy()
    y = pd.to_numeric(work[target], errors="coerce")
    rows: list[dict[str, Any]] = []
    abs_corrs: list[float] = []
    for feature in NUMERIC_FEATURES:
        if feature not in work.columns:
            continue
        x = pd.to_numeric(work[feature], errors="coerce")
        pair = pd.DataFrame({"x": x, "y": y}).dropna()
        if len(pair) < 10 or pair["x"].nunique() < 2:
            continue
        pearson_r, _ = stats.pearsonr(pair["x"], pair["y"])
        abs_r = abs(float(pearson_r))
        abs_corrs.append(abs_r)
        rows.append(
            {
                "feature": feature,
                "n_pairs": int(len(pair)),
                "pearson_r": round(float(pearson_r), 4),
                DYNASTY_ABS_CORR_COL: round(abs_r, 4),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    max_abs = max(abs_corrs) if abs_corrs else 1.0
    out["impact_0_100"] = (out[DYNASTY_ABS_CORR_COL] / max_abs * 100).round(1)
    return out.sort_values(DYNASTY_ABS_CORR_COL, ascending=False).reset_index(drop=True)


def _cqr_interval_row(
    x_row: pd.DataFrame,
    lo_model: HistGradientBoostingRegressor,
    hi_model: HistGradientBoostingRegressor,
    q_hat: float,
) -> tuple[float, float, float, float]:
    q_lo = float(lo_model.predict(x_row)[0])
    q_hi = float(hi_model.predict(x_row)[0])
    if q_lo > q_hi:
        q_lo, q_hi = q_hi, q_lo
    low = q_lo - q_hat
    high = q_hi + q_hat
    return low, high, q_lo, q_hi


def asymmetric_endpoint_chances(
    predicted: float,
    ppr_low: float,
    ppr_high: float,
    residuals: list[float] | np.ndarray | None,
    *,
    alpha: float = CQR_ALPHA,
) -> tuple[float, float]:
    """
    Position-calibrated P(Y < low) and P(Y > high) from residual ECDF.

    residuals are y - pred on the CQR calibration fold (fit-fold point model).
    Falls back to equal-tailed alpha/2 when residuals are unavailable.
    """
    fallback = round(100.0 * float(alpha) / 2.0, 1)
    if residuals is None:
        return fallback, fallback
    r = np.asarray(residuals, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return fallback, fallback
    low_gap = float(ppr_low) - float(predicted)
    high_gap = float(ppr_high) - float(predicted)
    # Finite-sample inclusive ECDF (add-one smoothing toward equal tails)
    bust = 100.0 * float((np.sum(r <= low_gap) + 1) / (r.size + 2))
    boom = 100.0 * float((np.sum(r >= high_gap) + 1) / (r.size + 2))
    bust = float(np.clip(bust, 0.5, 95.0))
    boom = float(np.clip(boom, 0.5, 95.0))
    return round(bust, 1), round(boom, 1)


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
    labeled_composites = _attach_adp_features(labeled_composites, master, id_cols)

    models_by_position: dict[str, HistGradientBoostingRegressor] = {}
    feature_cols_by_position: dict[str, list[str]] = {}
    best_params_by_position: dict[str, dict[str, Any]] = {}
    tuning_r_by_position: dict[str, float] = {}
    quantile_models_by_position: dict[str, dict[str, HistGradientBoostingRegressor]] = {}
    conformal_by_position: dict[str, dict[str, Any]] = {}
    adp_models_by_position: dict[str, HistGradientBoostingRegressor] = {}
    adp_params_by_position: dict[str, dict[str, Any]] = {}
    adp_tuning_r_by_position: dict[str, float | None] = {}

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

        final_model = _fit_point_model(pre_holdout, feature_cols, best_params)
        models_by_position[position] = final_model

        # Inductive CQR + predictive quartile (q25/q75) models
        cqr_fit, cqr_cal = _split_fit_cal(pre_holdout, val_season)
        if len(cqr_fit) >= 15 and len(cqr_cal) >= CQR_MIN_CAL:
            lo_model, hi_model = _fit_quantile_pair(cqr_fit, feature_cols, best_params)
            q25_model, q75_model = _fit_predictive_quartile_pair(
                cqr_fit, feature_cols, best_params
            )
            X_cal = _feature_matrix(cqr_cal, feature_cols)
            y_cal = pd.to_numeric(cqr_cal[TARGET], errors="coerce").to_numpy(dtype=float)
            q_lo_cal = lo_model.predict(X_cal)
            q_hi_cal = hi_model.predict(X_cal)
            # Ensure lo <= hi per row before scoring
            swap = q_lo_cal > q_hi_cal
            if np.any(swap):
                q_lo_cal, q_hi_cal = q_lo_cal.copy(), q_hi_cal.copy()
                q_lo_cal[swap], q_hi_cal[swap] = q_hi_cal[swap], q_lo_cal[swap]
            scores = np.maximum(q_lo_cal - y_cal, y_cal - q_hi_cal)
            q_hat = _conformal_q_hat(scores, CQR_ALPHA)

            # Residuals from fit-fold median model (fallback predictive IQR)
            point_fit = _fit_point_model(cqr_fit, feature_cols, best_params)
            pred_cal = point_fit.predict(X_cal)
            residuals = (y_cal - pred_cal).tolist()
            residuals = [round(float(v), 4) for v in residuals if np.isfinite(v)]
            r_arr = np.asarray(residuals, dtype=float)
            residual_q25 = float(np.percentile(r_arr, 25)) if r_arr.size else 0.0
            residual_q75 = float(np.percentile(r_arr, 75)) if r_arr.size else 0.0

            s_lo = (q_lo_cal - y_cal).tolist()
            s_hi = (y_cal - q_hi_cal).tolist()

            quantile_models_by_position[position] = {
                "lo": lo_model,
                "hi": hi_model,
                "q25": q25_model,
                "q75": q75_model,
            }
            conformal_by_position[position] = {
                "alpha": CQR_ALPHA,
                "q_lo": CQR_Q_LO,
                "q_hi": CQR_Q_HI,
                "q_hat": round(q_hat, 4),
                "n_cal": int(len(cqr_cal)),
                "n_fit": int(len(cqr_fit)),
                "residuals": residuals,
                "residual_q25": round(residual_q25, 4),
                "residual_q75": round(residual_q75, 4),
                "one_sided_scores_lo": [round(float(v), 4) for v in s_lo if np.isfinite(v)],
                "one_sided_scores_hi": [round(float(v), 4) for v in s_hi if np.isfinite(v)],
            }

        # ADP-only baseline: same position loop / holdout seasons; median HGB on ff_adp + ff_adp_rank
        adp_available = [c for c in ADP_FEATURE_COLS if c in pos_df.columns]
        if len(adp_available) == len(ADP_FEATURE_COLS):
            pre_holdout_adp = pre_holdout.dropna(subset=ADP_FEATURE_COLS)
            if len(pre_holdout_adp) >= 15:
                tune_train_adp, tune_val_adp = _split_train_val(pre_holdout_adp, val_season)
                _, adp_params, adp_tune_r = _tune_position_model(
                    position, tune_train_adp, tune_val_adp, ADP_FEATURE_COLS
                )
                adp_params_by_position[position] = adp_params
                adp_tuning_r_by_position[position] = (
                    round(adp_tune_r, 4) if pd.notna(adp_tune_r) else None
                )
                adp_models_by_position[position] = _fit_point_model(
                    pre_holdout_adp, ADP_FEATURE_COLS, adp_params
                )

    bundle = {
        "model_type": "position_specific",
        "point_estimate": "median_q50",
        "models_by_position": models_by_position,
        "feature_cols_by_position": feature_cols_by_position,
        "best_params_by_position": best_params_by_position,
        "holdout_rookie_seasons": list(HOLDOUT_ROOKIE_SEASONS),
        "quantile_models_by_position": quantile_models_by_position,
        "conformal_by_position": conformal_by_position,
        "cqr_alpha": CQR_ALPHA,
    }
    joblib.dump(bundle, MODELS_DIR / MODEL_FILE)

    percentile_lookup = _percentile_lookup(train_labeled)
    (MODELS_DIR / PERCENTILE_FILE).write_text(json.dumps(percentile_lookup), encoding="utf-8")

    ml_features = composites_all.copy()
    preds = np.full(len(ml_features), np.nan, dtype=float)
    ppr_lows = np.full(len(ml_features), np.nan, dtype=float)
    ppr_highs = np.full(len(ml_features), np.nan, dtype=float)
    pq25s = np.full(len(ml_features), np.nan, dtype=float)
    pq75s = np.full(len(ml_features), np.nan, dtype=float)
    for position, model in models_by_position.items():
        mask = ml_features["position"] == position
        if not mask.any():
            continue
        cols = feature_cols_by_position[position]
        X_pos = _feature_matrix(ml_features.loc[mask], cols)
        pos_preds = model.predict(X_pos).astype(float)
        q_models = quantile_models_by_position.get(position)
        conf = conformal_by_position.get(position)
        if q_models and conf:
            q_hat = float(conf["q_hat"])
            q_lo = q_models["lo"].predict(X_pos)
            q_hi = q_models["hi"].predict(X_pos)
            swap = q_lo > q_hi
            if np.any(swap):
                q_lo, q_hi = q_lo.copy(), q_hi.copy()
                q_lo[swap], q_hi[swap] = q_hi[swap], q_lo[swap]
            ppr_lows[mask.to_numpy()] = q_lo - q_hat
            ppr_highs[mask.to_numpy()] = q_hi + q_hat
            if "q25" in q_models and "q75" in q_models:
                pq25 = q_models["q25"].predict(X_pos).astype(float)
                pq75 = q_models["q75"].predict(X_pos).astype(float)
                swap_q = pq25 > pq75
                if np.any(swap_q):
                    pq25, pq75 = pq25.copy(), pq75.copy()
                    pq25[swap_q], pq75[swap_q] = pq75[swap_q], pq25[swap_q]
                pq25s[mask.to_numpy()] = pq25
                pq75s[mask.to_numpy()] = pq75
                pos_preds = np.array(
                    [_clamp_pred_to_iqr(p, a, b) for p, a, b in zip(pos_preds, pq25, pq75)],
                    dtype=float,
                )
        preds[mask.to_numpy()] = pos_preds
    ml_features["predicted_rookie_ppr"] = preds
    ml_features["ppr_low"] = ppr_lows
    ml_features["ppr_high"] = ppr_highs
    ml_features["predictive_q25"] = pq25s
    ml_features["predictive_q75"] = pq75s
    ml_features["success_score_0_100"] = [
        _to_success_score(pred, pos, percentile_lookup)
        for pred, pos in zip(ml_features["predicted_rookie_ppr"], ml_features["position"], strict=False)
    ]

    master_extra = [c for c in [TARGET, "first_stat_season", "draft_year"] if c in master.columns and c not in ml_features.columns]
    if master_extra:
        ml_features = ml_features.merge(master[id_cols + master_extra], on=id_cols, how="left")
    ml_features = _attach_adp_features(ml_features, master, id_cols)

    adp_preds = np.full(len(ml_features), np.nan, dtype=float)
    for position, adp_model in adp_models_by_position.items():
        mask = ml_features["position"] == position
        if not mask.any():
            continue
        X_adp = _feature_matrix(ml_features.loc[mask], ADP_FEATURE_COLS)
        # Score only rows with both ADP features (matches training dropna)
        row_ok = X_adp.notna().all(axis=1)
        if not row_ok.any():
            continue
        pred_adp = np.full(mask.sum(), np.nan, dtype=float)
        pred_adp[row_ok.to_numpy()] = adp_model.predict(X_adp.loc[row_ok]).astype(float)
        adp_preds[mask.to_numpy()] = pred_adp
    ml_features["predicted_adp_baseline"] = adp_preds

    metrics: dict[str, Any] = {
        "model_type": "position_specific",
        "point_estimate": "median_q50",
        "train_rows": int(len(train_labeled)),
        "holdout_rookie_seasons": list(HOLDOUT_ROOKIE_SEASONS),
        "tuning_validation_season": val_season,
        "tuning_r_by_position": tuning_r_by_position,
        "best_params_by_position": best_params_by_position,
        "feature_cols_by_position": feature_cols_by_position,
        "cqr_alpha": CQR_ALPHA,
        "conformal_by_position": conformal_by_position,
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

            # CQR coverage on holdout (rows with finite intervals)
            interval_df = holdout_df.dropna(subset=["ppr_low", "ppr_high", TARGET])
            if not interval_df.empty:
                yt = pd.to_numeric(interval_df[TARGET], errors="coerce")
                covered = (yt >= interval_df["ppr_low"]) & (yt <= interval_df["ppr_high"])
                below = yt < interval_df["ppr_low"]
                above = yt > interval_df["ppr_high"]
                widths = interval_df["ppr_high"] - interval_df["ppr_low"]
                metrics["holdout_interval_coverage"] = round(float(covered.mean()), 4)
                metrics["holdout_interval_width_mean"] = round(float(widths.mean()), 2)
                metrics["holdout_interval_n"] = int(len(interval_df))
                metrics["holdout_below_low_rate"] = round(float(below.mean()), 4)
                metrics["holdout_above_high_rate"] = round(float(above.mean()), 4)

                pred_bust: list[float] = []
                pred_boom: list[float] = []
                cov_by_pos: dict[str, Any] = {}
                for position, grp in interval_df.groupby("position"):
                    yt_p = pd.to_numeric(grp[TARGET], errors="coerce")
                    cov = (yt_p >= grp["ppr_low"]) & (yt_p <= grp["ppr_high"])
                    w = grp["ppr_high"] - grp["ppr_low"]
                    below_p = yt_p < grp["ppr_low"]
                    above_p = yt_p > grp["ppr_high"]
                    residuals = (conformal_by_position.get(str(position)) or {}).get("residuals")
                    busts: list[float] = []
                    booms: list[float] = []
                    for _, r in grp.iterrows():
                        b, m = asymmetric_endpoint_chances(
                            float(r["predicted_rookie_ppr"]),
                            float(r["ppr_low"]),
                            float(r["ppr_high"]),
                            residuals,
                        )
                        busts.append(b)
                        booms.append(m)
                        pred_bust.append(b)
                        pred_boom.append(m)
                    cov_by_pos[str(position)] = {
                        "n": int(len(grp)),
                        "coverage": round(float(cov.mean()), 4),
                        "width_mean": round(float(w.mean()), 2),
                        "below_low_rate": round(float(below_p.mean()), 4),
                        "above_high_rate": round(float(above_p.mean()), 4),
                        "mean_bust_chance_pct": round(float(np.mean(busts)), 2) if busts else None,
                        "mean_boom_chance_pct": round(float(np.mean(booms)), 2) if booms else None,
                    }
                metrics["holdout_interval_by_position"] = cov_by_pos
                if pred_bust:
                    metrics["holdout_mean_bust_chance_pct"] = round(float(np.mean(pred_bust)), 2)
                    metrics["holdout_mean_boom_chance_pct"] = round(float(np.mean(pred_boom)), 2)

        # ADP-only baseline metrics + same-row lift vs full model
        if adp_models_by_position and "predicted_adp_baseline" in ml_features.columns:
            adp_holdout = ml_features[_holdout_mask(ml_features)].dropna(
                subset=["predicted_adp_baseline", TARGET]
            )
            adp_block: dict[str, Any] = {
                "features": list(ADP_FEATURE_COLS),
                "point_estimate": "median_q50",
                "tuning_r_by_position": adp_tuning_r_by_position,
                "best_params_by_position": adp_params_by_position,
            }
            if not adp_holdout.empty:
                adp_overall = _holdout_metric_block(
                    adp_holdout[TARGET],
                    adp_holdout["predicted_adp_baseline"].to_numpy(),
                    adp_holdout["position"],
                )
                adp_block["holdout_rows"] = adp_overall["n"]
                adp_block["holdout_pearson_r"] = adp_overall["pearson_r"]
                adp_block["holdout_mae"] = adp_overall["mae"]
                adp_block["holdout_by_position"] = adp_overall["by_position"]

            # Headline lift: same holdout rows with ADP + full prediction + target
            comparable_df = ml_features[_holdout_mask(ml_features)].dropna(
                subset=["predicted_rookie_ppr", "predicted_adp_baseline", TARGET]
            )
            if not comparable_df.empty:
                y_cmp = pd.to_numeric(comparable_df[TARGET], errors="coerce")
                full_pred = comparable_df["predicted_rookie_ppr"].to_numpy()
                adp_pred = comparable_df["predicted_adp_baseline"].to_numpy()
                full_r = _pearson_r(y_cmp, full_pred)
                adp_r = _pearson_r(y_cmp, adp_pred)
                full_mae = float(mean_absolute_error(y_cmp, full_pred))
                adp_mae = float(mean_absolute_error(y_cmp, adp_pred))
                comparable: dict[str, Any] = {
                    "n": int(len(comparable_df)),
                    "full_r": round(full_r, 4) if pd.notna(full_r) else None,
                    "adp_r": round(adp_r, 4) if pd.notna(adp_r) else None,
                    "lift_r": round(full_r - adp_r, 4) if pd.notna(full_r) and pd.notna(adp_r) else None,
                    "full_mae": round(full_mae, 3),
                    "adp_mae": round(adp_mae, 3),
                    "lift_mae": round(full_mae - adp_mae, 3),
                }
                by_pos_cmp: dict[str, Any] = {}
                for position, grp in comparable_df.groupby("position"):
                    yt = pd.to_numeric(grp[TARGET], errors="coerce")
                    fp = grp["predicted_rookie_ppr"].to_numpy()
                    ap = grp["predicted_adp_baseline"].to_numpy()
                    fr = _pearson_r(yt, fp)
                    ar = _pearson_r(yt, ap)
                    fmae = float(mean_absolute_error(yt, fp))
                    amae = float(mean_absolute_error(yt, ap))
                    by_pos_cmp[str(position)] = {
                        "n": int(len(grp)),
                        "full_r": round(fr, 4) if pd.notna(fr) else None,
                        "adp_r": round(ar, 4) if pd.notna(ar) else None,
                        "lift_r": round(fr - ar, 4) if pd.notna(fr) and pd.notna(ar) else None,
                        "full_mae": round(fmae, 3),
                        "adp_mae": round(amae, 3),
                        "lift_mae": round(fmae - amae, 3),
                    }
                comparable["by_position"] = by_pos_cmp
                adp_block["comparable"] = comparable
                _export_adp_baseline_csv(comparable)

            metrics["baselines"] = {"adp_only": adp_block}

    (MODELS_DIR / METRICS_FILE).write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    return ml_features, composite_corr, metrics


def train_and_score_dynasty(
    master: pd.DataFrame,
    feature_corr: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Parallel dynasty trainer: redraft-style direct HGB + CQR/IQR for Y1, Y2, Y3, and total.

    Writes ONLY dynasty artifacts (hgb_dynasty_ppr.joblib, composite_weights_dynasty.json, …).
    Does not overwrite redraft model/metrics/CSV files.
    Implementation lives in rookie_ppr.dynasty_train (lazy import avoids circular deps).
    """
    from rookie_ppr.dynasty_train import train_and_score_dynasty as _impl

    return _impl(master, feature_corr)


def load_model_bundle(mode: str | AnalysisMode | None = None) -> dict:
    """Load redraft or dynasty joblib bundle (default: redraft)."""
    analysis = mode if isinstance(mode, AnalysisMode) else get_mode(mode or "redraft")
    path = MODELS_DIR / analysis.model_file
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Run python -m rookie_ppr.compile first."
        )
    return joblib.load(path)


def _resolve_analysis_mode(
    mode: str | AnalysisMode | None = None,
    bundle: dict | None = None,
) -> AnalysisMode:
    if isinstance(mode, AnalysisMode):
        return mode
    if mode:
        return get_mode(mode)
    if bundle and bundle.get("analysis_mode"):
        return get_mode(str(bundle["analysis_mode"]))
    if bundle and bundle.get("model_type") == "position_specific_adp_residual":
        return DYNASTY_MODE
    return REDRAFT_MODE


def _load_percentile_lookup_for_mode(analysis: AnalysisMode) -> dict[str, list[float]]:
    path = MODELS_DIR / analysis.percentile_file
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _predict_horizon_row(
    row: pd.DataFrame,
    position: str,
    horizon: dict[str, Any],
    *,
    fallback_feature_cols: list[str] | None = None,
) -> dict[str, Any]:
    """Redraft-style point + predictive IQR (+ CQR) for one dynasty horizon."""
    models = horizon.get("models_by_position") or horizon.get("models") or {}
    if position not in models:
        return {
            "predicted": None,
            "predictive_q25": None,
            "predictive_q75": None,
            "cqr_low": None,
            "cqr_high": None,
        }

    feature_cols_map = horizon.get("feature_cols_by_position") or {}
    feature_cols = list(
        feature_cols_map.get(position)
        or fallback_feature_cols
        or _position_feature_cols(position)
    )
    X = _feature_matrix(row, feature_cols)
    pred = float(models[position].predict(X)[0])

    q_models = (horizon.get("quantile_models_by_position") or {}).get(position)
    conf = (horizon.get("conformal_by_position") or {}).get(position)
    pq25 = pq75 = cqr_low = cqr_high = float("nan")
    if q_models and conf:
        cqr_low, cqr_high, _, _ = _cqr_interval_row(
            X, q_models["lo"], q_models["hi"], float(conf["q_hat"])
        )
        if "q25" in q_models and "q75" in q_models:
            pq25 = float(q_models["q25"].predict(X)[0])
            pq75 = float(q_models["q75"].predict(X)[0])
        else:
            pq25 = pred + float(conf.get("residual_q25", 0.0))
            pq75 = pred + float(conf.get("residual_q75", 0.0))
        if pq25 > pq75:
            pq25, pq75 = pq75, pq25
        pred = _clamp_pred_to_iqr(pred, pq25, pq75)

    return {
        "predicted": round(float(max(pred, 0.0)), 2) if np.isfinite(pred) else None,
        "predictive_q25": round(float(pq25), 2) if np.isfinite(pq25) else None,
        "predictive_q75": round(float(pq75), 2) if np.isfinite(pq75) else None,
        "cqr_low": round(float(cqr_low), 2) if np.isfinite(cqr_low) else None,
        "cqr_high": round(float(cqr_high), 2) if np.isfinite(cqr_high) else None,
        "bust_chance_pct": 25.0 if np.isfinite(pq25) else None,
        "boom_chance_pct": 25.0 if np.isfinite(pq75) else None,
    }


def _predict_year_breakdown(
    row: pd.DataFrame,
    position: str,
    bundle: dict,
    *,
    total: float | None = None,
) -> dict[str, Any]:
    """
    Predict Y1/Y2/Y3 with the same uncertainty stack as redraft.
    Years are independent (not rescaled to the total).
    """
    feature_cols_map = bundle.get("feature_cols_by_position", {}) or {}
    fallback_cols = list(
        feature_cols_map.get(position) or _position_feature_cols(position)
    )
    horizons = bundle.get("horizon_targets") or {}
    year_bundle = bundle.get("year_targets") or {}

    out: dict[str, Any] = {"year_breakdown_scaled_to_total": False}
    year_sum = 0.0
    n_years = 0
    for i, key in enumerate(("ppr_y1", "ppr_y2", "ppr_y3"), start=1):
        hz = horizons.get(key) or year_bundle.get(key) or {}
        # Normalize older secondary-style year bundles
        if "models_by_position" not in hz and "models" in hz:
            hz = {
                **hz,
                "models_by_position": hz.get("models") or {},
                "feature_cols_by_position": hz.get("feature_cols_by_position")
                or feature_cols_map,
            }
        pred_info = _predict_horizon_row(
            row, position, hz, fallback_feature_cols=fallback_cols
        )
        out[f"predicted_ppr_y{i}"] = pred_info["predicted"]
        out[f"predictive_q25_ppr_y{i}"] = pred_info["predictive_q25"]
        out[f"predictive_q75_ppr_y{i}"] = pred_info["predictive_q75"]
        out[f"ppr_low_ppr_y{i}"] = pred_info["cqr_low"]
        out[f"ppr_high_ppr_y{i}"] = pred_info["cqr_high"]
        out[f"bust_chance_pct_ppr_y{i}"] = pred_info.get("bust_chance_pct")
        out[f"boom_chance_pct_ppr_y{i}"] = pred_info.get("boom_chance_pct")
        if pred_info["predicted"] is not None:
            year_sum += float(pred_info["predicted"])
            n_years += 1

    out["predicted_year_sum"] = round(year_sum, 2) if n_years else None
    if total is not None and np.isfinite(total):
        out["predicted_total"] = round(float(total), 2)
    return out


def _predict_adp_residual_row(
    row: pd.DataFrame,
    position: str,
    bundle: dict,
) -> tuple[float, float, float]:
    """
    Dynasty point estimate: ADP expected + residual (or direct fallback).

    Returns (predicted_total, adp_expected, residual_or_nan).
    """
    feature_cols_map = bundle.get("feature_cols_by_position", {})
    residual_models = bundle.get("models_by_position", {})
    direct_models = bundle.get("direct_fallback_models_by_position", {}) or {}
    adp_models = bundle.get("adp_models_by_position", {}) or {}

    feature_cols = list(
        feature_cols_map.get(position) or _position_feature_cols_dynasty(position, list(row.columns))
    )
    X = _feature_matrix(row, feature_cols)
    adp_exp = _adp_expected_for_row(row, position, adp_models)

    if position in residual_models and np.isfinite(adp_exp):
        residual = float(residual_models[position].predict(X)[0])
        return adp_exp + residual, adp_exp, residual

    if position in direct_models:
        return float(direct_models[position].predict(X)[0]), adp_exp, float("nan")

    if position in residual_models:
        # Residual without ADP is not on total scale — refuse rather than mis-scale
        raise ValueError(
            f"Dynasty model for {position} needs ff_adp + ff_adp_rank (or a direct fallback)."
        )

    available = sorted(set(residual_models) | set(direct_models))
    raise ValueError(f"No trained dynasty model for position {position}. Available: {available}")


COMPOSITE_DISPLAY_NAMES: dict[str, str] = {
    "score_draft_capital": "Draft capital",
    "score_pre_draft_fantasy": "Pre-draft fantasy",
    "score_recruiting": "Recruiting",
    "score_timing": "Age / timing",
    "score_cfb_receiving": "CFB receiving",
    "score_cfb_rushing": "CFB rushing",
    "score_cfb_passing": "CFB passing",
    "score_combine_speed": "Combine speed",
    "score_combine_explosion": "Combine explosion",
    "score_combine_size": "Combine size",
    "score_team_context": "Team / opportunity",
}


def explain_prediction(
    composites_row: pd.DataFrame,
    position: str,
    *,
    predicted: float | None = None,
    bundle: dict | None = None,
    mode: str | AnalysisMode | None = None,
    top_n: int = 8,
    baseline: float = TYPICAL_COMPOSITE_BASELINE,
) -> list[dict[str, Any]]:
    """
    Rank composite features by leave-one-out impact on predicted PPR.

    For each populated composite used by the position model, re-predict with that
    feature set to a typical baseline (default 0 = position-average composite).
    Positive delta_ppr means the player's value raised the forecast vs typical.

    Legacy ADP-residual dynasty bundles explain on the total (ADP + residual) scale.
    """
    analysis = _resolve_analysis_mode(mode, bundle)
    bundle = bundle or load_model_bundle(analysis)
    is_residual = bundle.get("model_type") == "position_specific_adp_residual"

    row = composites_row.copy()
    if is_residual:
        row = _apply_cfb_gate(row)

    models = bundle.get("models_by_position", {})
    direct_models = bundle.get("direct_fallback_models_by_position", {}) or {}
    feature_cols_map = bundle.get("feature_cols_by_position", {})

    if is_residual:
        if position not in models and position not in direct_models:
            return []
        feature_cols = list(
            feature_cols_map.get(position)
            or _position_feature_cols_dynasty(position, list(row.columns))
        )
        if predicted is not None:
            base_pred = float(predicted)
        else:
            base_pred, _, _ = _predict_adp_residual_row(row, position, bundle)
    else:
        if position not in models:
            return []
        feature_cols = list(feature_cols_map.get(position) or _position_feature_cols(position))
        model = models[position]
        X_base = _feature_matrix(row, feature_cols)
        base_pred = float(predicted) if predicted is not None else float(model.predict(X_base)[0])

    X_base = _feature_matrix(row, feature_cols)
    drivers: list[dict[str, Any]] = []
    for col in feature_cols:
        if col not in X_base.columns:
            continue
        val = X_base.iloc[0][col]
        if val is None or (isinstance(val, float) and np.isnan(val)) or pd.isna(val):
            continue
        # Skip near-typical values — impact is noise at the baseline
        if abs(float(val) - float(baseline)) < 1e-9:
            continue
        alt_row = row.copy()
        alt_row[col] = float(baseline)

        if is_residual:
            alt_pred, _, _ = _predict_adp_residual_row(alt_row, position, bundle)
        else:
            X_alt = _feature_matrix(alt_row, feature_cols)
            alt_pred = float(models[position].predict(X_alt)[0])
        delta = base_pred - alt_pred
        drivers.append(
            {
                "feature": col,
                "label": COMPOSITE_DISPLAY_NAMES.get(
                    col, col.replace("score_", "").replace("_", " ").title()
                ),
                "value": round(float(val), 3),
                "delta_ppr": round(delta, 2),
                "abs_delta": round(abs(delta), 2),
            }
        )

    drivers.sort(key=lambda d: d["abs_delta"], reverse=True)
    if top_n > 0:
        drivers = drivers[:top_n]
    return drivers


def predict_from_composites(
    composites_row: pd.DataFrame,
    position: str,
    bundle: dict | None = None,
    percentile_lookup: dict[str, list[float]] | None = None,
    mode: str | AnalysisMode | None = None,
) -> pd.DataFrame:
    analysis = _resolve_analysis_mode(mode, bundle)
    bundle = bundle or load_model_bundle(analysis)
    if percentile_lookup is None:
        percentile_lookup = _load_percentile_lookup_for_mode(analysis)

    row = composites_row.copy()
    is_residual = bundle.get("model_type") == "position_specific_adp_residual"
    feature_cols_map = bundle.get("feature_cols_by_position", {})

    if is_residual:
        row = _apply_cfb_gate(row)
        residual_models = bundle.get("models_by_position", {})
        direct_models = bundle.get("direct_fallback_models_by_position", {}) or {}
        if position not in residual_models and position not in direct_models:
            available = sorted(set(residual_models) | set(direct_models))
            raise ValueError(
                f"No trained model for position {position}. Available: {available}"
            )
        feature_cols = list(
            feature_cols_map.get(position)
            or _position_feature_cols_dynasty(position, list(row.columns))
        )
        X = _feature_matrix(row, feature_cols)
        pred, adp_exp, residual = _predict_adp_residual_row(row, position, bundle)
        row["predicted_median"] = pred
        if np.isfinite(adp_exp):
            row[ADP_EXPECTED_COL] = adp_exp
        if np.isfinite(residual):
            row["predicted_dynasty_residual"] = residual

        q_models = (bundle.get("quantile_models_by_position") or {}).get(position)
        conf = (bundle.get("conformal_by_position") or {}).get(position)
        # Quantile/CQR models are fit on residual scale; shift by ADP expected for totals
        if q_models and conf and np.isfinite(adp_exp):
            cqr_low_res, cqr_high_res, q_lo, q_hi = _cqr_interval_row(
                X, q_models["lo"], q_models["hi"], float(conf["q_hat"])
            )
            cqr_low = adp_exp + cqr_low_res
            cqr_high = adp_exp + cqr_high_res
            row["cqr_low"] = cqr_low
            row["cqr_high"] = cqr_high
            row["ppr_low"] = cqr_low
            row["ppr_high"] = cqr_high
            row["quantile_lo"] = adp_exp + q_lo
            row["quantile_hi"] = adp_exp + q_hi
            row["cqr_alpha"] = float(conf.get("alpha", CQR_ALPHA))
            row["cqr_q_hat"] = float(conf["q_hat"])

            if "q25" in q_models and "q75" in q_models:
                pq25 = adp_exp + float(q_models["q25"].predict(X)[0])
                pq75 = adp_exp + float(q_models["q75"].predict(X)[0])
            else:
                pq25 = pred + float(conf.get("residual_q25", 0.0))
                pq75 = pred + float(conf.get("residual_q75", 0.0))
            if pq25 > pq75:
                pq25, pq75 = pq75, pq25
            row["predictive_q25"] = pq25
            row["predictive_q75"] = pq75
            pred = _clamp_pred_to_iqr(pred, pq25, pq75)
            row["bust_chance_pct"] = 25.0
            row["boom_chance_pct"] = 25.0
    else:
        models = bundle.get("models_by_position", {})
        if position not in models:
            raise ValueError(
                f"No trained model for position {position}. Available: {list(models)}"
            )

        feature_cols = feature_cols_map.get(position, _position_feature_cols(position))
        model = models[position]
        X = _feature_matrix(row, feature_cols)

        pred = float(model.predict(X)[0])
        row["predicted_median"] = pred

        q_models = (bundle.get("quantile_models_by_position") or {}).get(position)
        conf = (bundle.get("conformal_by_position") or {}).get(position)
        if q_models and conf:
            # 80% CQR band (model uncertainty metadata)
            cqr_low, cqr_high, q_lo, q_hi = _cqr_interval_row(
                X, q_models["lo"], q_models["hi"], float(conf["q_hat"])
            )
            row["cqr_low"] = cqr_low
            row["cqr_high"] = cqr_high
            row["ppr_low"] = cqr_low  # back-compat
            row["ppr_high"] = cqr_high
            row["quantile_lo"] = q_lo
            row["quantile_hi"] = q_hi
            row["cqr_alpha"] = float(conf.get("alpha", CQR_ALPHA))
            row["cqr_q_hat"] = float(conf["q_hat"])

            # Predictive IQR from error model: 25% chance below q25, 25% above q75
            if "q25" in q_models and "q75" in q_models:
                pq25 = float(q_models["q25"].predict(X)[0])
                pq75 = float(q_models["q75"].predict(X)[0])
            else:
                pq25 = pred + float(conf.get("residual_q25", 0.0))
                pq75 = pred + float(conf.get("residual_q75", 0.0))
            if pq25 > pq75:
                pq25, pq75 = pq75, pq25
            row["predictive_q25"] = pq25
            row["predictive_q75"] = pq75
            pred = _clamp_pred_to_iqr(pred, pq25, pq75)
            # By definition of predictive quartiles
            row["bust_chance_pct"] = 25.0
            row["boom_chance_pct"] = 25.0

    score = _to_success_score(pred, position, percentile_lookup)
    row[analysis.predicted_col] = pred
    row[analysis.success_score_col] = score
    # Stable aliases for callers that still expect redraft column names
    row["predicted_points"] = pred
    row["success_score"] = score
    if analysis.name == "redraft":
        row["predicted_rookie_ppr"] = pred
        row["success_score_0_100"] = score
    else:
        # Keep redraft-named aliases so older UI helpers keep working
        row["predicted_rookie_ppr"] = pred
        row["success_score_0_100"] = score
        year_break = _predict_year_breakdown(row, position, bundle, total=float(pred))
        for k, v in year_break.items():
            row[k] = v

    return row


if __name__ == "__main__":
    import sys

    from rookie_ppr.config import OUTPUT_DIR

    mode = (sys.argv[1] if len(sys.argv) > 1 else "dynasty").strip().lower()
    if mode != "dynasty":
        raise SystemExit("Usage: python -m rookie_ppr.model_score dynasty")

    master_path = OUTPUT_DIR / "players_master.csv"
    if not master_path.exists():
        raise SystemExit(f"Missing {master_path}; run compile first or point at master CSV.")

    master_df = pd.read_csv(master_path)
    redraft_mtime = None
    redraft_metrics = MODELS_DIR / METRICS_FILE
    if redraft_metrics.exists():
        redraft_mtime = redraft_metrics.stat().st_mtime

    ml_dyn, _, dyn_metrics = train_and_score_dynasty(master_df)
    hold_r = dyn_metrics.get("holdout_pearson_r")
    by_pos = (dyn_metrics.get("holdout") or {}).get("by_position") or dyn_metrics.get("holdout_by_position") or {}
    print(f"Dynasty holdout total r: {hold_r}")
    print(f"WR holdout r: {(by_pos.get('WR') or {}).get('pearson_r')}")
    for name, block in (dyn_metrics.get("year_targets") or {}).items():
        print(f"Year {name}: r={block.get('pearson_r')} mae={block.get('mae')} n={block.get('n')}")
    ecr = dyn_metrics.get("ecr") or {}
    print(f"ECR matched: {ecr.get('n_matched')} / loaded={ecr.get('loaded')} err={ecr.get('error')}")
    print(f"ml_features rows: {len(ml_dyn)}")

    if redraft_mtime is not None and redraft_metrics.exists():
        unchanged = abs(redraft_metrics.stat().st_mtime - redraft_mtime) < 1e-6
        print(f"Redraft model_metrics.json mtime unchanged: {unchanged}")
        try:
            redraft = json.loads(redraft_metrics.read_text(encoding="utf-8"))
            print(f"Redraft holdout_pearson_r: {redraft.get('holdout_pearson_r')}")
        except Exception as exc:  # noqa: BLE001
            print(f"Could not read redraft metrics: {exc}")
