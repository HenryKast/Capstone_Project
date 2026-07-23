"""
Dynasty training: same system as redraft (direct median HGB + CQR + predictive IQR),
applied to Y1 / Y2 / Y3 PPR and the Y1–Y3 total.

Writes ONLY dynasty artifacts. Does not touch redraft files.
"""
from __future__ import annotations

import json
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from rookie_ppr.analysis_config import DYNASTY_MODE
from rookie_ppr.config import CSV_OUTPUT_DIR, MODELS_DIR, SKILL_POSITIONS
from rookie_ppr.feature_composites import (
    ID_COLUMNS,
    apply_composites,
    build_composite_artifacts,
    build_composite_correlation,
    feature_corr_for_weights,
    save_composite_artifacts,
)
from rookie_ppr.model_score import (
    COMPOSITE_CORR_DYNASTY_CSV,
    CQR_ALPHA,
    CQR_MIN_CAL,
    CQR_Q_HI,
    CQR_Q_LO,
    DYNASTY_ABS_CORR_COL,
    DYNASTY_ECR_COLS,
    ML_FEATURES_DYNASTY_CSV,
    _attach_adp_features,
    _build_dynasty_feature_corr,
    _clamp_pred_to_iqr,
    _conformal_q_hat,
    _feature_matrix,
    _fit_point_model,
    _fit_predictive_quartile_pair,
    _fit_quantile_pair,
    _holdout_mask,
    _holdout_metric_block,
    _pearson_r,
    _percentile_lookup,
    _position_feature_cols,
    _split_fit_cal,
    _split_train_val,
    _to_success_score,
    _tune_position_model,
)


HORIZON_LABELS = {
    "ppr_y1": "Y1",
    "ppr_y2": "Y2",
    "ppr_y3": "Y3",
}


def _merge_cols(
    frame: pd.DataFrame,
    master: pd.DataFrame,
    id_cols: list[str],
    cols: list[str],
) -> pd.DataFrame:
    missing = [c for c in cols if c in master.columns and c not in frame.columns]
    if not missing or not id_cols:
        return frame
    keys = [c for c in id_cols if c in frame.columns and c in master.columns]
    if not keys:
        return frame
    return frame.merge(master[keys + missing].drop_duplicates(subset=keys), on=keys, how="left")


def _attach_dynasty_ecr(master: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Optional ECR columns if already on master (baseline reporting only)."""
    meta: dict[str, Any] = {
        "loaded": any(c in master.columns for c in DYNASTY_ECR_COLS),
        "n_matched": int(master["dynasty_ecr"].notna().sum()) if "dynasty_ecr" in master.columns else 0,
        "match_rate": 0.0,
        "caveat": "Current-market ECR; not used as a model feature.",
    }
    if meta["n_matched"] and len(master):
        meta["match_rate"] = round(meta["n_matched"] / max(len(master), 1), 4)
    return master, meta


def _train_direct_horizon(
    labeled: pd.DataFrame,
    target_col: str,
    holdout_seasons: tuple[int, ...],
    val_season: int,
) -> dict[str, Any]:
    """
    Redraft-style stack for one target: median HGB + CQR + predictive q25/q75
    per skill position.
    """
    models_by_position: dict[str, HistGradientBoostingRegressor] = {}
    feature_cols_by_position: dict[str, list[str]] = {}
    best_params_by_position: dict[str, dict[str, Any]] = {}
    tuning_r_by_position: dict[str, float | None] = {}
    quantile_models_by_position: dict[str, dict[str, HistGradientBoostingRegressor]] = {}
    conformal_by_position: dict[str, dict[str, Any]] = {}

    if target_col not in labeled.columns:
        return {
            "target": target_col,
            "models_by_position": models_by_position,
            "feature_cols_by_position": feature_cols_by_position,
            "best_params_by_position": best_params_by_position,
            "tuning_r_by_position": tuning_r_by_position,
            "quantile_models_by_position": quantile_models_by_position,
            "conformal_by_position": conformal_by_position,
        }

    work = labeled[pd.to_numeric(labeled[target_col], errors="coerce").notna()].copy()
    for position in SKILL_POSITIONS:
        pos_df = work[work["position"] == position].copy()
        if pos_df.empty:
            continue

        feature_cols = _position_feature_cols(position)
        feature_cols_by_position[position] = feature_cols

        pre_holdout = pos_df[~_holdout_mask(pos_df, holdout_seasons)]
        if len(pre_holdout) < 15:
            continue

        tune_train, tune_val = _split_train_val(pre_holdout, val_season)
        _, best_params, tune_r = _tune_position_model(
            position, tune_train, tune_val, feature_cols, target=target_col
        )
        best_params_by_position[position] = best_params
        tuning_r_by_position[position] = round(tune_r, 4) if pd.notna(tune_r) else None

        models_by_position[position] = _fit_point_model(
            pre_holdout, feature_cols, best_params, target=target_col
        )

        cqr_fit, cqr_cal = _split_fit_cal(pre_holdout, val_season)
        if len(cqr_fit) >= 15 and len(cqr_cal) >= CQR_MIN_CAL:
            lo_model, hi_model = _fit_quantile_pair(
                cqr_fit, feature_cols, best_params, target=target_col
            )
            q25_model, q75_model = _fit_predictive_quartile_pair(
                cqr_fit, feature_cols, best_params, target=target_col
            )
            X_cal = _feature_matrix(cqr_cal, feature_cols)
            y_cal = pd.to_numeric(cqr_cal[target_col], errors="coerce").to_numpy(dtype=float)
            q_lo_cal = lo_model.predict(X_cal)
            q_hi_cal = hi_model.predict(X_cal)
            swap = q_lo_cal > q_hi_cal
            if np.any(swap):
                q_lo_cal, q_hi_cal = q_lo_cal.copy(), q_hi_cal.copy()
                q_lo_cal[swap], q_hi_cal[swap] = q_hi_cal[swap], q_lo_cal[swap]
            scores = np.maximum(q_lo_cal - y_cal, y_cal - q_hi_cal)
            q_hat = _conformal_q_hat(scores, CQR_ALPHA)

            point_fit = _fit_point_model(cqr_fit, feature_cols, best_params, target=target_col)
            pred_cal = point_fit.predict(X_cal)
            residuals = [round(float(v), 4) for v in (y_cal - pred_cal).tolist() if np.isfinite(v)]
            r_arr = np.asarray(residuals, dtype=float)
            residual_q25 = float(np.percentile(r_arr, 25)) if r_arr.size else 0.0
            residual_q75 = float(np.percentile(r_arr, 75)) if r_arr.size else 0.0

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
                "one_sided_scores_lo": [
                    round(float(v), 4) for v in (q_lo_cal - y_cal).tolist() if np.isfinite(v)
                ],
                "one_sided_scores_hi": [
                    round(float(v), 4) for v in (y_cal - q_hi_cal).tolist() if np.isfinite(v)
                ],
            }

    return {
        "target": target_col,
        "models_by_position": models_by_position,
        "feature_cols_by_position": feature_cols_by_position,
        "best_params_by_position": best_params_by_position,
        "tuning_r_by_position": tuning_r_by_position,
        "quantile_models_by_position": quantile_models_by_position,
        "conformal_by_position": conformal_by_position,
    }


def _score_horizon_on_frame(
    ml_features: pd.DataFrame,
    horizon: dict[str, Any],
    *,
    predicted_col: str,
    q25_col: str,
    q75_col: str,
    low_col: str,
    high_col: str,
) -> None:
    """Write point + IQR + CQR columns for one horizon onto ml_features."""
    models = horizon.get("models_by_position") or {}
    feature_cols_by_position = horizon.get("feature_cols_by_position") or {}
    quantile_models_by_position = horizon.get("quantile_models_by_position") or {}
    conformal_by_position = horizon.get("conformal_by_position") or {}

    preds = np.full(len(ml_features), np.nan, dtype=float)
    pq25s = np.full(len(ml_features), np.nan, dtype=float)
    pq75s = np.full(len(ml_features), np.nan, dtype=float)
    lows = np.full(len(ml_features), np.nan, dtype=float)
    highs = np.full(len(ml_features), np.nan, dtype=float)

    for position, model in models.items():
        mask = ml_features["position"] == position
        if not mask.any():
            continue
        cols = feature_cols_by_position.get(position) or _position_feature_cols(position)
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
            lows[mask.to_numpy()] = q_lo - q_hat
            highs[mask.to_numpy()] = q_hi + q_hat
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

    ml_features[predicted_col] = preds
    ml_features[q25_col] = pq25s
    ml_features[q75_col] = pq75s
    ml_features[low_col] = lows
    ml_features[high_col] = highs


def train_and_score_dynasty(
    master: pd.DataFrame,
    feature_corr: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Dynasty trainer mirroring redraft: direct median HGB + CQR/IQR per position,
    for ppr_y1, ppr_y2, ppr_y3, and dynasty_ppr_y1_y3_total.
    """
    mode = DYNASTY_MODE
    target = mode.primary_target
    predicted_col = mode.predicted_col
    success_col = mode.success_score_col
    holdout_seasons = tuple(mode.holdout_seasons)
    val_season = int(mode.tuning_season)
    year_targets = tuple(getattr(mode, "year_targets", ()) or ())

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    CSV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if master.empty or target not in master.columns:
        return pd.DataFrame(), pd.DataFrame(), {}

    master_work, ecr_meta = _attach_dynasty_ecr(master)

    labeled = master_work[master_work[target].notna()].copy()
    if mode.require_dynasty_complete and "dynasty_seasons_complete" in labeled.columns:
        labeled = labeled[pd.to_numeric(labeled["dynasty_seasons_complete"], errors="coerce") == 1].copy()
    if labeled.empty:
        return pd.DataFrame(), pd.DataFrame(), {}

    if feature_corr is not None and not feature_corr.empty and DYNASTY_ABS_CORR_COL in feature_corr.columns:
        weight_corr = feature_corr_for_weights(feature_corr, source_abs_col=DYNASTY_ABS_CORR_COL)
    else:
        weight_corr = feature_corr_for_weights(
            _build_dynasty_feature_corr(master_work, target),
            source_abs_col=DYNASTY_ABS_CORR_COL,
        )

    holdout_mask = _holdout_mask(labeled, holdout_seasons)
    train_labeled = labeled[~holdout_mask].copy()

    artifacts = build_composite_artifacts(train_labeled, weight_corr)
    save_composite_artifacts(artifacts, MODELS_DIR / mode.composite_weights_file)

    composites_all = apply_composites(master_work, artifacts, weight_corr)
    composite_corr = build_composite_correlation(
        composites_all,
        master_work,
        target=target,
        abs_corr_col=DYNASTY_ABS_CORR_COL,
    )
    if not composite_corr.empty and DYNASTY_ABS_CORR_COL in composite_corr.columns:
        composite_corr = composite_corr.copy()
        composite_corr["abs_corr_rookie_ppr"] = composite_corr[DYNASTY_ABS_CORR_COL]

    id_cols = [c for c in ID_COLUMNS if c in master_work.columns]
    labeled_extra: list[str] = []
    for c in (
        target,
        "first_stat_season",
        "draft_year",
        "dynasty_seasons_complete",
        "draft_overall",
        *year_targets,
        *DYNASTY_ECR_COLS,
    ):
        if c in labeled.columns and c not in id_cols and c not in labeled_extra:
            labeled_extra.append(c)

    labeled_composites = composites_all.merge(
        labeled[id_cols + labeled_extra].drop_duplicates(subset=id_cols),
        on=id_cols,
        how="inner",
    )
    if "first_stat_season" not in labeled_composites.columns and "first_stat_season" in master_work.columns:
        labeled_composites = labeled_composites.merge(
            master_work[id_cols + ["first_stat_season"]], on=id_cols, how="left"
        )
    labeled_composites = _attach_adp_features(labeled_composites, master_work, id_cols)
    labeled_composites = _merge_cols(
        labeled_composites, master_work, id_cols, ["draft_overall", *year_targets, *DYNASTY_ECR_COLS]
    )

    # Also train year horizons on all master rows with that year's label (not only complete totals)
    year_labeled_frames: dict[str, pd.DataFrame] = {}
    for yt in year_targets:
        if yt not in master_work.columns:
            continue
        y_lab = master_work[master_work[yt].notna()].copy()
        if y_lab.empty:
            continue
        y_extra = [
            c
            for c in (yt, "first_stat_season", "draft_year", "draft_overall")
            if c in y_lab.columns and c not in id_cols
        ]
        y_frame = composites_all.merge(
            y_lab[id_cols + y_extra].drop_duplicates(subset=id_cols),
            on=id_cols,
            how="inner",
        )
        if "first_stat_season" not in y_frame.columns and "first_stat_season" in master_work.columns:
            y_frame = y_frame.merge(master_work[id_cols + ["first_stat_season"]], on=id_cols, how="left")
        year_labeled_frames[yt] = y_frame

    # Primary total — same recipe as redraft
    total_horizon = _train_direct_horizon(
        labeled_composites, target, holdout_seasons, val_season
    )

    horizon_targets: dict[str, Any] = {}
    for yt in year_targets:
        frame = year_labeled_frames.get(yt, labeled_composites)
        horizon_targets[yt] = _train_direct_horizon(frame, yt, holdout_seasons, val_season)

    bundle = {
        "model_type": "position_specific",
        "point_estimate": "median_q50",
        "analysis_mode": mode.name,
        "target": target,
        "models_by_position": total_horizon["models_by_position"],
        "feature_cols_by_position": total_horizon["feature_cols_by_position"],
        "best_params_by_position": total_horizon["best_params_by_position"],
        "holdout_first_stat_seasons": list(holdout_seasons),
        "quantile_models_by_position": total_horizon["quantile_models_by_position"],
        "conformal_by_position": total_horizon["conformal_by_position"],
        "cqr_alpha": CQR_ALPHA,
        "horizon_targets": horizon_targets,
        # Back-compat aliases used by older year-breakdown helpers
        "year_targets": {
            k: {
                "models": v.get("models_by_position") or {},
                "adp_models": {},
                "quantile_models_by_position": v.get("quantile_models_by_position") or {},
                "conformal_by_position": v.get("conformal_by_position") or {},
                "feature_cols_by_position": v.get("feature_cols_by_position") or {},
            }
            for k, v in horizon_targets.items()
        },
        "ecr_meta": ecr_meta,
    }
    joblib.dump(bundle, MODELS_DIR / mode.model_file)

    percentile_lookup = _percentile_lookup(train_labeled, target=target)
    percentile_lookup["by_target"] = {target: dict(percentile_lookup)}
    for yt in year_targets:
        src = year_labeled_frames.get(yt)
        if src is None or src.empty:
            continue
        y_train = src[~_holdout_mask(src, holdout_seasons)]
        percentile_lookup["by_target"][yt] = _percentile_lookup(y_train, target=yt)
    (MODELS_DIR / mode.percentile_file).write_text(
        json.dumps(percentile_lookup, default=list), encoding="utf-8"
    )

    ml_features = composites_all.copy()
    ml_features = _merge_cols(
        ml_features,
        master_work,
        id_cols,
        [
            "draft_overall",
            target,
            "first_stat_season",
            "draft_year",
            "dynasty_seasons_complete",
            *year_targets,
            *DYNASTY_ECR_COLS,
        ],
    )
    ml_features = _attach_adp_features(ml_features, master_work, id_cols)

    _score_horizon_on_frame(
        ml_features,
        total_horizon,
        predicted_col=predicted_col,
        q25_col="predictive_q25",
        q75_col="predictive_q75",
        low_col="ppr_low",
        high_col="ppr_high",
    )
    ml_features[success_col] = [
        _to_success_score(pred, pos, percentile_lookup)
        for pred, pos in zip(ml_features[predicted_col], ml_features["position"], strict=False)
    ]

    for yt in year_targets:
        hz = horizon_targets.get(yt) or {}
        _score_horizon_on_frame(
            ml_features,
            hz,
            predicted_col=f"predicted_{yt}",
            q25_col=f"predictive_q25_{yt}",
            q75_col=f"predictive_q75_{yt}",
            low_col=f"ppr_low_{yt}",
            high_col=f"ppr_high_{yt}",
        )
        y_lookup = (percentile_lookup.get("by_target") or {}).get(yt) or {}
        ml_features[f"success_score_{yt}"] = [
            _to_success_score(pred, pos, y_lookup)
            for pred, pos in zip(ml_features[f"predicted_{yt}"], ml_features["position"], strict=False)
        ]

    # --- Metrics ---
    metrics: dict[str, Any] = {
        "model_type": "position_specific",
        "analysis_mode": mode.name,
        "target": target,
        "point_estimate": "median_q50",
        "train_rows": int(len(train_labeled)),
        "holdout_first_stat_seasons": list(holdout_seasons),
        "tuning_validation_season": val_season,
        "tuning_r_by_position": total_horizon.get("tuning_r_by_position") or {},
        "best_params_by_position": total_horizon.get("best_params_by_position") or {},
        "feature_cols_by_position": total_horizon.get("feature_cols_by_position") or {},
        "cqr_alpha": CQR_ALPHA,
        "conformal_by_position": {
            pos: {k: v for k, v in conf.items() if k != "residuals"}
            for pos, conf in (total_horizon.get("conformal_by_position") or {}).items()
        },
        "ecr": ecr_meta,
    }

    hold = ml_features[_holdout_mask(ml_features, holdout_seasons)].dropna(
        subset=[target, predicted_col]
    )
    if not hold.empty:
        metrics["holdout"] = _holdout_metric_block(
            hold[target], hold[predicted_col].to_numpy(), hold["position"]
        )
        metrics["holdout_pearson_r"] = metrics["holdout"].get("pearson_r")
        metrics["holdout_mae"] = metrics["holdout"].get("mae")
        metrics["holdout_n"] = metrics["holdout"].get("n")

    year_metrics: dict[str, Any] = {}
    for yt in year_targets:
        pred_col = f"predicted_{yt}"
        block: dict[str, Any] = {
            "target": yt,
            "tuning_r_by_position": (horizon_targets.get(yt) or {}).get("tuning_r_by_position") or {},
        }
        if yt in ml_features.columns and pred_col in ml_features.columns:
            y_hold = ml_features[_holdout_mask(ml_features, holdout_seasons)].dropna(
                subset=[yt, pred_col]
            )
            if not y_hold.empty:
                block.update(
                    _holdout_metric_block(
                        y_hold[yt], y_hold[pred_col].to_numpy(), y_hold["position"]
                    )
                )
        year_metrics[yt] = block
    metrics["year_targets"] = year_metrics
    metrics["horizon_labels"] = HORIZON_LABELS

    # Optional inverted-ECR baseline (not used for scoring)
    if ecr_meta.get("loaded") and "dynasty_ecr" in ml_features.columns:
        ecr_hold = ml_features[_holdout_mask(ml_features, holdout_seasons)].dropna(
            subset=["dynasty_ecr", target]
        )
        if len(ecr_hold) >= 15:
            inv = -pd.to_numeric(ecr_hold["dynasty_ecr"], errors="coerce")
            metrics["baselines"] = {
                "dynasty_ecr": {
                    "n_matched_holdout": int(len(ecr_hold)),
                    "holdout_pearson_r_inverted_ecr": round(
                        _pearson_r(ecr_hold[target], inv.to_numpy()), 4
                    ),
                    "caveat": ecr_meta.get("caveat"),
                }
            }

    (MODELS_DIR / mode.metrics_file).write_text(
        json.dumps(metrics, indent=2, default=str), encoding="utf-8"
    )
    ml_features.to_csv(CSV_OUTPUT_DIR / ML_FEATURES_DYNASTY_CSV, index=False)
    if not composite_corr.empty:
        composite_corr.to_csv(CSV_OUTPUT_DIR / COMPOSITE_CORR_DYNASTY_CSV, index=False)

    return ml_features, composite_corr, metrics
