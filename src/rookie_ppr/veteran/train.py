"""
Train veteran next-season PPR models.

Writes ONLY veteran artifacts. Does not touch redraft or dynasty files.

Aligned with the league player-track recipe where it matters for the UI:
soft IQR clamp, injury-year q75 relief, RB monotonic constraints, a deeper
RB grid when ADP is available, and an elite-ADP lift toward the market curve.
"""
from __future__ import annotations

import json
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from rookie_ppr.league.adp_features import ADP_FEATURE_COLS, attach_adp_features
from rookie_ppr.league.backtest_projections import (
    ELITE_RB_ADP_MAX,
    ELITE_RB_MODEL_WEIGHT,
    _elite_rb_adp_lift,
    _rb_monotonic_cst,
)
from rookie_ppr.league.backtest_rosters import (
    _actual_season_ppr,
    _adp_with_actuals,
    adp_curve_value,
    fit_adp_curves,
)
from rookie_ppr.league.opportunity_features import (
    OPPORTUNITY_FEATURE_COLS,
    attach_opportunity_features,
)
from rookie_ppr.model_score import (
    CQR_ALPHA,
    CQR_MIN_CAL,
    DEFAULT_INJURY_IQR_RELIEF,
    DEFAULT_IQR_CLAMP_STRENGTH,
    PARAM_GRID,
    RB_PARAM_GRID_WITH_MARKET,
    _clamp_pred_to_iqr,
    _conformal_q_hat,
    _feature_matrix,
    _fit_point_model,
    _fit_predictive_quartile_pair,
    _fit_quantile_pair,
    _holdout_metric_block,
    _pearson_r,
    _tune_position_model,
)
from rookie_ppr.veteran.config import (
    CSV_OUTPUT_DIR,
    MODELS_DIR,
    POSITION_FEATURE_COLS,
    SKILL_POSITIONS,
    TARGET,
    VET_BASELINE_CSV,
    VET_FEATURES_CSV,
    VET_HOLDOUT_TARGET_SEASONS,
    VET_METRICS_FILE,
    VET_MODEL_FILE,
    VET_TUNING_TARGET_SEASON,
)
from rookie_ppr.veteran.features import feature_correlation_table
from rookie_ppr.veteran.metrics_tiers import tiered_holdout_metrics, weighted_star_holdout_r


def _param_grid(position: str) -> list[dict[str, Any]]:
    # ADP is always attached below; use the market-aware RB grid so elite tails
    # are not forced into the shallow mean-reverting surface.
    if position == "RB":
        return RB_PARAM_GRID_WITH_MARKET
    return PARAM_GRID


def _feature_cols(position: str) -> list[str]:
    return (
        list(POSITION_FEATURE_COLS.get(position, POSITION_FEATURE_COLS["WR"]))
        + list(ADP_FEATURE_COLS)
        + list(OPPORTUNITY_FEATURE_COLS)
    )


def _usable_feature_cols(frame: pd.DataFrame, cols: list[str]) -> list[str]:
    """Drop all-null / constant cols (HGB binning requires >=2 distinct values)."""
    usable: list[str] = []
    for c in cols:
        if c not in frame.columns:
            continue
        x = pd.to_numeric(frame[c], errors="coerce")
        if int(x.notna().sum()) == 0 or int(x.nunique(dropna=True)) < 2:
            continue
        usable.append(c)
    return usable


def _labeled(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out[TARGET] = pd.to_numeric(out[TARGET], errors="coerce")
    out["target_season"] = pd.to_numeric(out.get("target_season"), errors="coerce")
    return out[out[TARGET].notna() & out["target_season"].notna()].copy()


def _split_masks(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    ts = pd.to_numeric(frame["target_season"], errors="coerce")
    holdout = ts.isin(VET_HOLDOUT_TARGET_SEASONS)
    tune = ts == VET_TUNING_TARGET_SEASON
    train = (~holdout) & (~tune) & ts.notna()
    return train, tune, holdout


def _baseline_block(holdout_df: pd.DataFrame, pred_col: str, baseline_col: str) -> dict[str, Any]:
    y = pd.to_numeric(holdout_df[TARGET], errors="coerce")
    base = pd.to_numeric(holdout_df[baseline_col], errors="coerce")
    full = pd.to_numeric(holdout_df[pred_col], errors="coerce")
    pair = pd.DataFrame(
        {"y": y, "base": base, "full": full, "position": holdout_df["position"]}
    ).dropna()
    if len(pair) < 5:
        return {"n": int(len(pair))}
    out: dict[str, Any] = {
        "n": int(len(pair)),
        "full_r": round(_pearson_r(pair["y"], pair["full"].to_numpy()), 4),
        "baseline_r": round(_pearson_r(pair["y"], pair["base"].to_numpy()), 4),
        "full_mae": round(float(mean_absolute_error(pair["y"], pair["full"])), 3),
        "baseline_mae": round(float(mean_absolute_error(pair["y"], pair["base"])), 3),
    }
    out["lift_r"] = (
        round(out["full_r"] - out["baseline_r"], 4)
        if out["full_r"] is not None and out["baseline_r"] is not None
        else None
    )
    by_pos: dict[str, Any] = {}
    for pos, g in pair.groupby("position"):
        if len(g) < 5:
            continue
        by_pos[str(pos)] = {
            "n": int(len(g)),
            "full_r": round(_pearson_r(g["y"], g["full"].to_numpy()), 4),
            "baseline_r": round(_pearson_r(g["y"], g["base"].to_numpy()), 4),
            "full_mae": round(float(mean_absolute_error(g["y"], g["full"])), 3),
            "baseline_mae": round(float(mean_absolute_error(g["y"], g["base"])), 3),
        }
    out["by_position"] = by_pos
    return out


def _apply_elite_rb_lift(scored: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Lift elite-ADP RBs toward the walk-forward ADP curve (leakage-free)."""
    out = scored.copy()
    if "adp_rank" not in out.columns or "predicted_ppr_next" not in out.columns:
        return out, 0
    try:
        adp_actuals = _adp_with_actuals(_actual_season_ppr())
    except Exception:  # noqa: BLE001
        return out, 0

    n_lifted = 0
    targets = pd.to_numeric(out.get("target_season"), errors="coerce")
    for season in sorted(targets.dropna().unique()):
        season_i = int(season)
        curves = fit_adp_curves(adp_actuals, before_season=season_i)
        if "RB" not in curves:
            continue
        mask = (
            (out["position"] == "RB")
            & (targets == season_i)
            & pd.to_numeric(out["predicted_ppr_next"], errors="coerce").notna()
        )
        if not mask.any():
            continue
        idx = out.index[mask]
        for i in idx:
            pred = float(out.at[i, "predicted_ppr_next"])
            rank = pd.to_numeric(out.at[i, "adp_rank"], errors="coerce")
            rank_f = float(rank) if pd.notna(rank) else float("nan")
            curve = adp_curve_value(curves, "RB", rank_f)
            new_val = _elite_rb_adp_lift(pred, adp_rank=rank_f, curve=curve)
            if abs(new_val - pred) > 1e-6:
                n_lifted += 1
                out.at[i, "predicted_ppr_next"] = new_val
    return out, n_lifted


def train_veteran_models(feature_frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    features = attach_opportunity_features(attach_adp_features(feature_frame))
    labeled = _labeled(features)
    if labeled.empty:
        raise ValueError("No labeled veteran rows (need ppr_next).")

    train_m, tune_m, hold_m = _split_masks(labeled)
    train_df = labeled.loc[train_m].copy()
    tune_df = labeled.loc[tune_m].copy()

    models_by_pos: dict[str, HistGradientBoostingRegressor] = {}
    q_models: dict[str, dict[str, HistGradientBoostingRegressor]] = {}
    conformal: dict[str, Any] = {}
    tuning_r: dict[str, float] = {}
    best_params: dict[str, Any] = {}
    feature_cols_by_pos: dict[str, list[str]] = {}
    monotonic_by_pos: dict[str, bool] = {}

    for position in SKILL_POSITIONS:
        pos_train = train_df[train_df["position"] == position]
        pos_tune = tune_df[tune_df["position"] == position]
        if len(pos_train) < 20:
            print(f"  skip {position}: train n={len(pos_train)}")
            continue
        # If tune fold thin, use last 15% of train by target_season as val
        if len(pos_tune) < 8:
            ordered = pos_train.sort_values("target_season")
            cut = max(int(len(ordered) * 0.85), 15)
            pos_tune = ordered.iloc[cut:].copy()
            pos_train = ordered.iloc[:cut].copy()

        cols = _usable_feature_cols(pos_train, _feature_cols(position))
        if len(cols) < 3:
            print(f"  skip {position}: usable features={len(cols)}")
            continue
        feature_cols_by_pos[position] = cols
        mono = _rb_monotonic_cst(cols) if position == "RB" else None
        monotonic_by_pos[position] = bool(mono)

        _model, params, r = _tune_position_model(
            position,
            pos_train,
            pos_tune,
            cols,
            target=TARGET,
            param_grid=_param_grid(position),
            monotonic_cst=mono,
        )
        best_params[position] = params
        tuning_r[position] = round(float(r), 4) if pd.notna(r) else float("nan")

        # Fit/cal split inside non-holdout for CQR
        non_hold = labeled.loc[~hold_m & (labeled["position"] == position)].sort_values(
            "target_season"
        )
        if len(non_hold) < 25:
            fit_df, cal_df = non_hold, non_hold.iloc[0:0]
        else:
            cal_n = max(CQR_MIN_CAL, int(len(non_hold) * 0.2))
            fit_df = non_hold.iloc[:-cal_n].copy()
            cal_df = non_hold.iloc[-cal_n:].copy()
            if len(fit_df) < 15:
                fit_df, cal_df = non_hold, non_hold.iloc[0:0]

        fit_pool = fit_df if len(fit_df) >= 15 else non_hold
        point_fit = _fit_point_model(
            fit_pool, cols, params, target=TARGET, monotonic_cst=mono
        )
        models_by_pos[position] = point_fit
        q25, q75 = _fit_predictive_quartile_pair(
            fit_pool, cols, params, target=TARGET, monotonic_cst=mono
        )
        q_lo, q_hi = _fit_quantile_pair(
            fit_pool, cols, params, target=TARGET, monotonic_cst=mono
        )
        q_models[position] = {"q25": q25, "q75": q75, "q_lo": q_lo, "q_hi": q_hi}

        q_hat = 0.0
        residuals: list[float] = []
        if len(cal_df) >= CQR_MIN_CAL:
            X_cal = _feature_matrix(cal_df, cols)
            y_cal = pd.to_numeric(cal_df[TARGET], errors="coerce").to_numpy()
            pred_cal = point_fit.predict(X_cal)
            lo_cal = q_lo.predict(X_cal)
            hi_cal = q_hi.predict(X_cal)
            scores = np.maximum(lo_cal - y_cal, y_cal - hi_cal)
            scores = scores[np.isfinite(scores)]
            if len(scores):
                q_hat = _conformal_q_hat(scores, CQR_ALPHA)
            residuals = (y_cal - pred_cal).tolist()
        conformal[position] = {
            "q_hat": float(q_hat),
            "n_cal": int(len(cal_df)),
            "residuals_n": len(residuals),
        }

        print(
            f"  {position}: train={len(pos_train)} tune_r={tuning_r[position]} "
            f"features={len(cols)} mono={bool(mono)}"
        )

    # Score all rows (labeled + upcoming unlabeled for the UI board)
    scored = features.copy()
    scored["predicted_ppr_next"] = np.nan
    scored["predictive_q25"] = np.nan
    scored["predictive_q75"] = np.nan
    scored["ppr_low"] = np.nan
    scored["ppr_high"] = np.nan

    for position, model in models_by_pos.items():
        cols = feature_cols_by_pos[position]
        mask = scored["position"] == position
        if not mask.any():
            continue
        X = _feature_matrix(scored.loc[mask], cols)
        pred = model.predict(X)
        q = q_models[position]
        q25 = q["q25"].predict(X)
        q75 = q["q75"].predict(X)
        q_lo = q["q_lo"].predict(X)
        q_hi = q["q_hi"].predict(X)
        q_hat = float((conformal.get(position) or {}).get("q_hat") or 0.0)
        low = q_lo - q_hat
        high = q_hi + q_hat
        prior_games = pd.to_numeric(scored.loc[mask, "games"], errors="coerce").to_numpy()
        clamped = [
            _clamp_pred_to_iqr(
                float(p),
                float(a),
                float(b),
                strength=DEFAULT_IQR_CLAMP_STRENGTH,
                prior_games=float(g) if np.isfinite(g) else None,
                injury_relief=DEFAULT_INJURY_IQR_RELIEF,
            )
            for p, a, b, g in zip(pred, q25, q75, prior_games)
        ]
        scored.loc[mask, "predicted_ppr_next"] = clamped
        scored.loc[mask, "predictive_q25"] = q25
        scored.loc[mask, "predictive_q75"] = q75
        scored.loc[mask, "ppr_low"] = low
        scored.loc[mask, "ppr_high"] = high

    scored, n_elite_lifted = _apply_elite_rb_lift(scored)
    print(
        f"  elite RB ADP lift: raised {n_elite_lifted} projections "
        f"(ADP<={ELITE_RB_ADP_MAX}, model_weight={ELITE_RB_MODEL_WEIGHT})"
    )

    # Holdout metrics (mask on scored frame — labeled uses a filtered index)
    hold_mask_full = (
        pd.to_numeric(scored.get("target_season"), errors="coerce").isin(
            VET_HOLDOUT_TARGET_SEASONS
        )
        & pd.to_numeric(scored[TARGET], errors="coerce").notna()
        & pd.to_numeric(scored["predicted_ppr_next"], errors="coerce").notna()
    )
    hold_scored = scored.loc[hold_mask_full].copy()
    y_hold = pd.to_numeric(hold_scored[TARGET], errors="coerce")
    p_hold = pd.to_numeric(hold_scored["predicted_ppr_next"], errors="coerce")
    valid = y_hold.notna() & p_hold.notna()
    hold_metrics = _holdout_metric_block(
        y_hold[valid],
        p_hold[valid].to_numpy(),
        hold_scored.loc[valid, "position"],
    )

    by_year: dict[str, Any] = {}
    hold_valid = hold_scored.loc[valid]
    if not hold_valid.empty:
        for year, g in hold_valid.groupby(
            pd.to_numeric(hold_valid["target_season"], errors="coerce")
        ):
            if pd.isna(year) or len(g) < 5:
                continue
            by_year[str(int(year))] = _holdout_metric_block(
                g[TARGET],
                pd.to_numeric(g["predicted_ppr_next"], errors="coerce").to_numpy(),
                g["position"],
            )

    last_base = _baseline_block(hold_valid, "predicted_ppr_next", "baseline_last_ppr")
    trail_base = _baseline_block(hold_valid, "predicted_ppr_next", "baseline_trail3_ppr")

    tier_metrics = tiered_holdout_metrics(hold_valid)
    star_weighted = weighted_star_holdout_r(hold_valid)

    metrics: dict[str, Any] = {
        "model_type": "veteran_next_season_phase3",
        "phase": 3,
        "phase1_holdout_pearson_r_ref": 0.7411,
        "phase2_holdout_pearson_r_ref": 0.7482,
        "target": TARGET,
        "tuning_target_season": VET_TUNING_TARGET_SEASON,
        "holdout_target_seasons": list(VET_HOLDOUT_TARGET_SEASONS),
        "n_labeled": int(len(labeled)),
        "n_train": int(train_m.sum()),
        "n_tune": int(tune_m.sum()),
        "n_holdout": int(valid.sum()),
        "iqr_clamp_strength": DEFAULT_IQR_CLAMP_STRENGTH,
        "injury_iqr_relief": DEFAULT_INJURY_IQR_RELIEF,
        "elite_rb_lift": True,
        "elite_rb_adp_max": ELITE_RB_ADP_MAX,
        "elite_rb_model_weight": ELITE_RB_MODEL_WEIGHT,
        "elite_rb_lifted_n": int(n_elite_lifted),
        "monotonic_by_position": monotonic_by_pos,
        "use_adp_features": True,
        "use_opportunity_features": True,
        "tuning_r_by_position": tuning_r,
        "best_params_by_position": best_params,
        "feature_cols_by_position": feature_cols_by_pos,
        "conformal_by_position": conformal,
        "holdout": hold_metrics,
        "holdout_pearson_r": hold_metrics.get("pearson_r"),
        "holdout_mae": hold_metrics.get("mae"),
        "holdout_tiered_by_prior_ppr": tier_metrics,
        "holdout_weighted_star_r": star_weighted,
        "holdout_by_year": by_year,
        "baselines": {
            "last_season_ppr": last_base,
            "trail3_mean_ppr": trail_base,
        },
    }

    # Persist
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    CSV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {
        "models_by_position": models_by_pos,
        "quantile_models_by_position": q_models,
        "conformal_by_position": conformal,
        "feature_cols_by_position": feature_cols_by_pos,
        "target": TARGET,
        "iqr_clamp_strength": DEFAULT_IQR_CLAMP_STRENGTH,
        "injury_iqr_relief": DEFAULT_INJURY_IQR_RELIEF,
        "elite_rb_lift": {
            "adp_max": ELITE_RB_ADP_MAX,
            "model_weight": ELITE_RB_MODEL_WEIGHT,
        },
    }
    joblib.dump(artifact, MODELS_DIR / VET_MODEL_FILE)
    (MODELS_DIR / VET_METRICS_FILE).write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    scored.to_csv(CSV_OUTPUT_DIR / VET_FEATURES_CSV, index=False)

    corr = feature_correlation_table(labeled)
    if not corr.empty:
        from rookie_ppr.veteran.config import VET_CORR_CSV

        corr.to_csv(CSV_OUTPUT_DIR / VET_CORR_CSV, index=False)

    # Baseline comparison CSV
    rows = []
    for name, block in (("last_season_ppr", last_base), ("trail3_mean_ppr", trail_base)):
        rows.append(
            {
                "baseline": name,
                "scope": "overall",
                "position": "",
                "n": block.get("n"),
                "full_r": block.get("full_r"),
                "baseline_r": block.get("baseline_r"),
                "lift_r": block.get("lift_r"),
                "full_mae": block.get("full_mae"),
                "baseline_mae": block.get("baseline_mae"),
            }
        )
        for pos, stats_row in (block.get("by_position") or {}).items():
            rows.append(
                {
                    "baseline": name,
                    "scope": "by_position",
                    "position": pos,
                    "n": stats_row.get("n"),
                    "full_r": stats_row.get("full_r"),
                    "baseline_r": stats_row.get("baseline_r"),
                    "lift_r": round(stats_row["full_r"] - stats_row["baseline_r"], 4)
                    if stats_row.get("full_r") is not None
                    and stats_row.get("baseline_r") is not None
                    else None,
                    "full_mae": stats_row.get("full_mae"),
                    "baseline_mae": stats_row.get("baseline_mae"),
                }
            )
    pd.DataFrame(rows).to_csv(CSV_OUTPUT_DIR / VET_BASELINE_CSV, index=False)

    return scored, metrics
