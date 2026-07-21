from __future__ import annotations

import json

import numpy as np
import pandas as pd

from rookie_ppr.config import MODELS_DIR, OUTPUT_DIR
from rookie_ppr.feature_composites import SCORE_COLUMNS, apply_composites, load_composite_artifacts
from rookie_ppr.model_score import explain_prediction, predict_from_composites

MASTER_PATH = OUTPUT_DIR / "players_master.csv"
PERCENTILE_PATH = MODELS_DIR / "percentile_lookup.json"


def load_players_master() -> pd.DataFrame:
    if not MASTER_PATH.exists():
        raise FileNotFoundError(
            f"Missing {MASTER_PATH}. Run: python -m rookie_ppr.compile"
        )
    return pd.read_csv(MASTER_PATH)


def load_percentile_lookup() -> dict[str, list[float]]:
    if not PERCENTILE_PATH.exists():
        return {}
    return json.loads(PERCENTILE_PATH.read_text(encoding="utf-8"))


def position_ppr_distribution(position: str, lookup: dict[str, list[float]] | None = None) -> dict:
    """Same-position historical rookie PPR quartiles for the optional distribution plot."""
    lookup = lookup if lookup is not None else load_percentile_lookup()
    hist = lookup.get(str(position), [])
    if len(hist) < 4:
        return {
            "position": position,
            "n": len(hist),
            "min": float("nan"),
            "q1": float("nan"),
            "median": float("nan"),
            "q3": float("nan"),
            "max": float("nan"),
        }
    arr = np.asarray(hist, dtype=float)
    return {
        "position": position,
        "n": int(len(arr)),
        "min": float(np.min(arr)),
        "q1": float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)),
        "q3": float(np.percentile(arr, 75)),
        "max": float(np.max(arr)),
    }


def classify_boom_bust(success_score: float) -> str:
    """Boom/bust from success score: <25 bust, 25–75 neutral, >75 boom."""
    if success_score is None or (isinstance(success_score, float) and np.isnan(success_score)):
        return "unknown"
    if success_score > 75:
        return "boom"
    if success_score < 25:
        return "bust"
    return "neutral"


def load_model_metrics() -> dict:
    path = MODELS_DIR / "model_metrics.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _composite_importance(artifacts_weights: dict[str, dict[str, float]], score_col: str) -> float:
    members = artifacts_weights.get(score_col, {})
    if not members:
        return 1.0
    return float(sum(abs(float(v)) for v in members.values()))


def prediction_confidence_band(
    position: str,
    predicted: float,
    composite_populated: list[str],
    composite_missing: list[str],
    *,
    ppr_low: float | None = None,
    ppr_high: float | None = None,
    cqr_alpha: float | None = None,
    cqr_q_hat: float | None = None,
    bust_chance_pct: float | None = None,
    boom_chance_pct: float | None = None,
    cqr_low: float | None = None,
    cqr_high: float | None = None,
    threshold_method: str = "predictive_quartile",
) -> dict:
    """
    Boom/bust gauge from the prediction-error distribution.

    Default: predictive Q1/Q3 (25th/75th percentile models) around the point forecast.
    Bust/boom header % are ~25% / ~25% (chance below Q1 / above Q3 of that error model).
    Optional CQR 80% band kept as side metadata.
    """
    metrics = load_model_metrics()

    if ppr_low is not None and ppr_high is not None and np.isfinite(ppr_low) and np.isfinite(ppr_high):
        low = float(ppr_low)
        high = float(ppr_high)
        if low > high:
            low, high = high, low
        half = max(float(predicted) - low, high - float(predicted), 0.0)

        if bust_chance_pct is not None and boom_chance_pct is not None:
            bust_pct = float(bust_chance_pct)
            boom_pct = float(boom_chance_pct)
        else:
            bust_pct, boom_pct = 25.0, 25.0

        out = {
            "method": threshold_method,
            "half_width": round(half, 2),
            "ppr_low": round(low, 1),
            "ppr_high": round(high, 1),
            "boom_chance_pct": round(boom_pct, 1),
            "bust_chance_pct": round(bust_pct, 1),
            "chance_method": "predictive_quartile",
            "bust_definition": "predictive_q25",
            "boom_definition": "predictive_q75",
        }
        if cqr_low is not None and cqr_high is not None:
            out["cqr_low"] = round(float(cqr_low), 1)
            out["cqr_high"] = round(float(cqr_high), 1)
            out["cqr_alpha"] = float(cqr_alpha if cqr_alpha is not None else 0.2)
            out["q_hat"] = round(float(cqr_q_hat), 4) if cqr_q_hat is not None else None
            out["nominal_coverage_pct"] = round(
                100.0 * (1.0 - float(cqr_alpha if cqr_alpha is not None else 0.2)), 1
            )
        return out

    # Legacy MAE heuristic fallback
    by_pos = metrics.get("holdout_by_position") or {}
    pos_stats = by_pos.get(str(position)) or {}
    mae = float(pos_stats.get("mae") or metrics.get("holdout_mae") or 50.0)

    feature_cols = (metrics.get("feature_cols_by_position") or {}).get(str(position))
    if not feature_cols:
        feature_cols = list(SCORE_COLUMNS)

    weights: dict[str, dict[str, float]] = {}
    weights_path = MODELS_DIR / "composite_weights.json"
    if weights_path.exists():
        data = json.loads(weights_path.read_text(encoding="utf-8"))
        weights = data.get("weights") or {}

    populated = set(composite_populated)
    total_imp = 0.0
    missing_imp = 0.0
    for col in feature_cols:
        imp = _composite_importance(weights, col)
        total_imp += imp
        if col not in populated:
            missing_imp += imp

    missing_frac = (missing_imp / total_imp) if total_imp > 0 else 1.0
    multiplier = 1.0 + missing_frac
    half_width = mae * multiplier

    return {
        "method": "mae_heuristic",
        "mae_base": round(mae, 2),
        "missing_importance_frac": round(missing_frac, 4),
        "multiplier": round(multiplier, 3),
        "half_width": round(half_width, 2),
        "ppr_low": round(float(predicted) - half_width, 1),
        "ppr_high": round(float(predicted) + half_width, 1),
        "boom_chance_pct": 25.0,
        "bust_chance_pct": 25.0,
        "chance_method": "quartile_fallback",
    }


def dropdown_options(master: pd.DataFrame) -> dict[str, list[str]]:
    colleges = sorted(master["college"].dropna().astype(str).unique()) if "college" in master.columns else []
    teams = sorted(master["draft_team"].dropna().astype(str).unique()) if "draft_team" in master.columns else []
    return {
        "college": colleges,
        "draft_team": teams,
    }


def player_lookup_labels(master: pd.DataFrame) -> list[tuple[str, int]]:
    """Display label -> row index in master."""
    rows: list[tuple[str, int]] = []
    for idx, row in master.iterrows():
        name = row.get("player_name", "Unknown")
        pos = row.get("position", "")
        year = row.get("draft_year", "")
        rows.append((f"{name} ({pos}, {int(year) if pd.notna(year) else '?'})", int(idx)))
    return sorted(rows, key=lambda x: x[0].lower())


def row_from_player(master: pd.DataFrame, index: int) -> dict:
    """Extract scorable fields from a historical player row."""
    from rookie_ppr.scoring_fields import FIELD_SPECS

    row = master.loc[index]
    out: dict = {}
    for spec in FIELD_SPECS:
        col = spec.column
        if col not in row.index:
            continue
        val = row[col]
        if pd.isna(val) or val == "":
            continue
        if col == "birth_date":
            if "age_at_draft" not in out and "draft_year" in row.index and pd.notna(row.get("draft_year")):
                dob = pd.to_datetime(val, errors="coerce")
                if pd.notna(dob):
                    out["age_at_draft"] = int(row["draft_year"]) - dob.year
            continue
        if col == "age_at_draft" and pd.notna(val):
            out[col] = int(float(val))
            continue
        if isinstance(val, float) and val == int(val):
            out[col] = int(val)
        else:
            out[col] = val
    return out


def _clean_row(row: dict) -> dict:
    out: dict = {}
    for k, v in row.items():
        if v is None or v == "":
            continue
        if isinstance(v, str):
            v = v.strip()
            if not v:
                continue
        out[k] = v
    return out


def score_player(row: dict) -> dict:
    row = _clean_row(row)
    position = row.get("position")
    if not position:
        raise ValueError("Position is required (QB, RB, WR, or TE).")

    if not (MODELS_DIR / "hgb_rookie_ppr.joblib").exists():
        raise FileNotFoundError(
            "ML model not found. Run: python -m rookie_ppr.compile"
        )

    artifacts = load_composite_artifacts(MODELS_DIR / "composite_weights.json")
    raw = pd.DataFrame([row])
    composites = apply_composites(raw, artifacts)
    result = predict_from_composites(composites, position=str(position))

    populated = [
        c for c in SCORE_COLUMNS if c in result.columns and pd.notna(result.iloc[0].get(c))
    ]
    missing = [c for c in SCORE_COLUMNS if c not in populated]

    predicted = float(result["predicted_rookie_ppr"].iloc[0])
    success = float(result["success_score_0_100"].iloc[0])
    dist = position_ppr_distribution(str(position))
    score_drivers = explain_prediction(
        composites, str(position), predicted=predicted
    )
    composites_out = {
        c: round(float(result.iloc[0][c]), 3)
        for c in populated
        if c in result.columns and pd.notna(result.iloc[0].get(c))
    }

    def _cell(col: str):
        if col not in result.columns:
            return None
        val = result.iloc[0].get(col)
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return None
        return float(val)

    cqr_low = _cell("cqr_low") if _cell("cqr_low") is not None else _cell("ppr_low")
    cqr_high = _cell("cqr_high") if _cell("cqr_high") is not None else _cell("ppr_high")
    cqr_alpha = _cell("cqr_alpha")
    cqr_q_hat = _cell("cqr_q_hat")
    pq25 = _cell("predictive_q25")
    pq75 = _cell("predictive_q75")

    # Fallback: shift point forecast by calibration residual quartiles
    if pq25 is None or pq75 is None:
        metrics = load_model_metrics()
        conf_pos = (metrics.get("conformal_by_position") or {}).get(str(position)) or {}
        rq25 = conf_pos.get("residual_q25")
        rq75 = conf_pos.get("residual_q75")
        if rq25 is not None and rq75 is not None:
            pq25 = predicted + float(rq25)
            pq75 = predicted + float(rq75)

    if pq25 is not None and pq75 is not None:
        confidence = prediction_confidence_band(
            str(position),
            predicted,
            populated,
            missing,
            ppr_low=pq25,
            ppr_high=pq75,
            bust_chance_pct=25.0,
            boom_chance_pct=25.0,
            cqr_low=cqr_low,
            cqr_high=cqr_high,
            cqr_alpha=cqr_alpha,
            cqr_q_hat=cqr_q_hat,
            threshold_method="predictive_quartile",
        )
    else:
        confidence = prediction_confidence_band(
            str(position),
            predicted,
            populated,
            missing,
            ppr_low=cqr_low,
            ppr_high=cqr_high,
            cqr_alpha=cqr_alpha,
            cqr_q_hat=cqr_q_hat,
            threshold_method="cqr",
        )

    return {
        "position": position,
        "predicted_rookie_ppr": predicted,
        "success_score_0_100": success,
        "composite_populated": populated,
        "composite_missing": missing,
        "composites": composites_out,
        "score_drivers": score_drivers,
        "ppr_distribution": dist,
        "boom_bust": classify_boom_bust(success),
        "confidence": confidence,
    }
