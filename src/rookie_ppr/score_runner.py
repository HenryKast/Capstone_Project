from __future__ import annotations

import json

import numpy as np
import pandas as pd

from rookie_ppr.analysis_config import AnalysisMode, get_mode
from rookie_ppr.config import CSV_OUTPUT_DIR, MODELS_DIR, OUTPUT_DIR
from rookie_ppr.feature_composites import SCORE_COLUMNS, apply_composites, load_composite_artifacts
from rookie_ppr.model_score import explain_prediction, load_model_bundle, predict_from_composites

MASTER_PATH = OUTPUT_DIR / "players_master.csv"

# Raw columns reattached after apply_composites (IDs/composites only otherwise)
_DYNASTY_ATTACH_COLS = ("ff_adp", "ff_adp_rank", "draft_overall")


def _as_mode(mode: str | AnalysisMode | None = None) -> AnalysisMode:
    if isinstance(mode, AnalysisMode):
        return mode
    return get_mode(mode or "redraft")


def load_players_master() -> pd.DataFrame:
    if not MASTER_PATH.exists():
        raise FileNotFoundError(
            f"Missing {MASTER_PATH}. Run: python -m rookie_ppr.compile"
        )
    return pd.read_csv(MASTER_PATH)


def load_percentile_lookup(mode: str | AnalysisMode | None = None) -> dict[str, list[float]]:
    analysis = _as_mode(mode)
    path = MODELS_DIR / analysis.percentile_file
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def position_ppr_distribution(
    position: str,
    lookup: dict[str, list[float]] | None = None,
    mode: str | AnalysisMode | None = None,
) -> dict:
    """Same-position historical PPR quartiles for the optional distribution plot."""
    lookup = lookup if lookup is not None else load_percentile_lookup(mode)
    # Nested multi-target dynasty lookups store primary under by_target; ignore that key here
    if isinstance(lookup, dict) and position not in lookup and "by_target" in lookup:
        primary = _as_mode(mode).primary_target
        nested = lookup.get("by_target") or {}
        if isinstance(nested, dict) and primary in nested:
            lookup = nested[primary]
    hist = lookup.get(str(position), []) if isinstance(lookup, dict) else []
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


def load_model_metrics(mode: str | AnalysisMode | None = None) -> dict:
    analysis = _as_mode(mode)
    path = MODELS_DIR / analysis.metrics_file
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
    mode: str | AnalysisMode | None = None,
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
    analysis = _as_mode(mode)
    metrics = load_model_metrics(analysis)

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
    weights_path = MODELS_DIR / analysis.composite_weights_file
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

    # Dynasty year outcomes (for UI actual vs predicted breakdown)
    for col in ("ppr_y1", "ppr_y2", "ppr_y3", "dynasty_ppr_y1_y3_total"):
        if col in row.index and pd.notna(row.get(col)):
            out[col] = float(row[col])
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


def _reattach_gate_cols(composites: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """apply_composites keeps only IDs + scores; reattach ADP/draft when present on the raw row."""
    out = composites.copy()
    for col in _DYNASTY_ATTACH_COLS:
        if col in raw.columns:
            out[col] = pd.to_numeric(raw[col], errors="coerce").to_numpy()
    return out


def _try_precomputed_dynasty_row(row: dict, analysis: AnalysisMode) -> dict | None:
    """Fallback: look up a precomputed prediction in ml_features_dynasty.csv."""
    path = CSV_OUTPUT_DIR / "ml_features_dynasty.csv"
    if not path.exists():
        return None
    try:
        feats = pd.read_csv(path)
    except Exception:  # noqa: BLE001
        return None
    if analysis.predicted_col not in feats.columns:
        return None

    mask = pd.Series(True, index=feats.index)
    for key in ("gsis_id", "player_name", "position", "draft_year"):
        if key not in row or key not in feats.columns:
            continue
        val = row[key]
        if val is None or (isinstance(val, float) and pd.isna(val)):
            continue
        if key == "draft_year":
            mask &= pd.to_numeric(feats[key], errors="coerce") == float(val)
        else:
            mask &= feats[key].astype(str) == str(val)
    hits = feats.loc[mask]
    if hits.empty:
        return None
    hit = hits.iloc[0]
    predicted = hit.get(analysis.predicted_col)
    if predicted is None or (isinstance(predicted, float) and pd.isna(predicted)):
        return None
    success = hit.get(analysis.success_score_col)
    if success is None or (isinstance(success, float) and pd.isna(success)):
        success = float("nan")
    return {
        "predicted": float(predicted),
        "success": float(success),
        "source": "ml_features_dynasty",
        "row": hit,
    }


def score_player(row: dict, mode: str | AnalysisMode | None = None) -> dict:
    analysis = _as_mode(mode)
    row = _clean_row(row)
    position = row.get("position")
    if not position:
        raise ValueError("Position is required (QB, RB, WR, or TE).")

    model_path = MODELS_DIR / analysis.model_file
    if not model_path.exists():
        raise FileNotFoundError(
            f"ML model not found ({model_path.name}). Run: python -m rookie_ppr.compile"
        )

    artifacts = load_composite_artifacts(MODELS_DIR / analysis.composite_weights_file)
    raw = pd.DataFrame([row])
    composites = apply_composites(raw, artifacts)
    composites = _reattach_gate_cols(composites, raw)

    bundle = load_model_bundle(analysis)
    percentile_lookup = load_percentile_lookup(analysis)
    used_precomputed = False
    precomputed_row = None
    try:
        result = predict_from_composites(
            composites,
            position=str(position),
            bundle=bundle,
            percentile_lookup=percentile_lookup,
            mode=analysis,
        )
        predicted = float(result[analysis.predicted_col].iloc[0])
        success = float(result[analysis.success_score_col].iloc[0])
    except Exception as live_exc:  # noqa: BLE001
        if analysis.name != "dynasty":
            raise
        pre = _try_precomputed_dynasty_row(row, analysis)
        if pre is None:
            raise live_exc
        used_precomputed = True
        precomputed_row = pre["row"]
        predicted = float(pre["predicted"])
        success = float(pre["success"])
        result = composites.copy()
        result[analysis.predicted_col] = predicted
        result[analysis.success_score_col] = success
        pre_row = pre["row"]
        copy_cols = [
            "predictive_q25",
            "predictive_q75",
            "ppr_low",
            "ppr_high",
            "cqr_alpha",
            "cqr_q_hat",
        ]
        for i in (1, 2, 3):
            yt = f"ppr_y{i}"
            copy_cols.extend(
                [
                    f"predicted_{yt}",
                    f"predictive_q25_{yt}",
                    f"predictive_q75_{yt}",
                    f"ppr_low_{yt}",
                    f"ppr_high_{yt}",
                ]
            )
        for col in copy_cols:
            if col not in pre_row.index or pd.isna(pre_row.get(col)):
                continue
            # Normalize year IQR names to the live predict_from_composites scheme
            dest = col
            if col.startswith("predictive_q25_ppr_y"):
                dest = f"predictive_q25_ppr_y{col[-1]}"
            elif col.startswith("predictive_q75_ppr_y"):
                dest = f"predictive_q75_ppr_y{col[-1]}"
            elif col.startswith("ppr_low_ppr_y"):
                dest = f"ppr_low_ppr_y{col[-1]}"
            elif col.startswith("ppr_high_ppr_y"):
                dest = f"ppr_high_ppr_y{col[-1]}"
            elif col.startswith("predicted_ppr_y"):
                dest = f"predicted_ppr_y{col[-1]}"
            elif col == "ppr_low":
                dest = "cqr_low"
            elif col == "ppr_high":
                dest = "cqr_high"
            result[dest] = pre_row[col]
            result[col] = pre_row[col]

    populated = [
        c for c in SCORE_COLUMNS if c in result.columns and pd.notna(result.iloc[0].get(c))
    ]
    missing = [c for c in SCORE_COLUMNS if c not in populated]

    dist = position_ppr_distribution(str(position), lookup=percentile_lookup, mode=analysis)
    score_drivers = (
        []
        if used_precomputed
        else explain_prediction(
            composites, str(position), predicted=predicted, bundle=bundle, mode=analysis
        )
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
        metrics = load_model_metrics(analysis)
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
            mode=analysis,
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
            mode=analysis,
            ppr_low=cqr_low,
            ppr_high=cqr_high,
            cqr_alpha=cqr_alpha,
            cqr_q_hat=cqr_q_hat,
            threshold_method="cqr",
        )

    out = {
        "mode": analysis.name,
        "predicted_col": analysis.predicted_col,
        "success_score_col": analysis.success_score_col,
        "predicted_points": predicted,
        "success_score": success,
        analysis.predicted_col: predicted,
        analysis.success_score_col: success,
        # Aliases so existing UI helpers keep working
        "predicted_rookie_ppr": predicted,
        "success_score_0_100": success,
        "position": position,
        "composite_populated": populated,
        "composite_missing": missing,
        "composites": composites_out,
        "score_drivers": score_drivers,
        "ppr_distribution": dist,
        "boom_bust": classify_boom_bust(success),
        "confidence": confidence,
        "prediction_source": "precomputed" if used_precomputed else "live",
    }

    if analysis.name == "dynasty":
        years: dict[str, dict] = {}
        lookup = percentile_lookup
        by_target = lookup.get("by_target") if isinstance(lookup, dict) else None
        for i in (1, 2, 3):
            yt = f"ppr_y{i}"
            pred_y = _cell(f"predicted_ppr_y{i}")
            q25 = _cell(f"predictive_q25_ppr_y{i}")
            q75 = _cell(f"predictive_q75_ppr_y{i}")
            cqr_lo = _cell(f"ppr_low_ppr_y{i}")
            cqr_hi = _cell(f"ppr_high_ppr_y{i}")

            if precomputed_row is not None:
                def _pre(col: str):
                    if col in precomputed_row.index and pd.notna(precomputed_row.get(col)):
                        return float(precomputed_row[col])
                    return None

                if pred_y is None:
                    pred_y = _pre(f"predicted_{yt}")
                if q25 is None:
                    q25 = _pre(f"predictive_q25_{yt}")
                if q75 is None:
                    q75 = _pre(f"predictive_q75_{yt}")
                if cqr_lo is None:
                    cqr_lo = _pre(f"ppr_low_{yt}")
                if cqr_hi is None:
                    cqr_hi = _pre(f"ppr_high_{yt}")

            if pred_y is not None and np.isfinite(pred_y):
                pred_y = round(float(pred_y), 2)

            actual_y = None
            raw_actual = row.get(f"ppr_y{i}")
            if raw_actual is not None and raw_actual != "":
                try:
                    actual_y = float(raw_actual)
                except (TypeError, ValueError):
                    actual_y = None

            year_conf = None
            if pred_y is not None and q25 is not None and q75 is not None:
                year_conf = {
                    "method": "predictive_quartile",
                    "ppr_low": round(float(q25), 2),
                    "ppr_high": round(float(q75), 2),
                    "bust_chance_pct": 25.0,
                    "boom_chance_pct": 25.0,
                    "cqr_low": round(float(cqr_lo), 2) if cqr_lo is not None else None,
                    "cqr_high": round(float(cqr_hi), 2) if cqr_hi is not None else None,
                }
            elif pred_y is not None and cqr_lo is not None and cqr_hi is not None:
                year_conf = {
                    "method": "cqr",
                    "ppr_low": round(float(cqr_lo), 2),
                    "ppr_high": round(float(cqr_hi), 2),
                    "bust_chance_pct": 25.0,
                    "boom_chance_pct": 25.0,
                    "cqr_low": round(float(cqr_lo), 2),
                    "cqr_high": round(float(cqr_hi), 2),
                }

            y_lookup = (by_target or {}).get(yt) if isinstance(by_target, dict) else None
            y_dist = (
                position_ppr_distribution(str(position), lookup=y_lookup, mode=analysis)
                if y_lookup
                else {}
            )
            y_success = None
            if pred_y is not None and y_lookup:
                from rookie_ppr.model_score import _to_success_score

                y_success = _to_success_score(pred_y, str(position), y_lookup)

            years[f"y{i}"] = {
                "label": f"Y{i}",
                "target": yt,
                "predicted": pred_y,
                "actual": actual_y if actual_y is not None and np.isfinite(actual_y) else None,
                "success_score": round(float(y_success), 1) if y_success is not None and np.isfinite(y_success) else None,
                "confidence": year_conf,
                "ppr_distribution": y_dist,
            }

        out["year_by_year"] = years
        out["year_breakdown_scaled_to_total"] = False

    return out
