from __future__ import annotations

from pathlib import Path

import pandas as pd

from rookie_ppr.config import MODELS_DIR, OUTPUT_DIR
from rookie_ppr.feature_composites import SCORE_COLUMNS, apply_composites, load_composite_artifacts
from rookie_ppr.model_score import predict_from_composites

MASTER_PATH = OUTPUT_DIR / "players_master.csv"


def load_players_master() -> pd.DataFrame:
    if not MASTER_PATH.exists():
        raise FileNotFoundError(
            f"Missing {MASTER_PATH}. Run: python -m rookie_ppr.compile"
        )
    return pd.read_csv(MASTER_PATH)


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

    return {
        "position": position,
        "predicted_rookie_ppr": float(result["predicted_rookie_ppr"].iloc[0]),
        "success_score_0_100": float(result["success_score_0_100"].iloc[0]),
        "composite_populated": populated,
        "composite_missing": missing,
    }
