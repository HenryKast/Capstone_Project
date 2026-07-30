"""Veteran features: history, trends, team offense, SOS, Phase-2 OL/defense."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from rookie_ppr.veteran.config import FEATURE_COLS, TARGET


def add_history_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Lag / delta / trailing means within each player career."""
    if panel.empty:
        return panel
    out = panel.sort_values(["gsis_id", "season"]).copy()
    g = out.groupby("gsis_id", sort=False)

    out["lag1_ppr"] = g["ppr"].shift(1)
    out["lag2_ppr"] = g["ppr"].shift(2)
    out["delta_ppr"] = out["ppr"] - out["lag1_ppr"]
    out["lag1_touches"] = g["touches"].shift(1)
    out["delta_touches"] = out["touches"] - out["lag1_touches"]
    out["ppr_trail3_mean"] = (
        g["ppr"].transform(lambda s: s.shift(0).rolling(3, min_periods=1).mean())
    )
    # seasons played through T (1-indexed career count in panel)
    out["career_seasons"] = g.cumcount() + 1

    prior_team = g["team"].shift(1)
    out["team_changed"] = (
        (out["team"].notna() & prior_team.notna() & (out["team"] != prior_team)).astype(float)
    )
    out.loc[prior_team.isna(), "team_changed"] = np.nan

    # OL quality trend (team-level, still useful YoY for the player's line)
    if "ol_sack_rate" in out.columns:
        out["lag1_ol_sack_rate"] = g["ol_sack_rate"].shift(1)
        out["delta_ol_sack_rate"] = out["ol_sack_rate"] - out["lag1_ol_sack_rate"]
    else:
        out["delta_ol_sack_rate"] = np.nan

    if "off_snap_pct" in out.columns:
        out["lag1_off_snap_pct"] = g["off_snap_pct"].shift(1)
        out["delta_off_snap_pct"] = out["off_snap_pct"] - out["lag1_off_snap_pct"]
    else:
        out["delta_off_snap_pct"] = np.nan

    # Baselines for evaluation (not model-required, but useful columns)
    out["baseline_last_ppr"] = out["ppr"]
    out["baseline_trail3_ppr"] = out["ppr_trail3_mean"]
    return out


def build_feature_frame(panel: pd.DataFrame) -> pd.DataFrame:
    """Panel + history features; keeps unlabeled rows for future scoring."""
    return add_history_features(panel)


def feature_correlation_table(frame: pd.DataFrame, target: str = TARGET) -> pd.DataFrame:
    """Pearson abs-corr of features vs next-season PPR (labeled rows only)."""
    if frame.empty or target not in frame.columns:
        return pd.DataFrame()
    work = frame[pd.to_numeric(frame[target], errors="coerce").notna()].copy()
    y = pd.to_numeric(work[target], errors="coerce")
    rows: list[dict] = []
    for col in FEATURE_COLS:
        if col not in work.columns:
            continue
        x = pd.to_numeric(work[col], errors="coerce")
        pair = pd.DataFrame({"x": x, "y": y}).dropna()
        if len(pair) < 30 or pair["x"].nunique() < 2:
            continue
        r, p = stats.pearsonr(pair["x"], pair["y"])
        rows.append(
            {
                "feature": col,
                "n_pairs": int(len(pair)),
                "pearson_r": round(float(r), 4),
                "abs_corr": round(abs(float(r)), 4),
                "p_value": float(p),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("abs_corr", ascending=False).reset_index(drop=True)
