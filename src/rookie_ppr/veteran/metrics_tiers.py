"""Holdout metrics that emphasize top fantasy tiers (stars > depth)."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from rookie_ppr.model_score import _pearson_r

DEFAULT_TIERS = (20, 50, 100)


def _block(y: pd.Series, pred: np.ndarray) -> dict[str, Any]:
    y = pd.to_numeric(y, errors="coerce")
    pair = pd.DataFrame({"y": y, "p": pred}).dropna()
    n = int(len(pair))
    if n < 5:
        return {"n": n, "pearson_r": None, "mae": None}
    return {
        "n": n,
        "pearson_r": round(_pearson_r(pair["y"], pair["p"].to_numpy()), 4),
        "mae": round(float(np.mean(np.abs(pair["y"] - pair["p"]))), 3),
    }


def tiered_holdout_metrics(
    hold: pd.DataFrame,
    *,
    actual_col: str = "ppr_next",
    pred_col: str = "predicted_ppr_next",
    rank_col: str = "ppr",
    tiers: tuple[int, ...] = DEFAULT_TIERS,
) -> dict[str, Any]:
    """
    Pearson r on holdout subsets ranked by prior-season PPR (season T).

    Within each target_season × position, keep top K by ``rank_col`` and score
    how well the model ranks next-season outcomes for those tiers.
    """
    if hold.empty:
        return {}

    work = hold.copy()
    work[actual_col] = pd.to_numeric(work[actual_col], errors="coerce")
    work[pred_col] = pd.to_numeric(work[pred_col], errors="coerce")
    work[rank_col] = pd.to_numeric(work.get(rank_col), errors="coerce")
    work["target_season"] = pd.to_numeric(work.get("target_season"), errors="coerce")
    work = work[work[actual_col].notna() & work[pred_col].notna()].copy()
    if work.empty:
        return {}

    out: dict[str, Any] = {"tiers": list(tiers), "by_position": {}, "overall_pooled": {}}

    for pos, gpos in work.groupby("position"):
        pos_blocks: dict[str, Any] = {}
        for k in tiers:
            picks: list[pd.DataFrame] = []
            for _, gyear in gpos.groupby("target_season"):
                ranked = gyear.sort_values(rank_col, ascending=False, na_position="last")
                picks.append(ranked.head(int(k)))
            if not picks:
                continue
            pool = pd.concat(picks, ignore_index=True)
            pos_blocks[f"top_{k}"] = _block(pool[actual_col], pool[pred_col].to_numpy())
        if pos_blocks:
            out["by_position"][str(pos)] = pos_blocks

    for k in tiers:
        picks = []
        for (_, _), g in work.groupby(["target_season", "position"]):
            ranked = g.sort_values(rank_col, ascending=False, na_position="last")
            picks.append(ranked.head(int(k)))
        if picks:
            pool = pd.concat(picks, ignore_index=True)
            out["overall_pooled"][f"top_{k}"] = _block(pool[actual_col], pool[pred_col].to_numpy())

    # Overall all holdout rows (reference)
    out["all_holdout"] = _block(work[actual_col], work[pred_col].to_numpy())
    return out


def weighted_star_holdout_r(
    hold: pd.DataFrame,
    *,
    actual_col: str = "ppr_next",
    pred_col: str = "predicted_ppr_next",
    rank_col: str = "ppr",
) -> dict[str, Any]:
    """
    Weighted correlation: within each target_season × position pool,
    rank 1–20 weight 1.0, 21–50 → 0.55, 51–100 → 0.25, else 0.08.
    """
    work = hold.copy()
    work[actual_col] = pd.to_numeric(work[actual_col], errors="coerce")
    work[pred_col] = pd.to_numeric(work[pred_col], errors="coerce")
    work[rank_col] = pd.to_numeric(work.get(rank_col), errors="coerce")
    work = work[work[actual_col].notna() & work[pred_col].notna()].copy()
    if len(work) < 10:
        return {"n": int(len(work)), "weighted_pearson_r": None}

    weights = []
    ys = []
    ps = []
    for (_, _), g in work.groupby(["target_season", "position"]):
        ranked = g.sort_values(rank_col, ascending=False, na_position="last").reset_index(drop=True)
        for i, row in ranked.iterrows():
            rank = int(i) + 1
            if rank <= 20:
                w = 1.0
            elif rank <= 50:
                w = 0.55
            elif rank <= 100:
                w = 0.25
            else:
                w = 0.08
            weights.append(w)
            ys.append(float(row[actual_col]))
            ps.append(float(row[pred_col]))

    w = np.asarray(weights, dtype=float)
    y = np.asarray(ys, dtype=float)
    p = np.asarray(ps, dtype=float)
    if w.sum() <= 0 or len(y) < 5:
        return {"n": int(len(y)), "weighted_pearson_r": None}

    # Weighted Pearson via centered vectors
    y_m = np.average(y, weights=w)
    p_m = np.average(p, weights=w)
    num = np.sum(w * (y - y_m) * (p - p_m))
    den = np.sqrt(np.sum(w * (y - y_m) ** 2) * np.sum(w * (p - p_m) ** 2))
    r = float(num / den) if den > 0 else float("nan")
    return {"n": int(len(y)), "weighted_pearson_r": round(r, 4) if np.isfinite(r) else None}
