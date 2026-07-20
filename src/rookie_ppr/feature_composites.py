from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from rookie_ppr.analyze_correlation import _add_derived_columns

TARGET = "rookie_ppr"

# invert=True means multiply raw value by -1 before z-scoring (lower raw = better outcome)
COMPOSITE_SPECS: list[dict] = [
    {
        "name": "score_draft_capital",
        "members": [{"feature": "draft_overall", "invert": True}],
    },
    {
        "name": "score_pre_draft_fantasy",
        "members": [
            {"feature": "ff_adp", "invert": True},
            {"feature": "ff_adp_rank", "invert": True},
        ],
    },
    {
        "name": "score_recruiting",
        "members": [
            {"feature": "recruiting_rank", "invert": True},
            {"feature": "recruiting_stars", "invert": False},
        ],
    },
    {
        "name": "score_timing",
        "members": [
            {"feature": "age_at_draft", "invert": True},
            {"feature": "hs_class", "invert": False},
        ],
    },
    {
        "name": "score_cfb_receiving",
        "members": [
            {"feature": "cfb_rec", "invert": False},
            {"feature": "cfb_rec_yards", "invert": False},
            {"feature": "cfb_rec_td", "invert": False},
        ],
    },
    {
        "name": "score_cfb_rushing",
        "members": [
            {"feature": "cfb_rush_yards", "invert": False},
            {"feature": "cfb_rush_td", "invert": False},
        ],
    },
    {
        "name": "score_cfb_passing",
        "members": [
            {"feature": "cfb_pass_yards", "invert": False},
            {"feature": "cfb_pass_td", "invert": False},
        ],
    },
    {
        "name": "score_combine_speed",
        "members": [
            {"feature": "forty", "invert": True},
            {"feature": "cone", "invert": True},
            {"feature": "shuttle", "invert": True},
        ],
    },
    {
        "name": "score_combine_explosion",
        "members": [
            {"feature": "vertical", "invert": False},
            {"feature": "broad_jump", "invert": False},
        ],
    },
    {
        "name": "score_combine_size",
        "members": [
            {"feature": "ht_inches", "invert": False},
            {"feature": "wt", "invert": False},
        ],
    },
    {
        "name": "score_team_context",
        "members": [
            {"feature": "sos_opp_win_pct", "invert": True},
            {"feature": "team_opportunity_ppr", "invert": False},
            {"feature": "off_pass_rate_proxy", "invert": False},
            {"feature": "off_pass_yards", "invert": True},
            {"feature": "off_rush_yards", "invert": True},
        ],
    },
]

SCORE_COLUMNS = [spec["name"] for spec in COMPOSITE_SPECS]
ID_COLUMNS = ["gsis_id", "player_name", "position", "draft_year"]


@dataclass
class CompositeArtifacts:
    position_norms: dict[str, dict[str, dict[str, float]]]
    weights: dict[str, dict[str, float]]
    specs: list[dict]

    def to_json_dict(self) -> dict:
        return {
            "position_norms": self.position_norms,
            "weights": self.weights,
            "specs": self.specs,
        }

    @classmethod
    def from_json_dict(cls, data: dict) -> CompositeArtifacts:
        return cls(
            position_norms=data["position_norms"],
            weights=data["weights"],
            specs=data["specs"],
        )


def _weight_map(feature_corr: pd.DataFrame) -> dict[str, float]:
    if feature_corr is None or feature_corr.empty:
        return {}
    out = {}
    for _, row in feature_corr.iterrows():
        feat = row.get("feature")
        val = row.get("abs_corr_rookie_ppr")
        if pd.notna(feat) and pd.notna(val):
            out[str(feat)] = float(val)
    return out


def _position_z(
    series: pd.Series,
    position: pd.Series,
    norms: dict[str, dict[str, dict[str, float]]],
    feature: str,
    fit: bool,
) -> pd.Series:
    out = pd.Series(np.nan, index=series.index, dtype=float)
    for pos in position.dropna().unique():
        mask = position == pos
        vals = pd.to_numeric(series.loc[mask], errors="coerce")
        if fit:
            mean = float(vals.mean()) if vals.notna().any() else 0.0
            std = float(vals.std()) if vals.notna().sum() > 1 else 1.0
            if std == 0 or np.isnan(std):
                std = 1.0
            norms.setdefault(str(pos), {})[feature] = {"mean": mean, "std": std}
        pos_norm = norms.get(str(pos), {}).get(feature, {"mean": 0.0, "std": 1.0})
        mean = pos_norm.get("mean", 0.0)
        std = pos_norm.get("std", 1.0) or 1.0
        out.loc[mask] = (vals - mean) / std
    return out


def build_composite_artifacts(
    train_df: pd.DataFrame,
    feature_corr: pd.DataFrame | None = None,
) -> CompositeArtifacts:
    weights_src = _weight_map(feature_corr)
    weights: dict[str, dict[str, float]] = {}
    position_norms: dict[str, dict[str, dict[str, float]]] = {}

    work = _add_derived_columns(train_df)
    position = work["position"] if "position" in work.columns else pd.Series("UNK", index=work.index)

    for spec in COMPOSITE_SPECS:
        group_weights: dict[str, float] = {}
        for member in spec["members"]:
            feat = member["feature"]
            group_weights[feat] = weights_src.get(feat, 1.0)
            if feat not in work.columns:
                continue
            raw = pd.to_numeric(work[feat], errors="coerce")
            if member.get("invert"):
                raw = raw * -1
            _position_z(raw, position, position_norms, feat, fit=True)
        weights[spec["name"]] = group_weights

    return CompositeArtifacts(position_norms=position_norms, weights=weights, specs=COMPOSITE_SPECS)


def apply_composites(
    df: pd.DataFrame,
    artifacts: CompositeArtifacts,
    feature_corr: pd.DataFrame | None = None,
) -> pd.DataFrame:
    work = _add_derived_columns(df)
    position = work["position"] if "position" in work.columns else pd.Series("UNK", index=work.index)
    weights_src = _weight_map(feature_corr)
    out = work[[c for c in ID_COLUMNS if c in work.columns]].copy()

    for spec in artifacts.specs:
        name = spec["name"]
        z_cols: list[pd.Series] = []
        w_cols: list[float] = []
        for member in spec["members"]:
            feat = member["feature"]
            if feat not in work.columns:
                continue
            raw = pd.to_numeric(work[feat], errors="coerce")
            if member.get("invert"):
                raw = raw * -1
            z = _position_z(raw, position, artifacts.position_norms, feat, fit=False)
            w = artifacts.weights.get(name, {}).get(feat, weights_src.get(feat, 1.0))
            z_cols.append(z)
            w_cols.append(float(w))

        if not z_cols:
            out[name] = pd.NA
            out[f"{name}_coverage"] = 0.0
            continue

        z_frame = pd.concat(z_cols, axis=1)
        weights_arr = np.array(w_cols, dtype=float)
        mask = z_frame.notna()
        weighted = z_frame.where(mask).mul(weights_arr, axis=1)
        denom = mask.mul(weights_arr, axis=1).sum(axis=1)
        numer = weighted.sum(axis=1, skipna=True)
        out[name] = np.where(denom > 0, numer / denom, np.nan)
        out[f"{name}_coverage"] = mask.sum(axis=1) / len(z_cols)

    return out


def build_composite_correlation(composites: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    if composites.empty or master.empty or TARGET not in master.columns:
        return pd.DataFrame()

    merged = composites.merge(
        master[[c for c in ID_COLUMNS + [TARGET] if c in master.columns]],
        on=[c for c in ID_COLUMNS if c in composites.columns and c in master.columns],
        how="left",
    )
    target = pd.to_numeric(merged[TARGET], errors="coerce")
    rows = []
    abs_corrs: list[float] = []

    for col in SCORE_COLUMNS:
        if col not in merged.columns:
            continue
        x = pd.to_numeric(merged[col], errors="coerce")
        pair = pd.DataFrame({"x": x, "y": target}).dropna()
        n = len(pair)
        if n < 10:
            continue
        pearson_r, pearson_p = stats.pearsonr(pair["x"], pair["y"])
        spearman_r, spearman_p = stats.spearmanr(pair["x"], pair["y"])
        abs_r = abs(pearson_r)
        abs_corrs.append(abs_r)
        rows.append(
            {
                "composite": col,
                "n_pairs": n,
                "pearson_r": round(float(pearson_r), 4),
                "pearson_p": round(float(pearson_p), 6),
                "spearman_r": round(float(spearman_r), 4),
                "spearman_p": round(float(spearman_p), 6),
                "abs_corr_rookie_ppr": round(float(abs_r), 4),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    max_abs = max(abs_corrs) if abs_corrs else 1.0
    out["impact_0_100"] = (out["abs_corr_rookie_ppr"] / max_abs * 100).round(1)
    out = out.sort_values("abs_corr_rookie_ppr", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", range(1, len(out) + 1))
    return out


def save_composite_artifacts(artifacts: CompositeArtifacts, path) -> None:
    path = path if hasattr(path, "write_text") else None
    from pathlib import Path

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(artifacts.to_json_dict(), indent=2), encoding="utf-8")


def load_composite_artifacts(path) -> CompositeArtifacts:
    from pathlib import Path

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return CompositeArtifacts.from_json_dict(data)
