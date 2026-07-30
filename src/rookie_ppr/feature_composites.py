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
            {"feature": "vacated_touches", "invert": False},
            {"feature": "vacated_carries", "invert": False},
            {"feature": "incumbent_pos_ppr", "invert": True},
            {"feature": "incumbent_pos_carries", "invert": True},
            {"feature": "incumbent_pos_touches", "invert": True},
            {"feature": "off_pass_rate_proxy", "invert": False},
            {"feature": "off_pass_yards", "invert": False},
            {"feature": "off_rush_yards", "invert": False},
        ],
    },
]

# RB: emphasize vacated rushing volume + incumbent competition; drop noisy pass-rate proxies
COMPOSITE_MEMBER_OVERRIDES: dict[str, dict[str, list[dict]]] = {
    "RB": {
        "score_team_context": [
            {"feature": "vacated_touches", "invert": False},
            {"feature": "vacated_carries", "invert": False},
            {"feature": "incumbent_pos_ppr", "invert": True},
            {"feature": "incumbent_pos_carries", "invert": True},
            {"feature": "incumbent_pos_touches", "invert": True},
            {"feature": "incumbent_ff_adp", "invert": True},
            {"feature": "adp_vs_incumbent", "invert": False},
            {"feature": "sos_opp_win_pct", "invert": True},
            {"feature": "off_rush_yards", "invert": False},
        ],
    },
}

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


def feature_corr_for_weights(
    feature_corr: pd.DataFrame | None,
    *,
    source_abs_col: str = "abs_corr_rookie_ppr",
) -> pd.DataFrame:
    """
    Thin adapter so dynasty (or other) corr tables feed _weight_map unchanged.

    Copies source_abs_col into abs_corr_rookie_ppr when needed; redraft tables
    already use that column and pass through as-is.
    """
    if feature_corr is None or feature_corr.empty:
        return pd.DataFrame() if feature_corr is None else feature_corr.copy()
    out = feature_corr.copy()
    if source_abs_col != "abs_corr_rookie_ppr" and source_abs_col in out.columns:
        out["abs_corr_rookie_ppr"] = out[source_abs_col]
    return out


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


def _members_for_spec(spec: dict, position: str | None) -> list[dict]:
    pos = str(position) if position is not None else ""
    override = COMPOSITE_MEMBER_OVERRIDES.get(pos, {}).get(spec["name"])
    return list(override) if override else list(spec["members"])


def _all_member_features() -> list[dict]:
    """Union of default + override members (for fitting norms/weights)."""
    seen: set[str] = set()
    members: list[dict] = []
    for spec in COMPOSITE_SPECS:
        for member in spec["members"]:
            if member["feature"] not in seen:
                seen.add(member["feature"])
                members.append(member)
        for pos_map in COMPOSITE_MEMBER_OVERRIDES.values():
            for member in pos_map.get(spec["name"], []):
                if member["feature"] not in seen:
                    seen.add(member["feature"])
                    members.append(member)
    return members


def build_composite_artifacts(
    train_df: pd.DataFrame,
    feature_corr: pd.DataFrame | None = None,
) -> CompositeArtifacts:
    weights_src = _weight_map(feature_corr)
    weights: dict[str, dict[str, float]] = {}
    position_norms: dict[str, dict[str, dict[str, float]]] = {}

    work = _add_derived_columns(train_df)
    position = work["position"] if "position" in work.columns else pd.Series("UNK", index=work.index)

    # Fit norms for every feature that any position composite may use
    for member in _all_member_features():
        feat = member["feature"]
        if feat not in work.columns:
            continue
        raw = pd.to_numeric(work[feat], errors="coerce")
        if member.get("invert"):
            raw = raw * -1
        _position_z(raw, position, position_norms, feat, fit=True)

    for spec in COMPOSITE_SPECS:
        group_weights: dict[str, float] = {}
        for member in _all_member_features():
            # Only store weights for features in this composite (default or any override)
            used = {m["feature"] for m in spec["members"]}
            for pos_map in COMPOSITE_MEMBER_OVERRIDES.values():
                used.update(m["feature"] for m in pos_map.get(spec["name"], []))
            if member["feature"] in used:
                group_weights[member["feature"]] = weights_src.get(member["feature"], 1.0)
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
        scores = pd.Series(np.nan, index=work.index, dtype=float)
        coverage = pd.Series(0.0, index=work.index, dtype=float)

        positions = position.fillna("UNK").unique()
        for pos in positions:
            members = _members_for_spec(spec, pos)
            mask = position.fillna("UNK") == pos
            if not mask.any():
                continue

            z_cols: list[pd.Series] = []
            w_cols: list[float] = []
            for member in members:
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
                continue

            z_frame = pd.concat(z_cols, axis=1)
            weights_arr = np.array(w_cols, dtype=float)
            present = z_frame.notna()
            weighted = z_frame.where(present).mul(weights_arr, axis=1)
            denom = present.mul(weights_arr, axis=1).sum(axis=1)
            numer = weighted.sum(axis=1, skipna=True)
            pos_score = np.where(denom > 0, numer / denom, np.nan)
            scores.loc[mask] = pos_score[mask.to_numpy()]
            coverage.loc[mask] = (present.sum(axis=1) / len(z_cols)).loc[mask]

        out[name] = scores
        out[f"{name}_coverage"] = coverage

    return out


def build_composite_correlation(
    composites: pd.DataFrame,
    master: pd.DataFrame,
    *,
    target: str = TARGET,
    abs_corr_col: str = "abs_corr_rookie_ppr",
) -> pd.DataFrame:
    """Composite vs outcome correlation. Default target/column names preserve redraft schema."""
    if composites.empty or master.empty or target not in master.columns:
        return pd.DataFrame()

    merged = composites.merge(
        master[[c for c in ID_COLUMNS + [target] if c in master.columns]],
        on=[c for c in ID_COLUMNS if c in composites.columns and c in master.columns],
        how="left",
    )
    y = pd.to_numeric(merged[target], errors="coerce")
    rows = []
    abs_corrs: list[float] = []

    for col in SCORE_COLUMNS:
        if col not in merged.columns:
            continue
        x = pd.to_numeric(merged[col], errors="coerce")
        pair = pd.DataFrame({"x": x, "y": y}).dropna()
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
                abs_corr_col: round(float(abs_r), 4),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    max_abs = max(abs_corrs) if abs_corrs else 1.0
    out["impact_0_100"] = (out[abs_corr_col] / max_abs * 100).round(1)
    out = out.sort_values(abs_corr_col, ascending=False).reset_index(drop=True)
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
