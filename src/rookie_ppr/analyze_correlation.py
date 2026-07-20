from __future__ import annotations

import pandas as pd
from scipy import stats

TARGET = "rookie_ppr"
TARGET_BINARY = "rookie_success_top_half"

FEATURE_BLOCKS: dict[str, list[str]] = {
    "recruiting": ["recruiting_rank", "recruiting_stars"],
    "draft_capital": ["draft_round", "draft_overall"],
        "team_context": [
        "sos_opp_win_pct",
        "team_opportunity_ppr",
        "off_pass_rate_proxy",
        "off_pass_yards",
        "off_rush_yards",
    ],
    "combine": ["ht_inches", "wt", "forty", "bench", "vertical", "broad_jump", "cone", "shuttle"],
    "college_production": [
        "cfb_pass_yards",
        "cfb_pass_td",
        "cfb_rush_yards",
        "cfb_rush_td",
        "cfb_rec",
        "cfb_rec_yards",
        "cfb_rec_td",
    ],
    "pre_draft_fantasy": ["ff_adp", "ff_adp_rank"],
    "timing": ["hs_class", "age_at_draft"],
}

NUMERIC_FEATURES = sorted({f for feats in FEATURE_BLOCKS.values() for f in feats})


def _parse_ht_inches(val) -> float:
    if pd.isna(val):
        return float("nan")
    s = str(val).strip()
    if "-" in s:
        parts = s.split("-", 1)
        try:
            feet = int(parts[0])
            inches = int(parts[1])
            return float(feet * 12 + inches)
        except (ValueError, IndexError):
            return float("nan")
    return pd.to_numeric(s, errors="coerce")


def _column_series(frame: pd.DataFrame, col: str) -> pd.Series:
    """Return a 1-D Series even when duplicate column names exist."""
    sel = frame[col]
    if isinstance(sel, pd.DataFrame):
        sel = sel.iloc[:, 0]
    return sel.squeeze()


def _add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "age_at_draft" in out.columns and pd.to_numeric(out["age_at_draft"], errors="coerce").notna().any():
        out["age_at_draft"] = pd.to_numeric(out["age_at_draft"], errors="coerce")
    elif "birth_date" in out.columns and "draft_year" in out.columns:
        dob = pd.to_datetime(out["birth_date"], errors="coerce")
        out["age_at_draft"] = out["draft_year"] - dob.dt.year
    else:
        out["age_at_draft"] = pd.NA

    if "ht" in out.columns:
        out["ht_inches"] = out["ht"].map(_parse_ht_inches)
    else:
        out["ht_inches"] = pd.NA

    if TARGET in out.columns:
        out[TARGET_BINARY] = pd.NA
        mask = out[TARGET].notna()
        if mask.any() and "position" in out.columns and "draft_year" in out.columns:
            out.loc[mask, TARGET_BINARY] = (
                out.loc[mask]
                .groupby(["draft_year", "position"])[TARGET]
                .transform(lambda s: (s >= s.median()).astype(int))
            )
    return out


def _impact_score(abs_corr: float, max_abs: float) -> float:
    if pd.isna(abs_corr) or max_abs <= 0:
        return pd.NA
    return round(float(abs_corr / max_abs * 100), 1)


def _feature_block(feature: str) -> str:
    for block, feats in FEATURE_BLOCKS.items():
        if feature in feats:
            return block
    return "other"


def build_feature_correlation(master: pd.DataFrame) -> pd.DataFrame:
    df = _add_derived_columns(master)
    target = pd.to_numeric(df[TARGET], errors="coerce") if TARGET in df.columns else pd.Series(dtype=float)

    abs_corrs: list[float] = []
    pre_rows: list[dict] = []

    for feature in NUMERIC_FEATURES:
        if feature not in df.columns:
            continue
        x = pd.to_numeric(df[feature], errors="coerce")
        if x.notna().sum() < 10 or x.dropna().nunique() < 2:
            continue
        pair = pd.DataFrame({"x": x.astype(float), "y": target.astype(float)}).dropna()
        n = len(pair)
        missing_rate = 1 - (x.notna().sum() / len(df)) if len(df) else pd.NA

        pearson_r = pearson_p = spearman_r = spearman_p = pb_r = pb_p = pd.NA
        if n >= 10:
            pearson_r, pearson_p = stats.pearsonr(pair["x"], pair["y"])
            spearman_r, spearman_p = stats.spearmanr(pair["x"], pair["y"])

        if TARGET_BINARY in df.columns:
            bin_df = pd.DataFrame({"x": x, "y": df[TARGET_BINARY]}).dropna()
            bin_df["x"] = pd.to_numeric(bin_df["x"], errors="coerce")
            bin_df["y"] = pd.to_numeric(bin_df["y"], errors="coerce")
            bin_df = bin_df.dropna()
            if len(bin_df) >= 10 and bin_df["y"].nunique() == 2:
                pb_r, pb_p = stats.pointbiserialr(bin_df["y"].astype(float), bin_df["x"].astype(float))

        abs_primary = abs(pearson_r) if pd.notna(pearson_r) else pd.NA
        if pd.notna(abs_primary):
            abs_corrs.append(abs_primary)

        direction = "-"
        if pd.notna(pearson_r) and pearson_r >= 0:
            direction = "+"

        pre_rows.append(
            {
                "feature": feature,
                "dataset_block": _feature_block(feature),
                "n_pairs": n,
                "missing_rate": round(float(missing_rate), 4) if pd.notna(missing_rate) else pd.NA,
                "pearson_r": round(float(pearson_r), 4) if pd.notna(pearson_r) else pd.NA,
                "pearson_p": round(float(pearson_p), 6) if pd.notna(pearson_p) else pd.NA,
                "spearman_r": round(float(spearman_r), 4) if pd.notna(spearman_r) else pd.NA,
                "spearman_p": round(float(spearman_p), 6) if pd.notna(spearman_p) else pd.NA,
                "pointbiserial_r": round(float(pb_r), 4) if pd.notna(pb_r) else pd.NA,
                "pointbiserial_p": round(float(pb_p), 6) if pd.notna(pb_p) else pd.NA,
                "direction": direction,
                "abs_corr_rookie_ppr": round(float(abs_primary), 4) if pd.notna(abs_primary) else pd.NA,
            }
        )

    max_abs = max(abs_corrs) if abs_corrs else 1.0
    rows = []
    for row in pre_rows:
        row["impact_0_100"] = _impact_score(row["abs_corr_rookie_ppr"], max_abs)
        rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("abs_corr_rookie_ppr", ascending=False, na_position="last")
    return out.reset_index(drop=True)


def build_strongest_factors(feature_corr: pd.DataFrame) -> pd.DataFrame:
    if feature_corr.empty:
        return pd.DataFrame(
            columns=[
                "rank",
                "factor",
                "dataset_block",
                "abs_corr_rookie_ppr",
                "signed_corr_rookie_ppr",
                "impact_0_100",
                "n_pairs",
                "direction",
            ]
        )

    out = feature_corr.copy()
    out["signed_corr_rookie_ppr"] = out["pearson_r"]
    out = out.rename(columns={"feature": "factor"})
    out = out.sort_values("impact_0_100", ascending=False, na_position="last").reset_index(drop=True)
    out.insert(0, "rank", range(1, len(out) + 1))
    return out[
        [
            "rank",
            "factor",
            "dataset_block",
            "abs_corr_rookie_ppr",
            "signed_corr_rookie_ppr",
            "impact_0_100",
            "n_pairs",
            "direction",
        ]
    ]


def _incremental_r2(df: pd.DataFrame, block_features: list[str], baseline_cols: list[str]) -> float:
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    work = df.copy()
    y = pd.to_numeric(work[TARGET], errors="coerce")
    mask = y.notna()
    work = work.loc[mask].copy()
    y = y.loc[mask]
    if len(work) < 30:
        return pd.NA

    base_cols = list(dict.fromkeys(c for c in baseline_cols if c in work.columns))
    block_cols = list(
        dict.fromkeys(c for c in block_features if c in work.columns and c not in base_cols)
    )
    if not base_cols or not block_cols:
        return pd.NA

    def r2_for(cols: list[str]) -> float:
        use_cols = list(dict.fromkeys(c for c in cols if c in work.columns))
        if not use_cols:
            return pd.NA
        X = pd.DataFrame({c: _column_series(work, c) for c in use_cols}, index=work.index)
        num = []
        cat = []
        for c in use_cols:
            if c in NUMERIC_FEATURES:
                ser = pd.to_numeric(X[c], errors="coerce")
                X[c] = ser
                if ser.notna().sum() >= 5:
                    num.append(c)
            elif pd.api.types.is_numeric_dtype(X[c]):
                num.append(c)
            else:
                cat.append(c)
        transformers = []
        if num:
            transformers.append(("num", SimpleImputer(strategy="median"), num))
        if cat:
            transformers.append(
                (
                    "cat",
                    Pipeline(
                        [
                            ("imp", SimpleImputer(strategy="most_frequent")),
                            ("oh", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    cat,
                )
            )
        if not transformers:
            return pd.NA
        pipe = Pipeline([("prep", ColumnTransformer(transformers)), ("model", LinearRegression())])
        pipe.fit(X, y)
        return float(r2_score(y, pipe.predict(X)))

    r2_base = r2_for(base_cols)
    r2_full = r2_for(base_cols + block_cols)
    if pd.isna(r2_base) or pd.isna(r2_full):
        return pd.NA
    return max(float(r2_full - r2_base), 0.0)


def build_dataset_correlation(master: pd.DataFrame, feature_corr: pd.DataFrame) -> pd.DataFrame:
    df = _add_derived_columns(master)
    rows = []
    for block, features in FEATURE_BLOCKS.items():
        sub = feature_corr[feature_corr["dataset_block"] == block] if not feature_corr.empty else pd.DataFrame()
        abs_vals = sub["abs_corr_rookie_ppr"].dropna() if not sub.empty else pd.Series(dtype=float)
        mean_abs = float(abs_vals.mean()) if not abs_vals.empty else pd.NA
        max_abs = float(abs_vals.max()) if not abs_vals.empty else pd.NA
        incr_r2 = _incremental_r2(df, features, baseline_cols=["position", "draft_overall"])

        r2_scaled = float(incr_r2 * 100) if pd.notna(incr_r2) else 0.0
        mean_scaled = float(mean_abs * 100) if pd.notna(mean_abs) else 0.0
        dataset_score = round(0.7 * mean_scaled + 0.3 * min(r2_scaled, 100), 1) if pd.notna(mean_abs) else pd.NA

        rows.append(
            {
                "dataset_block": block,
                "feature_count": len(features),
                "mean_abs_corr": round(mean_abs, 4) if pd.notna(mean_abs) else pd.NA,
                "max_abs_corr": round(max_abs, 4) if pd.notna(max_abs) else pd.NA,
                "incremental_r2": round(float(incr_r2), 4) if pd.notna(incr_r2) else pd.NA,
                "dataset_correlation_score_0_100": dataset_score,
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("dataset_correlation_score_0_100", ascending=False, na_position="last")
        out.insert(0, "rank", range(1, len(out) + 1))
    return out.reset_index(drop=True)


def run_correlation_analysis(master: pd.DataFrame) -> dict[str, pd.DataFrame]:
    feature_corr = build_feature_correlation(master)
    strongest = build_strongest_factors(feature_corr)
    dataset_corr = build_dataset_correlation(master, feature_corr)
    return {
        "feature_correlation": feature_corr,
        "strongest_factors": strongest,
        "dataset_correlation": dataset_corr,
    }
