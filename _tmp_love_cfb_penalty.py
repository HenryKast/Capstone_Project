"""Investigate why Love's high CFB composites lower RB predicted PPR."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import partial_dependence

ROOT = Path(__file__).resolve().parent
ML = ROOT / "data/output/csv/ml_features.csv"
MASTER = ROOT / "data/output/csv/players_master.csv"
MODEL = ROOT / "data/output/models/hgb_rookie_ppr.joblib"
METRICS = ROOT / "data/output/models/model_metrics.json"
HOLDOUT = {2023, 2024, 2025, 2026}

RB_FEATS = [
    "score_draft_capital",
    "score_pre_draft_fantasy",
    "score_recruiting",
    "score_timing",
    "score_cfb_rushing",
    "score_cfb_receiving",
    "score_combine_speed",
    "score_team_context",
]


def main() -> None:
    ml = pd.read_csv(ML)
    master = pd.read_csv(MASTER)
    bundle = joblib.load(MODEL)
    model = bundle["models_by_position"]["RB"]
    feature_cols = bundle.get("feature_cols_by_position", {}).get("RB", RB_FEATS)

    love = ml[ml["player_name"] == "Jeremiyah Love"].iloc[0]
    print("=== Love composite vector (RB model features) ===")
    for c in feature_cols:
        print(f"  {c}: {love.get(c)}")
    print(f"  predicted_ppr: {love.get('predicted_ppr')}")
    print(f"  ff_adp: {love.get('ff_adp')}  ff_adp_rank: {love.get('ff_adp_rank')}")

    # Raw CFB from master
    mrow = master[master["player_name"] == "Jeremiyah Love"]
    if not mrow.empty:
        cols = [
            "player_name",
            "position",
            "draft_year",
            "draft_overall",
            "ff_adp",
            "ff_adp_rank",
            "cfb_rush_yards",
            "cfb_rush_td",
            "cfb_rec",
            "cfb_rec_yards",
            "cfb_rec_td",
            "rookie_ppr",
            "predicted_ppr",
        ]
        cols = [c for c in cols if c in mrow.columns]
        print("\n=== Love raw master fields ===")
        print(mrow[cols].T.to_string())

    # Price comparison if present
    price = ml[ml["player_name"].str.contains("Price", case=False, na=False) & (ml["position"] == "RB")]
    print("\n=== RB players matching Price ===")
    print(price[["player_name", "draft_year"] + [c for c in feature_cols if c in price.columns] +
          ([c for c in ["predicted_ppr"] if c in price.columns])].to_string(index=False))

    X_love = pd.DataFrame([{c: love.get(c) for c in feature_cols}])
    base = float(model.predict(X_love)[0])
    print(f"\n=== Predictions for Love (base={base:.2f}) ===")

    scenarios = {
        "cfb_rush=0, cfb_rec=0": {"score_cfb_rushing": 0.0, "score_cfb_receiving": 0.0},
        "cfb_rush=median, cfb_rec=median": None,  # filled below
        "cfb_only_rush_zero": {"score_cfb_rushing": 0.0},
        "cfb_only_rec_zero": {"score_cfb_receiving": 0.0},
        "cfb_rush=-1": {"score_cfb_rushing": -1.0},
        "cfb_rush=+1": {"score_cfb_rushing": 1.0},
        "cfb_rec=-1": {"score_cfb_receiving": -1.0},
        "cfb_rec=+1": {"score_cfb_receiving": 1.0},
        "drop_rush_to_nan": {"score_cfb_rushing": np.nan},
        "drop_rec_to_nan": {"score_cfb_receiving": np.nan},
        "both_cfb_nan": {"score_cfb_rushing": np.nan, "score_cfb_receiving": np.nan},
    }

    # Train RBs for medians / correlations
    rb = ml[ml["position"] == "RB"].copy()
    # Prefer master merge for draft_overall / target
    keys = [c for c in ["gsis_id", "player_name", "position", "draft_year"] if c in rb.columns and c in master.columns]
    rb = rb.merge(
        master[keys + [c for c in ["draft_overall", "rookie_ppr", "first_stat_season", "ff_adp"] if c in master.columns]],
        on=keys,
        how="left",
        suffixes=("", "_m"),
    )
    if "draft_overall" not in rb.columns and "draft_overall_m" in rb.columns:
        rb["draft_overall"] = rb["draft_overall_m"]
    if "rookie_ppr" not in rb.columns and "rookie_ppr_m" in rb.columns:
        rb["rookie_ppr"] = rb["rookie_ppr_m"]

    # holdout vs train by draft_year / first_stat_season
    season_col = "first_stat_season" if "first_stat_season" in rb.columns else "draft_year"
    rb["is_holdout"] = rb[season_col].isin(HOLDOUT) if season_col in rb.columns else False
    # Also treat 2026 prospects (no target) separately
    train_rb = rb[(~rb["is_holdout"]) & rb["rookie_ppr"].notna()].copy()
    hold_rb = rb[rb["is_holdout"] & rb["rookie_ppr"].notna()].copy()

    med_rush = float(train_rb["score_cfb_rushing"].median())
    med_rec = float(train_rb["score_cfb_receiving"].median())
    scenarios["cfb_rush=median, cfb_rec=median"] = {
        "score_cfb_rushing": med_rush,
        "score_cfb_receiving": med_rec,
    }

    # Price CFB if available
    if not price.empty:
        prow = price.iloc[0]
        scenarios["Price CFB scores"] = {
            "score_cfb_rushing": prow.get("score_cfb_rushing"),
            "score_cfb_receiving": prow.get("score_cfb_receiving"),
        }

    for name, overrides in scenarios.items():
        X = X_love.copy()
        for k, v in overrides.items():
            X[k] = v
        pred = float(model.predict(X)[0])
        print(f"  {name}: {pred:.2f}  (delta {pred - base:+.2f})")

    # Sweep CFB rushing while holding rest fixed
    print("\n=== Sweep score_cfb_rushing (rec fixed at Love) ===")
    for v in [-1.5, -1.0, -0.5, 0.0, 0.5, 0.85, 1.0, 1.5, 2.0]:
        X = X_love.copy()
        X["score_cfb_rushing"] = v
        pred = float(model.predict(X)[0])
        print(f"  rush={v:+.2f}: {pred:.2f} (delta {pred - base:+.2f})")

    print("\n=== Sweep score_cfb_receiving (rush fixed at Love) ===")
    for v in [-1.5, -1.0, -0.5, 0.0, 0.5, 0.66, 1.0, 1.5, 2.0]:
        X = X_love.copy()
        X["score_cfb_receiving"] = v
        pred = float(model.predict(X)[0])
        print(f"  rec={v:+.2f}: {pred:.2f} (delta {pred - base:+.2f})")

    # Correlations within draft tiers
    def tier_corr(df: pd.DataFrame, label: str) -> None:
        print(f"\n=== Corr CFB vs rookie_ppr within draft tiers ({label}, n={len(df)}) ===")
        df = df.copy()
        df["draft_overall"] = pd.to_numeric(df["draft_overall"], errors="coerce")
        tiers = [
            ("top50", df["draft_overall"] <= 50),
            ("51-100", (df["draft_overall"] > 50) & (df["draft_overall"] <= 100)),
            ("late100+", df["draft_overall"] > 100),
            ("all", df["draft_overall"].notna()),
        ]
        for tname, mask in tiers:
            sub = df.loc[mask]
            for feat in ["score_cfb_rushing", "score_cfb_receiving", "score_draft_capital", "score_pre_draft_fantasy"]:
                pair = sub[[feat, "rookie_ppr"]].dropna()
                if len(pair) < 8:
                    print(f"  {tname} {feat}: n={len(pair)} (skip)")
                    continue
                r = pair[feat].corr(pair["rookie_ppr"])
                print(f"  {tname} {feat}: r={r:+.3f} n={len(pair)}")

    tier_corr(train_rb, "train")
    tier_corr(hold_rb, "holdout")

    # Among early-draft RBs: high vs low CFB outcomes
    print("\n=== Early-draft RBs (draft_overall<=50) by CFB rushing tercile (train) ===")
    early = train_rb[pd.to_numeric(train_rb["draft_overall"], errors="coerce") <= 50].copy()
    early = early.dropna(subset=["score_cfb_rushing", "rookie_ppr"])
    if len(early) >= 9:
        early["cfb_tercile"] = pd.qcut(early["score_cfb_rushing"], 3, labels=["low", "mid", "high"], duplicates="drop")
        g = early.groupby("cfb_tercile", observed=True).agg(
            n=("rookie_ppr", "size"),
            mean_ppr=("rookie_ppr", "mean"),
            median_ppr=("rookie_ppr", "median"),
            mean_cfb_rush=("score_cfb_rushing", "mean"),
            mean_adp=("score_pre_draft_fantasy", "mean"),
            mean_draft=("score_draft_capital", "mean"),
        )
        print(g.to_string())
        print("\nPlayers in high CFB tercile (early draft):")
        hi = early[early["cfb_tercile"] == "high"].sort_values("rookie_ppr", ascending=False)
        show = [c for c in ["player_name", "draft_year", "draft_overall", "rookie_ppr", "score_cfb_rushing",
                            "score_cfb_receiving", "score_draft_capital", "score_pre_draft_fantasy"] if c in hi.columns]
        print(hi[show].to_string(index=False))
        print("\nPlayers in low CFB tercile (early draft):")
        lo = early[early["cfb_tercile"] == "low"].sort_values("rookie_ppr", ascending=False)
        print(lo[show].to_string(index=False))

    # Partial residual: CFB corr after controlling for draft+ADP
    print("\n=== Partial correlations (train RB): CFB residual vs PPR residual after draft+ADP ===")
    from sklearn.linear_model import LinearRegression

    def partial_r(df: pd.DataFrame, feat: str, controls: list[str]) -> None:
        cols = [feat, "rookie_ppr"] + controls
        sub = df[cols].dropna()
        if len(sub) < 20:
            print(f"  {feat}: n={len(sub)} skip")
            return
        Xc = sub[controls]
        y = sub["rookie_ppr"]
        xf = sub[[feat]]
        ry = y - LinearRegression().fit(Xc, y).predict(Xc)
        rf = xf[feat] - LinearRegression().fit(Xc, xf[feat]).predict(Xc)
        r = pd.Series(rf).corr(pd.Series(ry))
        # also among top50
        print(f"  {feat} | controls={controls}: partial_r={r:+.3f} n={len(sub)}")

    partial_r(train_rb, "score_cfb_rushing", ["score_draft_capital", "score_pre_draft_fantasy"])
    partial_r(train_rb, "score_cfb_receiving", ["score_draft_capital", "score_pre_draft_fantasy"])
    early_train = train_rb[pd.to_numeric(train_rb["draft_overall"], errors="coerce") <= 50]
    print("  (early top50 only)")
    partial_r(early_train, "score_cfb_rushing", ["score_draft_capital", "score_pre_draft_fantasy"])
    partial_r(early_train, "score_cfb_receiving", ["score_draft_capital", "score_pre_draft_fantasy"])

    # Feature importances
    print("\n=== HGB feature importances (RB) ===")
    if hasattr(model, "feature_importances_"):
        for name, imp in sorted(zip(feature_cols, model.feature_importances_), key=lambda x: -x[1]):
            print(f"  {name}: {imp:.4f}")
    else:
        # sklearn HGB may use permutation or no importances — try permutation on train
        from sklearn.inspection import permutation_importance

        Xtr = train_rb[feature_cols]
        ytr = train_rb["rookie_ppr"]
        mask = Xtr.notna().all(axis=1) & ytr.notna()
        # HGB handles NaN; use all with y
        mask = ytr.notna()
        Xtr2 = Xtr.loc[mask]
        ytr2 = ytr.loc[mask]
        r = permutation_importance(model, Xtr2, ytr2, n_repeats=20, random_state=0, scoring="r2")
        for name, imp, std in sorted(zip(feature_cols, r.importances_mean, r.importances_std), key=lambda x: -x[1]):
            print(f"  {name}: {imp:.4f} +/- {std:.4f}")

    # Compare Love-like peers: high draft + high ADP
    print("\n=== Train RBs similar to Love (draft_overall<=40 OR score_draft>=1.2) ===")
    sim = train_rb[
        (pd.to_numeric(train_rb["draft_overall"], errors="coerce") <= 40)
        | (train_rb["score_draft_capital"] >= 1.2)
    ].copy()
    show = [c for c in ["player_name", "draft_year", "draft_overall", "rookie_ppr",
                        "score_cfb_rushing", "score_cfb_receiving", "score_draft_capital",
                        "score_pre_draft_fantasy", "predicted_ppr"] if c in sim.columns]
    # predict for each
    if len(sim):
        preds = model.predict(sim[feature_cols])
        sim = sim.copy()
        sim["model_pred"] = preds
        # split high/low CFB
        med = sim["score_cfb_rushing"].median()
        print(f"median cfb_rush among peers: {med:.3f}")
        print("HIGH CFB rushing peers:")
        print(sim[sim["score_cfb_rushing"] >= med][show + ["model_pred"]].sort_values("rookie_ppr", ascending=False).to_string(index=False))
        print("LOW CFB rushing peers:")
        print(sim[sim["score_cfb_rushing"] < med][show + ["model_pred"]].sort_values("rookie_ppr", ascending=False).to_string(index=False))
        print("\nMean outcomes:")
        print(sim.assign(hi=sim["score_cfb_rushing"] >= med).groupby("hi").agg(
            n=("rookie_ppr", "size"),
            mean_ppr=("rookie_ppr", "mean"),
            mean_pred=("model_pred", "mean"),
            mean_cfb=("score_cfb_rushing", "mean"),
            mean_adp=("score_pre_draft_fantasy", "mean"),
        ).to_string())

    # Correlation of CFB with draft/ADP among RBs
    print("\n=== Collinearity: CFB vs draft/ADP (train RB) ===")
    for a in ["score_cfb_rushing", "score_cfb_receiving"]:
        for b in ["score_draft_capital", "score_pre_draft_fantasy"]:
            pair = train_rb[[a, b]].dropna()
            print(f"  corr({a}, {b}) = {pair[a].corr(pair[b]):+.3f} n={len(pair)}")

    # Missingness check for Love combine_speed
    print("\n=== Missingness on Love row ===")
    for c in feature_cols:
        v = love.get(c)
        print(f"  {c}: {v}  nan={pd.isna(v)}")

    # PDP for CFB features on train
    print("\n=== Partial dependence (CFB rushing / receiving) on train RBs ===")
    X_pdp = train_rb[feature_cols]
    for feat in ["score_cfb_rushing", "score_cfb_receiving"]:
        try:
            pd_res = partial_dependence(model, X_pdp, [feat], kind="average", grid_resolution=12)
            grid = pd_res["grid_values"][0]
            avg = pd_res["average"][0]
            print(f"  {feat}:")
            for g, a in zip(grid, avg):
                print(f"    x={g:+.3f}  avg_pred={a:.2f}")
        except Exception as e:
            print(f"  {feat} PDP failed: {e}")

    # Conditional PDP-like: only among high draft capital
    print("\n=== Mean predicted PPR when swapping CFB on early-draft train RBs ===")
    early2 = train_rb[pd.to_numeric(train_rb["draft_overall"], errors="coerce") <= 50].copy()
    if len(early2) >= 5:
        base_preds = model.predict(early2[feature_cols])
        X0 = early2[feature_cols].copy()
        X0["score_cfb_rushing"] = 0.0
        X0["score_cfb_receiving"] = 0.0
        zero_preds = model.predict(X0)
        Xhi = early2[feature_cols].copy()
        Xhi["score_cfb_rushing"] = float(love["score_cfb_rushing"])
        Xhi["score_cfb_receiving"] = float(love["score_cfb_receiving"])
        hi_preds = model.predict(Xhi)
        print(f"  n={len(early2)}")
        print(f"  mean pred actual CFB: {base_preds.mean():.2f}")
        print(f"  mean pred CFB=0: {zero_preds.mean():.2f} (delta {zero_preds.mean()-base_preds.mean():+.2f})")
        print(f"  mean pred Love CFB: {hi_preds.mean():.2f} (delta {hi_preds.mean()-base_preds.mean():+.2f})")
        print(f"  mean Love-vs-0 effect: {hi_preds.mean()-zero_preds.mean():+.2f}")


if __name__ == "__main__":
    main()
