"""Inspect RB HGB tree thresholds around Love's CFB rushing score."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "data/output/models/hgb_rookie_ppr.joblib"
ML = ROOT / "data/output/csv/ml_features.csv"
MASTER = ROOT / "data/output/csv/players_master.csv"
WEIGHTS = ROOT / "data/output/models/composite_weights.json"

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
    import json

    bundle = joblib.load(MODEL)
    model = bundle["models_by_position"]["RB"]
    feats = bundle.get("feature_cols_by_position", {}).get("RB", RB_FEATS)

    # Confirm composite direction via norms
    arts = json.loads(WEIGHTS.read_text())
    rb_norms = arts["position_norms"]["RB"]
    print("=== RB CFB norms (after invert, so mean of signed raw) ===")
    for f in ["cfb_rush_yards", "cfb_rush_td", "cfb_rec", "cfb_rec_yards", "cfb_rec_td"]:
        print(f"  {f}: {rb_norms.get(f)}")

    # Love raw -> expected z
    love_raw = {"cfb_rush_yards": 1372.0, "cfb_rush_td": 18.0, "cfb_rec": 27.0, "cfb_rec_yards": 280.0, "cfb_rec_td": 3.0}
    print("\n=== Love member z-scores (higher = better production) ===")
    for f, v in love_raw.items():
        m = rb_norms[f]["mean"]
        s = rb_norms[f]["std"]
        # invert False for CFB -> z = (raw - mean)/std; note norms stored on possibly inverted series
        # For non-inverted features, mean is of raw values
        z = (v - m) / s
        print(f"  {f}: raw={v} mean={m:.1f} std={s:.1f} z={z:+.3f}")

    # Walk trees for splits on cfb_rushing
    rush_idx = feats.index("score_cfb_rushing")
    draft_idx = feats.index("score_draft_capital")
    adp_idx = feats.index("score_pre_draft_fantasy")
    rec_idx = feats.index("score_cfb_receiving")

    print(f"\n=== Feature indices: rush={rush_idx} rec={rec_idx} draft={draft_idx} adp={adp_idx} ===")

    # sklearn HistGradientBoosting: model._predictors is list of lists of TreePredictor
    n_splits_rush = 0
    thresholds = []
    for i, preds in enumerate(model._predictors):
        for tree in preds:
            nodes = tree.nodes
            for node in nodes:
                # node is numpy void with fields
                feat = int(node["feature_idx"])
                if feat == rush_idx:
                    thr = float(node["num_threshold"])
                    thresholds.append((i, thr, int(node["count"])))
                    n_splits_rush += 1

    print(f"\nSplits on score_cfb_rushing: {n_splits_rush}")
    thr_arr = np.array([t[1] for t in thresholds])
    print(f"  threshold min/median/max: {thr_arr.min():.3f} / {np.median(thr_arr):.3f} / {thr_arr.max():.3f}")
    print("  thresholds near Love cliff (0.4 to 1.2):")
    near = sorted([t for t in thresholds if 0.4 <= t[1] <= 1.2], key=lambda x: x[1])
    for it, thr, cnt in near[:30]:
        print(f"    iter={it} thr={thr:.4f} n={cnt}")

    # For Love's feature vector, compare leaf values when rush is 0.5 vs 0.85
    # Use decision_path if available — HGB doesn't expose easily; use staged predictions
    ml = pd.read_csv(ML)
    love = ml[ml["player_name"] == "Jeremiyah Love"].iloc[0]
    X = pd.DataFrame([{c: love[c] for c in feats}])

    print("\n=== Staged prediction: cumulative contribution by boosting iteration ===")
    # Compare two variants
    variants = {
        "Love": X.copy(),
        "rush=0.5": X.assign(**{"score_cfb_rushing": 0.5}),
        "rush=0.0": X.assign(**{"score_cfb_rushing": 0.0}),
        "Price CFB": X.assign(**{"score_cfb_rushing": -0.275145, "score_cfb_receiving": -0.587867}),
    }
    for name, Xv in variants.items():
        # raw predict
        print(f"  {name}: pred={model.predict(Xv)[0]:.2f}")

    # Interaction: effect of high CFB rush at different draft levels
    print("\n=== Interaction grid: delta(pred at rush=0.85 vs rush=0) by draft/ADP ===")
    drafts = [0.0, 0.5, 1.0, 1.5, 1.9]
    adps = [0.0, 0.5, 1.0, 1.2]
    base_row = {c: float(love[c]) if pd.notna(love[c]) else np.nan for c in feats}
    # hold other Love features; vary draft and adp
    print("draft\\adp", " ".join(f"{a:>8.1f}" for a in adps))
    for d in drafts:
        row_parts = []
        for a in adps:
            Xlo = pd.DataFrame([{**base_row, "score_draft_capital": d, "score_pre_draft_fantasy": a, "score_cfb_rushing": 0.0}])
            Xhi = pd.DataFrame([{**base_row, "score_draft_capital": d, "score_pre_draft_fantasy": a, "score_cfb_rushing": 0.85}])
            delta = float(model.predict(Xhi)[0] - model.predict(Xlo)[0])
            row_parts.append(f"{delta:+8.1f}")
        print(f"{d:>5.1f}   ", " ".join(row_parts))

    # Same with receiving
    print("\n=== Interaction: delta(rec=0.66 vs rec=0) by draft/ADP (rush fixed Love) ===")
    print("draft\\adp", " ".join(f"{a:>8.1f}" for a in adps))
    for d in drafts:
        row_parts = []
        for a in adps:
            Xlo = pd.DataFrame([{**base_row, "score_draft_capital": d, "score_pre_draft_fantasy": a, "score_cfb_receiving": 0.0}])
            Xhi = pd.DataFrame([{**base_row, "score_draft_capital": d, "score_pre_draft_fantasy": a, "score_cfb_receiving": 0.66}])
            delta = float(model.predict(Xhi)[0] - model.predict(Xlo)[0])
            row_parts.append(f"{delta:+8.1f}")
        print(f"{d:>5.1f}   ", " ".join(row_parts))

    # Bust examples with high CFB among early picks that train the leaf
    master = pd.read_csv(MASTER)
    rb = ml[ml["position"] == "RB"].merge(
        master[["gsis_id", "player_name", "position", "draft_year", "draft_overall", "rookie_ppr"]].drop_duplicates(),
        on=["gsis_id", "player_name", "position", "draft_year"],
        how="left",
        suffixes=("", "_y"),
    )
    train = rb[(rb["draft_year"] < 2023) & rb["rookie_ppr"].notna()].copy()
    early_hi = train[
        (pd.to_numeric(train["draft_overall"], errors="coerce") <= 50)
        & (train["score_cfb_rushing"] >= 0.8)
    ].sort_values("rookie_ppr")
    print("\n=== Early-draft train RBs with CFB rush >= 0.8 (sorted by rookie_ppr) ===")
    cols = ["player_name", "draft_year", "draft_overall", "rookie_ppr", "score_cfb_rushing", "score_cfb_receiving",
            "score_draft_capital", "score_pre_draft_fantasy"]
    print(early_hi[cols].to_string(index=False))
    print(f"\nn={len(early_hi)} mean_ppr={early_hi['rookie_ppr'].mean():.1f} median={early_hi['rookie_ppr'].median():.1f}")

    early_lo = train[
        (pd.to_numeric(train["draft_overall"], errors="coerce") <= 50)
        & (train["score_cfb_rushing"] < 0.5)
    ]
    print(f"Early-draft CFB rush < 0.5: n={len(early_lo)} mean_ppr={early_lo['rookie_ppr'].mean():.1f} median={early_lo['rookie_ppr'].median():.1f}")

    # Composite correlation table if exists
    comp = ROOT / "data/output/csv/composite_correlation.csv"
    if comp.exists():
        print("\n=== composite_correlation.csv (CFB rows) ===")
        cdf = pd.read_csv(comp)
        print(cdf[cdf["composite"].str.contains("cfb", case=False)].to_string(index=False))


if __name__ == "__main__":
    main()
