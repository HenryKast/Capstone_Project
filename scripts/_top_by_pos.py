import pandas as pd
from rookie_ppr.config import OUTPUT_DIR, INCOMING_DRAFT_YEAR

ml = pd.read_csv(OUTPUT_DIR / "csv" / "ml_features.csv")
master = pd.read_csv(OUTPUT_DIR / "players_master.csv")
if "player_name" not in ml.columns:
    ml = ml.merge(
        master[["gsis_id", "player_name", "draft_year"]].drop_duplicates(),
        on=["gsis_id", "draft_year"] if "draft_year" in ml.columns else ["gsis_id"],
        how="left",
    )

inc = ml[ml["draft_year"] == INCOMING_DRAFT_YEAR].copy()
print(f"=== Incoming {INCOMING_DRAFT_YEAR} (by predicted PPR) ===")
for pos in ["QB", "RB", "WR", "TE"]:
    sub = (
        inc[inc["position"] == pos]
        .dropna(subset=["predicted_rookie_ppr"])
        .sort_values("predicted_rookie_ppr", ascending=False)
    )
    if sub.empty:
        print(pos, "none")
        continue
    top = sub.iloc[0]
    print(
        f"{pos}: {top['player_name']}  "
        f"pred={top['predicted_rookie_ppr']:.1f}  "
        f"success={top['success_score_0_100']:.1f}  "
        f"count={len(sub)}"
    )
    for r in sub.iloc[1:3].itertuples():
        print(f"   - {r.player_name} ({r.predicted_rookie_ppr:.0f})")

print()
print("=== All years in model (by predicted PPR) ===")
for pos in ["QB", "RB", "WR", "TE"]:
    sub = (
        ml[ml["position"] == pos]
        .dropna(subset=["predicted_rookie_ppr"])
        .sort_values("predicted_rookie_ppr", ascending=False)
    )
    top = sub.iloc[0]
    print(
        f"{pos}: {top['player_name']} ({int(top['draft_year'])})  "
        f"pred={top['predicted_rookie_ppr']:.1f}  "
        f"success={top['success_score_0_100']:.1f}"
    )
