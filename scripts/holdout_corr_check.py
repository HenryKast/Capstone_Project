import pandas as pd
from scipy import stats

master = pd.read_csv("data/output/players_master.csv")
ml = pd.read_csv("data/output/csv/ml_features.csv")

holdout_years = [2023, 2024]
h = master[master["draft_year"].isin(holdout_years)].copy()
ml_h = ml[ml["draft_year"].isin(holdout_years)].copy()

id_cols = ["gsis_id", "player_name", "position", "draft_year"]
score_cols = [c for c in ml_h.columns if c.startswith("score_") and not c.endswith("_coverage")]
h = h.merge(
    ml_h[id_cols + ["predicted_rookie_ppr", "success_score_0_100"] + score_cols],
    on=id_cols,
    how="left",
)

target = pd.to_numeric(h["rookie_ppr"], errors="coerce")
h = h[target.notna()].copy()
target = target[target.notna()]
print(f"Holdout labeled rookies: n={len(h)} (2023-2024)\n")


def corr_row(name, series, block=""):
    x = pd.to_numeric(series, errors="coerce")
    pair = pd.DataFrame({"x": x, "y": target}).dropna()
    if len(pair) < 8:
        return None
    r, p = stats.pearsonr(pair["x"], pair["y"])
    rs, _ = stats.spearmanr(pair["x"], pair["y"])
    return {
        "name": name,
        "block": block,
        "n": len(pair),
        "pearson_r": r,
        "abs_r": abs(r),
        "spearman_r": rs,
        "p": p,
    }


rows = []
for col, block in [("predicted_rookie_ppr", "ml_model"), ("success_score_0_100", "ml_model")]:
    row = corr_row(col, h[col], block)
    if row:
        rows.append(row)

if "birth_date" in h.columns:
    dob = pd.to_datetime(h["birth_date"], errors="coerce")
    h["age_at_draft"] = h["draft_year"] - dob.dt.year

raw_features = [
    "draft_overall",
    "draft_round",
    "ff_adp_rank",
    "ff_adp",
    "age_at_draft",
    "cfb_pass_td",
    "cfb_pass_yards",
    "recruiting_stars",
    "recruiting_rank",
    "cone",
    "forty",
    "shuttle",
    "cfb_rec_yards",
    "cfb_rec",
    "cfb_rush_td",
    "cfb_rush_yards",
    "broad_jump",
    "wt",
    "hs_class",
    "sos_opp_win_pct",
    "team_opportunity_ppr",
    "off_pass_rate_proxy",
    "off_pass_yards",
    "off_rush_yards",
]
for feat in raw_features:
    if feat in h.columns:
        row = corr_row(feat, h[feat], "raw")
        if row:
            rows.append(row)

for col in score_cols:
    row = corr_row(col, h[col], "composite")
    if row:
        rows.append(row)

df = pd.DataFrame(rows).sort_values("abs_r", ascending=False)
model_abs = df[df["block"] == "ml_model"]["abs_r"].max()
print("=== Top 20 vs actual rookie_ppr (holdout only) ===")
print(df.head(20).to_string(index=False))
print()
print(f"Model best |r| = {model_abs:.3f}")
beats = df[(df["abs_r"] > model_abs) & (df["block"] != "ml_model")]
print(f"Beating model on holdout: {len(beats)}")
if len(beats):
    print(beats[["name", "block", "n", "pearson_r", "abs_r"]].to_string(index=False))
else:
    print("None — ML model still strongest overall on holdout.")

block_map = {
    "draft_overall": "draft_capital",
    "draft_round": "draft_capital",
    "ff_adp": "pre_draft_fantasy",
    "ff_adp_rank": "pre_draft_fantasy",
    "recruiting_rank": "recruiting",
    "recruiting_stars": "recruiting",
    "score_draft_capital": "draft_capital",
    "score_pre_draft_fantasy": "pre_draft_fantasy",
    "score_recruiting": "recruiting",
    "score_timing": "timing",
    "score_cfb_receiving": "college",
    "score_cfb_rushing": "college",
    "score_cfb_passing": "college",
    "score_combine_speed": "combine",
    "score_combine_explosion": "combine",
    "score_combine_size": "combine",
    "score_team_context": "team_context",
    "predicted_rookie_ppr": "ml_model",
    "success_score_0_100": "ml_model",
}
df["dataset_block"] = df["name"].map(block_map).fillna("other")
print("\n=== Best per dataset block (holdout) ===")
for block, g in df.groupby("dataset_block"):
    best = g.loc[g["abs_r"].idxmax()]
    print(f"{block:22s} {best['name']:28s} r={best['pearson_r']:+.3f}  n={int(best['n'])}")
