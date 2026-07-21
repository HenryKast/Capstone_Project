# Rookie PPR Success Dataset

Compile high-school classes **2010–2023** skill-position players (QB, RB, WR, TE) into one master workbook and matching CSVs for studying **rookie-season PPR** fantasy outcomes and **predicting upcoming draft classes** (including the **2026–27** season).

This repository includes environment setup, the data compile pipeline, correlation analysis, correlation-weighted composite features, position-tuned ML scoring, and a desktop scorer UI.

---

## New machine setup (start here)

### Requirements

- Python **3.11+** (tested through 3.14)
- Network access on first compile (nflverse downloads)
- Optional: free [CollegeFootballData](https://collegefootballdata.com/) API key for CFB production stats



### Windows

```powershell


# 1) Create .venv and install dependencies
powershell -ExecutionPolicy Bypass -File .\env\setup_env.ps1

# 2) Activate (if scripts are blocked, see note below)
.\.venv\Scripts\Activate.ps1

# 3) Optional: college production
copy .env.example .env
# Edit .env and set CFBD_API_KEY=...

# 4) Optional: fetch On3 recruiting (HS 2010–2023)
python -m rookie_ppr.compile --fetch-on3

# Or compile without re-fetching if recruiting CSVs already exist:
python -m rookie_ppr.compile

# 5) Open the scorer UI
python -m rookie_ppr.ui
```

If activation fails with *running scripts is disabled*:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Or skip activation and call the venv Python directly:

```powershell
.\.venv\Scripts\python.exe -m rookie_ppr.compile
.\.venv\Scripts\python.exe -m rookie_ppr.ui
```



### macOS / Linux

```bash
cd /path/to/Capstone\ Project
bash env/setup_env.sh
source .venv/bin/activate
# optional: copy .env.example → .env and set CFBD_API_KEY
python -m rookie_ppr.compile --fetch-on3
python -m rookie_ppr.ui
```

More environment detail: `[env/README.md](env/README.md)`.

### Everyday use (after first setup)

```powershell

.\.venv\Scripts\Activate.ps1
python -m rookie_ppr.ui          # scorer GUI
python -m rookie_ppr.compile     # refresh data / retrain models
python -m rookie_ppr.score --position WR --draft-overall 4 --ff-adp 61.5
```

---



## Analysis process (correlation → ML → refinement)

The project was built in stages. Each stage feeds the next.

```
1. Compile raw sources
        ↓
2. Correlation analysis (what predicts rookie PPR?)
        ↓
3. Composite features (merge redundant correlated inputs)
        ↓
4. ML training (predict rookie PPR → success score 0–100)
        ↓
5. Refinement (position-specific models, holdout by rookie season, tuning)
        ↓
6. Application (CLI + GUI scoring; incoming 2026 rookies sheet)
```



### Stage 1 — Compile

`python -m rookie_ppr.compile` pulls and joins:


| Source                                     | Provides                                            |
| ------------------------------------------ | --------------------------------------------------- |
| nflverse draft / combine / rosters / stats | Draft capital, measurables, birth dates, rookie PPR |
| On3 recruiting CSVs (`hs_2010`…`hs_2023`)  | Rank, stars, HS class                               |
| CFBD (optional API key)                    | Final college season production                     |
| School roster scrape (fallback)            | Missing CFB stats when CFBD has no match            |
| FantasyPros Overall ADP (`data/manual/`)   | Pre-draft fantasy ADP / rank                        |
| NFL.com player stats (2025+ opportunity)   | Prior-season team usage when nflverse lags          |


Outputs: `data/output/rookie_ppr_master.xlsx`, `data/output/csv/*`, `data/output/models/*`.

### Stage 2 — Correlation analysis

Run automatically at the end of compile on labeled rookies (players with `rookie_ppr`).

**Goals**

- Measure which **pre-rookie** inputs associate with rookie-season PPR
- Rank individual features and dataset blocks
- Drive composite weights and mark **(Important)** UI fields

**Why correlation before ML:** Draft round vs overall pick, ADP vs ADP rank, and receiving yards vs receptions are highly redundant. Correlation shows standalone signal, exposes weak blocks (e.g. team context), and prevents over-weighting duplicates in modeling.

**Outputs:** `feature_correlation`, `strongest_factors`, `dataset_correlation`, `group_averages`.

### Stage 3 — Composite features

Highly correlated raw fields are merged into **11** `score_`* **columns** (correlation-weighted averages of position-normalized z-scores). Example: `cfb_rec` + `cfb_rec_yards` + `cfb_rec_td` → `score_cfb_receiving`.

Validated afterward in `composite_correlation`.

### Stage 4 — Initial ML training

A HistGradientBoosting model predicted `rookie_ppr` from composites, then mapped predictions to a **success score 0–100** (same-position historical percentile). Later refinements (Stage 5) added median point estimates, CQR intervals, position-specific tuning grids, opportunity depth, and the ADP baseline.

### Stage 5 — Refinement


| Refinement                                                       | Why                                            | Result                                          |
| ---------------------------------------------------------------- | ---------------------------------------------- | ----------------------------------------------- |
| Drop exact duplicate columns (`ff_ecr`, `draft_pick`, …)         | Remove identical / redundant fields            | Cleaner master + composites                     |
| **One model per position** + position-specific composite subsets | QB/RB/WR/TE use different signals              | Better overall holdout fit                      |
| Hyperparameter tuning on **2022** rookie season                  | Honest validation before holdout               | Per-position `max_depth`, `learning_rate`, etc. |
| Holdout by `first_stat_season` **2023–2026**                     | Match NFL fantasy seasons; include late debuts | Current holdout **r ≈ 0.68**                    |
| Expand HS cohort to **2023** + ADP recovery for 2026             | Include early declarers (e.g. Carnell Tate)    | ~68 incoming 2026 skill rookies in workbook     |
| **Median (q50) point estimate** + predictive IQR clamp          | Robust central forecast; keep median inside error Q1–Q3 | `predicted_rookie_ppr` = median HGB, bounded by `predictive_q25`/`predictive_q75` |
| **CQR 80% intervals** (conformalized quantile regression)        | Calibrated uncertainty on holdout              | `ppr_low` / `ppr_high`; holdout coverage ≈ 0.80 |
| **Predictive Q25/Q75** error quartiles                           | Boom/bust gauge zones in UI                    | ~25% below Q1 / above Q3 by construction        |
| **Position-specific tuning grids** (RB / TE / WR)                | Shallow trees + higher L2 on small-n roles     | RB `max_depth`≤3; TE drops combine from features |
| **Deeper opportunity features** + RB team-context override       | Landing spot matters more for RBs              | `vacated_touches` / `vacated_carries`; NFL.com 2025 fill when nflverse lags |
| **ADP-only baseline** on comparable holdout rows                 | Separate class strength from model lift        | `predicted_adp_baseline`; `model_vs_adp_baseline.csv` |
| **CFB ingest hardening** (aliases + school-site fallback)        | Fill missing 2026 college production           | `cfb_name_aliases.csv`; roster scrape fallback  |




### Stage 6 — Application

- **GUI:** `python -m rookie_ppr.ui` — lookup 2026 rookies by default, searchable dropdowns, compare popups, edit stats and recalculate
- **CLI:** `python -m rookie_ppr.score --position …`
- **Workbook tab:** `incoming_rookies_2026` with predictions (rookie PPR blank until the season completes)

---



## What you get


| Output                               | Description                                             |
| ------------------------------------ | ------------------------------------------------------- |
| `data/output/rookie_ppr_master.xlsx` | Master workbook                                         |
| `data/output/csv/*.csv`              | Same tables as CSVs                                     |
| `data/output/players_master.csv`     | Wide joined modeling table                              |
| `data/output/models/`                | Per-position models, composite weights, holdout metrics |




### Workbook sheets

- `players` — cohort identity
- `recruiting` — Industry Composite rank (Rivals fallback)
- `draft` — college, pick, team
- `combine` — NFL combine measurables
- `college_production` — CFB stats (needs `CFBD_API_KEY`)
- `team_context` — SOS, offense, landing-spot opportunity, returning incumbent competition (2025+ seasons use NFL.com player stats when nflverse lags)
- `pre_draft_fantasy` — FantasyPros overall ADP
- `fantasy_rookie` — rookie-season PPR
- `players_master` — full join
- `incoming_rookies_2026` — **2026 draft class** inputs + ML predictions
- `group_averages` — mean rookie PPR by position, round, recruiting band, etc.
- `strongest_factors` / `feature_correlation` / `dataset_correlation` / `composite_correlation`
- `ml_features` — composites, predicted points, ADP baseline, CQR band, predictive Q1/Q3, success score 0–100
- `data_dictionary` — column definitions

---



## Correlation analysis (detail)

Labeled cohort: players in `players_master` with rookie fantasy data. Target = `rookie_ppr` (total PPR in first NFL season).

### Success definitions


| Target                    | Definition                                                   | Used for           |
| ------------------------- | ------------------------------------------------------------ | ------------------ |
| `rookie_ppr`              | Total PPR in first NFL season                                | Pearson / Spearman |
| `rookie_success_top_half` | 1 if ≥ median `rookie_ppr` within same draft year + position | Point-biserial     |




### Per-feature (`feature_correlation`, `strongest_factors`)

For each numeric feature with enough data:

1. Pearson r and Spearman ρ vs `rookie_ppr`
2. Point-biserial vs top-half success
3. **Impact 0–100** = Pearson r scaled to the strongest feature
4. Direction from the sign of Pearson r

Derived helpers: `ht` → `ht_inches`; `age_at_draft` = draft year − birth year.

### Per-dataset-block (`dataset_correlation`)

Blocks: recruiting, draft capital, team context, combine, college production, pre-draft fantasy, timing.


| Metric                            | Meaning                                                        |
| --------------------------------- | -------------------------------------------------------------- |
| `mean_abs_corr`                   | Typical strength of the block                                  |
| `max_abs_corr`                    | Best single feature in the block                               |
| `incremental_r2`                  | Lift beyond baseline `position` + `draft_overall`              |
| `dataset_correlation_score_0_100` | `0.7 × mean_abs_corr×100 + 0.3 × min(incremental_r2×100, 100)` |


**Typical block ranking:** draft capital → pre-draft fantasy → timing → college production → recruiting → combine → team context.

### Strongest raw factors (illustrative)


| Rank | Factor          | Block             | r (approx.) | Impact |
| ---- | --------------- | ----------------- | ----------- | ------ |
| 1    | `draft_overall` | draft capital     | −0.57       | 100    |
| 2    | `draft_round`   | draft capital     | −0.56       | 98     |
| 3    | `ff_adp_rank`   | pre-draft fantasy | −0.54       | 94     |
| 4    | `ff_adp`        | pre-draft fantasy | −0.51       | 89     |
| 5    | `age_at_draft`  | timing            | −0.26       | 45     |


Negative r means earlier pick / lower ADP associates with higher rookie PPR. Full tables are in the CSV exports.

### Redundant columns removed


| Removed              | Kept                   | Reason                        |
| -------------------- | ---------------------- | ----------------------------- |
| `draft_pick`         | `draft_overall`        | Identical                     |
| `opportunity_proxy`  | `team_opportunity_ppr` | Same values; clearer name     |
| `draft_category`     | `position`             | Identical                     |
| `rookie_season`      | `draft_year`           | Identical in cohort           |
| `hs_class_estimated` | `hs_class`             | Estimate is fallback only     |
| `ff_ecr`             | `ff_adp_rank`          | Duplicate of FantasyPros rank |


---



## Machine learning (detail)



### Pipeline

1. Correlation → composite weights
2. Build 11 `score_*` features
3. Tune and train **one HistGradientBoostingRegressor per position** (position-specific feature subsets + param grids)
4. Fit **median (q50)** point models, **predictive q25/q75** quartile models, and **CQR** lo/hi quantile models with conformal calibration
5. Score all rows → clamp median into predictive IQR; attach `ppr_low` / `ppr_high` (80% CQR band)
6. Train **ADP-only baseline** (`ff_adp`, `ff_adp_rank`) on the same pre-holdout set for evaluation lift
7. Map predicted points → **success score 0–100** (same-position historical percentile)



### Composite construction

```
oriented_value = raw × (-1 if lower-is-better else +1)
z = (oriented_value - position_mean) / position_std
composite = Σ(weight_i × z_i) / Σ(weight_i)   over non-null members
```

Weights = Pearson r vs `rookie_ppr` from correlation. Norms fit on **training rows only**.


| Composite                 | Members                                     |
| ------------------------- | ------------------------------------------- |
| `score_draft_capital`     | `draft_overall`                             |
| `score_pre_draft_fantasy` | `ff_adp`, `ff_adp_rank`                     |
| `score_recruiting`        | `recruiting_rank` (inv), `recruiting_stars` |
| `score_timing`            | `age_at_draft` (inv), `hs_class`            |
| `score_cfb_receiving`     | `cfb_rec`, `cfb_rec_yards`, `cfb_rec_td`    |
| `score_cfb_rushing`       | `cfb_rush_yards`, `cfb_rush_td`             |
| `score_cfb_passing`       | `cfb_pass_yards`, `cfb_pass_td`             |
| `score_combine_speed`     | `forty`, `cone`, `shuttle` (inv)            |
| `score_combine_explosion` | `vertical`, `broad_jump`                    |
| `score_combine_size`      | `ht_inches`, `wt`                           |
| `score_team_context`      | SOS, vacated touches/carries, incumbent competition, offense proxies (RB uses a rushing-focused member subset) |




### Position-specific feature subsets

Models use correlation-weighted composites, not raw columns. Combine blocks are omitted where they added noise on holdout.


| Position | Model composites                                                                 | Dropped from this role's model        |
| -------- | -------------------------------------------------------------------------------- | ------------------------------------- |
| QB       | draft, fantasy, recruiting, timing, passing, size, speed, team                   | receiving, rushing, explosion         |
| RB       | draft, fantasy, recruiting, timing, rushing, receiving, **speed only**, team     | passing, combine explosion/size       |
| WR       | draft, fantasy, recruiting, timing, receiving, speed, explosion, size, team      | passing, rushing                      |
| TE       | draft, fantasy, recruiting, timing, receiving, team                              | passing, rushing, **all combine**   |

RB `score_team_context` is built from vacated volume + incumbent carries/touches + SOS + team rush yards (pass-rate proxies dropped for RB).




### Holdout and tuning

Holdout uses **rookie NFL season** (`first_stat_season`, else `draft_year`):


| Fantasy season | `first_stat_season` |
| -------------- | ------------------- |
| 2023–24        | 2023                |
| 2024–25        | 2024                |
| 2025–26        | 2025                |
| 2026–27        | 2026                |


```
Labeled rookies
├── Train / fit composites: rookie season ≤ 2022
├── Tune hyperparameters: rookie season = 2022 (per position, max Pearson r)
└── Holdout report only: rookie season ∈ {2023, 2024, 2025, 2026}
```

Configured in `[src/rookie_ppr/config.py](src/rookie_ppr/config.py)` as `HOLDOUT_ROOKIE_SEASONS` and `TUNING_ROOKIE_SEASON`.

**Current holdout** (see `data/output/models/model_metrics.json`; re-run compile to refresh):


| Metric            | Value                                             |
| ----------------- | ------------------------------------------------- |
| Point estimate    | **Median (q50)** HGB, clamped to predictive Q1–Q3 |
| Holdout Pearson r | **0.68** (n = 169)                                |
| Holdout MAE       | 46.3 points                                       |
| Holdout R²        | 0.44                                              |
| CQR 80% coverage  | **0.80** on holdout (nominal α = 0.20)            |
| CQR mean width    | 141 points                                        |
| Holdout below low | 6.5% (bust-side misses)                           |
| Holdout above high| 13.6% (boom-side misses)                          |



| Position | n  | Holdout r | MAE   |
| -------- | -- | --------- | ----- |
| QB       | 27 | 0.79      | 49.2  |
| RB       | 43 | 0.62      | 48.7  |
| WR       | 69 | 0.68      | 45.6  |
| TE       | 30 | 0.69      | 41.8  |


#### ADP-only baseline

Draft classes vary in talent; raw PPR is hard to compare across years without a market anchor. An **ADP-only** position-specific model (`ff_adp`, `ff_adp_rank` only; same median HGB / holdout seasons) estimates what FantasyPros ADP alone would predict.

Headline evaluation uses **comparable holdout rows** that have ADP + outcome + full-model prediction (`baselines.adp_only.comparable` in metrics; CSV: `data/output/csv/model_vs_adp_baseline.csv`). ~19 holdout rookies lack ADP and are excluded from the lift table.


| Scope    | n   | Full r | ADP r | lift_r | Full MAE | ADP MAE | lift_mae |
| -------- | --- | ------ | ----- | ------ | -------- | ------- | -------- |
| Overall  | 150 | 0.67   | 0.68  | ~0     | 48.7     | 49.2    | −0.5     |
| QB       | 23  | 0.77   | 0.76  | +0.01  | 54.4     | 56.5    | −2.1     |
| RB       | 42  | 0.61   | 0.64  | −0.03  | 49.3     | 48.8    | +0.4     |
| WR       | 62  | 0.69   | 0.68  | +0.01  | 46.2     | 45.8    | +0.4     |
| TE       | 23  | 0.67   | 0.52  | **+0.15** | 48.6  | 51.6    | −3.1     |


- **lift_r** = full r − ADP r (positive = full model ranks rookies better than ADP)
- **lift_mae** = full MAE − ADP MAE (negative = full model is closer on average)

Overall correlation is roughly ADP-tied; clearest lift is at **TE**. Use this to separate “beats market expectation” from “looks good because the class was strong.”

### Uncertainty & score drivers

- **`predicted_rookie_ppr`** — median (50th percentile) forecast in points (PPR scoring format).
- **`predictive_q25` / `predictive_q75`** — model error quartiles; UI boom/bust zones use these (not historical peer PPR).
- **`ppr_low` / `ppr_high`** — conformalized 80% interval from CQR (wider than the IQR gauge).
- **`predicted_adp_baseline`** — what ADP-only would predict for the same player.
- **Compare score drivers** — leave-one-out: each composite reset to **0** (position-typical) to show marginal lift on the forecast.

**Known behavior:** shallow gradient-boosted trees can learn **negative interactions** between correlated inputs (e.g. elite draft capital + very high college rushing on early RBs). Treat large driver swings on highly drafted players as a model limitation, not a scouting verdict.

**Important:** Predictions backfilled on the full historical cohort look much stronger than holdout (~0.9 r) because most of those players were in training. Always use **holdout metrics** for evaluation.

Re-run `python -m rookie_ppr.compile` after each NFL season so new rookie PPR enters the holdout and models update.

### Model artifacts


| File                         | Contents                                      |
| ---------------------------- | --------------------------------------------- |
| `hgb_rookie_ppr.joblib`      | Per-position models + feature lists           |
| `composite_weights.json`     | Weights + position z-score norms              |
| `percentile_lookup.json`     | Historical PPR for success score              |
| `model_metrics.json`         | Holdout, tuning, and ADP-baseline metrics     |
| `csv/model_vs_adp_baseline.csv` | Full vs ADP-only r/MAE/lift (overall + by pos)|


---



## Scorer UI

```powershell
python -m rookie_ppr.ui
# or: rookie-ppr-ui
```

**Features**

- Chinese blue theme, white text
- **Default player lookup = 2026 rookies** (2026–27 season); check **Include past players** for the full history
- Type-to-filter dropdowns (player, college, NFL team, HS class, stars, age, position)
- **(Important)** labels on strongest correlates; range hints under each field
- Edit any field and click **Calculate score** again — values are kept (no need to clear the form)
- **Calculate** keeps missing composite info in the main window and opens a **prediction side popup** with an ESPN-style **boom/bust gauge**:
  - Box-and-whisker styled like the historical peers plot: Bust / IQR / Boom zones from **prediction-error Q1/Q3** (not peer history), yellow median = projected points
  - **25% / 25%** chance below Q1 / above Q3 by construction of the error-model quartiles
  - Optional model **80% CQR** band shown as text under the header
  - Peer **success score** shown as text (vs historical same-position rookies)
  - Optional checkbox: **historical same-position PPR** box-and-whisker
- Searching a new player closes the prediction popup unless **compare** windows are open
- **Compare players** — same boom/bust gauge, confidence band, optional PPR distribution

Blank fields = missing data (composites use whatever is available).

**Later:** dynasty formats.
### CLI scoring

```powershell
python -m rookie_ppr.score --position WR --draft-overall 4 --recruiting-rank 41 --ff-adp 61.5
```

---



## Predicting the 2026 draft class (2026–27)

After compile, open sheet `incoming_rookies_2026` (also `data/output/csv/incoming_rookies_2026.csv`).


| Source                      | Status for 2026 class                                               |
| --------------------------- | ------------------------------------------------------------------- |
| nflverse draft + combine    | Available                                                           |
| On3 recruiting (HS 2021–23) | Via `--fetch-on3` / local CSVs                                      |
| CFBD college                | Available with `CFBD_API_KEY`                                       |
| FantasyPros 2026 ADP        | `data/manual/FantasyPros_2026_Overall_ADP_Rankings.csv`             |
| ADP recovery                | 2026 draftees on the ADP list are kept even if HS class was missing |
| Rookie PPR / 2026 team SOS  | Fills in after the 2026 NFL season                                  |


Each row includes `predicted_rookie_ppr`, `predicted_adp_baseline`, uncertainty columns (`ppr_low`, `ppr_high`, `predictive_q25`, `predictive_q75`), and `success_score_0_100`.

---



## Cohort rules

- HS classes **2010–2023** (recruiting match, else birthdate estimate)
- Incoming draft-year players on FantasyPros Overall ADP kept even without HS class
- Positions: QB, RB, WR, TE
- Fantasy target: PPR, **rookie season only**
- College: school listed at NFL draft



### Recruiting fetch

```powershell
python -m rookie_ppr.fetch_on3_recruiting
# or
python -m rookie_ppr.compile --fetch-on3
```

CSVs: `data/raw/recruiting/hs_2010.csv` … `hs_2023.csv`.

Ranking preference: Industry Composite → Rivals-only → unranked.

---



## Project layout

```
env/                         # portable environment setup
src/rookie_ppr/              # compile, correlation, ML, UI, score CLI
data/raw/recruiting/         # On3 HS class CSVs
data/manual/                 # FantasyPros ADP + match overrides
data/output/                 # xlsx, csv, models
```

---



## License / data attribution

NFL data via [nflverse](https://github.com/nflverse) release CSVs (generally CC-BY 4.0). Recruiting data is subject to On3/Rivals terms; supply those files yourself (or use `--fetch-on3` for personal/academic use under applicable terms).

Environment installs on **Python 3.11–3.14** using pandas + direct nflverse CSV downloads (no pyarrow/nflreadpy required).