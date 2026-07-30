# Rookie / Dynasty PPR Scorer

Predict **PPR fantasy points** for skill-position rookies (QB, RB, WR, TE) in two formats:

- **Redraft** — first NFL season (`rookie_ppr`)
- **Dynasty** — years 1–3 separately, plus Y1–Y3 total

One desktop UI with browser-style **Redraft | Dynasty** tabs. Models are trained separately and do not overwrite each other.

---

## New machine setup

**Requirements:** Python **3.11+**, network on first compile (nflverse). Optional free [CollegeFootballData](https://collegefootballdata.com/) API key for CFB stats. More env detail: [`env/README.md`](env/README.md).

### Windows

```powershell
powershell -ExecutionPolicy Bypass -File .\env\setup_env.ps1
.\.venv\Scripts\Activate.ps1

# Optional CFB production
copy .env.example .env
# Edit .env → CFBD_API_KEY=...

# Optional: fetch On3 recruiting (HS 2010–2023), then compile + train
python -m rookie_ppr.compile --fetch-on3

# Or compile using recruiting CSVs already in the repo
python -m rookie_ppr.compile

# Scorer UI (tabs: Redraft | Dynasty)
python -m rookie_ppr.ui
```

If activation fails (*running scripts is disabled*):

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Or skip activation:

```powershell
.\.venv\Scripts\python.exe -m rookie_ppr.compile
.\.venv\Scripts\python.exe -m rookie_ppr.ui
```

### macOS / Linux

```bash
bash env/setup_env.sh
source .venv/bin/activate
# optional: .env.example → .env with CFBD_API_KEY
python -m rookie_ppr.compile --fetch-on3
python -m rookie_ppr.ui
```

### Everyday commands

```powershell
python -m rookie_ppr.ui                 # Redraft | Dynasty tabs
python -m rookie_ppr.ui --mode dynasty  # start on Dynasty tab
python -m rookie_ppr.compile            # refresh data + retrain redraft (+ dynasty sheets/models)
python -m rookie_ppr.model_score dynasty  # retrain dynasty only
python -m rookie_ppr.score --position WR --draft-overall 4 --ff-adp 61.5

# ForeKast API (weekly playoff/title odds + injuries/trades for the league site)
python -m rookie_ppr.api
# Open http://127.0.0.1:8000/docs — contract: docs/FOREKAST_API.md
# Hosting: docs/HOSTING.md (Render/Railway + what to send the website owner)
```

---

## Compiling data (including new players)

**All data currently in this dataset is up to date.** Re-run compile when you need a refresh or when adding players/years not already covered.

`python -m rookie_ppr.compile` rebuilds the master workbook/CSVs and retrains the redraft model (dynasty artifacts are built in the same pipeline when configured).

| Source | Provides |
| --- | --- |
| nflverse draft / combine / rosters / stats | Draft capital, measurables, birth dates, NFL fantasy |
| On3 recruiting CSVs (`data/raw/recruiting/hs_*.csv`) | Rank, stars, HS class |
| CFBD (optional API key) | Final college season production |
| School roster scrape (fallback) | CFB stats when CFBD has no match |
| FantasyPros Overall ADP (`data/manual/`) | Pre-draft fantasy ADP / rank |
| NFL.com player pages (when needed) | Recent-season fills when nflverse lags |

**Adding players or seasons not yet in the set**

1. Ensure recruiting CSVs cover the HS class (`python -m rookie_ppr.compile --fetch-on3` or manual CSVs).
2. Put any ADP / ID overrides in `data/manual/` (see existing templates).
3. Set `CFBD_API_KEY` if you need college production.
4. Run `python -m rookie_ppr.compile` (add `--fetch-on3` only when recruiting must be re-scraped).
5. For dynasty-only retrain after compile: `python -m rookie_ppr.model_score dynasty`.

**Main outputs:** `data/output/rookie_ppr_master.xlsx`, `data/output/players_master.csv`, `data/output/csv/`, `data/output/models/` (redraft + `*_dynasty*` twins).

---

## How the models work (short)

```
Compile → correlation → composite features → per-position HistGradientBoosting
         → point forecast + predictive Q1/Q3 (boom/bust) + CQR band → UI / CLI
```

- **Redraft target:** rookie-season PPR. Holdout by first NFL season.
- **Dynasty targets:** `ppr_y1`, `ppr_y2`, `ppr_y3`, and `dynasty_ppr_y1_y3_total` — same recipe as redraft (median HGB + predictive IQR + CQR), trained separately per horizon.
- **Success score (0–100):** empirical percentile of the prediction vs same-position historical outcomes.
- **UI:** lookup incoming rookies, edit inputs, Calculate / Compare; dynasty shows Y1–Y3 + total gauges.

---

## Holdout performance: redraft vs dynasty

Current holdout Pearson **r** (higher = better rank-order of outcomes):

| Model | Holdout r | MAE (points) | Notes |
| --- | ---: | ---: | --- |
| **Redraft** (Y1 only) | **0.68** | ~46 | One season |
| **Dynasty total** (Y1–Y3 sum) | **0.47** | ~172 | Three seasons combined |
| Dynasty Y1 | **0.69** | — | Similar difficulty to redraft |
| Dynasty Y2 | **0.48** | — | Harder |
| Dynasty Y3 | **0.31** | — | Hardest |

**Why dynasty total r is lower — and why that is expected**

1. **More years ⇒ more variance.** Injuries, coaching changes, free agency, role changes, and QB/scheme churn compound after year 1. A strong Y1 does not lock in Y2–Y3.
2. **Errors add.** Total = Y1+Y2+Y3. Noise in later seasons widens absolute error (MAE ~172 vs ~46) even when early-season signal is decent.
3. **Y1 alone still looks like redraft.** Dynasty Y1 holdout r (~0.69) matches redraft (~0.68). The drop shows up when we ask about **later** years and the **sum**.
4. **Smaller labeled set.** Dynasty needs complete Y1–Y3 labels; fewer players and a different holdout window than redraft.

So a lower dynasty total r is not a failed copy of redraft — it is the cost of forecasting a noisier, longer horizon with the same pre-draft information.

---

## Biggest factors (predictions and quartiles)

Both formats use the **same composite families**. Point forecasts and predictive **Q1/Q3** (the boom/bust box) are separate quantile models on those composites; the strongest inputs drive both the center and the band.

### Composite strength (correlation with the training target)

**Redraft** (vs `rookie_ppr`):

| Rank | Composite | Abs corr (impact) |
| --- | --- | ---: |
| 1 | Draft capital | 0.57 |
| 2 | Pre-draft fantasy (ADP) | 0.55 |
| 3 | Timing (age / class) | 0.26 |
| 4 | CFB receiving | 0.24 |
| 5 | Team context (opportunity + incumbents) | 0.19 |

**Dynasty** (vs Y1–Y3 total):

| Rank | Composite | Abs corr (impact) |
| --- | --- | ---: |
| 1 | Pre-draft fantasy (ADP) | 0.44 |
| 2 | Draft capital | 0.44 |
| 3 | Timing | 0.26 |
| 4 | CFB receiving | 0.22 |
| 5 | CFB rushing | 0.20 |

Market consensus (draft pick + ADP) dominates **both** formats. College production and landing-spot context matter more as secondary structure; combine size is near-noise.

### Raw features (illustrative)

Top raw correlates are the same story: **`draft_overall` / `draft_round`**, then **`ff_adp` / `ff_adp_rank`**, then age and position-relevant CFB counting stats. Full tables: `data/output/csv/strongest_factors.csv` and `strongest_factors_dynasty.csv`.

### Predictions vs quartiles

- The **yellow point** is the median (q50) model.
- The **box (Q1–Q3)** is from separate q25/q75 models on the same features — not “prediction ± fixed error.”
- Rookie PPR is **right-skewed**, so the median can sit near or above Q3 for boom-prone profiles; that is expected, not a drawing bug.
- Leave-one-out **score drivers** in the UI show which populated composites moved the point forecast most for that player.

---

## Limitations (where models go wrong)

- **Pre-draft only.** No in-season injuries, depth-chart news after the draft, or mid-year role changes.
- **ADP / draft capital overweight.** Undrafted or late-round breakouts and early busts of “consensus” picks are systematically hard.
- **Sparse or missing inputs.** No CFB match, blank combine, or empty team context → wider bands and less reliable drivers.
- **Landing spot is partial.** Opportunity helps, but scheme fit, OL quality, and coaching are only weakly proxied.
- **Dynasty years 2–3.** Holdout r falls sharply; treat Y2/Y3 gauges as directional, not precise.
- **Right-skew / boom seasons.** Mean/median vs quartile disagreement is common for high-upside profiles.
- **Position imbalance.** Sample sizes differ (WR largest; QB smallest); position-specific r varies.
- **Holdout is temporal.** Metrics are on held-out first-stat seasons, not a random shuffle — future regimes can drift.
- **Name / ID joins.** Fuzzy matching and manual overrides can mis-attach CFB or recruiting rows.

Use the success score and IQR as **uncertainty context**, not as a guarantee.


## Project layout

```
env/                 # setup scripts + requirements
src/rookie_ppr/      # compile, join, composites, train, score, UI
data/raw/            # nflverse / recruiting / caches
data/manual/         # ADP, overrides
data/output/         # workbook, CSVs, models
```

Key modules: `compile.py`, `model_score.py` / `dynasty_train.py`, `score_runner.py`, `ui.py`, `analysis_config.py`.
