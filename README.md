# Rookie PPR Success Dataset

Compile high-school classes **2010–2020** skill-position players (QB, RB, WR, TE) into one master workbook and matching CSVs for studying **rookie-season PPR** fantasy outcomes.

This repository currently includes **environment setup** and the **data compile / export** pipeline. Correlation analysis and the ML success scorer are planned next and are not run by the compile step.

## What you get

After compile:

| Output | Description |
| --- | --- |
| `data/output/rookie_ppr_master.xlsx` | Master workbook (one sheet per dataset page) |
| `data/output/csv/*.csv` | Same tables as individual CSV files |
| `data/output/players_master.csv` | Wide joined modeling table |

### Workbook sheets

- `players` — cohort identity
- `recruiting` — Industry Composite rank (Rivals fallback) when On3 CSVs are provided
- `draft` — college at draft, pick, team
- `combine` — NFL combine measurables
- `college_production` — CFB stats (requires optional `CFBD_API_KEY`)
- `team_context` — SOS, offensive environment, landing-spot opportunity
- `pre_draft_fantasy` — fantasy ADP/ECR when available
- `fantasy_rookie` — rookie-season PPR
- `players_master` — full join
- `data_dictionary` — column definitions

## Setup (for a new machine)

### Windows

```powershell
powershell -ExecutionPolicy Bypass -File .\env\setup_env.ps1
.\.venv\Scripts\Activate.ps1
pip install -e .
```

### macOS / Linux

```bash
bash env/setup_env.sh
source .venv/bin/activate
pip install -e .
```

Details: [`env/README.md`](env/README.md)

## Compile data

With the venv active:

```powershell
python -m rookie_ppr.compile
```

First run downloads nflverse datasets (needs network).

### Recruiting files (On3)

Auto-download Industry Comparison (+ Industry Player backfill) for HS classes 2010–2020:

```powershell
python -m rookie_ppr.fetch_on3_recruiting
# or:
python -m rookie_ppr.compile --fetch-on3
```

CSVs land in `data/raw/recruiting/` as `hs_2010.csv` … `hs_2020.csv`.

Ranking rule:

1. Rivals Industry Composite when ranked  
2. Else Rivals-only  
3. Else unranked  

### Optional college production

Copy `.env.example` → `.env` and set `CFBD_API_KEY` from [collegefootballdata.com](https://collegefootballdata.com/).

## Project layout

```
env/                      # portable environment setup
src/rookie_ppr/           # compile pipeline package
data/raw/recruiting/      # user-supplied On3 CSVs
data/manual/              # optional match overrides
data/output/              # xlsx + csv outputs
```

## Cohort rules

- HS classes 2010–2020 (from recruiting match, or birthdate estimate when recruiting files are absent)
- Positions: QB, RB, WR, TE
- Fantasy: PPR, **rookie season only**
- College: school listed at NFL draft

## License / data attribution

NFL data via [nflverse](https://github.com/nflverse) release CSVs (generally CC-BY 4.0). Recruiting data remains subject to On3/Rivals terms; you must supply those files yourself.

### Python note

The environment is tested to install on **Python 3.14** using pandas + direct nflverse CSV downloads (no pyarrow/nflreadpy required).
