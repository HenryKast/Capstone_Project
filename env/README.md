# Environment setup

This folder holds everything needed to recreate the Python environment for the Rookie PPR project.

## Requirements

- Python **3.11+** (works on 3.11–3.14; see `.python-version`)
- Internet access the first time you compile data (nflverse GitHub release CSVs)

Dependencies intentionally avoid `pyarrow` / `polars` / `nflreadpy` so installs succeed on newer Python versions where those wheels may be missing.

## Quick start (Windows)

From the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File .\env\setup_env.ps1
.\.venv\Scripts\Activate.ps1
```

## Quick start (macOS / Linux)

```bash
bash env/setup_env.sh
source .venv/bin/activate
```

## Conda alternative

From the repository root:

```bash
conda env create -f environment.yml
conda activate rookie-ppr
```

## Files

| File | Purpose |
| --- | --- |
| `requirements.txt` | Runtime dependencies |
| `requirements-dev.txt` | Runtime + lint/test tools |
| `setup_env.ps1` | Windows venv bootstrap |
| `setup_env.sh` | macOS/Linux venv bootstrap |
| `.python-version` | Hint for pyenv users |

## Optional API keys / cookies

### College Football Data (CFBD)

College production stats require a free API key:

1. Create a key at https://collegefootballdata.com/
2. Copy `.env.example` to `.env` in the repo root
3. Set `CFBD_API_KEY=your_key`

Without a key, college-production columns are left blank and the rest of the compile still runs. Compiled CSVs already include college features when they were built with a key.

### ESPN private league (optional refresh)

Pre-built league tables live under `data/output/csv/league_*.csv`, so you do **not** need ESPN cookies to use the app.

To re-pull raw ESPN history you must set in `.env`:

- `ESPN_LEAGUE_ID`
- `ESPN_S2` (espn_s2 cookie)
- `ESPN_SWID` (SWID cookie)

Never commit `.env`. The raw `data/raw/espn_cache/` folder is gitignored.

## After setup

```powershell
# Optional full rebuild (not required if using shipped CSVs + models)
python -m rookie_ppr.compile

# Launch UI (uses data/output/csv and data/output/models)
python -m rookie_ppr.ui

# ForeKast HTTP API for nerds-united.vercel.app
pip install fastapi "uvicorn[standard]" pydantic   # or: pip install -r env/requirements.txt
python -m rookie_ppr.api
# Docs: http://127.0.0.1:8000/docs — contract: docs/FOREKAST_API.md

# After new ESPN weeks (needs cookies in .env):
python -m rookie_ppr.api.refresh_forekast --seasons 2025
```

Outputs land in `data/output/`. Trained models ship under `data/output/models/`.
