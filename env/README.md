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

## Optional API key (college production)

College Football Data (CFBD) college stats require a free API key:

1. Create a key at https://collegefootballdata.com/
2. Copy `.env.example` to `.env` in the repo root
3. Set `CFBD_API_KEY=your_key`

Without a key, college-production columns are left blank and the rest of the compile still runs.

## After setup

```powershell
python -m rookie_ppr.compile
```

Outputs land in `data/output/`.
