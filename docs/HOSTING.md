# Hosting the ForeKast API

The API is a FastAPI app that reads shipped CSVs under `data/output/csv/`
(and models under `data/output/models/` for future score endpoints).
**No ESPN cookies or `.env` secrets are required** to serve historical ForeKast data.

Contract for the website: [`docs/FOREKAST_API.md`](FOREKAST_API.md)

## What to give the website owner

1. **Public base URL** after deploy (example: `https://nerds-united-forekast-api.onrender.com`)
2. This contract doc (`FOREKAST_API.md`)
3. Primary endpoint: `GET {BASE}/v1/forekast/snapshot`

They do **not** need your `.env` or the Capstone UI — only the API URL + contract.

## Option A — Render (Dockerfile)

1. Push this repo to GitHub
2. [Render](https://dashboard.render.com) → New → Web Service → connect repo
3. Runtime: **Docker** (uses root `Dockerfile`)
4. Health check path: `/health`
5. After deploy, open `{BASE}/docs` and `{BASE}/v1/forekast/snapshot`

`render.yaml` is included for Blueprint deploys.

## Option B — Railway

1. Push repo to GitHub
2. [Railway](https://railway.app) → New Project → Deploy from GitHub
3. Railway detects `Dockerfile` (or uses `Procfile`)
4. Set no secrets for read-only ForeKast
5. Generate a public domain; hit `/health`

## Option C — local smoke test before push

```powershell
.\.venv\Scripts\pip.exe install -r env\requirements.txt
.\.venv\Scripts\python.exe -m rookie_ppr.api
# http://127.0.0.1:8000/health
# http://127.0.0.1:8000/v1/forekast/snapshot
```

## After you push — commit checklist

Include at least:

- `src/rookie_ppr/api/`
- `Dockerfile`, `.dockerignore`, `Procfile`, `render.yaml`
- `docs/FOREKAST_API.md`, `docs/HOSTING.md`
- `env/requirements.txt`, `pyproject.toml`, `README.md`, `env/README.md`
- `.gitignore`, `.env.example` (placeholders only)

**Never commit:** `.env`, `data/raw/espn_cache/`, `.venv/`

Models + `data/output/csv/league_*` are already tracked on this branch so clones/deploys work without recompile.

## Weekly refresh (optional, your machine)

When a new NFL week is complete and ESPN cookies are in local `.env`:

```powershell
python -m rookie_ppr.league.ingest   # if rosters need refresh
python -m rookie_ppr.api.refresh_forekast --seasons 2025
git add data/output/csv/league_weekly_odds.csv data/output/csv/league_injury_events.csv data/output/csv/league_trade_events.csv
git commit -m "Refresh ForeKast weekly odds"
git push
# Redeploy host (or auto-deploy from main/dynasty)
```

## CORS

Allowed origins already include `https://nerds-united.vercel.app`. If the site uses another domain, add it in `src/rookie_ppr/api/app.py` (`ALLOWED_ORIGINS`).
