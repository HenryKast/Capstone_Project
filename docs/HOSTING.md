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

Models + `data/output/csv/league_*` (including `league_managers.csv`) are already
tracked so clones/deploys resolve `ownerName` without ESPN cookies.

## New season (after the draft)

The odds simulation replays head-to-head matchups, so playoff and title odds
depend on `league_matchups.csv`. Measured on 2025, swapping the schedule while
holding rosters fixed moves preseason playoff odds by ~13 points per team, so
odds published against a draft schedule will visibly change if the schedule is
edited afterwards. Team scoring projections are schedule-free.

Check readiness at any time:

```powershell
python -m rookie_ppr.league.target_season --check
```

Once the schedule is final, re-ingest (ESPN responses are cached, so `--force`
is required to pick up a changed schedule) and publish:

```powershell
python -m rookie_ppr.league.ingest --seasons 2026 --force
python -m rookie_ppr.league.target_season --run
```

Drop `--skip-rosters` here even before kickoff: ESPN answers every upcoming week
with the roster as it stands today, and the preseason odds read from that
snapshot so post-draft waiver and trade activity shows up on the site.

A season-scoped ingest replaces only that season's rows, and a table it skipped
is left on disk rather than truncated, so this cannot drop league history.

`--run` refuses to publish while the schedule is incomplete, and records a
fingerprint of the schedule it used; a later `--check` reports if it changed.
Publishing makes the new season the API's `latest`, at `asOfWeek: 0`.
`/v1/forekast/season` keeps defaulting to the newest *finished* season, since
that board compares projected finish against actual.

Before kickoff, `asOfWeek: 0` describes the league as it stands right now. Once
week 1 is played it reverts to its historical meaning — the draft-day view — and
week 1 onward becomes the live one, matching what completed seasons publish.

### Draft grades

Both grades are schedule-free, so they can be published as soon as the draft is
in `league_backtest_rosters.csv` — no need to wait for the schedule:

```powershell
python -m rookie_ppr.league.draft_grades --seasons 2026
```

That writes `league_draft_grades.csv` and rebuilds
`league_draft_grade_calibration.csv` across every graded season. Only the named
seasons are replaced. The grade rows carry `playoff_odds` / `title_odds` too,
but those stay empty until the season's odds are published, so re-run grades
after `target_season --run` to fill them in. `refresh_forekast` runs grades last
for the same reason; pass `--skip-grades` to opt out.

## Weekly refresh (optional, your machine)

When a new NFL week is complete and ESPN cookies are in local `.env`, these two
commands are the whole loop:

```powershell
python -m rookie_ppr.league.ingest --seasons 2026 --force
python -m rookie_ppr.api.refresh_forekast --seasons 2026
git add data/output/csv/league_rosters.csv data/output/csv/league_matchups.csv data/output/csv/league_teams.csv data/output/csv/league_weekly_odds.csv data/output/csv/league_injury_events.csv data/output/csv/league_trade_events.csv data/output/csv/league_draft_grades.csv data/output/csv/league_draft_grade_calibration.csv
git commit -m "Refresh ForeKast weekly odds"
git push
# Redeploy host (or auto-deploy from main/dynasty)
```

`--force` on the ingest is what makes the refresh weekly rather than a no-op:
ESPN responses are cached on disk, so without it you re-read last week's
rosters and scores.

`refresh_forekast` picks up the new week on its own. It simulates through the
newest week that has actually been played, so roster moves, injuries and trades
land as soon as they are in the roster snapshots. Remaining-week player
projections also update from in-season usage (targets, carries, dropbacks) and
scoring, shrunk toward the preseason prior: one spike is about a 9% blend,
four consistent weeks about 29%, ten weeks 50%. Usage is weighted more than
raw points. It also builds any missing drafted rosters rather than failing
inside the simulation, and it now defaults to every season including the one
under way, so a bare run does not skip 2026.

A season-scoped refresh only replaces that season's rows; every other season in
`league_weekly_odds.csv`, the event tables and `league_managers.csv` is kept.

Odds carry roughly ±3 points of Monte Carlo noise per team at the default sim
count, so small week-to-week wobble is sampling, not signal. Raise `--sims` on
`target_season` if you want steadier published numbers.

## CORS

Allowed origins already include `https://nerds-united.vercel.app`. If the site uses another domain, add it in `src/rookie_ppr/api/app.py` (`ALLOWED_ORIGINS`).
