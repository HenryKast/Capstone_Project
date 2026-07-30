"""FastAPI app: Henry ForeKast weekly odds, injuries, trades for the league site."""
from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from rookie_ppr.api import forekast
from rookie_ppr.api.csv_store import clear_cache

ALLOWED_ORIGINS = [
    "https://nerds-united.vercel.app",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app = FastAPI(
    title="Nerds United ForeKast API",
    description=(
        "Weekly playoff/title odds, major injuries, and trades for "
        "https://nerds-united.vercel.app (Henry ForeKast)."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return forekast.health_payload()


@app.get("/v1/meta")
def meta() -> dict:
    payload = forekast.health_payload()
    seasons = payload.get("seasons") or []
    weeks_by_season = {}
    for s in seasons:
        try:
            weeks_by_season[str(s)] = forekast.weeks_for_season(int(s))
        except FileNotFoundError:
            weeks_by_season[str(s)] = []
    return {
        **payload,
        "weeksBySeason": weeks_by_season,
        "endpoints": [
            "/health",
            "/v1/meta",
            "/v1/forekast/weekly",
            "/v1/forekast/injuries",
            "/v1/forekast/trades",
            "/v1/forekast/snapshot",
            "/v1/forekast/season",
        ],
    }


@app.post("/v1/admin/reload")
def reload_cache() -> dict:
    """Clear in-memory CSV cache after regenerating artifacts."""
    clear_cache()
    return {"ok": True, "cleared": True}


@app.get("/v1/forekast/weekly")
def weekly_odds(
    season: int | None = Query(None, description="Season year; default latest"),
    week: int | None = Query(
        None, description="as_of_week; default latest week for the season"
    ),
) -> dict:
    try:
        return forekast.build_weekly(season, week)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/v1/forekast/injuries")
def injuries(
    season: int | None = Query(None),
    week: int | None = Query(None),
    exact: bool = Query(
        False,
        description="If true, only events at as_of_week; else all events up to week",
    ),
) -> dict:
    try:
        return forekast.build_injuries(season, week, exact=exact)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/v1/forekast/trades")
def trades(
    season: int | None = Query(None),
    week: int | None = Query(None),
    exact: bool = Query(
        False,
        description="If true, only events at as_of_week; else all events up to week",
    ),
) -> dict:
    try:
        return forekast.build_trades(season, week, exact=exact)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/v1/forekast/snapshot")
def snapshot(
    season: int | None = Query(None),
    week: int | None = Query(None),
) -> dict:
    """One round-trip for /takes: odds + injuries + trades at a week."""
    try:
        return forekast.build_snapshot(season, week)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/v1/forekast/season")
def season_board(
    season: int | None = Query(None, description="Season year; default latest"),
) -> dict:
    """Season finish board shaped like the site's Henry ForeKast table."""
    try:
        return forekast.build_season_forekast(season)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
