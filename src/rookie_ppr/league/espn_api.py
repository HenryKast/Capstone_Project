"""Authenticated reads from the ESPN fantasy API, cached to disk as JSON.

Two URL shapes serve the same league. Recent seasons live under
``/seasons/{year}/segments/0/leagues/{id}``; older ones only answer through
``/leagueHistory/{id}?seasonId={year}``, which returns a single-element list and
carries less per-week detail. Callers should not care which one applies, so
:func:`fetch_view` tries the richer endpoint first and remembers what worked.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import requests

from rookie_ppr.league.config import (
    ESPN_CACHE_DIR,
    ESPN_LEAGUE_ID,
    ESPN_S2,
    ESPN_SWID,
)

BASE = "https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl"
TIMEOUT = 60

ENDPOINT_SEASON = "seasons"
ENDPOINT_HISTORY = "leagueHistory"

# Which URL shape answered for a season, so we only probe once per process.
_ENDPOINT_STYLE: dict[int, str] = {}


class EspnAuthError(RuntimeError):
    """Credentials are missing, expired, or rejected by ESPN."""


def _require_credentials() -> tuple[str, str, str]:
    missing = [
        name
        for name, value in (
            ("ESPN_LEAGUE_ID", ESPN_LEAGUE_ID),
            ("ESPN_S2", ESPN_S2),
            ("ESPN_SWID", ESPN_SWID),
        )
        if not value
    ]
    if missing:
        raise EspnAuthError(
            f"Missing {', '.join(missing)} in .env. Sign in at fantasy.espn.com, "
            "copy the espn_s2 and SWID cookies, and add them alongside the "
            "league id."
        )
    return ESPN_LEAGUE_ID, ESPN_S2, ESPN_SWID


def _headers(fantasy_filter: dict | None = None) -> dict[str, str]:
    _, s2, swid = _require_credentials()
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
        "Cookie": f"espn_s2={s2}; SWID={swid}",
    }
    if fantasy_filter:
        headers["x-fantasy-filter"] = json.dumps(fantasy_filter)
    return headers


def _url(season: int, style: str, views: list[str], params: dict[str, int]) -> str:
    league_id, _, _ = _require_credentials()
    query = [f"view={v}" for v in views]
    query += [f"{k}={v}" for k, v in sorted(params.items())]
    if style == ENDPOINT_SEASON:
        stem = f"{BASE}/seasons/{season}/segments/0/leagues/{league_id}"
    else:
        stem = f"{BASE}/leagueHistory/{league_id}"
        query.insert(0, f"seasonId={season}")
    return f"{stem}?{'&'.join(query)}"


def _cache_path(season: int, views: list[str], params: dict[str, int]) -> Path:
    league_id, _, _ = _require_credentials()
    parts = [str(league_id), str(season), "-".join(sorted(views))]
    parts += [f"{k}{v}" for k, v in sorted(params.items())]
    name = "__".join(parts).replace("/", "_") + ".json"
    return ESPN_CACHE_DIR / name


def _node(payload: Any) -> dict:
    """leagueHistory wraps the league in a list; the season endpoint does not."""
    if isinstance(payload, list):
        return payload[0] if payload else {}
    return payload or {}


def _get(url: str, fantasy_filter: dict | None = None) -> Any:
    resp = requests.get(url, headers=_headers(fantasy_filter), timeout=TIMEOUT)
    if resp.status_code in (401, 403):
        raise EspnAuthError(
            "ESPN rejected the stored cookies (HTTP "
            f"{resp.status_code}). They expire when you log out; grab fresh "
            "espn_s2 / SWID values and update .env."
        )
    resp.raise_for_status()
    return resp.json()


def endpoint_style(season: int) -> str:
    """Which URL shape serves this season. Cheap after the first call."""
    if season in _ENDPOINT_STYLE:
        return _ENDPOINT_STYLE[season]
    for style in (ENDPOINT_SEASON, ENDPOINT_HISTORY):
        try:
            payload = _get(_url(season, style, ["mSettings"], {}))
        except EspnAuthError:
            raise
        except Exception:  # noqa: BLE001 - fall through to the other URL shape
            continue
        if (_node(payload).get("settings") or {}).get("name"):
            _ENDPOINT_STYLE[season] = style
            return style
    raise RuntimeError(
        f"Neither ESPN endpoint returned league {ESPN_LEAGUE_ID} for {season}."
    )


def fetch_view(
    season: int,
    views: list[str],
    *,
    scoring_period: int | None = None,
    matchup_period: int | None = None,
    fantasy_filter: dict | None = None,
    force: bool = False,
) -> dict:
    """Return one league payload, reading from (and filling) the disk cache."""
    params: dict[str, int] = {}
    if scoring_period is not None:
        params["scoringPeriodId"] = scoring_period
    if matchup_period is not None:
        params["matchupPeriodId"] = matchup_period

    cache_path = _cache_path(season, views, params)
    if cache_path.exists() and not force:
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            cache_path.unlink(missing_ok=True)

    payload = _get(
        _url(season, endpoint_style(season), views, params), fantasy_filter
    )
    node = _node(payload)
    ESPN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(node), encoding="utf-8")
    return node


def fetch_player_names(season: int, player_ids: list[int]) -> dict[int, dict]:
    """Best-effort ESPN id -> {name, position id, pro team id} for draft picks.

    Draft payloads carry only player ids. This uses the league-independent
    player endpoint, which is the only place to look up someone who was drafted
    and then dropped before any roster snapshot we keep.
    """
    ids = sorted({int(pid) for pid in player_ids if pid and int(pid) > 0})
    if not ids:
        return {}

    cache_path = ESPN_CACHE_DIR / f"players__{season}__{len(ids)}.json"
    if cache_path.exists():
        try:
            raw = json.loads(cache_path.read_text(encoding="utf-8"))
            return {int(k): v for k, v in raw.items()}
        except json.JSONDecodeError:
            cache_path.unlink(missing_ok=True)

    out: dict[int, dict] = {}
    chunk = 300
    for start in range(0, len(ids), chunk):
        batch = ids[start : start + chunk]
        url = f"{BASE}/seasons/{season}/players?view=players_wl"
        try:
            payload = _get(url, {"filterIds": {"value": batch}})
        except EspnAuthError:
            raise
        except Exception:  # noqa: BLE001 - names are a nicety, ids are the key
            continue
        rows = payload if isinstance(payload, list) else (payload or {}).get("players") or []
        for row in rows:
            player = row.get("player") if isinstance(row, dict) and "player" in row else row
            if not isinstance(player, dict):
                continue
            pid = player.get("id")
            if pid is None:
                continue
            out[int(pid)] = {
                "full_name": player.get("fullName"),
                "default_position_id": player.get("defaultPositionId"),
                "pro_team_id": player.get("proTeamId"),
            }

    if out:
        ESPN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps({str(k): v for k, v in out.items()}), encoding="utf-8"
        )
    return out


__all__ = [
    "ENDPOINT_HISTORY",
    "ENDPOINT_SEASON",
    "EspnAuthError",
    "endpoint_style",
    "fetch_player_names",
    "fetch_view",
]
