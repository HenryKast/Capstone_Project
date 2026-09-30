"""Apply ESPN-accepted trades that have not processed onto live rosters yet.

The latest ForeKast as-of week uses current ESPN ``mRoster`` membership for
remaining-season strength. ``TRADE_ACCEPT`` transactions still sitting in
``mPendingTransactions`` are treated as already completed: players move to the
destination team, and explicit drops leave the snapshot.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from rookie_ppr.league.config import (
    BENCH_SLOT_IDS,
    LINEUP_SLOT_NAMES,
    NFL_TEAM_BY_ESPN_ID,
    POSITION_BY_ESPN_ID,
    STAT_SOURCE_PROJECTED,
)
from rookie_ppr.league.espn_api import EspnAuthError, fetch_view
from rookie_ppr.league.ingest import ROSTER_COLS, attach_gsis_ids

ACCEPTED_TRADE_TYPES = {"TRADE_ACCEPT"}
ACCEPTED_STATUSES = {"PENDING", "ACCEPTED"}
DROP_ITEM_TYPES = {"DROP", "WAIVER"}


def accepted_pending_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Parse TRADE_ACCEPT rows from an ESPN pending-transactions payload."""
    rows: list[dict[str, Any]] = []
    buckets: list[Any] = []
    for key in ("transactions", "pendingTransactions"):
        buckets.extend(payload.get(key) or [])
    seen: set[tuple[int, str, int | None, int | None]] = set()
    for tx in buckets:
        if not isinstance(tx, dict):
            continue
        ttype = str(tx.get("type") or "")
        if ttype not in ACCEPTED_TRADE_TYPES:
            continue
        status = str(tx.get("status") or "PENDING")
        if status not in ACCEPTED_STATUSES:
            continue
        for item in tx.get("items") or []:
            if not isinstance(item, dict):
                continue
            pid = item.get("playerId")
            if pid is None:
                continue
            itype = str(item.get("type") or "TRADE").upper()
            from_id = item.get("fromTeamId")
            to_id = item.get("toTeamId")
            key = (
                int(pid),
                itype,
                int(from_id) if from_id is not None else None,
                int(to_id) if to_id is not None else None,
            )
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "espn_player_id": int(pid),
                    "from_team_id": int(from_id) if from_id is not None else None,
                    "to_team_id": int(to_id) if to_id is not None else None,
                    "item_type": itype,
                }
            )
    return rows


def fetch_accepted_pending_items(season: int, *, force: bool = True) -> list[dict[str, Any]]:
    node = fetch_view(season, ["mPendingTransactions"], force=force)
    return accepted_pending_items(node)


def apply_accepted_pending_trades(
    rosters: pd.DataFrame,
    *,
    season: int,
    week: int,
    items: list[dict[str, Any]],
) -> pd.DataFrame:
    """Move or drop players on one season-week snapshot."""
    if not items or rosters.empty:
        return rosters
    out = rosters.copy()
    out["espn_player_id"] = pd.to_numeric(out["espn_player_id"], errors="coerce")
    mask_week = (out["season"] == season) & (out["week"] == week)
    drop_ids: list[int] = []
    for item in items:
        pid = int(item["espn_player_id"])
        itype = str(item.get("item_type") or "TRADE").upper()
        if itype in DROP_ITEM_TYPES:
            drop_ids.append(pid)
            continue
        dest = item.get("to_team_id")
        if dest is None:
            continue
        hit = mask_week & (out["espn_player_id"] == pid)
        out.loc[hit, "team_id"] = int(dest)
    if drop_ids:
        out = out.loc[~(mask_week & out["espn_player_id"].isin(drop_ids))].reset_index(
            drop=True
        )
    return out


def fetch_live_roster_week(
    season: int, week: int, *, force: bool = True
) -> pd.DataFrame:
    """Current ESPN mRoster, stamped as ``week`` so remaining-season sims use it."""
    node = fetch_view(season, ["mRoster"], force=force)
    scoring_period = node.get("scoringPeriodId")
    rows: list[dict[str, Any]] = []
    for team in node.get("teams") or []:
        tid = int(team["id"])
        for entry in (team.get("roster") or {}).get("entries") or []:
            player = (entry.get("playerPoolEntry") or {}).get("player") or {}
            slot_id = entry.get("lineupSlotId")
            projected = None
            for stat in player.get("stats") or []:
                if stat.get("scoringPeriodId") != scoring_period:
                    continue
                if stat.get("statSourceId") != STAT_SOURCE_PROJECTED:
                    continue
                if stat.get("appliedTotal") is not None:
                    projected = float(stat["appliedTotal"])
            rows.append(
                {
                    "season": season,
                    "week": week,
                    "team_id": tid,
                    "espn_player_id": int(entry.get("playerId") or player.get("id") or 0),
                    "player_name": player.get("fullName"),
                    "position": POSITION_BY_ESPN_ID.get(player.get("defaultPositionId")),
                    "pro_team": NFL_TEAM_BY_ESPN_ID.get(player.get("proTeamId")),
                    "lineup_slot_id": slot_id,
                    "lineup_slot": LINEUP_SLOT_NAMES.get(slot_id),
                    "is_starter": slot_id not in BENCH_SLOT_IDS,
                    "actual_points": np.nan,
                    "espn_projected_points": projected,
                }
            )
    live = pd.DataFrame(rows)
    if live.empty:
        return pd.DataFrame(columns=ROSTER_COLS)
    return attach_gsis_ids(live)[ROSTER_COLS]


def overlay_latest_week_rosters(
    league_rosters: pd.DataFrame,
    *,
    season: int,
    week: int,
    force: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Replace one week with live ESPN rosters plus unprocessed TRADE_ACCEPT moves.

    Ingested historical weeks stay untouched. ESPN auth failures leave the
    ingested snapshot in place.
    """
    info: dict[str, Any] = {
        "live": False,
        "pending_moves": 0,
        "scoring_period": None,
        "error": None,
    }
    items: list[dict[str, Any]] = []
    try:
        items = fetch_accepted_pending_items(season, force=force)
        info["pending_moves"] = len(items)
    except EspnAuthError as exc:
        info["error"] = str(exc)
        return league_rosters, info
    except Exception as exc:  # noqa: BLE001 - overlay is best-effort
        info["error"] = f"{type(exc).__name__}: {exc}"

    try:
        live = fetch_live_roster_week(season, week, force=force)
        info["live"] = True
        if items:
            live = apply_accepted_pending_trades(
                live, season=season, week=week, items=items
            )
        keep = league_rosters[
            ~((league_rosters["season"] == season) & (league_rosters["week"] == week))
        ]
        return pd.concat([keep, live], ignore_index=True), info
    except EspnAuthError as exc:
        info["error"] = str(exc)
    except Exception as exc:  # noqa: BLE001
        info["error"] = f"{type(exc).__name__}: {exc}"

    if items:
        league_rosters = apply_accepted_pending_trades(
            league_rosters, season=season, week=week, items=items
        )
    return league_rosters, info
