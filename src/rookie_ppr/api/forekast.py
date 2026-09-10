"""Site-shaped Henry ForeKast adapters over league CSVs."""
from __future__ import annotations

from typing import Any

import pandas as pd

from rookie_ppr.api.csv_store import json_safe, load_csv
from rookie_ppr.league.config import (
    LEAGUE_DRAFT_GRADE_CALIBRATION_CSV,
    LEAGUE_DRAFT_GRADES_CSV,
    LEAGUE_FINISH_PROJ_VS_ACTUAL_CSV,
    LEAGUE_INJURY_EVENTS_CSV,
    LEAGUE_MANAGERS_CSV,
    LEAGUE_TEAMS_CSV,
    LEAGUE_TRADE_EVENTS_CSV,
    LEAGUE_WEEKLY_ODDS_CSV,
)
from rookie_ppr.league.draft_reaches import load_manager_map


def _ascii(name: object) -> str:
    return str(name).encode("ascii", "ignore").decode("ascii").strip()


def available_seasons() -> list[int]:
    odds = load_csv(LEAGUE_WEEKLY_ODDS_CSV)
    return sorted(int(s) for s in odds["season"].dropna().unique())


def latest_season() -> int:
    seasons = available_seasons()
    if not seasons:
        raise FileNotFoundError("No seasons in weekly odds CSV")
    return seasons[-1]


def weeks_for_season(season: int) -> list[int]:
    odds = load_csv(LEAGUE_WEEKLY_ODDS_CSV)
    sdf = odds[odds["season"] == int(season)]
    return sorted(int(w) for w in sdf["as_of_week"].dropna().unique())


def latest_week(season: int) -> int:
    weeks = weeks_for_season(season)
    if not weeks:
        raise FileNotFoundError(f"No weeks for season {season}")
    return weeks[-1]


def resolve_season_week(
    season: int | None = None,
    week: int | None = None,
) -> tuple[int, int]:
    s = int(season) if season is not None else latest_season()
    w = int(week) if week is not None else latest_week(s)
    return s, w


def _manager_lookup(season: int) -> dict[int, str]:
    """Prefer shipped league_managers.csv (works on Railway without ESPN cache)."""
    try:
        managers = load_csv(LEAGUE_MANAGERS_CSV)
        sdf = managers[managers["season"] == int(season)]
        if not sdf.empty:
            return {
                int(r.team_id): _ascii(r.manager_name)
                for r in sdf.itertuples(index=False)
                if pd.notna(r.manager_name) and _ascii(r.manager_name)
            }
    except FileNotFoundError:
        pass
    try:
        managers = load_manager_map([season])
    except Exception:
        return {}
    if managers.empty:
        return {}
    return {
        int(r.team_id): _ascii(r.manager_name)
        for r in managers.itertuples(index=False)
    }


def _team_name_lookup(season: int) -> dict[int, str]:
    teams = load_csv(LEAGUE_TEAMS_CSV)
    sdf = teams[teams["season"] == int(season)]
    return {
        int(r.team_id): _ascii(r.team_name)
        for r in sdf.itertuples(index=False)
    }


def build_weekly(
    season: int | None = None,
    week: int | None = None,
) -> dict[str, Any]:
    s, w = resolve_season_week(season, week)
    odds = load_csv(LEAGUE_WEEKLY_ODDS_CSV)
    sdf = odds[(odds["season"] == s) & (odds["as_of_week"] == w)].copy()
    sdf = sdf.sort_values("team_id")
    managers = _manager_lookup(s)
    teams: list[dict[str, Any]] = []
    for row in sdf.itertuples(index=False):
        tid = int(row.team_id)
        teams.append(
            {
                "teamId": str(tid),
                "ownerName": managers.get(tid) or None,
                "teamName": _ascii(row.team_name),
                "winsToDate": json_safe(getattr(row, "wins_to_date", None)),
                "pointsToDate": json_safe(getattr(row, "points_to_date", None)),
                "simulatedWins": json_safe(row.sim_wins_mean),
                "playoffOddsPct": json_safe(row.playoff_odds),
                "titleOddsPct": json_safe(row.title_odds),
                "simSeedMean": json_safe(getattr(row, "sim_seed_mean", None)),
            }
        )
    return {
        "season": s,
        "asOfWeek": w,
        "generatedFrom": LEAGUE_WEEKLY_ODDS_CSV,
        "teams": teams,
    }


def build_injuries(
    season: int | None = None,
    week: int | None = None,
    *,
    exact: bool = False,
) -> dict[str, Any]:
    s, w = resolve_season_week(season, week)
    inj = load_csv(LEAGUE_INJURY_EVENTS_CSV)
    sdf = inj[inj["season"] == s].copy()
    if exact:
        sdf = sdf[sdf["as_of_week"] == w]
    else:
        sdf = sdf[sdf["as_of_week"] <= w]
    sdf = sdf.sort_values(["as_of_week", "team_id", "player_name"])
    managers = _manager_lookup(s)
    team_names = _team_name_lookup(s)
    events: list[dict[str, Any]] = []
    for row in sdf.itertuples(index=False):
        tid = int(row.team_id)
        events.append(
            {
                "season": s,
                "asOfWeek": int(row.as_of_week),
                "endWeek": json_safe(getattr(row, "end_week", None)),
                "nWeeks": json_safe(getattr(row, "n_weeks", None)),
                "teamId": str(tid),
                "ownerName": managers.get(tid) or None,
                "teamName": team_names.get(tid) or None,
                "playerName": _ascii(row.player_name),
                "kind": json_safe(getattr(row, "kind", None)),
                "label": _ascii(row.label),
                "overallPick": json_safe(getattr(row, "overall_pick", None)),
                "pickLabel": json_safe(getattr(row, "pick_label", None)),
            }
        )
    return {
        "season": s,
        "asOfWeek": w,
        "exact": exact,
        "generatedFrom": LEAGUE_INJURY_EVENTS_CSV,
        "events": events,
    }


def build_trades(
    season: int | None = None,
    week: int | None = None,
    *,
    exact: bool = False,
) -> dict[str, Any]:
    s, w = resolve_season_week(season, week)
    trades = load_csv(LEAGUE_TRADE_EVENTS_CSV)
    sdf = trades[trades["season"] == s].copy()
    if exact:
        sdf = sdf[sdf["as_of_week"] == w]
    else:
        sdf = sdf[sdf["as_of_week"] <= w]
    sdf = sdf.sort_values(["as_of_week", "team_id", "direction", "player_name"])
    managers = _manager_lookup(s)
    team_names = _team_name_lookup(s)
    events: list[dict[str, Any]] = []
    for row in sdf.itertuples(index=False):
        tid = int(row.team_id)
        ctid = getattr(row, "counterparty_team_id", None)
        ctid_i = int(ctid) if pd.notna(ctid) else None
        events.append(
            {
                "season": s,
                "asOfWeek": int(row.as_of_week),
                "teamId": str(tid),
                "ownerName": managers.get(tid) or None,
                "teamName": team_names.get(tid) or None,
                "counterpartyTeamId": str(ctid_i) if ctid_i is not None else None,
                "counterpartyOwnerName": (
                    managers.get(ctid_i) if ctid_i is not None else None
                ),
                "direction": json_safe(row.direction),
                "playerName": _ascii(row.player_name),
                "label": _ascii(row.label),
                "packageId": json_safe(getattr(row, "package_id", None)),
                "kind": json_safe(getattr(row, "kind", None)),
                "overallPick": json_safe(getattr(row, "overall_pick", None)),
                "pickLabel": json_safe(getattr(row, "pick_label", None)),
            }
        )
    return {
        "season": s,
        "asOfWeek": w,
        "exact": exact,
        "generatedFrom": LEAGUE_TRADE_EVENTS_CSV,
        "events": events,
    }


def build_snapshot(
    season: int | None = None,
    week: int | None = None,
) -> dict[str, Any]:
    s, w = resolve_season_week(season, week)
    return {
        "season": s,
        "asOfWeek": w,
        "odds": build_weekly(s, w),
        "injuries": build_injuries(s, w, exact=False),
        "trades": build_trades(s, w, exact=False),
    }


def latest_finish_season() -> int:
    """Latest season with a finish table.

    This board compares projected finish against actual finish, so it only
    exists once a season is over. Defaulting it to the newest season in the odds
    CSV would blank the board the moment an upcoming season is published.
    """
    finish = load_csv(LEAGUE_FINISH_PROJ_VS_ACTUAL_CSV)
    seasons = sorted(int(s) for s in finish["season"].dropna().unique())
    if not seasons:
        raise FileNotFoundError("No seasons in finish table")
    return seasons[-1]


def build_season_forekast(season: int | None = None) -> dict[str, Any]:
    """Adapter for the site's Henry ForeKast finish table."""
    s = int(season) if season is not None else latest_finish_season()
    finish = load_csv(LEAGUE_FINISH_PROJ_VS_ACTUAL_CSV)
    sdf = finish[finish["season"] == s].copy()
    managers = _manager_lookup(s)
    teams: list[dict[str, Any]] = []
    for row in sdf.sort_values("our_rs").itertuples(index=False):
        tid = int(row.team_id)
        our_rs = json_safe(row.our_rs)
        act_rs = json_safe(row.act_rs)
        espn = json_safe(row.espn_bos)
        our_po = json_safe(row.our_po)
        act_po = json_safe(row.act_po)
        sim_w = json_safe(row.sim_wins_mean)
        act_w = json_safe(row.wins)
        rs_delta = None
        if our_rs is not None and act_rs is not None:
            rs_delta = int(act_rs) - int(our_rs)
        po_delta = None
        if our_po is not None and act_po is not None:
            po_delta = int(act_po) - int(our_po)
        espn_delta = None
        if espn is not None and act_rs is not None:
            espn_delta = int(act_rs) - int(espn)
        wins_delta = None
        if sim_w is not None and act_w is not None:
            wins_delta = float(act_w) - float(sim_w)
        teams.append(
            {
                "season": s,
                "teamId": str(tid),
                "ownerName": managers.get(tid) or None,
                "teamName": _ascii(row.team),
                "espnBosRank": espn,
                "henryRegularSeasonRank": our_rs,
                "actualRegularSeasonRank": act_rs,
                "henryRegularSeasonDelta": rs_delta,
                "espnBosRegularSeasonDelta": espn_delta,
                "henryPlayoffRank": our_po,
                "actualPlayoffRank": act_po,
                "playoffDelta": po_delta,
                "simulatedWins": sim_w,
                "actualWins": act_w,
                "winsDelta": wins_delta,
                "playoffOddsPct": json_safe(row.playoff_odds),
                "titleOddsPct": json_safe(row.title_odds),
                "henryRegularSeasonMae": json_safe(getattr(row, "err_our_rs", None)),
                "espnBosRegularSeasonMae": json_safe(getattr(row, "err_espn_rs", None)),
                "henryPlayoffMae": json_safe(getattr(row, "err_our_po", None)),
            }
        )
    return {
        "season": s,
        "generatedFrom": LEAGUE_FINISH_PROJ_VS_ACTUAL_CSV,
        "teams": teams,
    }


CALIBRATION_NOTE = (
    "What each letter has actually been worth across completed seasons "
    "(2018-2025). Draft grades are weak predictors by nature: the strongest "
    "draft-time signal explains roughly a tenth of the variance in wins, and a "
    "manager's grade one year barely predicts the next. Read the letters as a "
    "description of the draft, not a forecast of the season."
)


def draft_grade_seasons() -> list[int]:
    grades = load_csv(LEAGUE_DRAFT_GRADES_CSV)
    return sorted(int(s) for s in grades["season"].dropna().unique())


def _calibration_payload() -> dict[str, Any]:
    try:
        frame = load_csv(LEAGUE_DRAFT_GRADE_CALIBRATION_CSV)
    except FileNotFoundError:
        return {"note": CALIBRATION_NOTE, "value": [], "roster": []}
    out: dict[str, Any] = {"note": CALIBRATION_NOTE}
    for kind in ("value", "roster"):
        sdf = frame[frame["grade_kind"] == kind]
        out[kind] = [
            {
                "grade": json_safe(r.grade),
                "nTeamSeasons": json_safe(r.n_team_seasons),
                "avgWins": json_safe(r.avg_wins),
                "avgFinalRank": json_safe(r.avg_final_rank),
                "playoffRatePct": json_safe(r.playoff_rate),
                "titleRatePct": json_safe(r.title_rate),
            }
            for r in sdf.itertuples(index=False)
        ]
    return out


def _pick_payload(row: Any, prefix: str) -> dict[str, Any]:
    return {
        "playerName": _ascii(getattr(row, f"{prefix}_player", "")),
        "position": json_safe(getattr(row, f"{prefix}_position", None)),
        "overallPick": json_safe(getattr(row, f"{prefix}_overall", None)),
        "adpRank": json_safe(getattr(row, f"{prefix}_adp", None)),
        "valueAdded": json_safe(getattr(row, f"{prefix}_value", None)),
    }


def build_draft_grades(season: int | None = None) -> dict[str, Any]:
    """Draft value and roster grades, with the calibration that qualifies them.

    Both grades are schedule-free and available as soon as a draft ends. Playoff
    and title odds stay null until that season's odds have been published.
    """
    grades = load_csv(LEAGUE_DRAFT_GRADES_CSV)
    seasons = draft_grade_seasons()
    if not seasons:
        raise FileNotFoundError("No graded drafts available")
    s = int(season) if season is not None else seasons[-1]
    sdf = grades[grades["season"] == s].copy()
    if sdf.empty:
        raise FileNotFoundError(f"No draft grades for season {s}")
    sdf = sdf.sort_values("value_points", ascending=False)
    managers = _manager_lookup(s)

    teams: list[dict[str, Any]] = []
    for row in sdf.itertuples(index=False):
        tid = int(row.team_id)
        teams.append(
            {
                "teamId": str(tid),
                "ownerName": managers.get(tid) or _ascii(row.manager_name) or None,
                "teamName": _ascii(row.team_name),
                "valueGrade": json_safe(row.value_grade),
                "valuePoints": json_safe(row.value_points),
                "valueZ": json_safe(row.value_z),
                "rosterGrade": json_safe(row.roster_grade),
                "rosterPointsPerWeek": json_safe(row.roster_points_per_week),
                "rosterZ": json_safe(row.roster_z),
                "playoffOddsPct": json_safe(row.playoff_odds),
                "titleOddsPct": json_safe(row.title_odds),
                "skillPicks": json_safe(row.n_skill_picks),
                "firstPick": json_safe(row.first_pick),
                "bestPick": _pick_payload(row, "best_pick"),
                "biggestReach": _pick_payload(row, "reach"),
            }
        )
    return {
        "season": s,
        "availableSeasons": seasons,
        "generatedFrom": LEAGUE_DRAFT_GRADES_CSV,
        "calibration": _calibration_payload(),
        "teams": teams,
    }


def health_payload() -> dict[str, Any]:
    from rookie_ppr.api.csv_store import csv_path

    names = [
        LEAGUE_WEEKLY_ODDS_CSV,
        LEAGUE_INJURY_EVENTS_CSV,
        LEAGUE_TRADE_EVENTS_CSV,
        LEAGUE_FINISH_PROJ_VS_ACTUAL_CSV,
        LEAGUE_TEAMS_CSV,
        LEAGUE_MANAGERS_CSV,
        LEAGUE_DRAFT_GRADES_CSV,
        LEAGUE_DRAFT_GRADE_CALIBRATION_CSV,
    ]
    artifacts = {}
    for name in names:
        path = csv_path(name)
        artifacts[name] = {
            "exists": path.exists(),
            "bytes": path.stat().st_size if path.exists() else 0,
        }
    seasons: list[int] = []
    latest: dict[str, Any] = {}
    try:
        seasons = available_seasons()
        if seasons:
            s = seasons[-1]
            w = latest_week(s)
            latest = {"season": s, "asOfWeek": w}
    except FileNotFoundError:
        pass
    return {
        "ok": all(a["exists"] for a in artifacts.values()),
        "service": "rookie-ppr-forekast",
        "artifacts": artifacts,
        "seasons": seasons,
        "latest": latest,
    }
