"""Pull this ESPN league's history into tidy CSVs.

Six tables come out of here: per-season settings and scoring rules, teams with
final records and seeds, every matchup with actual scores, weekly rosters with
lineup slots and ESPN's own weekly projections (2020+ only), and draft results.
Everything is keyed on ESPN ids and carries ``gsis_id`` where a match exists, so
the league tables join straight onto the veteran/rookie projections.
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from rookie_ppr.config import ensure_directories
from rookie_ppr.league.config import (
    BENCH_SLOT_IDS,
    CSV_OUTPUT_DIR,
    ESPN_STAT_NAMES,
    ESPN_SWID,
    LEAGUE_DRAFT_CSV,
    LEAGUE_MATCHUPS_CSV,
    LEAGUE_ROSTER_SEASON_MIN,
    LEAGUE_ROSTERS_CSV,
    LEAGUE_SCORING_CSV,
    LEAGUE_SEASONS,
    LEAGUE_SETTINGS_CSV,
    LEAGUE_TEAMS_CSV,
    LINEUP_SLOT_NAMES,
    MAX_SCORING_PERIOD,
    NFL_TEAM_BY_ESPN_ID,
    POSITION_BY_ESPN_ID,
    STAT_SOURCE_ACTUAL,
    STAT_SOURCE_PROJECTED,
)
from rookie_ppr.league.espn_api import endpoint_style, fetch_player_names, fetch_view
from rookie_ppr.league.standings import reconcile_starter_points, validate_seeding_rule
from rookie_ppr.nflverse_http import try_read_release_csv
from rookie_ppr.utils import normalize_name

SETTINGS_COLS = [
    "season",
    "league_name",
    "size",
    "reg_weeks",
    "playoff_teams",
    "playoff_seeding_rule",
    "playoff_reseed",
    "divisions_json",
    "lineup_slots_json",
    "endpoint",
]
TEAM_COLS = [
    "season",
    "team_id",
    "team_name",
    "team_abbrev",
    "division_id",
    "is_my_team",
    "wins",
    "losses",
    "ties",
    "points_for",
    "points_against",
    "playoff_seed",
    "final_rank",
]
MATCHUP_COLS = [
    "season",
    "week",
    "team_id",
    "opp_team_id",
    "is_home",
    "points",
    "opp_points",
    "result",
    "adjustment",
    "adjustment_reason",
    "playoff_tier",
    "is_bye",
]
ROSTER_COLS = [
    "season",
    "week",
    "team_id",
    "espn_player_id",
    "gsis_id",
    "player_name",
    "position",
    "pro_team",
    "lineup_slot_id",
    "lineup_slot",
    "is_starter",
    "actual_points",
    "espn_projected_points",
]
DRAFT_COLS = [
    "season",
    "overall_pick",
    "round",
    "round_pick",
    "team_id",
    "espn_player_id",
    "gsis_id",
    "player_name",
    "position",
    "pro_team",
    "is_keeper",
    "bid_amount",
]


def _team_name(team: dict) -> str:
    name = team.get("name")
    if name and str(name).strip():
        return str(name).strip()
    joined = f"{team.get('location', '')} {team.get('nickname', '')}".strip()
    return joined or f"Team {team.get('id')}"


def _is_my_team(team: dict) -> bool:
    if not ESPN_SWID:
        return False
    owners = team.get("owners") or []
    return ESPN_SWID in owners or ESPN_SWID == (team.get("primaryOwner") or "")


# ---------------------------------------------------------------- settings


def build_settings_and_scoring(
    seasons: list[int], *, force: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    settings_rows: list[dict] = []
    scoring_rows: list[dict] = []

    for season in seasons:
        node = fetch_view(season, ["mSettings"], force=force)
        settings = node.get("settings") or {}
        schedule = settings.get("scheduleSettings") or {}
        roster = (settings.get("rosterSettings") or {}).get("lineupSlotCounts") or {}
        slots = {LINEUP_SLOT_NAMES.get(int(k), k): v for k, v in roster.items() if v}

        settings_rows.append(
            {
                "season": season,
                "league_name": settings.get("name"),
                "size": settings.get("size"),
                "reg_weeks": schedule.get("matchupPeriodCount"),
                "playoff_teams": schedule.get("playoffTeamCount"),
                "playoff_seeding_rule": schedule.get("playoffSeedingRule"),
                "playoff_reseed": schedule.get("playoffReseed"),
                "divisions_json": json.dumps(
                    [
                        {"id": d.get("id"), "name": d.get("name"), "size": d.get("size")}
                        for d in schedule.get("divisions") or []
                    ]
                ),
                "lineup_slots_json": json.dumps(slots),
                "endpoint": endpoint_style(season),
            }
        )

        for item in (settings.get("scoringSettings") or {}).get("scoringItems") or []:
            stat_id = item.get("statId")
            points = item.get("points")
            if stat_id is None or not points:
                continue
            scoring_rows.append(
                {
                    "season": season,
                    "stat_id": int(stat_id),
                    "stat_name": ESPN_STAT_NAMES.get(int(stat_id)),
                    "points": float(points),
                }
            )

    settings_df = pd.DataFrame(settings_rows, columns=SETTINGS_COLS)
    scoring_df = pd.DataFrame(
        scoring_rows, columns=["season", "stat_id", "stat_name", "points"]
    )
    return settings_df, scoring_df


# ------------------------------------------------------------------- teams


def build_teams(seasons: list[int], *, force: bool = False) -> pd.DataFrame:
    rows: list[dict] = []
    for season in seasons:
        node = fetch_view(season, ["mTeam"], force=force)
        for team in node.get("teams") or []:
            overall = ((team.get("record") or {}).get("overall") or {})
            rows.append(
                {
                    "season": season,
                    "team_id": team.get("id"),
                    "team_name": _team_name(team),
                    "team_abbrev": (team.get("abbrev") or "").strip() or None,
                    "division_id": team.get("divisionId"),
                    "is_my_team": _is_my_team(team),
                    "wins": overall.get("wins"),
                    "losses": overall.get("losses"),
                    "ties": overall.get("ties"),
                    "points_for": overall.get("pointsFor"),
                    "points_against": overall.get("pointsAgainst"),
                    "playoff_seed": team.get("playoffSeed"),
                    "final_rank": team.get("rankCalculatedFinal"),
                }
            )
    return pd.DataFrame(rows, columns=TEAM_COLS)


# ---------------------------------------------------------------- matchups


def build_matchups(seasons: list[int], *, force: bool = False) -> pd.DataFrame:
    rows: list[dict] = []
    for season in seasons:
        node = fetch_view(season, ["mMatchupScore"], force=force)
        for game in node.get("schedule") or []:
            week = game.get("matchupPeriodId")
            tier = game.get("playoffTierType")
            home = game.get("home") or {}
            away = game.get("away") or {}
            # Playoff byes come back with only one side populated.
            sides = [(home, away, True), (away, home, False)]
            for side, other, is_home in sides:
                if not side:
                    continue
                points = side.get("totalPoints")
                opp_points = other.get("totalPoints") if other else None
                if not other:
                    result = "BYE"
                elif points is None or opp_points is None:
                    result = None
                elif points > opp_points:
                    result = "W"
                elif points < opp_points:
                    result = "L"
                else:
                    result = "T"
                rows.append(
                    {
                        "season": season,
                        "week": week,
                        "team_id": side.get("teamId"),
                        "opp_team_id": other.get("teamId") if other else None,
                        "is_home": is_home,
                        "points": points,
                        "opp_points": opp_points,
                        "result": result,
                        # Commissioner corrections land here, not in the roster,
                        # so a team's score can differ from its started players.
                        "adjustment": side.get("adjustment") or 0.0,
                        "adjustment_reason": side.get("adjustmentReason"),
                        "playoff_tier": tier,
                        "is_bye": not bool(other),
                    }
                )
    return pd.DataFrame(rows, columns=MATCHUP_COLS)


# ----------------------------------------------------------------- rosters


def _weekly_points(player: dict, week: int) -> tuple[float | None, float | None]:
    """(actual, ESPN projection) for this player in this week."""
    actual = projected = None
    for stat in player.get("stats") or []:
        if stat.get("scoringPeriodId") != week:
            continue
        total = stat.get("appliedTotal")
        if total is None:
            continue
        source = stat.get("statSourceId")
        if source == STAT_SOURCE_ACTUAL:
            actual = float(total)
        elif source == STAT_SOURCE_PROJECTED:
            projected = float(total)
    return actual, projected


def build_rosters(
    seasons: list[int], *, force: bool = False, max_week: int = MAX_SCORING_PERIOD
) -> pd.DataFrame:
    """Weekly lineups with actual and ESPN-projected points.

    Skips seasons before :data:`LEAGUE_ROSTER_SEASON_MIN`, where ESPN returns a
    frozen end-of-season roster for every week.
    """
    rows: list[dict] = []
    for season in seasons:
        if season < LEAGUE_ROSTER_SEASON_MIN:
            continue
        for week in range(1, max_week + 1):
            node = fetch_view(
                season,
                ["mBoxscore", "mMatchupScore"],
                scoring_period=week,
                matchup_period=week,
                force=force,
            )
            for game in node.get("schedule") or []:
                if game.get("matchupPeriodId") != week:
                    continue
                for key in ("home", "away"):
                    side = game.get(key) or {}
                    if not side:
                        continue
                    team_id = side.get("teamId")
                    roster = side.get("rosterForCurrentScoringPeriod") or side.get(
                        "rosterForMatchupPeriod"
                    ) or {}
                    for entry in roster.get("entries") or []:
                        player = (entry.get("playerPoolEntry") or {}).get("player") or {}
                        slot_id = entry.get("lineupSlotId")
                        actual, projected = _weekly_points(player, week)
                        rows.append(
                            {
                                "season": season,
                                "week": week,
                                "team_id": team_id,
                                "espn_player_id": entry.get("playerId")
                                or player.get("id"),
                                "player_name": player.get("fullName"),
                                "position": POSITION_BY_ESPN_ID.get(
                                    player.get("defaultPositionId")
                                ),
                                "pro_team": NFL_TEAM_BY_ESPN_ID.get(
                                    player.get("proTeamId")
                                ),
                                "lineup_slot_id": slot_id,
                                "lineup_slot": LINEUP_SLOT_NAMES.get(slot_id),
                                "is_starter": slot_id not in BENCH_SLOT_IDS,
                                "actual_points": actual,
                                "espn_projected_points": projected,
                            }
                        )
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=ROSTER_COLS)
    return attach_gsis_ids(df)[ROSTER_COLS]


# ------------------------------------------------------------------- draft


def _team_defense_info(player_id: int) -> dict:
    """ESPN encodes a team defense as player id ``-16000 - proTeamId``."""
    pro_team_id = -int(player_id) - 16000
    team = NFL_TEAM_BY_ESPN_ID.get(pro_team_id)
    return {
        "full_name": f"{team} D/ST" if team else None,
        "default_position_id": 16,
        "pro_team_id": pro_team_id,
    }


def build_draft(seasons: list[int], *, force: bool = False) -> pd.DataFrame:
    rows: list[dict] = []
    for season in seasons:
        node = fetch_view(season, ["mDraftDetail"], force=force)
        detail = node.get("draftDetail") or {}
        if not detail.get("drafted"):
            continue
        picks = [p for p in detail.get("picks") or [] if p.get("playerId")]
        names = fetch_player_names(
            season, [p["playerId"] for p in picks if p["playerId"] > 0]
        )
        for pick in picks:
            pid = pick.get("playerId")
            if pid < 0:
                info = _team_defense_info(pid)
            else:
                info = names.get(int(pid), {})
            rows.append(
                {
                    "season": season,
                    "overall_pick": pick.get("overallPickNumber"),
                    "round": pick.get("roundId"),
                    "round_pick": pick.get("roundPickNumber"),
                    "team_id": pick.get("teamId"),
                    "espn_player_id": pid,
                    "player_name": info.get("full_name"),
                    "position": POSITION_BY_ESPN_ID.get(info.get("default_position_id")),
                    "pro_team": NFL_TEAM_BY_ESPN_ID.get(info.get("pro_team_id")),
                    "is_keeper": bool(pick.get("keeper")),
                    "bid_amount": pick.get("bidAmount"),
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=DRAFT_COLS)
    return attach_gsis_ids(df)[DRAFT_COLS]


# ------------------------------------------------------- id reconciliation


def espn_to_gsis_map() -> pd.DataFrame:
    """ESPN player id -> gsis_id, plus a normalized name for fallback joins."""
    players = try_read_release_csv(
        [("players", "players.csv"), ("players", "players.csv.gz")]
    )
    empty = pd.DataFrame(columns=["espn_player_id", "gsis_id", "name_norm"])
    if players.empty:
        return empty

    espn_col = next(
        (c for c in ("espn_id", "espnId", "espn_player_id") if c in players.columns), None
    )
    gsis_col = next((c for c in ("gsis_id", "gsisId") if c in players.columns), None)
    name_col = next(
        (c for c in ("display_name", "full_name", "football_name", "player_name")
         if c in players.columns),
        None,
    )
    if not gsis_col:
        return empty

    keep = [c for c in (espn_col, gsis_col, name_col) if c]
    out = players[keep].copy()
    rename = {gsis_col: "gsis_id"}
    if espn_col:
        rename[espn_col] = "espn_player_id"
    if name_col:
        rename[name_col] = "player_name_src"
    out = out.rename(columns=rename)

    if "espn_player_id" in out.columns:
        out["espn_player_id"] = pd.to_numeric(
            out["espn_player_id"], errors="coerce"
        ).astype("Int64")
    else:
        out["espn_player_id"] = pd.Series([pd.NA] * len(out), dtype="Int64")
    out["name_norm"] = (
        out["player_name_src"].map(normalize_name) if "player_name_src" in out.columns else ""
    )
    return out[["espn_player_id", "gsis_id", "name_norm"]].dropna(subset=["gsis_id"])


def attach_gsis_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``gsis_id`` by ESPN id, falling back to a normalized-name match."""
    out = df.copy()
    out["gsis_id"] = pd.NA
    id_map = espn_to_gsis_map()
    if id_map.empty:
        return out

    out["espn_player_id"] = pd.to_numeric(
        out["espn_player_id"], errors="coerce"
    ).astype("Int64")

    paired = id_map.dropna(subset=["espn_player_id"]).drop_duplicates(
        subset=["espn_player_id"]
    )
    by_id = {int(k): v for k, v in zip(paired["espn_player_id"], paired["gsis_id"])}
    out["gsis_id"] = [
        by_id.get(int(pid)) if pd.notna(pid) else None for pid in out["espn_player_id"]
    ]

    # D/ST rows have no nflverse player, so only chase real names.
    missing = out["gsis_id"].isna() & out["player_name"].notna()
    if missing.any():
        named = id_map[id_map["name_norm"].astype(bool)].drop_duplicates(
            subset=["name_norm"]
        )
        by_name = dict(zip(named["name_norm"], named["gsis_id"]))
        out.loc[missing, "gsis_id"] = [
            by_name.get(normalize_name(name))
            for name in out.loc[missing, "player_name"]
        ]
    return out


# ------------------------------------------------------------------ driver


def ingest_league(
    seasons: list[int] | None = None,
    *,
    force: bool = False,
    include_rosters: bool = True,
) -> dict[str, pd.DataFrame]:
    seasons = list(seasons or LEAGUE_SEASONS)
    settings, scoring = build_settings_and_scoring(seasons, force=force)
    tables = {
        "settings": settings,
        "scoring": scoring,
        "teams": build_teams(seasons, force=force),
        "matchups": build_matchups(seasons, force=force),
        "draft": build_draft(seasons, force=force),
    }
    tables["rosters"] = (
        build_rosters(seasons, force=force)
        if include_rosters
        else pd.DataFrame(columns=ROSTER_COLS)
    )
    return tables


def save_league_tables(tables: dict[str, pd.DataFrame]) -> dict[str, str]:
    ensure_directories()
    targets = {
        "settings": LEAGUE_SETTINGS_CSV,
        "scoring": LEAGUE_SCORING_CSV,
        "teams": LEAGUE_TEAMS_CSV,
        "matchups": LEAGUE_MATCHUPS_CSV,
        "rosters": LEAGUE_ROSTERS_CSV,
        "draft": LEAGUE_DRAFT_CSV,
    }
    written: dict[str, str] = {}
    for key, filename in targets.items():
        df = tables.get(key)
        if df is None:
            continue
        path = CSV_OUTPUT_DIR / filename
        df.to_csv(path, index=False)
        written[key] = str(path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest ESPN fantasy league history into data/output/csv"
    )
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        default=list(LEAGUE_SEASONS),
        help="Seasons to pull (default: every completed season)",
    )
    parser.add_argument(
        "--force", action="store_true", help="Refetch instead of using the JSON cache"
    )
    parser.add_argument(
        "--skip-rosters",
        action="store_true",
        help="Skip the per-week roster pull (the slow part)",
    )
    args = parser.parse_args()

    tables = ingest_league(
        args.seasons, force=args.force, include_rosters=not args.skip_rosters
    )
    written = save_league_tables(tables)

    for key in ("settings", "scoring", "teams", "matchups", "rosters", "draft"):
        df = tables.get(key)
        if df is None:
            continue
        print(f"{key:<9} rows={len(df):<6} -> {written.get(key, '(not written)')}")

    rosters = tables["rosters"]
    if not rosters.empty:
        seasons_with_rosters = sorted(rosters["season"].unique())
        matched = rosters["gsis_id"].notna().mean()
        skill = rosters[rosters["position"].isin(["QB", "RB", "WR", "TE"])]
        skill_matched = skill["gsis_id"].notna().mean() if not skill.empty else float("nan")
        print(
            f"\nroster seasons: {seasons_with_rosters[0]}-{seasons_with_rosters[-1]}"
            f" | gsis matched: all={matched:.1%} skill-position={skill_matched:.1%}"
        )
        proj = rosters["espn_projected_points"].notna().mean()
        print(f"weeks with an ESPN projection on the player row: {proj:.1%}")

        print("\nstarter points vs ESPN team score:")
        print(
            reconcile_starter_points(rosters, tables["matchups"]).to_string(index=False)
        )

    print("\nseeding rule check (derived seeds vs ESPN's):")
    print(validate_seeding_rule(tables["teams"]).to_string(index=False))


if __name__ == "__main__":
    main()
