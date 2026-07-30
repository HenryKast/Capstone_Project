"""Build player × season panel with next-season PPR labels."""
from __future__ import annotations

import numpy as np
import pandas as pd

from rookie_ppr.ingest_nfl import (
    _ppr_series,
    load_player_season_stats,
    load_rosters,
    load_schedules,
    load_team_season_stats,
)
from rookie_ppr.ingest_opportunity import build_team_offensive_environment
from rookie_ppr.features_sos import compute_team_sos
from rookie_ppr.utils import normalize_name, normalize_position, normalize_team_abbr
from rookie_ppr.veteran.config import (
    SKILL_POSITIONS,
    STAT_LABEL_COLS,
    TARGET,
    VET_SEASON_MAX,
    VET_SEASON_MIN,
)
from rookie_ppr.veteran.pbp_context import (
    attach_next_season_opponent_defense,
    attach_schedule_opponent_defense,
    build_team_pbp_metrics,
)
from rookie_ppr.veteran.role_context import (
    build_depth_features,
    build_ngs_features,
    build_snap_features,
)


def _season_list(start: int = VET_SEASON_MIN, end: int = VET_SEASON_MAX) -> list[int]:
    return list(range(int(start), int(end) + 1))


def _player_season_table(stats: pd.DataFrame) -> pd.DataFrame:
    """One row per gsis_id × season with counting stats + PPR."""
    if stats.empty:
        return pd.DataFrame()

    s = stats.copy()
    id_col = next((c for c in ("gsis_id", "player_id", "gsisId") if c in s.columns), None)
    if id_col is None or "season" not in s.columns:
        return pd.DataFrame()

    if id_col != "gsis_id":
        s = s.rename(columns={id_col: "gsis_id"})

    s["position"] = s["position"].map(normalize_position) if "position" in s.columns else pd.NA
    s = s[s["position"].isin(SKILL_POSITIONS)].copy()
    if s.empty:
        return pd.DataFrame()

    team_col = next((c for c in ("recent_team", "team", "posteam") if c in s.columns), None)
    s["team"] = s[team_col].map(normalize_team_abbr) if team_col else pd.NA

    name_col = next((c for c in ("player_display_name", "player_name", "full_name") if c in s.columns), None)
    s["player_name"] = s[name_col].astype(str) if name_col else ""
    s["player_name_norm"] = s["player_name"].map(normalize_name)

    s["ppr"] = pd.to_numeric(_ppr_series(s), errors="coerce")
    games_col = next((c for c in ("games", "season_games", "games_played") if c in s.columns), None)
    s["games"] = pd.to_numeric(s[games_col], errors="coerce") if games_col else pd.NA

    for col in (
        "carries",
        "targets",
        "receptions",
        "rushing_yards",
        "receiving_yards",
        "passing_yards",
        "rushing_tds",
        "receiving_tds",
        "passing_tds",
        "interceptions",
        "fumbles_lost",
    ):
        if col in s.columns:
            s[col] = pd.to_numeric(s[col], errors="coerce")
        else:
            s[col] = np.nan

    # If weekly was already aggregated, may have duplicate gsis/season/team from name flips —
    # collapse to one row per player-season (prefer max PPR row's team).
    s["season"] = pd.to_numeric(s["season"], errors="coerce")
    s = s.dropna(subset=["gsis_id", "season", "position"])
    s = s.sort_values(["gsis_id", "season", "ppr"], ascending=[True, True, False])
    keep = [
        "gsis_id",
        "player_name",
        "player_name_norm",
        "position",
        "season",
        "team",
        "ppr",
        "games",
        "carries",
        "targets",
        "receptions",
        "rushing_yards",
        "receiving_yards",
        "passing_yards",
        "rushing_tds",
        "receiving_tds",
        "passing_tds",
        "interceptions",
        "fumbles_lost",
    ]
    out = s[keep].drop_duplicates(subset=["gsis_id", "season"], keep="first")
    out["touches"] = out["carries"].fillna(0) + out["receptions"].fillna(0)
    both_miss = out["carries"].isna() & out["receptions"].isna()
    out.loc[both_miss, "touches"] = np.nan
    out["ppr_per_game"] = out["ppr"] / out["games"].replace({0: np.nan})
    return out.reset_index(drop=True)


def _age_lookup(rosters: pd.DataFrame) -> pd.DataFrame:
    """gsis_id → birth_date for age-at-season."""
    if rosters is None or rosters.empty:
        return pd.DataFrame(columns=["gsis_id", "birth_date"])
    r = rosters.copy()
    id_col = next((c for c in ("gsis_id", "player_id") if c in r.columns), None)
    if id_col is None or "birth_date" not in r.columns:
        return pd.DataFrame(columns=["gsis_id", "birth_date"])
    r = r[[id_col, "birth_date"]].dropna(subset=[id_col]).copy()
    if id_col != "gsis_id":
        r = r.rename(columns={id_col: "gsis_id"})
    r["birth_date"] = pd.to_datetime(r["birth_date"], errors="coerce")
    return r.dropna(subset=["birth_date"]).drop_duplicates("gsis_id", keep="first")


def build_veteran_panel(
    *,
    season_min: int = VET_SEASON_MIN,
    season_max: int = VET_SEASON_MAX,
) -> pd.DataFrame:
    """
    Player×season rows with Phase-1/2 context and ppr_next = PPR in season+1.

    Rows without a completed next season keep ppr_next null (scorable later).
    """
    seasons = _season_list(season_min, season_max)
    # Schedules one year past max for next-season opponent difficulty
    schedule_seasons = _season_list(season_min, season_max + 1)
    print(f"  loading player stats {seasons[0]}–{seasons[-1]} ...")
    stats = load_player_season_stats(seasons)
    print(f"  player-stat rows: {len(stats)}")
    panel = _player_season_table(stats)
    print(f"  skill player-seasons: {len(panel)}")
    if panel.empty:
        return panel

    print("  loading schedules / SOS ...")
    schedules = load_schedules(schedule_seasons)
    sos = compute_team_sos(schedules)
    print(f"  sos rows: {len(sos)}")

    print("  loading team offense ...")
    team_stats = load_team_season_stats(seasons)
    offense = build_team_offensive_environment(team_stats)
    print(f"  offense env rows: {len(offense)}")

    print("  loading pbp defense / OL proxies (Phase 2) ...")
    team_pbp = build_team_pbp_metrics(seasons)
    print(f"  team pbp metric rows: {len(team_pbp)}")
    opp_def = attach_schedule_opponent_defense(schedules, team_pbp)
    next_opp_def = attach_next_season_opponent_defense(schedules, team_pbp)
    print(f"  opp-def rows: {len(opp_def)}; next-opp-def rows: {len(next_opp_def)}")

    print("  loading snaps / depth / NGS (Phase 3) ...")
    snaps = build_snap_features(seasons)
    depth = build_depth_features(seasons)
    ngs = build_ngs_features(seasons)
    print(f"  snap rows: {len(snaps)}; depth rows: {len(depth)}; ngs rows: {len(ngs)}")

    print("  loading rosters (age) ...")
    # Rosters can be large; sample recent + a few early years for birth dates
    roster_years = sorted(set([season_min, 2005, 2010, 2015, 2020, season_max] + seasons[-8:]))
    roster_years = [y for y in roster_years if season_min <= y <= season_max]
    rosters = load_rosters(roster_years)
    ages = _age_lookup(rosters)

    # Next-season labels: PPR plus the components a week-by-week build needs
    label_cols = ["ppr", *STAT_LABEL_COLS]
    label_cols = [c for c in label_cols if c in panel.columns]
    nxt = panel[["gsis_id", "season", *label_cols]].rename(
        columns={
            "season": "target_season",
            "ppr": TARGET,
            **{c: f"{c}_next" for c in label_cols if c != "ppr"},
        }
    )
    panel = panel.copy()
    panel["target_season"] = pd.to_numeric(panel["season"], errors="coerce") + 1
    panel = panel.merge(nxt, how="left", on=["gsis_id", "target_season"])

    # Team context for season T
    if not sos.empty:
        panel = panel.merge(
            sos.rename(columns={"team": "team"}),
            how="left",
            on=["season", "team"],
        )
    else:
        panel["sos_opp_win_pct"] = np.nan
        panel["games_scheduled"] = np.nan

    if not offense.empty:
        panel = panel.merge(offense, how="left", on=["season", "team"])
    else:
        panel["off_pass_yards"] = np.nan
        panel["off_rush_yards"] = np.nan
        panel["off_pass_rate_proxy"] = np.nan

    pbp_cols = [
        "def_epa_per_play",
        "def_pass_epa",
        "def_rush_epa",
        "off_epa_per_play",
        "off_pass_epa",
        "off_rush_epa",
        "ol_sack_rate",
        "ol_qb_hit_rate",
    ]
    if not team_pbp.empty:
        keep = ["season", "team"] + [c for c in pbp_cols if c in team_pbp.columns]
        panel = panel.merge(team_pbp[keep], how="left", on=["season", "team"])
    else:
        for c in pbp_cols:
            panel[c] = np.nan

    if not opp_def.empty:
        panel = panel.merge(opp_def, how="left", on=["season", "team"])
    else:
        panel["opp_def_epa_faced"] = np.nan
        panel["opp_def_pass_epa_faced"] = np.nan
        panel["opp_def_rush_epa_faced"] = np.nan

    if not next_opp_def.empty:
        panel = panel.merge(next_opp_def, how="left", on=["season", "team"])
    else:
        panel["next_opp_def_epa_faced"] = np.nan
        panel["next_opp_def_pass_epa_faced"] = np.nan
        panel["next_opp_def_rush_epa_faced"] = np.nan

    if not snaps.empty:
        panel = panel.merge(snaps, how="left", on=["gsis_id", "season"])
    else:
        panel["off_snap_pct"] = np.nan
        panel["off_snaps"] = np.nan
        panel["snap_games"] = np.nan

    if not depth.empty:
        panel = panel.merge(depth, how="left", on=["gsis_id", "season"])
    else:
        panel["depth_rank"] = np.nan

    if not ngs.empty:
        panel = panel.merge(ngs, how="left", on=["gsis_id", "season"])

    if not ages.empty:
        panel = panel.merge(ages, how="left", on="gsis_id")
        birth = panel["birth_date"]
        # Age as of Sept 1 of season year
        season_anchor = pd.to_datetime(panel["season"].astype("Int64").astype(str) + "-09-01", errors="coerce")
        panel["age"] = (season_anchor - birth).dt.days / 365.25
        panel = panel.drop(columns=["birth_date"])
    else:
        panel["age"] = np.nan

    return panel.sort_values(["position", "gsis_id", "season"]).reset_index(drop=True)
