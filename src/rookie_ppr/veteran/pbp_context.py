"""Phase 2: team defense + OL proxies from nflverse play-by-play (1999+)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from rookie_ppr.nflverse_http import try_read_release_csv
from rookie_ppr.utils import normalize_team_abbr

PBP_USECOLS = [
    "season",
    "season_type",
    "week",
    "posteam",
    "defteam",
    "epa",
    "play_type",
    "pass",
    "rush",
    "pass_attempt",
    "rush_attempt",
    "sack",
    "qb_hit",
    "complete_pass",
    "interception",
    "yards_gained",
]


def load_pbp_season(season: int) -> pd.DataFrame:
    """Load one season of pbp with a slim column set."""
    return try_read_release_csv(
        [
            ("pbp", f"play_by_play_{season}.csv.gz"),
            ("pbp", f"play_by_play_{season}.csv"),
        ],
        usecols=PBP_USECOLS,
    )


def _is_reg(df: pd.DataFrame) -> pd.Series:
    if "season_type" not in df.columns:
        return pd.Series(True, index=df.index)
    return df["season_type"].astype(str).str.upper().isin(["REG", "REGULAR"])


def build_team_pbp_metrics(seasons: list[int]) -> pd.DataFrame:
    """
    Per team-season:
      def_epa_per_play / def_pass_epa / def_rush_epa  (EPA allowed; higher = worse D)
      ol_sack_rate / ol_qb_hit_rate                   (pass-play pressure allowed)
      off_epa_per_play / off_pass_epa / off_rush_epa  (own offense EPA)
    """
    rows: list[pd.DataFrame] = []
    for season in seasons:
        print(f"    pbp {season} ...", flush=True)
        raw = load_pbp_season(int(season))
        if raw.empty:
            continue
        p = raw.copy()
        if "season" not in p.columns:
            p["season"] = season
        p = p.loc[_is_reg(p)].copy()
        if "week" in p.columns:
            p = p.loc[pd.to_numeric(p["week"], errors="coerce").fillna(99) <= 18].copy()

        p["posteam"] = p["posteam"].map(normalize_team_abbr) if "posteam" in p.columns else pd.NA
        p["defteam"] = p["defteam"].map(normalize_team_abbr) if "defteam" in p.columns else pd.NA
        p["epa"] = pd.to_numeric(p.get("epa"), errors="coerce")

        # Play filters
        if "pass" in p.columns:
            is_pass = pd.to_numeric(p["pass"], errors="coerce").fillna(0).astype(bool)
        elif "pass_attempt" in p.columns:
            is_pass = pd.to_numeric(p["pass_attempt"], errors="coerce").fillna(0).astype(bool)
        else:
            is_pass = p.get("play_type", pd.Series(index=p.index)).astype(str).str.lower().eq("pass")

        if "rush" in p.columns:
            is_rush = pd.to_numeric(p["rush"], errors="coerce").fillna(0).astype(bool)
        elif "rush_attempt" in p.columns:
            is_rush = pd.to_numeric(p["rush_attempt"], errors="coerce").fillna(0).astype(bool)
        else:
            is_rush = p.get("play_type", pd.Series(index=p.index)).astype(str).str.lower().eq("run")

        # Exclude non-scrimmage / no EPA
        scrimmage = p["epa"].notna() & p["posteam"].notna() & p["defteam"].notna()
        scrimmage &= is_pass | is_rush
        p = p.loc[scrimmage].copy()
        if p.empty:
            continue

        sack = pd.to_numeric(p["sack"], errors="coerce").fillna(0) if "sack" in p.columns else 0.0
        qb_hit = pd.to_numeric(p["qb_hit"], errors="coerce").fillna(0) if "qb_hit" in p.columns else 0.0
        p["_sack"] = sack
        p["_qb_hit"] = qb_hit
        p["_is_pass"] = is_pass.loc[p.index].astype(bool)
        p["_is_rush"] = is_rush.loc[p.index].astype(bool)

        # Defense: EPA allowed
        def_all = (
            p.groupby(["season", "defteam"], as_index=False)
            .agg(def_epa_per_play=("epa", "mean"), def_plays=("epa", "count"))
            .rename(columns={"defteam": "team"})
        )
        def_pass = (
            p.loc[p["_is_pass"]]
            .groupby(["season", "defteam"], as_index=False)
            .agg(def_pass_epa=("epa", "mean"))
            .rename(columns={"defteam": "team"})
        )
        def_rush = (
            p.loc[p["_is_rush"]]
            .groupby(["season", "defteam"], as_index=False)
            .agg(def_rush_epa=("epa", "mean"))
            .rename(columns={"defteam": "team"})
        )

        # Offense / OL
        off_all = (
            p.groupby(["season", "posteam"], as_index=False)
            .agg(off_epa_per_play=("epa", "mean"), off_plays=("epa", "count"))
            .rename(columns={"posteam": "team"})
        )
        off_pass = (
            p.loc[p["_is_pass"]]
            .groupby(["season", "posteam"], as_index=False)
            .agg(
                off_pass_epa=("epa", "mean"),
                ol_sack_rate=("_sack", "mean"),
                ol_qb_hit_rate=("_qb_hit", "mean"),
                pass_plays=("epa", "count"),
            )
            .rename(columns={"posteam": "team"})
        )
        off_rush = (
            p.loc[p["_is_rush"]]
            .groupby(["season", "posteam"], as_index=False)
            .agg(off_rush_epa=("epa", "mean"))
            .rename(columns={"posteam": "team"})
        )

        team = def_all.merge(def_pass, on=["season", "team"], how="outer")
        team = team.merge(def_rush, on=["season", "team"], how="outer")
        team = team.merge(off_all, on=["season", "team"], how="outer")
        team = team.merge(off_pass, on=["season", "team"], how="outer")
        team = team.merge(off_rush, on=["season", "team"], how="outer")
        rows.append(team)

    if not rows:
        return pd.DataFrame(
            columns=[
                "season",
                "team",
                "def_epa_per_play",
                "def_pass_epa",
                "def_rush_epa",
                "off_epa_per_play",
                "off_pass_epa",
                "off_rush_epa",
                "ol_sack_rate",
                "ol_qb_hit_rate",
            ]
        )
    out = pd.concat(rows, ignore_index=True)
    out["team"] = out["team"].map(normalize_team_abbr)
    return out.drop_duplicates(subset=["season", "team"], keep="last")


def attach_schedule_opponent_defense(
    schedules: pd.DataFrame,
    team_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each team-season, average opponents' defensive EPA allowed that season.
    Higher opp_def_epa_faced => easier slate (opponents allow more EPA).
    """
    cols = ["season", "team", "opp_def_epa_faced", "opp_def_pass_epa_faced", "opp_def_rush_epa_faced"]
    if schedules.empty or team_metrics.empty:
        return pd.DataFrame(columns=cols)

    s = schedules.copy()
    if "game_type" in s.columns:
        s = s[s["game_type"].astype(str).str.upper().isin(["REG", "REGULAR"])].copy()
    if "week" in s.columns:
        s = s[pd.to_numeric(s["week"], errors="coerce").fillna(99) <= 18].copy()

    needed = {"season", "home_team", "away_team"}
    if not needed.issubset(s.columns):
        return pd.DataFrame(columns=cols)

    s["home_team"] = s["home_team"].map(normalize_team_abbr)
    s["away_team"] = s["away_team"].map(normalize_team_abbr)
    s["season"] = pd.to_numeric(s["season"], errors="coerce")

    def_cols = ["def_epa_per_play", "def_pass_epa", "def_rush_epa"]
    d = team_metrics[["season", "team"] + [c for c in def_cols if c in team_metrics.columns]].copy()
    d["season"] = pd.to_numeric(d["season"], errors="coerce")

    home = s[["season", "home_team", "away_team"]].rename(
        columns={"home_team": "team", "away_team": "opp"}
    )
    away = s[["season", "away_team", "home_team"]].rename(
        columns={"away_team": "team", "home_team": "opp"}
    )
    matchups = pd.concat([home, away], ignore_index=True)
    matchups = matchups.merge(
        d.rename(columns={"team": "opp", **{c: f"opp_{c}" for c in def_cols if c in d.columns}}),
        how="left",
        on=["season", "opp"],
    )

    agg_map = {}
    if "opp_def_epa_per_play" in matchups.columns:
        agg_map["opp_def_epa_faced"] = ("opp_def_epa_per_play", "mean")
    if "opp_def_pass_epa" in matchups.columns:
        agg_map["opp_def_pass_epa_faced"] = ("opp_def_pass_epa", "mean")
    if "opp_def_rush_epa" in matchups.columns:
        agg_map["opp_def_rush_epa_faced"] = ("opp_def_rush_epa", "mean")
    if not agg_map:
        return pd.DataFrame(columns=cols)

    out = matchups.groupby(["season", "team"], as_index=False).agg(**agg_map)
    return out


def attach_next_season_opponent_defense(
    schedules: pd.DataFrame,
    team_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """
    For season T rows: difficulty of season T+1 slate using opponents' season-T defense.

    Output keyed by (season=T, team) with next_opp_def_epa_faced.
    """
    cols = ["season", "team", "next_opp_def_epa_faced", "next_opp_def_pass_epa_faced", "next_opp_def_rush_epa_faced"]
    if schedules.empty or team_metrics.empty:
        return pd.DataFrame(columns=cols)

    s = schedules.copy()
    if "game_type" in s.columns:
        s = s[s["game_type"].astype(str).str.upper().isin(["REG", "REGULAR"])].copy()
    if "week" in s.columns:
        s = s[pd.to_numeric(s["week"], errors="coerce").fillna(99) <= 18].copy()
    needed = {"season", "home_team", "away_team"}
    if not needed.issubset(s.columns):
        return pd.DataFrame(columns=cols)

    s["home_team"] = s["home_team"].map(normalize_team_abbr)
    s["away_team"] = s["away_team"].map(normalize_team_abbr)
    s["season"] = pd.to_numeric(s["season"], errors="coerce")

    def_cols = ["def_epa_per_play", "def_pass_epa", "def_rush_epa"]
    d = team_metrics[["season", "team"] + [c for c in def_cols if c in team_metrics.columns]].copy()
    d["season"] = pd.to_numeric(d["season"], errors="coerce")

    # T+1 matchups
    home = s[["season", "home_team", "away_team"]].rename(
        columns={"home_team": "team", "away_team": "opp", "season": "next_season"}
    )
    away = s[["season", "away_team", "home_team"]].rename(
        columns={"away_team": "team", "home_team": "opp", "season": "next_season"}
    )
    matchups = pd.concat([home, away], ignore_index=True)
    matchups["season"] = matchups["next_season"] - 1  # feature season T

    # opponents rated by season-T defense
    d_t = d.rename(columns={"team": "opp", **{c: f"opp_{c}" for c in def_cols if c in d.columns}})
    matchups = matchups.merge(d_t, how="left", left_on=["season", "opp"], right_on=["season", "opp"])

    agg_map = {}
    if "opp_def_epa_per_play" in matchups.columns:
        agg_map["next_opp_def_epa_faced"] = ("opp_def_epa_per_play", "mean")
    if "opp_def_pass_epa" in matchups.columns:
        agg_map["next_opp_def_pass_epa_faced"] = ("opp_def_pass_epa", "mean")
    if "opp_def_rush_epa" in matchups.columns:
        agg_map["next_opp_def_rush_epa_faced"] = ("opp_def_rush_epa", "mean")
    if not agg_map:
        return pd.DataFrame(columns=cols)

    out = matchups.groupby(["season", "team"], as_index=False).agg(**agg_map)
    return out
