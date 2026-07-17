from __future__ import annotations

import pandas as pd

from rookie_ppr.config import DRAFT_YEAR_MAX, DRAFT_YEAR_MIN, SKILL_POSITIONS
from rookie_ppr.nflverse_http import try_read_release_csv
from rookie_ppr.utils import estimate_hs_class_from_birth, normalize_name, normalize_position


def load_draft_picks() -> pd.DataFrame:
    draft = try_read_release_csv(
        [
            ("draft_picks", "draft_picks.csv"),
            ("draft_picks", "draft_picks.csv.gz"),
        ]
    )
    if draft.empty:
        return draft

    years = list(range(DRAFT_YEAR_MIN, DRAFT_YEAR_MAX + 1))
    season_col = "season" if "season" in draft.columns else None
    if season_col:
        draft = draft[draft[season_col].isin(years)].copy()

    pos_col = "position" if "position" in draft.columns else "pos"
    draft["position"] = draft[pos_col].map(normalize_position)
    draft = draft[draft["position"].isin(SKILL_POSITIONS)].copy()

    name_col = next(
        (
            c
            for c in (
                "pfr_player_name",
                "football_name",
                "player_name",
                "cfb_player_name",
                "pfr_name",
            )
            if c in draft.columns
        ),
        None,
    )
    if name_col is None:
        draft["player_name"] = ""
    else:
        draft["player_name"] = draft[name_col].astype(str)
    draft["player_name_norm"] = draft["player_name"].map(normalize_name)

    keep = {
        "season": "draft_year",
        "round": "draft_round",
        "pick": "draft_pick",
        "overall": "draft_overall",
        "team": "draft_team",
        "gsis_id": "gsis_id",
        "pfr_player_id": "pfr_player_id",
        "cfb_id": "cfb_id",
        "college": "college",
        "category": "draft_category",
        "side": "draft_side",
        "hof": "hof",
    }
    out = pd.DataFrame(
        {
            "player_name": draft["player_name"],
            "player_name_norm": draft["player_name_norm"],
            "position": draft["position"],
        }
    )
    for src, dst in keep.items():
        out[dst] = draft[src] if src in draft.columns else pd.NA

    if out["draft_overall"].isna().all() and "pick" in draft.columns:
        out["draft_overall"] = draft["pick"]

    return out.reset_index(drop=True)


def load_rosters(seasons: list[int]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for season in seasons:
        df = try_read_release_csv(
            [
                ("rosters", f"roster_{season}.csv"),
                ("rosters", f"roster_{season}.csv.gz"),
                ("weekly_rosters", f"roster_weekly_{season}.csv"),
            ]
        )
        if df.empty:
            continue
        df["season"] = season
        frames.append(df)
    if not frames:
        # Fallback: full players table
        players = try_read_release_csv([("players", "players.csv"), ("players", "players.csv.gz")])
        return players
    return pd.concat(frames, ignore_index=True)


def _aggregate_weekly_to_season(weekly: pd.DataFrame) -> pd.DataFrame:
    """Aggregate nflverse weekly player_stats to season totals (REG only when available)."""
    if weekly.empty:
        return weekly
    w = weekly.copy()
    if "season_type" in w.columns:
        w = w[w["season_type"].astype(str).str.upper().isin(["REG", "REGULAR"])].copy()
    if "fantasy_points_ppr" not in w.columns:
        return pd.DataFrame()

    id_col = next((c for c in ("player_id", "gsis_id") if c in w.columns), None)
    if id_col is None or "season" not in w.columns:
        return pd.DataFrame()

    w["fantasy_points_ppr"] = pd.to_numeric(w["fantasy_points_ppr"], errors="coerce").fillna(0)
    # Count a game if the player appears in the weekly file for that week
    group_cols = [id_col, "season"]
    for optional in ("player_name", "player_display_name", "position", "recent_team"):
        if optional in w.columns:
            group_cols.append(optional)

    named_aggs: dict = {
        "fantasy_points_ppr": ("fantasy_points_ppr", "sum"),
        "games": ("week" if "week" in w.columns else "fantasy_points_ppr", "count"),
    }
    for col in ("receptions", "rushing_yards", "receiving_yards", "passing_yards"):
        if col in w.columns:
            named_aggs[col] = (col, "sum")

    agg = w.groupby(group_cols, as_index=False).agg(**named_aggs)
    if id_col != "gsis_id":
        agg = agg.rename(columns={id_col: "gsis_id"})
    return agg


def load_player_season_stats(seasons: list[int]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for season in seasons:
        df = try_read_release_csv(
            [
                ("player_stats", f"player_stats_{season}.csv"),
                ("player_stats", f"player_stats_{season}.csv.gz"),
                ("stats_player", f"stats_player_season_{season}.csv"),
            ]
        )
        if df.empty:
            continue
        if "season" not in df.columns:
            df["season"] = season
        frames.append(df)
    if not frames:
        return pd.DataFrame()

    weekly = pd.concat(frames, ignore_index=True)
    # Weekly files include a week column; season files usually do not
    if "week" in weekly.columns:
        return _aggregate_weekly_to_season(weekly)
    return weekly


def build_rookie_fantasy(draft: pd.DataFrame, stats: pd.DataFrame, rosters: pd.DataFrame) -> pd.DataFrame:
    """Compute PPR for each player's first NFL season."""
    if draft.empty:
        return pd.DataFrame()

    id_col = next((c for c in ("player_id", "gsis_id", "gsisId") if c in stats.columns), None)
    roster_id = next((c for c in ("gsis_id", "player_id") if c in rosters.columns), None)

    draft = draft.copy()
    draft["rookie_season"] = draft["draft_year"]

    if not rosters.empty and roster_id and "gsis_id" in draft.columns:
        r = rosters.copy()
        cols = [c for c in (roster_id, "season", "full_name", "display_name", "birth_date", "age") if c in r.columns]
        r_small = r[cols].drop_duplicates()
        birth_map = (
            r_small.dropna(subset=[roster_id])
            .sort_values("season" if "season" in r_small.columns else roster_id)
            .groupby(roster_id, as_index=False)
            .first()
        )
        merge_cols = [roster_id] + [c for c in ("birth_date", "age") if c in birth_map.columns]
        draft = draft.merge(birth_map[merge_cols], how="left", left_on="gsis_id", right_on=roster_id)
        if roster_id != "gsis_id" and roster_id in draft.columns:
            draft = draft.drop(columns=[roster_id])

    if "birth_date" in draft.columns:
        draft["hs_class_estimated"] = draft["birth_date"].map(estimate_hs_class_from_birth)
    else:
        draft["hs_class_estimated"] = pd.NA

    if stats.empty or id_col is None:
        fantasy = draft.copy()
        fantasy["rookie_ppr"] = pd.NA
        fantasy["rookie_games"] = pd.NA
        fantasy["rookie_ppr_per_game"] = pd.NA
        return fantasy

    s = stats.copy()
    if "fantasy_points_ppr" in s.columns:
        s["ppr"] = pd.to_numeric(s["fantasy_points_ppr"], errors="coerce")
    else:
        def col(name: str, default: float = 0.0) -> pd.Series:
            if name in s.columns:
                return pd.to_numeric(s[name], errors="coerce").fillna(0)
            return pd.Series(default, index=s.index)

        s["ppr"] = (
            col("receptions")
            + col("rushing_yards") / 10.0
            + col("receiving_yards") / 10.0
            + col("passing_yards") / 25.0
            + col("rushing_tds") * 6.0
            + col("receiving_tds") * 6.0
            + col("passing_tds") * 4.0
            - col("interceptions") * 2.0
            - col("rushing_fumbles_lost") * 2.0
            - col("receiving_fumbles_lost") * 2.0
        )

    games_col = next((c for c in ("games", "season_games", "games_played") if c in s.columns), None)
    if "season" not in s.columns:
        fantasy = draft.copy()
        fantasy["rookie_ppr"] = pd.NA
        fantasy["rookie_games"] = pd.NA
        fantasy["rookie_ppr_per_game"] = pd.NA
        return fantasy

    s = s.sort_values([id_col, "season"])
    first = s.groupby(id_col, as_index=False).first()
    rename = {id_col: "gsis_id", "season": "first_stat_season", "ppr": "rookie_ppr"}
    if games_col:
        rename[games_col] = "rookie_games"
    first = first.rename(columns=rename)
    keep_cols = [c for c in ("gsis_id", "first_stat_season", "rookie_ppr", "rookie_games") if c in first.columns]
    first = first[keep_cols]

    merged = draft.merge(first, how="left", on="gsis_id")
    use_ppr = merged["first_stat_season"].isna() | (
        merged["first_stat_season"].astype("Float64") <= merged["draft_year"].astype("Float64") + 1
    )
    if "rookie_ppr" in merged.columns:
        merged.loc[~use_ppr, "rookie_ppr"] = pd.NA
    if "rookie_games" in merged.columns:
        merged.loc[~use_ppr, "rookie_games"] = pd.NA
    else:
        merged["rookie_games"] = pd.NA
    merged["rookie_ppr_per_game"] = merged["rookie_ppr"] / merged["rookie_games"].replace({0: pd.NA})
    merged["rookie_ppr_rank_pos"] = merged.groupby(["draft_year", "position"])["rookie_ppr"].rank(
        ascending=False, method="min"
    )
    return merged


def load_combine() -> pd.DataFrame:
    combine = try_read_release_csv(
        [
            ("combine", "combine.csv"),
            ("combine", "combine.csv.gz"),
        ]
    )
    if combine.empty:
        return combine
    name_col = next((c for c in ("player_name", "pfr_player_name", "name") if c in combine.columns), None)
    if name_col:
        combine["player_name_norm"] = combine[name_col].map(normalize_name)
    if "pos" in combine.columns and "position" not in combine.columns:
        combine["position"] = combine["pos"].map(normalize_position)
    elif "position" in combine.columns:
        combine["position"] = combine["position"].map(normalize_position)
    return combine


def load_schedules(seasons: list[int]) -> pd.DataFrame:
    schedules = try_read_release_csv(
        [
            ("schedules", "games.csv"),
            ("schedules", "schedules.csv"),
            ("schedules", "games.csv.gz"),
        ]
    )
    if schedules.empty or "season" not in schedules.columns:
        return schedules
    return schedules[schedules["season"].isin(seasons)].copy()


def load_depth_charts(seasons: list[int]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for season in seasons:
        df = try_read_release_csv(
            [
                ("depth_charts", f"depth_charts_{season}.csv"),
                ("depth_charts", f"depth_charts_{season}.csv.gz"),
            ]
        )
        if df.empty:
            continue
        df["season"] = season
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_ff_rankings() -> pd.DataFrame:
    return try_read_release_csv(
        [
            ("ff_rankings", "rankings.csv"),
            ("dynastyprocess", "db_fpecr_latest.csv"),
        ]
    )


def load_team_season_stats(seasons: list[int]) -> pd.DataFrame:
    """
    Build team-season offensive proxies from weekly player_stats when
    dedicated team-stats release assets are unavailable.
    """
    frames: list[pd.DataFrame] = []
    for season in seasons:
        df = try_read_release_csv(
            [
                ("player_stats", f"player_stats_{season}.csv"),
                ("player_stats", f"player_stats_{season}.csv.gz"),
            ]
        )
        if df.empty:
            continue
        if "season_type" in df.columns:
            df = df[df["season_type"].astype(str).str.upper().isin(["REG", "REGULAR"])].copy()
        team_col = "recent_team" if "recent_team" in df.columns else None
        if team_col is None:
            continue
        if "season" not in df.columns:
            df["season"] = season
        for c in ("passing_yards", "rushing_yards", "attempts", "carries"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
            else:
                df[c] = 0
        g = (
            df.groupby(["season", team_col], as_index=False)
            .agg(
                passing_yards=("passing_yards", "sum"),
                rushing_yards=("rushing_yards", "sum"),
                passing_attempts=("attempts", "sum"),
                rushing_attempts=("carries", "sum"),
            )
            .rename(columns={team_col: "team"})
        )
        frames.append(g)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
