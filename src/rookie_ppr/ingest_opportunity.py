from __future__ import annotations

import pandas as pd

from rookie_ppr.utils import normalize_position


def build_team_offensive_environment(team_stats: pd.DataFrame) -> pd.DataFrame:
    """Derive simple offensive environment metrics from team season stats."""
    if team_stats.empty:
        return pd.DataFrame(columns=["season", "team", "off_pass_yards", "off_rush_yards", "off_pass_rate_proxy"])

    t = team_stats.copy()
    team_col = "team" if "team" in t.columns else ("recent_team" if "recent_team" in t.columns else None)
    season_col = "season" if "season" in t.columns else None
    if team_col is None or season_col is None:
        return pd.DataFrame(columns=["season", "team", "off_pass_yards", "off_rush_yards", "off_pass_rate_proxy"])

    pass_y = next((c for c in ("passing_yards", "pass_yards") if c in t.columns), None)
    rush_y = next((c for c in ("rushing_yards", "rush_yards") if c in t.columns), None)
    pass_att = next((c for c in ("passing_attempts", "attempts", "pass_attempts") if c in t.columns), None)
    rush_att = next((c for c in ("rushing_attempts", "carries", "rush_attempts") if c in t.columns), None)

    out = pd.DataFrame(
        {
            "season": t[season_col],
            "team": t[team_col],
            "off_pass_yards": pd.to_numeric(t[pass_y], errors="coerce") if pass_y else pd.NA,
            "off_rush_yards": pd.to_numeric(t[rush_y], errors="coerce") if rush_y else pd.NA,
        }
    )
    if pass_att and rush_att:
        pa = pd.to_numeric(t[pass_att], errors="coerce")
        ra = pd.to_numeric(t[rush_att], errors="coerce")
        out["off_pass_rate_proxy"] = pa / (pa + ra)
    else:
        out["off_pass_rate_proxy"] = pd.NA
    return out.drop_duplicates(subset=["season", "team"])


def build_landing_opportunity(stats: pd.DataFrame, draft: pd.DataFrame) -> pd.DataFrame:
    """
    Approximate vacated usage: prior-season positional fantasy production on the draft team.

    Uses previous season PPR by team/position as an opportunity proxy.
    """
    cols = [
        "draft_year",
        "draft_team",
        "position",
        "prior_team_pos_ppr",
        "opportunity_proxy",
    ]
    if draft.empty or stats.empty:
        return pd.DataFrame(columns=cols)

    s = stats.copy()
    id_ok = "player_id" in s.columns or "gsis_id" in s.columns
    if not id_ok or "season" not in s.columns:
        return pd.DataFrame(columns=cols)

    team_col = next((c for c in ("recent_team", "team", "posteam") if c in s.columns), None)
    pos_col = "position" if "position" in s.columns else None
    if team_col is None or pos_col is None:
        return pd.DataFrame(columns=cols)

    if "fantasy_points_ppr" in s.columns:
        s["ppr"] = s["fantasy_points_ppr"]
    else:
        return pd.DataFrame(columns=cols)

    s["position"] = s[pos_col].map(normalize_position)
    s = s[s["position"].isin(["QB", "RB", "WR", "TE"])]
    team_pos = (
        s.groupby(["season", team_col, "position"], as_index=False)["ppr"]
        .sum()
        .rename(columns={team_col: "draft_team", "ppr": "prior_team_pos_ppr"})
    )
    team_pos["season_next"] = team_pos["season"] + 1

    d = draft[["draft_year", "draft_team", "position"]].copy()
    merged = d.merge(
        team_pos,
        how="left",
        left_on=["draft_year", "draft_team", "position"],
        right_on=["season_next", "draft_team", "position"],
    )
    merged["opportunity_proxy"] = merged["prior_team_pos_ppr"]
    return merged[["draft_year", "draft_team", "position", "prior_team_pos_ppr", "opportunity_proxy"]].drop_duplicates()
