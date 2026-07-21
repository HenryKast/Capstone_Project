from __future__ import annotations

import numpy as np
import pandas as pd

from rookie_ppr.utils import normalize_position, normalize_team_abbr


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
            "team": t[team_col].map(normalize_team_abbr),
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


def _ensure_carries(s: pd.DataFrame) -> pd.Series:
    if "carries" in s.columns:
        return pd.to_numeric(s["carries"], errors="coerce")
    if "rushing_attempts" in s.columns:
        return pd.to_numeric(s["rushing_attempts"], errors="coerce")
    return pd.Series(np.nan, index=s.index)


def _stats_team_pos_usage(stats: pd.DataFrame) -> pd.DataFrame:
    """Season × team × position: PPR + carries/targets/receptions/touches."""
    cols = [
        "season",
        "draft_team",
        "position",
        "team_opportunity_ppr",
        "team_pos_carries",
        "team_pos_targets",
        "team_pos_receptions",
        "team_pos_touches",
    ]
    if stats.empty:
        return pd.DataFrame(columns=cols)

    s = stats.copy()
    team_col = next((c for c in ("recent_team", "team", "posteam") if c in s.columns), None)
    pos_col = "position" if "position" in s.columns else None
    if team_col is None or pos_col is None or "season" not in s.columns:
        return pd.DataFrame(columns=cols)
    if "fantasy_points_ppr" not in s.columns:
        return pd.DataFrame(columns=cols)

    s["position"] = s[pos_col].map(normalize_position)
    s = s[s["position"].isin(["QB", "RB", "WR", "TE"])].copy()
    s["ppr"] = pd.to_numeric(s["fantasy_points_ppr"], errors="coerce")
    s["_team"] = s[team_col].map(normalize_team_abbr)
    s["_carries"] = _ensure_carries(s)
    s["_targets"] = pd.to_numeric(s["targets"], errors="coerce") if "targets" in s.columns else np.nan
    s["_rec"] = pd.to_numeric(s["receptions"], errors="coerce") if "receptions" in s.columns else np.nan
    s["_touches"] = s["_carries"].fillna(0) + s["_rec"].fillna(0)
    # If both carries and rec missing, touches stay NaN
    both_miss = s["_carries"].isna() & s["_rec"].isna()
    s.loc[both_miss, "_touches"] = np.nan

    return (
        s.groupby(["season", "_team", "position"], as_index=False)
        .agg(
            team_opportunity_ppr=("ppr", "sum"),
            team_pos_carries=("_carries", "sum"),
            team_pos_targets=("_targets", "sum"),
            team_pos_receptions=("_rec", "sum"),
            team_pos_touches=("_touches", "sum"),
        )
        .rename(columns={"_team": "draft_team"})
    )


def _prior_season_map(years: pd.Series, available: list[float]) -> pd.Series:
    available_set = set(available)

    def _prior(y: float) -> float:
        if pd.isna(y):
            return float("nan")
        y = int(y)
        preferred = y - 1
        if preferred in available_set:
            return float(preferred)
        earlier = [s for s in available if s < y]
        return float(max(earlier)) if earlier else float("nan")

    return years.map(_prior)


def build_landing_opportunity(stats: pd.DataFrame, draft: pd.DataFrame) -> pd.DataFrame:
    """
    Prior-season positional usage on the draft team (vacated-usage proxy).

    Includes fantasy points plus carries/targets/receptions/touches for depth.
    Uses draft_year - 1 when available; otherwise the latest stats season strictly
    before draft_year (needed when nflverse lags).
    """
    cols = [
        "draft_year",
        "draft_team",
        "position",
        "team_opportunity_ppr",
        "team_pos_carries",
        "team_pos_targets",
        "team_pos_receptions",
        "team_pos_touches",
    ]
    if draft.empty or stats.empty:
        return pd.DataFrame(columns=cols)

    team_pos = _stats_team_pos_usage(stats)
    if team_pos.empty:
        return pd.DataFrame(columns=cols)

    available = sorted(pd.to_numeric(team_pos["season"], errors="coerce").dropna().unique())
    if not available:
        return pd.DataFrame(columns=cols)

    d = draft[["draft_year", "draft_team", "position"]].drop_duplicates().copy()
    d["draft_year"] = pd.to_numeric(d["draft_year"], errors="coerce")
    d["draft_team"] = d["draft_team"].map(normalize_team_abbr)
    d["prior_season"] = _prior_season_map(d["draft_year"], available)

    merged = d.merge(
        team_pos,
        how="left",
        left_on=["prior_season", "draft_team", "position"],
        right_on=["season", "draft_team", "position"],
    )
    return merged[cols].drop_duplicates()


def build_incumbent_competition(
    stats: pd.DataFrame,
    draft: pd.DataFrame,
    rosters: pd.DataFrame,
) -> pd.DataFrame:
    """
    Returning same-team / same-position competition.

    Among players who produced at that team+position in the prior available season
    and still appear on that team's roster afterward: max/sum PPR and max carries/targets.
    """
    cols = [
        "draft_year",
        "draft_team",
        "position",
        "incumbent_pos_ppr",
        "incumbent_pos_ppr_sum",
        "incumbent_pos_carries",
        "incumbent_pos_targets",
        "incumbent_pos_touches",
    ]
    if draft.empty or stats.empty:
        return pd.DataFrame(columns=cols)

    s = stats.copy()
    team_col = next((c for c in ("recent_team", "team", "posteam") if c in s.columns), None)
    if team_col is None or "season" not in s.columns or "fantasy_points_ppr" not in s.columns:
        return pd.DataFrame(columns=cols)

    id_col = next((c for c in ("gsis_id", "player_id") if c in s.columns), None)
    if id_col is None:
        return pd.DataFrame(columns=cols)

    s["position"] = s["position"].map(normalize_position) if "position" in s.columns else pd.NA
    s = s[s["position"].isin(["QB", "RB", "WR", "TE"])].copy()
    s["ppr"] = pd.to_numeric(s["fantasy_points_ppr"], errors="coerce")
    s["season"] = pd.to_numeric(s["season"], errors="coerce")
    s["_team"] = s[team_col].map(normalize_team_abbr)
    s["_carries"] = _ensure_carries(s)
    s["_targets"] = pd.to_numeric(s["targets"], errors="coerce") if "targets" in s.columns else np.nan
    s["_rec"] = pd.to_numeric(s["receptions"], errors="coerce") if "receptions" in s.columns else np.nan
    s["_touches"] = s["_carries"].fillna(0) + s["_rec"].fillna(0)
    both_miss = s["_carries"].isna() & s["_rec"].isna()
    s.loc[both_miss, "_touches"] = np.nan

    player_season = (
        s.groupby([id_col, "season", "_team", "position"], as_index=False)
        .agg(
            ppr=("ppr", "sum"),
            carries=("_carries", "sum"),
            targets=("_targets", "sum"),
            touches=("_touches", "sum"),
        )
        .rename(columns={id_col: "gsis_id", "_team": "draft_team"})
    )
    available_stats = sorted(player_season["season"].dropna().unique())
    if not available_stats:
        return pd.DataFrame(columns=cols)

    roster_keys = pd.DataFrame(columns=["gsis_id", "roster_season", "draft_team"])
    if rosters is not None and not rosters.empty:
        r = rosters.copy()
        r_team = next((c for c in ("team", "recent_team", "club_code") if c in r.columns), None)
        r_id = next((c for c in ("gsis_id", "player_id") if c in r.columns), None)
        if r_team and r_id and "season" in r.columns:
            roster_keys = (
                r[[r_id, "season", r_team]]
                .dropna()
                .drop_duplicates()
                .rename(columns={r_id: "gsis_id", "season": "roster_season", r_team: "draft_team"})
            )
            roster_keys["roster_season"] = pd.to_numeric(roster_keys["roster_season"], errors="coerce")
            roster_keys["draft_team"] = roster_keys["draft_team"].map(normalize_team_abbr)

    available_rosters = (
        sorted(roster_keys["roster_season"].dropna().unique()) if not roster_keys.empty else []
    )

    d = draft[["draft_year", "draft_team", "position"]].drop_duplicates().copy()
    d["draft_year"] = pd.to_numeric(d["draft_year"], errors="coerce")
    d["draft_team"] = d["draft_team"].map(normalize_team_abbr)

    rows: list[dict] = []
    for _, slot in d.iterrows():
        y = slot["draft_year"]
        team = slot["draft_team"]
        pos = slot["position"]
        empty = {
            "draft_year": y,
            "draft_team": team,
            "position": pos,
            "incumbent_pos_ppr": np.nan,
            "incumbent_pos_ppr_sum": np.nan,
            "incumbent_pos_carries": np.nan,
            "incumbent_pos_targets": np.nan,
            "incumbent_pos_touches": np.nan,
        }
        if pd.isna(y) or pd.isna(team) or pd.isna(pos):
            continue
        y = int(y)
        earlier_stats = [s for s in available_stats if s < y]
        if not earlier_stats:
            rows.append({**empty, "draft_year": y})
            continue
        prior = max(earlier_stats)
        roster_candidates = [rs for rs in available_rosters if prior < rs <= y]
        roster_season = max(roster_candidates) if roster_candidates else (
            max([rs for rs in available_rosters if rs <= y], default=prior)
        )

        prod = player_season[
            (player_season["season"] == prior)
            & (player_season["draft_team"] == team)
            & (player_season["position"] == pos)
        ]
        if prod.empty:
            rows.append({**empty, "draft_year": y})
            continue

        if not roster_keys.empty and pd.notna(roster_season):
            returning_ids = set(
                roster_keys.loc[
                    (roster_keys["roster_season"] == roster_season)
                    & (roster_keys["draft_team"] == team),
                    "gsis_id",
                ].astype(str)
            )
            prod = prod[prod["gsis_id"].astype(str).isin(returning_ids)]

        if prod.empty:
            rows.append({**empty, "draft_year": y})
            continue

        ppr_vals = pd.to_numeric(prod["ppr"], errors="coerce").dropna()
        car_vals = pd.to_numeric(prod["carries"], errors="coerce").dropna()
        tgt_vals = pd.to_numeric(prod["targets"], errors="coerce").dropna()
        touch_vals = pd.to_numeric(prod["touches"], errors="coerce").dropna()
        rows.append(
            {
                "draft_year": y,
                "draft_team": team,
                "position": pos,
                "incumbent_pos_ppr": float(ppr_vals.max()) if len(ppr_vals) else np.nan,
                "incumbent_pos_ppr_sum": float(ppr_vals.sum()) if len(ppr_vals) else np.nan,
                "incumbent_pos_carries": float(car_vals.max()) if len(car_vals) else np.nan,
                "incumbent_pos_targets": float(tgt_vals.max()) if len(tgt_vals) else np.nan,
                "incumbent_pos_touches": float(touch_vals.max()) if len(touch_vals) else np.nan,
            }
        )

    return pd.DataFrame(rows, columns=cols).drop_duplicates(
        subset=["draft_year", "draft_team", "position"]
    )


def attach_team_season_with_fallback(
    players: pd.DataFrame,
    season_frame: pd.DataFrame,
    value_cols: list[str],
    *,
    primary_season_col: str = "draft_year",
) -> pd.DataFrame:
    """
    Left-join season_frame on (draft_year, draft_team), then fill from the latest
    available season strictly before draft_year (not only draft_year-1).
    """
    out = players.copy()
    for c in value_cols:
        if c not in out.columns:
            out[c] = pd.NA
    if season_frame.empty or "draft_team" not in out.columns or primary_season_col not in out.columns:
        return out

    sf = season_frame.copy()
    if "team" in sf.columns and "draft_team" not in sf.columns:
        sf = sf.rename(columns={"team": "draft_team"})
    if "season" not in sf.columns or "draft_team" not in sf.columns:
        return out

    sf["season"] = pd.to_numeric(sf["season"], errors="coerce")
    keep = ["season", "draft_team"] + [c for c in value_cols if c in sf.columns]
    sf = sf[keep].drop_duplicates(subset=["season", "draft_team"])
    available = sorted(sf["season"].dropna().unique())

    key = out[[primary_season_col, "draft_team"]].copy()
    key["_row"] = np.arange(len(out))
    key[primary_season_col] = pd.to_numeric(key[primary_season_col], errors="coerce")
    key["draft_team"] = key["draft_team"].map(normalize_team_abbr)
    key["join_season"] = _prior_season_map(key[primary_season_col], available)
    # Also try exact draft_year first (rare but keeps behavior when data exists)
    key["exact_season"] = key[primary_season_col]

    sf["draft_team"] = sf["draft_team"].map(normalize_team_abbr)

    exact = key.merge(
        sf,
        how="left",
        left_on=["exact_season", "draft_team"],
        right_on=["season", "draft_team"],
    )
    prior = key.merge(
        sf,
        how="left",
        left_on=["join_season", "draft_team"],
        right_on=["season", "draft_team"],
    )

    for c in value_cols:
        if c not in exact.columns and c not in prior.columns:
            continue
        filled = exact[c].copy() if c in exact.columns else pd.Series(np.nan, index=exact.index)
        if c in prior.columns:
            filled = filled.where(filled.notna(), prior[c])
        tmp = exact[["_row"]].copy()
        tmp[c] = filled
        tmp = tmp.drop_duplicates("_row").set_index("_row")[c]
        out[c] = tmp.reindex(range(len(out))).to_numpy()
    return out
