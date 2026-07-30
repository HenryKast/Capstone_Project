"""Phase 3: snap share, depth-chart role, Next Gen Stats (when available)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from rookie_ppr.nflverse_http import try_read_release_csv
from rookie_ppr.utils import normalize_position


def _pfr_to_gsis_map() -> pd.DataFrame:
    """Map pfr_player_id → gsis_id from nflverse players register."""
    players = try_read_release_csv(
        [
            ("players", "players.csv"),
            ("players", "players.csv.gz"),
        ]
    )
    if players.empty:
        return pd.DataFrame(columns=["pfr_player_id", "gsis_id"])
    pfr = next((c for c in ("pfr_id", "pfr_player_id") if c in players.columns), None)
    gsis = next((c for c in ("gsis_id", "gsisId") if c in players.columns), None)
    if not pfr or not gsis:
        return pd.DataFrame(columns=["pfr_player_id", "gsis_id"])
    out = players[[pfr, gsis]].dropna().drop_duplicates()
    return out.rename(columns={pfr: "pfr_player_id", gsis: "gsis_id"})


def build_snap_features(seasons: list[int]) -> pd.DataFrame:
    """Season offense snap share (2012+)."""
    id_map = _pfr_to_gsis_map()
    frames: list[pd.DataFrame] = []
    for season in seasons:
        if int(season) < 2012:
            continue
        df = try_read_release_csv(
            [
                ("snap_counts", f"snap_counts_{season}.csv.gz"),
                ("snap_counts", f"snap_counts_{season}.csv"),
            ]
        )
        if df.empty or "pfr_player_id" not in df.columns:
            continue
        s = df.copy()
        if "season" not in s.columns:
            s["season"] = season
        if "game_type" in s.columns:
            s = s[s["game_type"].astype(str).str.upper().isin(["REG", "REGULAR"])].copy()
        s["offense_snaps"] = pd.to_numeric(s.get("offense_snaps"), errors="coerce")
        s["offense_pct"] = pd.to_numeric(s.get("offense_pct"), errors="coerce")
        if s["offense_pct"].notna().any() and float(s["offense_pct"].dropna().max()) > 1.5:
            s["offense_pct"] = s["offense_pct"] / 100.0
        g = s.groupby(["season", "pfr_player_id"], as_index=False).agg(
            off_snaps=("offense_snaps", "sum"),
            off_snap_pct=("offense_pct", "mean"),
            snap_games=("offense_pct", "count"),
        )
        if not id_map.empty:
            g = g.merge(id_map, how="left", on="pfr_player_id")
        else:
            g["gsis_id"] = pd.NA
        frames.append(g)

    if not frames:
        return pd.DataFrame(columns=["gsis_id", "season", "off_snap_pct", "off_snaps", "snap_games"])
    out = pd.concat(frames, ignore_index=True)
    out = out[out["gsis_id"].notna()].copy()
    return out[["gsis_id", "season", "off_snap_pct", "off_snaps", "snap_games"]].drop_duplicates(
        subset=["gsis_id", "season"], keep="last"
    )


def build_depth_features(seasons: list[int]) -> pd.DataFrame:
    """Early-season offensive depth rank (lower = closer to starter)."""
    frames: list[pd.DataFrame] = []
    for season in seasons:
        if int(season) < 2001:
            continue
        df = try_read_release_csv(
            [
                ("depth_charts", f"depth_charts_{season}.csv.gz"),
                ("depth_charts", f"depth_charts_{season}.csv"),
            ]
        )
        if df.empty or "gsis_id" not in df.columns:
            continue
        d = df.copy()
        if "season" not in d.columns:
            d["season"] = season
        if "week" in d.columns:
            d["week"] = pd.to_numeric(d["week"], errors="coerce")
            d = d[(d["week"].isna()) | (d["week"] <= 4)].copy()
        if "formation" in d.columns:
            form = d["formation"].astype(str).str.lower()
            d = d[form.str.contains("off", na=True) | (form == "") | (form == "nan")].copy()

        # nflverse uses depth_team (1=starter, 2=backup, ...); depth_position is a label (WR/RB).
        rank_col = next(
            (c for c in ("depth_team", "depth_order", "dt_depth", "rank") if c in d.columns),
            None,
        )
        if rank_col:
            d["depth_rank"] = pd.to_numeric(d[rank_col], errors="coerce")
        elif "depth_position" in d.columns:
            raw = d["depth_position"]
            num = pd.to_numeric(raw, errors="coerce")
            parsed = raw.astype(str).str.extract(r"(\d+)\s*$", expand=False)
            d["depth_rank"] = num.fillna(pd.to_numeric(parsed, errors="coerce"))
        else:
            continue

        g = d.groupby(["gsis_id", "season"], as_index=False).agg(depth_rank=("depth_rank", "min"))
        frames.append(g)

    if not frames:
        return pd.DataFrame(columns=["gsis_id", "season", "depth_rank"])
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["gsis_id", "season"], keep="last")


def _ngs_rollup(df: pd.DataFrame, metric_map: dict[str, str], seasons: set[int]) -> pd.DataFrame:
    if df.empty or "player_gsis_id" not in df.columns:
        return pd.DataFrame(columns=["gsis_id", "season"])
    r = df.copy()
    r["season"] = pd.to_numeric(r["season"], errors="coerce")
    r = r[r["season"].isin(seasons)].copy()
    if r.empty:
        return pd.DataFrame(columns=["gsis_id", "season"])

    available = {src: dst for src, dst in metric_map.items() if src in r.columns}
    if not available:
        return pd.DataFrame(columns=["gsis_id", "season"])

    if "week" in r.columns:
        w = pd.to_numeric(r["week"], errors="coerce")
        season_rows = r[w.fillna(-1) == 0]
    else:
        season_rows = r.iloc[0:0]

    if not season_rows.empty:
        cols = ["player_gsis_id", "season"] + list(available.keys())
        out = season_rows[cols].copy()
    else:
        for src in available:
            r[src] = pd.to_numeric(r[src], errors="coerce")
        out = r.groupby(["player_gsis_id", "season"], as_index=False).agg(
            **{dst: (src, "mean") for src, dst in available.items()}
        )
        return out.rename(columns={"player_gsis_id": "gsis_id"})

    out = out.rename(columns={"player_gsis_id": "gsis_id", **available})
    return out


def build_ngs_features(seasons: list[int]) -> pd.DataFrame:
    """Season-level NGS efficiency metrics (2016+)."""
    season_set = {int(s) for s in seasons if int(s) >= 2016}
    if not season_set:
        return pd.DataFrame(columns=["gsis_id", "season"])

    rush = try_read_release_csv(
        [("nextgen_stats", "ngs_rushing.csv.gz"), ("nextgen_stats", "ngs_rushing.csv")]
    )
    rec = try_read_release_csv(
        [("nextgen_stats", "ngs_receiving.csv.gz"), ("nextgen_stats", "ngs_receiving.csv")]
    )
    pas = try_read_release_csv(
        [("nextgen_stats", "ngs_passing.csv.gz"), ("nextgen_stats", "ngs_passing.csv")]
    )

    pieces = [
        _ngs_rollup(
            rush,
            {
                "rush_yards_over_expected": "ngs_ryoe",
                "rush_yards_over_expected_per_att": "ngs_ryoe_per_att",
                "efficiency": "ngs_rush_efficiency",
            },
            season_set,
        ),
        _ngs_rollup(
            rec,
            {
                "avg_separation": "ngs_avg_separation",
                "avg_cushion": "ngs_avg_cushion",
                "avg_yac_above_expectation": "ngs_yac_oe",
            },
            season_set,
        ),
        _ngs_rollup(
            pas,
            {
                "avg_time_to_throw": "ngs_avg_ttt",
                "completion_percentage_above_expectation": "ngs_cpoe",
                "aggressiveness": "ngs_aggressiveness",
            },
            season_set,
        ),
    ]
    pieces = [p for p in pieces if not p.empty]
    if not pieces:
        return pd.DataFrame(columns=["gsis_id", "season"])
    out = pieces[0]
    for extra in pieces[1:]:
        out = out.merge(extra, on=["gsis_id", "season"], how="outer")
    return out.drop_duplicates(subset=["gsis_id", "season"], keep="last")
