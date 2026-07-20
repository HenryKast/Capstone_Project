from __future__ import annotations

import pandas as pd

TARGET = "rookie_ppr"


def _band_recruiting(rank: float) -> str:
    if pd.isna(rank):
        return "unranked"
    r = float(rank)
    if r <= 50:
        return "top_50"
    if r <= 100:
        return "51_100"
    if r <= 300:
        return "101_300"
    return "301_plus"


def _band_sos(val: float) -> str:
    if pd.isna(val):
        return "unknown"
    v = float(val)
    if v < 0.45:
        return "easy_lt_0.45"
    if v < 0.50:
        return "mid_0.45_0.50"
    if v < 0.55:
        return "mid_0.50_0.55"
    return "hard_ge_0.55"


def _summarize(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if TARGET not in df.columns or df.empty:
        return pd.DataFrame()
    g = df.groupby(group_cols, dropna=False)
    out = g.agg(
        n_players=(TARGET, "count"),
        mean_rookie_ppr=(TARGET, "mean"),
        median_rookie_ppr=(TARGET, "median"),
    ).reset_index()
    if "rookie_ppr_per_game" in df.columns:
        extra = g["rookie_ppr_per_game"].mean().reset_index(name="mean_rookie_ppr_per_game")
        out = out.merge(extra, on=group_cols, how="left")
    return out.sort_values("mean_rookie_ppr", ascending=False, na_position="last")


def build_group_averages(master: pd.DataFrame) -> pd.DataFrame:
    df = master.copy()
    frames: list[pd.DataFrame] = []

    if "position" in df.columns:
        pos = _summarize(df, ["position"])
        pos.insert(0, "group_type", "position")
        pos = pos.rename(columns={"position": "group_value"})
        frames.append(pos)

    if "draft_round" in df.columns:
        rnd = _summarize(df, ["draft_round"])
        rnd.insert(0, "group_type", "draft_round")
        rnd = rnd.rename(columns={"draft_round": "group_value"})
        frames.append(rnd)

    if "recruiting_rank" in df.columns:
        df["recruiting_band"] = df["recruiting_rank"].map(_band_recruiting)
        rec = _summarize(df, ["recruiting_band"])
        rec.insert(0, "group_type", "recruiting_band")
        rec = rec.rename(columns={"recruiting_band": "group_value"})
        frames.append(rec)

    if "sos_opp_win_pct" in df.columns:
        df["sos_band"] = df["sos_opp_win_pct"].map(_band_sos)
        sos = _summarize(df, ["sos_band"])
        sos.insert(0, "group_type", "sos_band")
        sos = sos.rename(columns={"sos_band": "group_value"})
        frames.append(sos)

    if "college" in df.columns:
        col = _summarize(df, ["college"])
        col = col[col["n_players"] >= 3]
        col.insert(0, "group_type", "college")
        col = col.rename(columns={"college": "group_value"})
        frames.append(col.head(100))

    if "draft_team" in df.columns:
        tm = _summarize(df, ["draft_team"])
        tm = tm[tm["n_players"] >= 3]
        tm.insert(0, "group_type", "draft_team")
        tm = tm.rename(columns={"draft_team": "group_value"})
        frames.append(tm)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
