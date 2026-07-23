from __future__ import annotations

import pandas as pd

from rookie_ppr.config import DYNASTY_YEARS


def _rank_pct_within(series: pd.Series) -> pd.Series:
    """Higher PPR → higher percentile (0–100)."""
    if series.dropna().empty:
        return pd.Series(pd.NA, index=series.index)
    return series.rank(pct=True, ascending=True, method="average") * 100.0


def build_dynasty_trajectory(master: pd.DataFrame) -> pd.DataFrame:
    """
    Year-by-year ranks and late-bloomer flags for complete Y1–Y3 windows.

    Designed for dynasty exploration: slow Y1/Y2 with strong Y3, etc.
    Does not feed redraft models.
    """
    if master.empty:
        return pd.DataFrame()

    id_cols = [c for c in ("gsis_id", "player_name", "position", "draft_year", "first_stat_season") if c in master.columns]
    ppr_cols = [f"ppr_y{i}" for i in range(1, DYNASTY_YEARS + 1) if f"ppr_y{i}" in master.columns]
    if not ppr_cols or "dynasty_seasons_complete" not in master.columns:
        return pd.DataFrame()

    work = master[id_cols + ppr_cols + ["dynasty_seasons_complete"]].copy()
    for col in (
        "dynasty_ppr_y1_y3_total",
        "dynasty_ppr_y1_y3_avg_season",
        "dynasty_ppr_y1_y3_avg_game",
    ):
        if col in master.columns:
            work[col] = master[col]

    # Within draft class + position percentiles (complete windows only for rank pool)
    for i, col in enumerate(ppr_cols, start=1):
        pct_col = f"y{i}_rank_pct_pos"
        work[pct_col] = pd.NA
        if "draft_year" not in work.columns or "position" not in work.columns:
            continue
        complete_mask = work["dynasty_seasons_complete"] == 1
        for _, grp in work.loc[complete_mask].groupby(["draft_year", "position"]):
            work.loc[grp.index, pct_col] = _rank_pct_within(grp[col]).to_numpy()

    if "ppr_y1" in work.columns and "ppr_y3" in work.columns:
        work["y3_vs_y1_delta"] = pd.to_numeric(work["ppr_y3"], errors="coerce") - pd.to_numeric(
            work["ppr_y1"], errors="coerce"
        )
    else:
        work["y3_vs_y1_delta"] = pd.NA
    if "ppr_y2" in work.columns and "ppr_y3" in work.columns:
        work["y3_vs_y2_delta"] = pd.to_numeric(work["ppr_y3"], errors="coerce") - pd.to_numeric(
            work["ppr_y2"], errors="coerce"
        )
    else:
        work["y3_vs_y2_delta"] = pd.NA

    y1 = pd.to_numeric(work.get("y1_rank_pct_pos"), errors="coerce")
    y2 = pd.to_numeric(work.get("y2_rank_pct_pos"), errors="coerce")
    y3 = pd.to_numeric(work.get("y3_rank_pct_pos"), errors="coerce")
    complete = work["dynasty_seasons_complete"] == 1

    work["slow_start"] = complete & y1.notna() & (y1 <= 33.0)
    work["strong_y3"] = complete & y3.notna() & (y3 >= 67.0)
    work["late_bloomer_y3"] = (
        work["slow_start"]
        & work["strong_y3"]
        & work["y3_vs_y1_delta"].notna()
        & (work["y3_vs_y1_delta"] > 0)
    )
    work["slow_y1_y2_strong_y3"] = (
        complete
        & y1.notna()
        & y2.notna()
        & y3.notna()
        & (y1 < 50.0)
        & (y2 < 50.0)
        & (y3 >= 50.0)
    )

    archetype = pd.Series("limited_sample", index=work.index, dtype=object)
    archetype.loc[complete] = "steady"
    early = complete & y1.notna() & y3.notna() & (y1 >= 67.0) & (y3 <= 33.0)
    decline = complete & y1.notna() & y3.notna() & (y3 + 20.0 < y1)
    archetype.loc[decline | early] = "early_peak"
    archetype.loc[work["late_bloomer_y3"] | work["slow_y1_y2_strong_y3"]] = "late_bloomer_y3"
    work["trajectory_archetype"] = archetype

    return work.reset_index(drop=True)
