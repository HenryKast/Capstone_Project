"""Pre-season market features (FantasyPros ADP) keyed to the season being predicted.

A panel row describes season Y-1 and predicts season Y. Drafts for season Y
happen before a snap of it is played, so that year's ADP is legitimately
available to a forecast of Y and carries no outcome information.

The market is included as a rank rather than a price. Rank is comparable across
years, whereas average draft position depends on league size and format.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from rookie_ppr.ingest_fantasypros_adp import load_fantasypros_adp
from rookie_ppr.utils import normalize_name

ADP_FEATURE_COLS = ["adp_log_rank", "adp_pos_rank", "adp_rank"]

# Rank assigned to a player the market did not rank at all, so "undrafted" is a
# real value the model can split on rather than a gap.
UNRANKED_RANK = 250.0


def build_adp_features() -> pd.DataFrame:
    """(name_norm, position, season) -> market features for that season."""
    adp = load_fantasypros_adp()
    if adp.empty:
        return pd.DataFrame(columns=["name_norm", "position", "season", *ADP_FEATURE_COLS])

    out = adp.dropna(subset=["overall_rank"]).copy()
    out["adp_rank"] = pd.to_numeric(out["overall_rank"], errors="coerce")
    out = out.dropna(subset=["adp_rank"])
    out = out.sort_values("adp_rank").drop_duplicates(
        subset=["season", "player_name_norm", "position"]
    )
    out["adp_pos_rank"] = out.groupby(["season", "position"])["adp_rank"].rank(method="first")
    out["adp_log_rank"] = np.log(out["adp_rank"])
    return out.rename(columns={"player_name_norm": "name_norm"})[
        ["name_norm", "position", "season", *ADP_FEATURE_COLS]
    ]


def attach_adp_features(features: pd.DataFrame) -> pd.DataFrame:
    """Join market features for each row's ``target_season``.

    Matching is on normalized name plus position, then name alone for the
    handful the position field disagrees on. Anyone the market never ranked gets
    the sentinel deep rank instead of a null.
    """
    market = build_adp_features()
    out = features.copy()
    for col in ADP_FEATURE_COLS:
        out[col] = np.nan
    if market.empty:
        return out

    out["_name_norm"] = out["player_name"].map(normalize_name)
    target = pd.to_numeric(out.get("target_season"), errors="coerce")

    with_pos = {
        (name, position, season): (log_rank, pos_rank, rank)
        for name, position, season, log_rank, pos_rank, rank in market.itertuples(index=False)
    }
    name_only: dict[tuple[str, int], tuple[float, float, float]] = {}
    for name, _position, season, log_rank, pos_rank, rank in market.itertuples(index=False):
        key = (name, season)
        # Keep the best (lowest) rank when a name appears at two positions.
        if key not in name_only or rank < name_only[key][2]:
            name_only[key] = (log_rank, pos_rank, rank)

    values: list[tuple[float, float, float]] = []
    for name, position, season in zip(out["_name_norm"], out["position"], target):
        if pd.isna(season):
            values.append((np.nan, np.nan, np.nan))
            continue
        season = int(season)
        hit = with_pos.get((name, position, season)) or name_only.get((name, season))
        if hit is None:
            values.append((np.log(UNRANKED_RANK), np.nan, UNRANKED_RANK))
        else:
            values.append(hit)

    out[ADP_FEATURE_COLS[0]] = [v[0] for v in values]
    out[ADP_FEATURE_COLS[1]] = [v[1] for v in values]
    out[ADP_FEATURE_COLS[2]] = [v[2] for v in values]
    return out.drop(columns=["_name_norm"])


__all__ = ["ADP_FEATURE_COLS", "attach_adp_features", "build_adp_features"]
