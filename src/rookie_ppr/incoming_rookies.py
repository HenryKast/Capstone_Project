from __future__ import annotations

import pandas as pd

from rookie_ppr.config import INCOMING_DRAFT_YEAR
from rookie_ppr.feature_composites import ID_COLUMNS, SCORE_COLUMNS


def build_incoming_rookies_sheet(
    master: pd.DataFrame,
    ml_features: pd.DataFrame | None = None,
    draft_year: int = INCOMING_DRAFT_YEAR,
) -> pd.DataFrame:
    """Pre-rookie feature set + ML predictions for an upcoming draft class."""
    if master.empty or "draft_year" not in master.columns:
        return pd.DataFrame()

    incoming = master[pd.to_numeric(master["draft_year"], errors="coerce") == draft_year].copy()
    if incoming.empty:
        return incoming

    if ml_features is not None and not ml_features.empty:
        id_cols = [c for c in ID_COLUMNS if c in incoming.columns and c in ml_features.columns]
        extra = [
            c
            for c in ml_features.columns
            if c in SCORE_COLUMNS
            or c in ("predicted_rookie_ppr", "success_score_0_100")
        ]
        merge_cols = id_cols + [c for c in extra if c not in incoming.columns]
        if id_cols and merge_cols:
            incoming = incoming.merge(
                ml_features[merge_cols].drop_duplicates(id_cols),
                on=id_cols,
                how="left",
            )

    sort_cols = [c for c in ("position", "draft_overall", "player_name") if c in incoming.columns]
    if sort_cols:
        incoming = incoming.sort_values(sort_cols, na_position="last")
    return incoming.reset_index(drop=True)
