"""Team-relative opportunity features: usage share and volume context.

``targets`` and ``carries`` alone do not say whether a player was the lead back
or the third option on a run-heavy team. Dividing by the team's total that
season turns absolute volume into a role signal the model can split on.

All of these are computed from the prior season already on the panel row
(season Y-1 predicting Y), so nothing looks ahead. Depth rank and team-change
flags already live on the feature frame; this module only adds the shares.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

OPPORTUNITY_FEATURE_COLS = [
    "target_share",
    "carry_share",
    "touch_share",
    "pos_target_share",
    "pos_carry_share",
    "pos_touch_share",
    "team_targets",
    "team_carries",
    "team_touches",
]


def attach_opportunity_features(features: pd.DataFrame) -> pd.DataFrame:
    """Add team-relative and position-relative usage shares."""
    out = features.copy()
    for col in ("targets", "carries", "touches"):
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")

    needed = {"team", "season", "position"}
    if not needed.issubset(out.columns):
        for col in OPPORTUNITY_FEATURE_COLS:
            out[col] = np.nan
        return out

    team_totals = (
        out.groupby(["season", "team"], dropna=True)[["targets", "carries", "touches"]]
        .sum(min_count=1)
        .rename(
            columns={
                "targets": "team_targets",
                "carries": "team_carries",
                "touches": "team_touches",
            }
        )
        .reset_index()
    )
    pos_totals = (
        out.groupby(["season", "team", "position"], dropna=True)[
            ["targets", "carries", "touches"]
        ]
        .sum(min_count=1)
        .rename(
            columns={
                "targets": "pos_targets",
                "carries": "pos_carries",
                "touches": "pos_touches",
            }
        )
        .reset_index()
    )
    out = out.merge(team_totals, on=["season", "team"], how="left")
    out = out.merge(pos_totals, on=["season", "team", "position"], how="left")

    out["target_share"] = out["targets"] / out["team_targets"].replace({0: np.nan})
    out["carry_share"] = out["carries"] / out["team_carries"].replace({0: np.nan})
    out["touch_share"] = out["touches"] / out["team_touches"].replace({0: np.nan})
    out["pos_target_share"] = out["targets"] / out["pos_targets"].replace({0: np.nan})
    out["pos_carry_share"] = out["carries"] / out["pos_carries"].replace({0: np.nan})
    out["pos_touch_share"] = out["touches"] / out["pos_touches"].replace({0: np.nan})
    return out.drop(columns=["pos_targets", "pos_carries", "pos_touches"], errors="ignore")


__all__ = ["OPPORTUNITY_FEATURE_COLS", "attach_opportunity_features"]
