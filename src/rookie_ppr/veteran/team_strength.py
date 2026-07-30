"""Score fantasy rosters from the unified draft board (10-team league helper)."""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from rookie_ppr.utils import normalize_name
from rookie_ppr.veteran.draft_board import load_draft_board

DEFAULT_STARTERS = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1}


def _norm_name(s: str) -> str:
    return normalize_name(s)


def score_team_roster(
    player_names: list[str],
    board: pd.DataFrame,
    *,
    starters: Mapping[str, int] | None = None,
) -> dict:
    """
    Sum projected_ppr_17 for a best-ball style starting lineup from the board.

    FLEX fills from remaining RB/WR/TE by projection.
    """
    starters = dict(starters or DEFAULT_STARTERS)
    if "projected_ppr_17" not in board.columns:
        raise KeyError("Draft board is missing projected_ppr_17 — rebuild it.")
    lookup = board.copy()
    lookup["match_key"] = lookup["player_name"].map(_norm_name)
    name_set = {_norm_name(n) for n in player_names}
    pool = lookup[lookup["match_key"].isin(name_set)].copy()
    if pool.empty:
        return {"total_ppr_17": 0.0, "matched_players": 0, "lineup": []}

    used: set[str] = set()
    lineup: list[dict] = []
    total = 0.0

    def take(pos: str, n: int, slot: str) -> None:
        nonlocal total
        cand = pool[(pool["position"] == pos) & (~pool["match_key"].isin(used))].sort_values(
            "projected_ppr_17", ascending=False
        )
        for row in cand.head(n).itertuples(index=False):
            used.add(row.match_key)
            pts = float(row.projected_ppr_17)
            total += pts
            lineup.append({"slot": slot, "player_name": row.player_name, "projected_ppr_17": pts})

    take("QB", starters.get("QB", 1), "QB")
    take("RB", starters.get("RB", 2), "RB")
    take("WR", starters.get("WR", 2), "WR")
    take("TE", starters.get("TE", 1), "TE")
    flex_n = starters.get("FLEX", 1)
    if flex_n:
        flex = pool[
            pool["position"].isin(["RB", "WR", "TE"]) & (~pool["match_key"].isin(used))
        ].sort_values("projected_ppr_17", ascending=False)
        for row in flex.head(flex_n).itertuples(index=False):
            used.add(row.match_key)
            pts = float(row.projected_ppr_17)
            total += pts
            lineup.append({"slot": "FLEX", "player_name": row.player_name, "projected_ppr_17": pts})

    return {
        "total_ppr_17": round(total, 2),
        "matched_players": int(len(used)),
        "lineup": lineup,
    }


def rank_fantasy_teams(
    teams: Mapping[str, list[str]],
    board: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Rank named teams by projected starter strength."""
    if board is None:
        board = load_draft_board()
    rows = []
    for team_name, roster in teams.items():
        scored = score_team_roster(roster, board)
        rows.append(
            {
                "team": team_name,
                "projected_starter_ppr_17": scored["total_ppr_17"],
                "matched_players": scored["matched_players"],
            }
        )
    out = pd.DataFrame(rows).sort_values("projected_starter_ppr_17", ascending=False)
    out["strength_rank"] = np.arange(1, len(out) + 1)
    return out.reset_index(drop=True)
