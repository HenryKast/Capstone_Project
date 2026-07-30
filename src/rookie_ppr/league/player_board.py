"""Production player ranking board from the ADP+opportunity walk-forward model.

Team simulation keeps the base model for blend diversification. Player ranks
use this board instead: on drafted skill players it is the strongest standalone
forecast we have (walk-forward r ≈ 0.54 vs the market's 0.57, and closer than
the base model).
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from rookie_ppr.ingest_fantasypros_adp import load_fantasypros_adp
from rookie_ppr.league.config import (
    CSV_OUTPUT_DIR,
    LEAGUE_PLAYER_BOARD_CSV,
    LEAGUE_PLAYER_PROJECTIONS_CSV,
    LEAGUE_SEASONS,
    LEAGUE_TARGET_SEASON,
    MODELED_POSITIONS,
)
from rookie_ppr.utils import normalize_name
from rookie_ppr.veteran.config import VET_FEATURES_CSV
from rookie_ppr.veteran.draft_board import expected_games_table

# Replacement ranks for a 10-team league with QB/RB/RB/WR/WR/TE/FLEX.
REPLACEMENT_RANK = {"QB": 10, "RB": 25, "WR": 30, "TE": 10}
REG_SEASON_GAMES = 17
FALLBACK_EXPECTED_GAMES = {"QB": 13.3, "RB": 12.9, "WR": 13.8, "TE": 13.0}


def _attach_adp(board: pd.DataFrame) -> pd.DataFrame:
    adp = load_fantasypros_adp()
    out = board.copy()
    out["adp_rank"] = np.nan
    if adp.empty:
        return out
    ranked = (
        adp.dropna(subset=["overall_rank"])
        .sort_values("overall_rank")
        .drop_duplicates(subset=["season", "player_name_norm"])
    )
    lookup = ranked.set_index(["season", "player_name_norm"])["overall_rank"].to_dict()
    names = out["player_name"].map(normalize_name)
    out["adp_rank"] = [
        lookup.get((int(season), name), np.nan)
        for season, name in zip(out["target_season"], names)
    ]
    return out


def build_player_board(
    projections: pd.DataFrame | None = None,
    *,
    seasons: list[int] | None = None,
) -> pd.DataFrame:
    """Rank players by ADP+opportunity projected PPR, with VORP and ADP."""
    if projections is None:
        path = CSV_OUTPUT_DIR / LEAGUE_PLAYER_PROJECTIONS_CSV
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run the league compile (player-track projections first)."
            )
        projections = pd.read_csv(path)

    seasons = list(seasons or LEAGUE_SEASONS)
    work = projections[
        projections["target_season"].isin(seasons)
        & projections["position"].isin(MODELED_POSITIONS)
        & projections["predicted_ppr"].notna()
    ].copy()
    if work.empty:
        return pd.DataFrame()

    work["projected_season_ppr"] = pd.to_numeric(work["predicted_ppr"], errors="coerce")
    # Season EV already prices in missed games. Put it on a 17-game public-board
    # pace using the position's historical starter availability — never the
    # player's own prior-season games, which would inflate returning injured stars.
    try:
        features = pd.read_csv(
            CSV_OUTPUT_DIR / VET_FEATURES_CSV,
            usecols=["gsis_id", "position", "season", "games", "ppr"],
            low_memory=False,
        )
        eg_table = expected_games_table(features)
    except Exception:  # noqa: BLE001 - board still works with fallbacks
        eg_table = {}
    work["expected_games"] = work["position"].map(
        lambda p: eg_table.get(str(p), FALLBACK_EXPECTED_GAMES.get(str(p), 13.0))
    )
    work["projected_ppr_17"] = work["projected_season_ppr"] * (
        REG_SEASON_GAMES / work["expected_games"]
    )

    frames: list[pd.DataFrame] = []
    for season, group in work.groupby("target_season", sort=True):
        g = group.sort_values("projected_ppr_17", ascending=False).reset_index(drop=True)
        g["overall_rank"] = np.arange(1, len(g) + 1)
        g["position_rank"] = g.groupby("position")["projected_ppr_17"].rank(
            ascending=False, method="first"
        ).astype(int)

        replacement: dict[str, float] = {}
        for pos, pg in g.groupby("position"):
            cutoff = REPLACEMENT_RANK.get(str(pos), 24)
            ordered = pg.sort_values("projected_ppr_17", ascending=False)["projected_ppr_17"]
            if len(ordered) >= cutoff:
                replacement[str(pos)] = float(ordered.iloc[cutoff - 1])
            elif len(ordered):
                replacement[str(pos)] = float(ordered.iloc[-1])
            else:
                replacement[str(pos)] = 0.0
        g["vorp"] = [
            float(row.projected_ppr_17) - replacement.get(str(row.position), 0.0)
            for row in g.itertuples(index=False)
        ]
        g["vorp_rank"] = g["vorp"].rank(ascending=False, method="first").astype(int)
        frames.append(g)

    board = pd.concat(frames, ignore_index=True)
    board = _attach_adp(board)
    board["adp_vs_model"] = board["adp_rank"] - board["overall_rank"]

    cols = [
        "target_season",
        "overall_rank",
        "vorp_rank",
        "position_rank",
        "gsis_id",
        "player_name",
        "position",
        "team",
        "projected_season_ppr",
        "projected_ppr_17",
        "expected_games",
        "vorp",
        "adp_rank",
        "adp_vs_model",
        "prior_ppr",
        "prior_games",
        "actual_ppr",
        "pred_q25",
        "pred_q75",
    ]
    return board[[c for c in cols if c in board.columns]]


def save_player_board(board: pd.DataFrame) -> str:
    CSV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = CSV_OUTPUT_DIR / LEAGUE_PLAYER_BOARD_CSV
    board.to_csv(path, index=False)
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the production player ranking board (ADP+opportunity model)"
    )
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        default=list(LEAGUE_SEASONS) + [LEAGUE_TARGET_SEASON],
    )
    args = parser.parse_args()

    board = build_player_board(seasons=args.seasons)
    path = save_player_board(board)
    print(f"rows={len(board)} -> {path}")
    latest = int(board["target_season"].max()) if len(board) else None
    if latest is not None:
        top = board[board["target_season"] == latest].nsmallest(15, "vorp_rank")
        print(f"\ntop 15 by VORP for {latest}:")
        print(
            top[
                ["vorp_rank", "player_name", "position", "projected_ppr_17", "vorp", "adp_rank"]
            ]
            .round(1)
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
