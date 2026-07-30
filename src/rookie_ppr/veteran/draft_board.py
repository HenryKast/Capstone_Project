"""
Unified redraft draft board: veterans (next-season ML) + rookies (redraft ML).

Two projection columns per player:
  projected_season_ppr — model expected value; averages over missed games.
  projected_ppr_17     — same value put on a 17-game pace, which is the basis
                         most public projections (ESPN, etc.) publish.

The pace conversion uses expected games played, estimated from history by
position and prior-season tier, because starters miss ~3 games per season on
average and the model's expected value already prices that in.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rookie_ppr.config import CSV_OUTPUT_DIR, INCOMING_DRAFT_YEAR
from rookie_ppr.veteran.config import VET_DRAFT_BOARD_CSV, VET_FEATURES_CSV

REG_SEASON_GAMES = 17
ROOKIE_ML_FEATURES = "ml_features.csv"

# Availability is estimated from starter-caliber seasons only (prior-season top 24
# at the position) and applied uniformly within a position, so the pace conversion
# never reorders players or rewards a backup for an expected-games denominator.
STARTER_RANK_CUTOFF = 24
EXPECTED_GAMES_MIN = 10.0
FALLBACK_EXPECTED_GAMES = {"QB": 13.3, "RB": 12.9, "WR": 13.8, "TE": 13.0}
AVAILABILITY_SEASON_MIN = 2012

# Replacement level for a 10-team league (QB/RB/RB/WR/WR/TE/FLEX starters),
# expressed as the positional rank whose projection counts as freely available.
REPLACEMENT_RANK = {"QB": 10, "RB": 25, "WR": 30, "TE": 10}


def expected_games_table(features: pd.DataFrame) -> dict[str, float]:
    """Mean games played in season+1 for prior-season starters, by position."""
    need = {"gsis_id", "position", "season", "games", "ppr"}
    if features.empty or not need.issubset(features.columns):
        return {}

    d = features[["gsis_id", "position", "season", "games", "ppr"]].copy()
    d["season"] = pd.to_numeric(d["season"], errors="coerce")
    d["games"] = pd.to_numeric(d["games"], errors="coerce")
    d["ppr"] = pd.to_numeric(d["ppr"], errors="coerce")
    d = d.dropna(subset=["gsis_id", "season"]).sort_values(["gsis_id", "season"])

    d["games_next"] = d.groupby("gsis_id")["games"].shift(-1)
    d["season_next"] = d.groupby("gsis_id")["season"].shift(-1)
    d = d[(d["season_next"] == d["season"] + 1) & (d["season"] >= AVAILABILITY_SEASON_MIN)]
    if d.empty:
        return {}

    d["pos_rank"] = d.groupby(["season", "position"])["ppr"].rank(ascending=False, method="first")
    starters = d[d["pos_rank"] <= STARTER_RANK_CUTOFF].dropna(subset=["games_next"])

    table: dict[str, float] = {}
    for pos, series in starters.groupby("position")["games_next"]:
        if len(series) < 20:
            continue
        table[str(pos)] = float(np.clip(series.mean(), EXPECTED_GAMES_MIN, REG_SEASON_GAMES))
    return table


def _expected_games(position: str, table: dict[str, float]) -> float:
    val = table.get(position)
    if val is not None:
        return val
    return FALLBACK_EXPECTED_GAMES.get(position, 13.0)


def _replacement_pace(board: pd.DataFrame) -> dict[str, float]:
    """Pace of the replacement-level player at each position."""
    out: dict[str, float] = {}
    for pos, g in board.groupby("position"):
        cutoff = REPLACEMENT_RANK.get(str(pos), 24)
        ordered = g.sort_values("projected_ppr_17", ascending=False)["projected_ppr_17"]
        if ordered.empty:
            continue
        idx = min(int(cutoff), len(ordered)) - 1
        out[str(pos)] = float(ordered.iloc[idx])
    return out


def _veteran_projection_rows(veteran: pd.DataFrame) -> tuple[pd.DataFrame, int | None]:
    """
    Rows from the newest season in the panel, i.e. the only season whose
    forecast is still unplayed. Returns (rows, source_season).
    """
    if veteran.empty or "predicted_ppr_next" not in veteran.columns:
        return pd.DataFrame(), None

    v = veteran.copy()
    for col in ("season", "target_season", "predicted_ppr_next", "games", "ppr"):
        if col in v.columns:
            v[col] = pd.to_numeric(v[col], errors="coerce")
    v = v[v["predicted_ppr_next"].notna()]
    if v.empty:
        return pd.DataFrame(), None

    source_season = int(v["season"].max())
    latest = v[v["season"] == source_season].copy()
    # Drop players with no snap of prior-season production (retired / never active)
    if "games" in latest.columns:
        latest = latest[latest["games"].fillna(0) > 0]
    if latest.empty:
        return pd.DataFrame(), source_season

    latest = latest.sort_values("predicted_ppr_next", ascending=False)
    latest = latest.drop_duplicates(subset=["gsis_id"], keep="first")

    out = pd.DataFrame(
        {
            "gsis_id": latest.get("gsis_id"),
            "player_name": latest.get("player_name"),
            "position": latest.get("position"),
            "team": latest.get("team"),
            "player_type": "veteran",
            "source_season": source_season,
            "prior_ppr": latest.get("ppr"),
            "prior_games": latest.get("games"),
            "projected_season_ppr": latest["predicted_ppr_next"],
            "ppr_low": latest.get("ppr_low"),
            "ppr_high": latest.get("ppr_high"),
            "draft_year": np.nan,
        }
    ).reset_index(drop=True)
    return out, source_season


def _rookie_projection_rows(rookies: pd.DataFrame, *, draft_year: int) -> pd.DataFrame:
    if rookies.empty:
        return pd.DataFrame()

    r = rookies.copy()
    r["draft_year"] = pd.to_numeric(r.get("draft_year"), errors="coerce")
    r = r[r["draft_year"] == int(draft_year)]
    pred_col = next(
        (c for c in ("predicted_rookie_ppr", "predicted_ppr") if c in r.columns), None
    )
    if pred_col is None:
        return pd.DataFrame()
    r[pred_col] = pd.to_numeric(r[pred_col], errors="coerce")
    r = r[r[pred_col].notna()]
    if r.empty:
        return pd.DataFrame()

    team_col = next((c for c in ("draft_team", "team", "nfl_team") if c in r.columns), None)
    return pd.DataFrame(
        {
            "gsis_id": r.get("gsis_id"),
            "player_name": r.get("player_name"),
            "position": r.get("position"),
            "team": r[team_col] if team_col else "",
            "player_type": "rookie",
            "source_season": np.nan,
            "prior_ppr": np.nan,
            "prior_games": np.nan,
            "projected_season_ppr": r[pred_col],
            "ppr_low": r.get("ppr_low"),
            "ppr_high": r.get("ppr_high"),
            "draft_year": int(draft_year),
        }
    ).reset_index(drop=True)


def build_draft_board(
    veteran_features: pd.DataFrame | None = None,
    *,
    draft_year: int | None = None,
    rookie_features_path: Path | None = None,
) -> pd.DataFrame:
    """Merge veteran + rookie projections; rank by 17-game pace (synthetic ADP)."""
    if veteran_features is None:
        path = CSV_OUTPUT_DIR / VET_FEATURES_CSV
        veteran_features = pd.read_csv(path, low_memory=False) if path.exists() else pd.DataFrame()

    vet, source_season = _veteran_projection_rows(veteran_features)
    target_season = (source_season + 1) if source_season is not None else INCOMING_DRAFT_YEAR
    draft_year = int(draft_year or target_season)

    rookie_path = rookie_features_path or (CSV_OUTPUT_DIR / ROOKIE_ML_FEATURES)
    rook = pd.DataFrame()
    if rookie_path.exists():
        rook = _rookie_projection_rows(
            pd.read_csv(rookie_path, low_memory=False), draft_year=draft_year
        )

    board = pd.concat([f for f in (vet, rook) if not f.empty], ignore_index=True)
    if board.empty:
        return board

    board["projected_season_ppr"] = pd.to_numeric(board["projected_season_ppr"], errors="coerce")
    board = board[board["projected_season_ppr"].notna()].copy()

    table = expected_games_table(veteran_features)
    board["expected_games"] = [_expected_games(str(pos), table) for pos in board["position"]]
    board["projected_ppr_17"] = (
        board["projected_season_ppr"] / board["expected_games"] * REG_SEASON_GAMES
    ).round(2)
    board["projected_ppg"] = (board["projected_ppr_17"] / REG_SEASON_GAMES).round(2)

    # Rank by value over replacement so the board is draftable; raw points still
    # sit alongside it for comparison against public projections.
    replacement = _replacement_pace(board)
    board["replacement_ppr_17"] = [
        replacement.get(str(pos), np.nan) for pos in board["position"]
    ]
    board["vorp_17"] = (board["projected_ppr_17"] - board["replacement_ppr_17"]).round(2)

    board["position_rank"] = (
        board.groupby("position")["projected_ppr_17"].rank(ascending=False, method="first").astype(int)
    )
    board = board.sort_values(
        ["vorp_17", "projected_ppr_17"], ascending=False
    ).reset_index(drop=True)
    board["synthetic_adp_rank"] = np.arange(1, len(board) + 1)
    board["synthetic_adp"] = board["synthetic_adp_rank"].astype(float)

    board["target_season"] = target_season
    board["projected_season_ppr"] = board["projected_season_ppr"].round(2)
    board["expected_games"] = board["expected_games"].round(2)
    cols = [
        "synthetic_adp_rank",
        "player_name",
        "position",
        "position_rank",
        "team",
        "player_type",
        "projected_ppr_17",
        "projected_ppg",
        "vorp_17",
        "projected_season_ppr",
        "expected_games",
        "replacement_ppr_17",
        "prior_ppr",
        "prior_games",
        "ppr_low",
        "ppr_high",
        "target_season",
        "source_season",
        "draft_year",
        "gsis_id",
        "synthetic_adp",
    ]
    return board[[c for c in cols if c in board.columns]]


def save_draft_board(board: pd.DataFrame, path: Path | None = None) -> Path:
    dest = path or (CSV_OUTPUT_DIR / VET_DRAFT_BOARD_CSV)
    dest.parent.mkdir(parents=True, exist_ok=True)
    board.to_csv(dest, index=False)
    return dest


def load_draft_board(path: Path | None = None) -> pd.DataFrame:
    dest = path or (CSV_OUTPUT_DIR / VET_DRAFT_BOARD_CSV)
    if not dest.exists():
        raise FileNotFoundError(f"Missing draft board at {dest}. Run veteran compile.")
    return pd.read_csv(dest, low_memory=False)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build unified draft board CSV (no retrain)")
    parser.add_argument("--draft-year", type=int, default=None)
    args = parser.parse_args()
    board = build_draft_board(draft_year=args.draft_year)
    if board.empty:
        raise SystemExit("Board empty — run: python -m rookie_ppr.veteran.compile")
    path = save_draft_board(board)
    target = int(board["target_season"].iloc[0])
    print(f"Wrote {len(board)} players (target season {target}) to {path}")
    print(board.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
