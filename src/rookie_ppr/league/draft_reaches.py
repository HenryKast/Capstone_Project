"""Biggest ADP reaches / falls from each league draft.

For every season, take the 20 skill-player picks with the largest absolute gap
between FantasyPros ADP rank and actual overall pick, then order them from
greatest reach (drafted earlier than ADP) to greatest fall (drafted later).
"""
from __future__ import annotations

import argparse

import pandas as pd

from rookie_ppr.league.config import (
    CSV_OUTPUT_DIR,
    LEAGUE_BACKTEST_ROSTERS_CSV,
    LEAGUE_DRAFT_REACHES_CSV,
    LEAGUE_MANAGERS_CSV,
    LEAGUE_SEASONS,
    LEAGUE_TEAMS_CSV,
    MODELED_POSITIONS,
)
from rookie_ppr.league.espn_api import fetch_view
from rookie_ppr.veteran.config import VET_FEATURES_CSV

TOP_N_PER_SEASON = 20
# Only count market-relevant picks; deep ADP is too noisy for reach/fall ranking.
MAX_ADP_RANK = 122.0
# Ignore late-round noise: only grade picks drafted overall before pick 100.
MAX_OVERALL_PICK = 99

REACH_COLS = [
    "season",
    "team_id",
    "manager_name",
    "team_name",
    "player_name",
    "position",
    "adp_rank",
    "overall_pick",
    "draft_delta",
    "abs_delta",
    "finish_label",
    "avg_ppg",
    "actual_season_ppr",
    "games",
]


def _ascii(name: object) -> str:
    return str(name).encode("ascii", "ignore").decode("ascii").strip()


def load_manager_map(seasons: list[int] | None = None) -> pd.DataFrame:
    """(season, team_id) -> primary owner display name.

    Prefers shipped ``league_managers.csv`` (no ESPN cookies). Falls back to
    the local ESPN mTeam cache when regenerating on a machine that has it.
    """
    seasons = list(seasons or LEAGUE_SEASONS)
    path = CSV_OUTPUT_DIR / LEAGUE_MANAGERS_CSV
    if path.exists():
        frame = pd.read_csv(path)
        if not frame.empty and "manager_name" in frame.columns:
            out = frame[frame["season"].isin(seasons)].copy()
            out["manager_name"] = out["manager_name"].map(_ascii)
            return out.reset_index(drop=True)

    rows: list[dict] = []
    for season in seasons:
        try:
            node = fetch_view(season, ["mTeam"], force=False)
        except Exception:
            continue
        members = {
            m.get("id"): m for m in (node.get("members") or []) if m.get("id")
        }
        for team in node.get("teams") or []:
            tid = team.get("id")
            if tid is None:
                continue
            owner = members.get(team.get("primaryOwner") or "", {})
            first = str(owner.get("firstName") or "").strip()
            last = str(owner.get("lastName") or "").strip()
            name = _ascii(f"{first} {last}".strip())
            if not name:
                name = f"Team {int(tid)}"
            rows.append(
                {
                    "season": int(season),
                    "team_id": int(tid),
                    "manager_name": name,
                }
            )
    return pd.DataFrame(rows)


def export_league_managers(seasons: list[int] | None = None) -> pd.DataFrame:
    """Build and write ``league_managers.csv`` from ESPN mTeam cache.

    Seasons outside ``seasons`` are preserved from the existing file, so a
    scoped refresh (say, just the upcoming season) never drops league history.
    """
    # Force ESPN path even if an older CSV exists
    seasons = list(seasons or LEAGUE_SEASONS)
    rows: list[dict] = []
    for season in seasons:
        try:
            node = fetch_view(season, ["mTeam"], force=False)
        except Exception:
            continue
        members = {
            m.get("id"): m for m in (node.get("members") or []) if m.get("id")
        }
        for team in node.get("teams") or []:
            tid = team.get("id")
            if tid is None:
                continue
            owner = members.get(team.get("primaryOwner") or "", {})
            first = str(owner.get("firstName") or "").strip()
            last = str(owner.get("lastName") or "").strip()
            name = _ascii(f"{first} {last}".strip())
            if not name:
                name = f"Team {int(tid)}"
            rows.append(
                {
                    "season": int(season),
                    "team_id": int(tid),
                    "manager_name": name,
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    path = CSV_OUTPUT_DIR / LEAGUE_MANAGERS_CSV
    if path.exists():
        prior = pd.read_csv(path)
        if not prior.empty:
            kept = prior[~prior["season"].isin(out["season"].unique())]
            out = pd.concat([kept, out], ignore_index=True)
    out = out.sort_values(["season", "team_id"]).drop_duplicates(
        subset=["season", "team_id"], keep="last"
    )
    CSV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out.reset_index(drop=True)


def _season_actuals() -> pd.DataFrame:
    """Per-player season PPR, games, PPG, and position finish label (e.g. WR3)."""
    path = CSV_OUTPUT_DIR / VET_FEATURES_CSV
    frame = pd.read_csv(
        path,
        low_memory=False,
        usecols=["gsis_id", "season", "position", "ppr", "games", "ppr_per_game"],
    )
    frame = frame[frame["position"].isin(MODELED_POSITIONS)].copy()
    frame["season"] = pd.to_numeric(frame["season"], errors="coerce")
    frame["ppr"] = pd.to_numeric(frame["ppr"], errors="coerce").fillna(0.0)
    frame["games"] = pd.to_numeric(frame["games"], errors="coerce")
    frame["avg_ppg"] = pd.to_numeric(frame["ppr_per_game"], errors="coerce")
    missing_ppg = frame["avg_ppg"].isna() & frame["games"].notna() & (frame["games"] > 0)
    frame.loc[missing_ppg, "avg_ppg"] = (
        frame.loc[missing_ppg, "ppr"] / frame.loc[missing_ppg, "games"]
    )
    frame = frame.dropna(subset=["gsis_id", "season"])
    frame = frame.drop_duplicates(subset=["gsis_id", "season"], keep="first")
    frame["pos_rank"] = (
        frame.groupby(["season", "position"])["ppr"]
        .rank(ascending=False, method="min")
        .astype(int)
    )
    frame["finish_label"] = frame["position"] + frame["pos_rank"].astype(str)
    return frame[
        ["gsis_id", "season", "ppr", "games", "avg_ppg", "finish_label"]
    ].rename(columns={"ppr": "actual_season_ppr"})


def build_draft_reaches(
    seasons: list[int] | None = None,
    top_n: int = TOP_N_PER_SEASON,
) -> pd.DataFrame:
    seasons = list(seasons or LEAGUE_SEASONS)
    draft = pd.read_csv(CSV_OUTPUT_DIR / LEAGUE_BACKTEST_ROSTERS_CSV)
    draft = draft[
        draft["season"].isin(seasons) & draft["position"].isin(MODELED_POSITIONS)
    ].copy()
    draft["adp_rank"] = pd.to_numeric(draft["adp_rank"], errors="coerce")
    draft["overall_pick"] = pd.to_numeric(draft["overall_pick"], errors="coerce")
    draft = draft.dropna(subset=["adp_rank", "overall_pick"])
    draft = draft[
        (draft["adp_rank"] <= MAX_ADP_RANK)
        & (draft["overall_pick"] <= MAX_OVERALL_PICK)
    ]
    draft["draft_delta"] = draft["adp_rank"] - draft["overall_pick"]
    draft["abs_delta"] = draft["draft_delta"].abs()

    managers = load_manager_map(seasons)
    teams = pd.read_csv(CSV_OUTPUT_DIR / LEAGUE_TEAMS_CSV)
    team_names = teams[["season", "team_id", "team_name"]].copy()
    team_names["team_name"] = team_names["team_name"].map(_ascii)

    actuals = _season_actuals()
    draft = draft.drop(columns=["actual_season_ppr"], errors="ignore")
    out = draft.merge(managers, on=["season", "team_id"], how="left")
    out = out.merge(team_names, on=["season", "team_id"], how="left")
    out = out.merge(actuals, on=["gsis_id", "season"], how="left")
    out["manager_name"] = out["manager_name"].fillna(
        "Team " + out["team_id"].astype(str)
    )
    out["team_name"] = out["team_name"].map(
        lambda v: _ascii(v) if pd.notna(v) else ""
    )
    out["player_name"] = out["player_name"].map(_ascii)
    out["finish_label"] = out["finish_label"].fillna("-")

    tops: list[pd.DataFrame] = []
    for season, g in out.groupby("season"):
        top = g.nlargest(int(top_n), "abs_delta").copy()
        top = top.sort_values(
            ["draft_delta", "abs_delta", "overall_pick"],
            ascending=[False, False, True],
        )
        tops.append(top)

    if not tops:
        return pd.DataFrame(columns=REACH_COLS)

    result = pd.concat(tops, ignore_index=True)
    # Combined default view: greatest reach → greatest fall across all seasons
    result = result.sort_values(
        ["draft_delta", "abs_delta", "season", "overall_pick"],
        ascending=[False, False, True, True],
    )
    keep = [c for c in REACH_COLS if c in result.columns]
    return result[keep].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Biggest FantasyPros ADP reaches / falls per league draft"
    )
    parser.add_argument("--seasons", type=int, nargs="+", default=list(LEAGUE_SEASONS))
    parser.add_argument("--top-n", type=int, default=TOP_N_PER_SEASON)
    args = parser.parse_args()
    out = build_draft_reaches(args.seasons, top_n=args.top_n)
    path = CSV_OUTPUT_DIR / LEAGUE_DRAFT_REACHES_CSV
    CSV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print(f"rows={len(out)} seasons={out['season'].nunique() if not out.empty else 0} -> {path}")
    if not out.empty:
        print(
            out.groupby("season")
            .size()
            .rename("n")
            .to_string()
        )
        print("--- top reaches ---")
        print(
            out.head(5)[
                ["season", "player_name", "adp_rank", "overall_pick", "draft_delta", "finish_label"]
            ].to_string(index=False)
        )
        print("--- top falls ---")
        print(
            out.tail(5)[
                ["season", "player_name", "adp_rank", "overall_pick", "draft_delta", "finish_label"]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
