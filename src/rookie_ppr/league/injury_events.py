"""Detect major fantasy starter injury weeks from league roster churn.

Events annotate week-by-week odds charts (bandaid markers):

1. **Out for season** — early-drafted / heavily started skill player leaves the
   roster and never returns in the regular season.
2. **3+ week absence** — drafted or starting skill player is effectively out
   for 3+ consecutive regular-season weeks (ESPN proj ~0 and not started,
   excluding NFL bye weeks), then may return.
"""
from __future__ import annotations

import argparse

import pandas as pd

from rookie_ppr.league.config import (
    CSV_OUTPUT_DIR,
    LEAGUE_DRAFT_CSV,
    LEAGUE_ROSTERS_CSV,
    LEAGUE_SEASONS,
    LEAGUE_SETTINGS_CSV,
    MODELED_POSITIONS,
)
from rookie_ppr.league.simulate_season import team_games_and_byes
from rookie_ppr.utils import normalize_team_abbr

LEAGUE_INJURY_EVENTS_CSV = "league_injury_events.csv"

# Top-4-round-ish draft capital counts as "major" for season-ending exits.
MAJOR_OVERALL_PICK = 40
MIN_STARTS = 4
MIN_STARTER_PROJ = 10.0

# 3+ week absences: any drafted skill player who had been a real starter.
MIN_ABSENCE_WEEKS = 3
INACTIVE_PROJ = 1.5
MIN_PRIOR_STARTER_PROJ = 8.0


def _pick_label(round_n: int | None, round_pick: int | None) -> str | None:
    if round_n is None or round_pick is None:
        return None
    return f"{int(round_n)}.{int(round_pick):02d}"


def _ascii(name: object) -> str:
    return str(name).encode("ascii", "ignore").decode("ascii").strip()


def _is_inactive_week(
    *,
    proj: float,
    is_starter: bool,
    week: int,
    pro_team: object,
    byes: dict[str, set[int]],
) -> bool:
    abbr = normalize_team_abbr(pro_team)
    if abbr and week in byes.get(abbr, set()):
        return False
    return (not bool(is_starter)) and proj <= INACTIVE_PROJ


def detect_injury_events(
    seasons: list[int] | None = None,
) -> pd.DataFrame:
    seasons = list(seasons or LEAGUE_SEASONS)
    rosters = pd.read_csv(CSV_OUTPUT_DIR / LEAGUE_ROSTERS_CSV)
    draft = pd.read_csv(CSV_OUTPUT_DIR / LEAGUE_DRAFT_CSV)
    settings = pd.read_csv(CSV_OUTPUT_DIR / LEAGUE_SETTINGS_CSV)
    reg_map = dict(zip(settings["season"].astype(int), settings["reg_weeks"].astype(int)))

    rosters = rosters[
        rosters["season"].isin(seasons) & rosters["position"].isin(MODELED_POSITIONS)
    ].copy()
    rosters["gsis_key"] = rosters["gsis_id"].astype(str)
    draft = draft[draft["season"].isin(seasons)].copy()
    draft["gsis_key"] = draft["gsis_id"].astype(str)

    events: list[dict] = []
    for season in seasons:
        reg_weeks = int(reg_map[season])
        _, byes = team_games_and_byes(season)
        season_r = rosters[rosters["season"] == season]
        season_d = draft[draft["season"] == season]

        for gsis_key, g in season_r.dropna(subset=["gsis_id"]).groupby("gsis_key"):
            g = g.sort_values(["week", "team_id"])
            drow = season_d[season_d["gsis_key"] == gsis_key]
            drafted = not drow.empty
            overall = int(drow.iloc[0]["overall_pick"]) if drafted else None
            round_n = int(drow.iloc[0]["round"]) if drafted else None
            round_pick = int(drow.iloc[0]["round_pick"]) if drafted else None
            name = _ascii(
                drow.iloc[0]["player_name"] if drafted else g.iloc[0]["player_name"]
            )
            pick = _pick_label(round_n, round_pick)

            starter_rows = g[g["is_starter"] == True]  # noqa: E712
            starter_proj = pd.to_numeric(
                starter_rows["espn_projected_points"], errors="coerce"
            )
            mean_starter_proj = float(starter_proj.mean()) if len(starter_proj) else 0.0
            n_starts = int(len(starter_rows))

            is_major = (
                (overall is not None and overall <= MAJOR_OVERALL_PICK)
                or (n_starts >= MIN_STARTS and mean_starter_proj >= MIN_STARTER_PROJ)
            )
            # 3+ week absences: drafted starters, or major waiver stars
            is_drafted_starter = drafted and n_starts >= 1 and mean_starter_proj >= MIN_PRIOR_STARTER_PROJ
            if not (is_major or is_drafted_starter):
                continue

            # --- Path A: vanishes for rest of regular season (majors only) ---
            last_week = int(g["week"].max())
            out_for_season = False
            if is_major and 2 <= last_week < reg_weeks:
                prior_starts = g[(g["week"] < last_week) & (g["is_starter"] == True)]  # noqa: E712
                if len(prior_starts) >= 2:
                    prior_proj = pd.to_numeric(
                        prior_starts["espn_projected_points"], errors="coerce"
                    ).mean()
                    if float(prior_proj) >= MIN_PRIOR_STARTER_PROJ:
                        later = season_r[
                            (season_r["gsis_key"] == gsis_key)
                            & (season_r["week"] > last_week)
                            & (season_r["week"] <= reg_weeks)
                        ]
                        if later.empty:
                            team_id = int(g[g["week"] == last_week].iloc[-1]["team_id"])
                            as_of = last_week + 1
                            label = f"{name} out for season"
                            if pick:
                                label = f"{label} ({pick})"
                            events.append(
                                {
                                    "season": season,
                                    "team_id": team_id,
                                    "as_of_week": as_of,
                                    "end_week": reg_weeks,
                                    "n_weeks": reg_weeks - last_week,
                                    "player_name": name,
                                    "overall_pick": overall,
                                    "pick_label": pick,
                                    "kind": "out_for_season",
                                    "label": label,
                                }
                            )
                            out_for_season = True

            if out_for_season:
                continue

            # --- Path B: 3+ consecutive inactive weeks (then may return) ---
            g2 = g[g["week"] <= reg_weeks].copy()
            if g2.empty:
                continue
            g2["proj"] = pd.to_numeric(
                g2["espn_projected_points"], errors="coerce"
            ).fillna(0.0)
            # One row per week (prefer starter row if duplicated)
            g2 = g2.sort_values(["week", "is_starter"], ascending=[True, False])
            g2 = g2.drop_duplicates(subset=["week"], keep="first")

            week_map = {
                int(row.week): row for row in g2.itertuples(index=False)
            }
            weeks = list(range(1, reg_weeks + 1))

            # Build inactive flags; missing from roster mid-season also counts
            # as inactive if they had started earlier (dropped while hurt).
            had_start_before: dict[int, bool] = {}
            seen_start = False
            inactive: dict[int, bool] = {}
            team_at: dict[int, int] = {}
            for week in weeks:
                row = week_map.get(week)
                if row is not None and bool(row.is_starter):
                    seen_start = True
                had_start_before[week] = seen_start
                if row is None:
                    inactive[week] = seen_start  # gone after starting
                    continue
                team_at[week] = int(row.team_id)
                inactive[week] = _is_inactive_week(
                    proj=float(row.proj),
                    is_starter=bool(row.is_starter),
                    week=week,
                    pro_team=row.pro_team,
                    byes=byes,
                ) and seen_start

            # Find runs of inactive weeks length >= MIN_ABSENCE_WEEKS
            run_start: int | None = None
            for week in weeks + [reg_weeks + 1]:
                is_out = inactive.get(week, False) if week <= reg_weeks else False
                if is_out:
                    if run_start is None:
                        run_start = week
                else:
                    if run_start is not None:
                        run_end = week - 1
                        n = run_end - run_start + 1
                        if n >= MIN_ABSENCE_WEEKS:
                            # Require a healthy starter week before the run
                            prior = g2[
                                (g2["week"] < run_start) & (g2["is_starter"] == True)  # noqa: E712
                            ]
                            if not prior.empty and float(prior["proj"].mean()) >= MIN_PRIOR_STARTER_PROJ:
                                team_id = team_at.get(
                                    run_start - 1,
                                    team_at.get(run_start, int(g2.iloc[0]["team_id"])),
                                )
                                # Prefer team from last active week
                                if run_start - 1 in team_at:
                                    team_id = team_at[run_start - 1]
                                elif run_start in team_at:
                                    team_id = team_at[run_start]
                                label = f"{name} out {n} weeks (W{run_start}-W{run_end})"
                                if pick:
                                    label = f"{label} ({pick})"
                                events.append(
                                    {
                                        "season": season,
                                        "team_id": int(team_id),
                                        "as_of_week": run_start,
                                        "end_week": run_end,
                                        "n_weeks": n,
                                        "player_name": name,
                                        "overall_pick": overall,
                                        "pick_label": pick,
                                        "kind": "out_3plus",
                                        "label": label,
                                    }
                                )
                        run_start = None

    out = pd.DataFrame(events)
    if out.empty:
        return out
    out["overall_pick"] = pd.to_numeric(out["overall_pick"], errors="coerce")
    out = out.sort_values(
        ["season", "team_id", "as_of_week", "overall_pick"],
        ascending=[True, True, True, True],
    )
    out = out.drop_duplicates(
        subset=["season", "team_id", "as_of_week", "player_name", "kind"]
    )
    return out.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect major starter injury weeks")
    parser.add_argument("--seasons", type=int, nargs="+", default=list(LEAGUE_SEASONS))
    args = parser.parse_args()
    out = detect_injury_events(args.seasons)
    path = CSV_OUTPUT_DIR / LEAGUE_INJURY_EVENTS_CSV
    CSV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print(f"events={len(out)} -> {path}")
    if not out.empty:
        print(out.groupby("kind").size().to_string())
        print(out[out["season"] == 2025][["as_of_week", "team_id", "kind", "label"]].to_string(index=False))


if __name__ == "__main__":
    main()
