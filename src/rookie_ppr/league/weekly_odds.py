"""Week-by-week playoff and title odds, conditioned on results so far.

For each as-of week W:
  * lock actual H2H results for weeks 1..W
  * rebuild remaining-season strength from that week's ESPN roster
    (trades/waivers show up as roster membership; soft injury discount when
    ESPN projects ~0 and the player is not started, excluding NFL bye weeks)
  * update remaining-week player pace from in-season usage and scoring, shrunk
    toward the preseason prior so one spike cannot re-rate a player
  * Monte Carlo the rest of the regular season + playoff bracket

Week 0 is the preseason view: current rosters before kickoff, draft-day
rosters once week 1 has been played.
"""
from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import pandas as pd

from rookie_ppr.league.blend import (
    DEFAULT_BLEND_MODEL_WEIGHT,
    blend_weight_map,
    fit_blend_weights_walkforward,
)
from rookie_ppr.league.config import (
    CSV_OUTPUT_DIR,
    LEAGUE_BACKTEST_ROSTERS_CSV,
    LEAGUE_MATCHUPS_CSV,
    LEAGUE_ROSTERS_CSV,
    LEAGUE_SEASONS,
    LEAGUE_SETTINGS_CSV,
    LEAGUE_TARGET_SEASON,
    LEAGUE_TEAMS_CSV,
    MODELED_POSITIONS,
)
from rookie_ppr.league.in_season import (
    apply_in_season_update,
    load_rates_for_season,
    load_weekly_player_stats,
)
from rookie_ppr.league.simulate_season import (
    DEFAULT_SIMS,
    PLAYOFF_ROUNDS,
    PLAYOFF_TEAMS,
    _run_bracket,
    _seed_order,
    calibrate_weekly_noise,
    slot_average_stats,
    team_games_and_byes,
    weekly_team_moments,
)
from rookie_ppr.utils import normalize_team_abbr

LEAGUE_WEEKLY_ODDS_CSV = "league_weekly_odds.csv"
# Fewer sims than the full preseason run — many (season × week) cells.
DEFAULT_WEEKLY_SIMS = 2000
INJURY_PROJ_CEILING = 1.0
INJURY_MULT = 0.10


def _ascii_name(name: object, team_id: int) -> str:
    text = str(name).encode("ascii", "ignore").decode("ascii").strip()
    return text or f"Team {team_id}"


def completed_weeks(matchups: pd.DataFrame, season: int, reg_weeks: int) -> int:
    """Highest regular-season week that has actually been played.

    An upcoming season has a full schedule but no scores, so this returns 0 and
    the only honest as-of week is the preseason one.
    """
    played = matchups[
        (matchups["season"] == season)
        & (matchups["week"] <= reg_weeks)
        & matchups["opp_team_id"].notna()
        & matchups["points"].notna()
        & matchups["opp_points"].notna()
    ]
    # A scheduled-but-unplayed matchup comes back as 0-0.
    played = played[(played["points"] > 0) | (played["opp_points"] > 0)]
    return int(played["week"].max()) if not played.empty else 0


def _actual_wins_points(
    matchups: pd.DataFrame,
    *,
    season: int,
    as_of_week: int,
    reg_weeks: int,
    team_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative wins and points through ``as_of_week`` (regular season only)."""
    index_of = {int(t): i for i, t in enumerate(team_ids)}
    wins = np.zeros(len(team_ids), dtype=float)
    points = np.zeros(len(team_ids), dtype=float)
    if as_of_week <= 0:
        return wins, points

    games = matchups[
        (matchups["season"] == season)
        & (matchups["week"] <= as_of_week)
        & (matchups["week"] <= reg_weeks)
        & matchups["opp_team_id"].notna()
        & matchups["points"].notna()
        & matchups["opp_points"].notna()
    ]
    # Each H2H appears twice (once per team). Count from each team's row.
    for row in games.itertuples(index=False):
        i = index_of.get(int(row.team_id))
        if i is None:
            continue
        pts = float(row.points)
        points[i] += pts
        if pts > float(row.opp_points):
            wins[i] += 1.0
        elif pts == float(row.opp_points):
            wins[i] += 0.5
    return wins, points


def _draft_roster_frame(backtest: pd.DataFrame, draft: pd.DataFrame, season: int) -> pd.DataFrame:
    roster = backtest[backtest["season"] == season].copy()
    draft_teams = draft[draft["season"] == season][["overall_pick", "pro_team"]]
    if "pro_team" not in roster.columns:
        roster = roster.merge(draft_teams, on="overall_pick", how="left")
    return roster


def _live_roster_frame(
    league_rosters: pd.DataFrame,
    backtest: pd.DataFrame,
    *,
    season: int,
    week: int,
    byes: dict[str, set[int]],
    default_games: int,
    as_of_week: int | None = None,
    weekly_stats: pd.DataFrame | None = None,
    rates: dict[str, dict[str, float]] | None = None,
) -> pd.DataFrame:
    """Skill-player roster at ``week``, with season pace + soft injury discount."""
    snap = league_rosters[
        (league_rosters["season"] == season) & (league_rosters["week"] == week)
    ].copy()
    skill = snap[snap["position"].isin(MODELED_POSITIONS)].copy()
    if skill.empty:
        return skill

    proj = (
        backtest[backtest["season"] == season][
            ["gsis_id", "proj_season_ppr", "adp_curve_ppr", "proj_source"]
        ]
        .dropna(subset=["gsis_id"])
        .drop_duplicates(subset=["gsis_id"])
    )
    skill["gsis_id"] = skill["gsis_id"].astype(str)
    proj = proj.copy()
    proj["gsis_id"] = proj["gsis_id"].astype(str)
    merged = skill.merge(proj, on="gsis_id", how="left")

    espn_week = pd.to_numeric(merged["espn_projected_points"], errors="coerce").fillna(0.0)
    missing = merged["proj_season_ppr"].isna()
    merged.loc[missing, "proj_season_ppr"] = espn_week[missing].clip(lower=0.0) * default_games
    merged.loc[missing, "adp_curve_ppr"] = merged.loc[missing, "proj_season_ppr"]
    merged.loc[missing, "proj_source"] = "espn_weekly"

    update_through = as_of_week if as_of_week is not None else week
    merged = apply_in_season_update(
        merged,
        season=season,
        as_of_week=update_through,
        league_rosters=league_rosters,
        default_games=default_games,
        weekly_stats=weekly_stats,
        rates=rates,
    )

    # Soft injury / inactive: ESPN projects ~0 and player is not started, and
    # it is not their NFL bye week.
    for idx, row in merged.iterrows():
        team = normalize_team_abbr(row.get("pro_team"))
        on_bye = bool(team and week in byes.get(team, set()))
        if on_bye:
            continue
        proj_w = float(row["espn_projected_points"]) if pd.notna(row["espn_projected_points"]) else 0.0
        started = bool(row.get("is_starter"))
        if proj_w <= INJURY_PROJ_CEILING and not started:
            merged.at[idx, "proj_season_ppr"] = float(row["proj_season_ppr"]) * INJURY_MULT
            if pd.notna(row.get("adp_curve_ppr")):
                merged.at[idx, "adp_curve_ppr"] = float(row["adp_curve_ppr"]) * INJURY_MULT

    merged["proj_source"] = merged["proj_source"].fillna("espn_weekly")
    return merged


def simulate_as_of_week(
    season: int,
    as_of_week: int,
    *,
    backtest: pd.DataFrame,
    draft: pd.DataFrame,
    league_rosters: pd.DataFrame,
    matchups: pd.DataFrame,
    teams: pd.DataFrame,
    reg_weeks: int,
    n_sims: int,
    rng: np.random.Generator,
    blend_weight: float,
    noise: dict[str, tuple[float, float]],
    slot_stats: dict[str, tuple[float, float]],
    games: dict[str, int],
    byes: dict[str, set[int]],
    use_live_preseason: bool = False,
    weekly_stats: pd.DataFrame | None = None,
    rates: dict[str, dict[str, float]] | None = None,
) -> pd.DataFrame:
    season_teams = teams[teams["season"] == season].sort_values("team_id")
    team_ids = season_teams["team_id"].to_numpy()
    divisions = season_teams["division_id"].to_numpy()
    names = {
        int(r.team_id): _ascii_name(r.team_name, int(r.team_id))
        for r in season_teams.itertuples(index=False)
    }
    index_of = {int(t): i for i, t in enumerate(team_ids)}
    n_teams = len(team_ids)
    default_games = 17 if season >= 2021 else 16
    weeks = list(range(1, reg_weeks + PLAYOFF_ROUNDS + 1))

    season_roster_weeks = pd.to_numeric(
        league_rosters.loc[league_rosters["season"] == season, "week"], errors="coerce"
    ).dropna()
    max_roster_week = int(season_roster_weeks.max()) if not season_roster_weeks.empty else 0

    if max_roster_week <= 0 or (as_of_week <= 0 and not use_live_preseason):
        # For a season already under way, week 0 means draft day, so the drafted
        # roster is the right view even though later snapshots exist.
        roster_src = _draft_roster_frame(backtest, draft, season)
    else:
        snap_week = max(1, min(as_of_week, max_roster_week))
        roster_src = _live_roster_frame(
            league_rosters,
            backtest,
            season=season,
            week=snap_week,
            byes=byes,
            default_games=default_games,
            as_of_week=as_of_week,
            weekly_stats=weekly_stats,
            rates=rates,
        )

    means = np.zeros((n_teams, len(weeks)))
    sds = np.zeros((n_teams, len(weeks)))
    for team_id, group in roster_src.groupby("team_id"):
        idx = index_of.get(int(team_id))
        if idx is None:
            continue
        m, s = weekly_team_moments(
            group,
            "blend",
            weeks=weeks,
            games=games,
            byes=byes,
            noise=noise,
            slot_stats=slot_stats,
            default_games=default_games,
            blend_weight=blend_weight,
        )
        means[idx] = m
        sds[idx] = s

    draws = rng.normal(
        loc=means, scale=np.maximum(sds, 1e-6), size=(n_sims, n_teams, len(weeks))
    )
    np.clip(draws, 0.0, None, out=draws)

    wins0, points0 = _actual_wins_points(
        matchups,
        season=season,
        as_of_week=as_of_week,
        reg_weeks=reg_weeks,
        team_ids=team_ids,
    )
    wins = np.broadcast_to(wins0, (n_sims, n_teams)).copy()
    points_for = np.broadcast_to(points0, (n_sims, n_teams)).copy()

    schedule = matchups[
        (matchups["season"] == season)
        & (matchups["week"] > as_of_week)
        & (matchups["week"] <= reg_weeks)
        & matchups["opp_team_id"].notna()
    ]
    # Deduplicate to one row per matchup (home side only) to avoid double counting.
    seen: set[tuple[int, int, int]] = set()
    for row in schedule.itertuples(index=False):
        a, b = int(row.team_id), int(row.opp_team_id)
        key = (int(row.week), min(a, b), max(a, b))
        if key in seen:
            continue
        seen.add(key)
        i = index_of.get(a)
        j = index_of.get(b)
        if i is None or j is None:
            continue
        w_idx = int(row.week) - 1
        a_pts = draws[:, i, w_idx]
        b_pts = draws[:, j, w_idx]
        wins[:, i] += (a_pts > b_pts).astype(float)
        wins[:, j] += (b_pts > a_pts).astype(float)
        wins[:, i] += 0.5 * (a_pts == b_pts).astype(float)
        wins[:, j] += 0.5 * (b_pts == a_pts).astype(float)
        points_for[:, i] += a_pts
        points_for[:, j] += b_pts

    seed_order = _seed_order(wins, points_for, divisions)
    seeds = np.empty_like(seed_order)
    np.put_along_axis(
        seeds, seed_order, np.tile(np.arange(1, n_teams + 1), (n_sims, 1)), axis=1
    )
    champion = _run_bracket(seed_order, draws[:, :, reg_weeks:])
    champion_counts = np.zeros((n_sims, n_teams))
    champion_counts[np.arange(n_sims), champion] = 1.0

    return pd.DataFrame(
        {
            "season": season,
            "as_of_week": as_of_week,
            "team_id": team_ids,
            "team_name": [names[int(t)] for t in team_ids],
            "wins_to_date": wins0,
            "points_to_date": points0,
            "sim_wins_mean": wins.mean(axis=0),
            "playoff_odds": (seeds <= PLAYOFF_TEAMS).mean(axis=0),
            "title_odds": champion_counts.mean(axis=0),
            "sim_seed_mean": seeds.mean(axis=0),
        }
    )


def run_weekly_odds(
    seasons: list[int] | None = None,
    *,
    n_sims: int = DEFAULT_WEEKLY_SIMS,
    seed: int = 42,
) -> pd.DataFrame:
    seasons = list(seasons or LEAGUE_SEASONS)
    backtest = pd.read_csv(CSV_OUTPUT_DIR / LEAGUE_BACKTEST_ROSTERS_CSV)
    draft = pd.read_csv(CSV_OUTPUT_DIR / "league_draft.csv")
    if "pro_team" not in backtest.columns:
        backtest = backtest.merge(
            draft[["season", "overall_pick", "pro_team"]],
            on=["season", "overall_pick"],
            how="left",
        )
    league_rosters = pd.read_csv(CSV_OUTPUT_DIR / LEAGUE_ROSTERS_CSV)
    matchups = pd.read_csv(CSV_OUTPUT_DIR / LEAGUE_MATCHUPS_CSV)
    teams = pd.read_csv(CSV_OUTPUT_DIR / LEAGUE_TEAMS_CSV)
    settings = pd.read_csv(CSV_OUTPUT_DIR / LEAGUE_SETTINGS_CSV)
    reg_weeks_map = dict(zip(settings["season"].astype(int), settings["reg_weeks"].astype(int)))

    blend_weights = fit_blend_weights_walkforward(backtest, seasons)
    weights = blend_weight_map(blend_weights)

    rng = np.random.default_rng(seed)
    frames: list[pd.DataFrame] = []
    for season in seasons:
        if season not in reg_weeks_map:
            raise KeyError(
                f"{season} is not in {LEAGUE_SETTINGS_CSV}; ingest the season first: "
                f"python -m rookie_ppr.league.ingest --seasons {season}"
            )
        # An empty roster frame would simulate every team at zero points and
        # quietly emit coin-flip odds, so fail loudly instead.
        if not (backtest["season"] == season).any():
            raise ValueError(
                f"No {season} rows in {LEAGUE_BACKTEST_ROSTERS_CSV}; build them first: "
                f"python -m rookie_ppr.league.target_season --run --season {season}"
            )
        reg_weeks = int(reg_weeks_map[season])
        games, byes = team_games_and_byes(season)
        noise = calibrate_weekly_noise(league_rosters, before_season=season) or {}
        slot_stats = slot_average_stats(league_rosters, before_season=season) or {}
        w = weights.get(int(season), DEFAULT_BLEND_MODEL_WEIGHT)
        weekly_stats = load_weekly_player_stats(
            season, force=season >= LEAGUE_TARGET_SEASON
        )
        rates = load_rates_for_season(season)
        # Only weeks that have been played carry information; an upcoming
        # season stops at the preseason (week 0) view.
        last_week = min(reg_weeks, completed_weeks(matchups, season, reg_weeks))
        # Nothing has kicked off yet, so week 0 is the league as it stands right
        # now rather than a draft-day retrospective, and it should track waiver
        # and trade activity. Once week 1 is in the books it reverts to meaning
        # draft day, which is what the completed seasons already publish.
        preseason = last_week == 0
        for as_of in range(0, last_week + 1):
            frames.append(
                simulate_as_of_week(
                    season,
                    as_of,
                    backtest=backtest,
                    draft=draft,
                    league_rosters=league_rosters,
                    matchups=matchups,
                    teams=teams,
                    reg_weeks=reg_weeks,
                    n_sims=n_sims,
                    rng=rng,
                    blend_weight=w,
                    noise=noise,
                    slot_stats=slot_stats,
                    games=games,
                    byes=byes,
                    use_live_preseason=preseason,
                    weekly_stats=weekly_stats,
                    rates=rates,
                )
            )
        source = "current rosters" if preseason else "drafted rosters at W0"
        extra = ""
        if last_week > 0:
            extra = "; remaining pace updates from in-season usage/scoring"
        print(f"  weekly odds {season} weeks 0..{last_week} ({source}{extra})")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Week-by-week playoff and title odds with live roster snapshots"
    )
    parser.add_argument("--seasons", type=int, nargs="+", default=list(LEAGUE_SEASONS))
    parser.add_argument("--sims", type=int, default=DEFAULT_WEEKLY_SIMS)
    args = parser.parse_args()

    print(f"computing weekly odds ({args.sims} sims / week):")
    out = run_weekly_odds(args.seasons, n_sims=args.sims)
    path = CSV_OUTPUT_DIR / LEAGUE_WEEKLY_ODDS_CSV
    CSV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print(f"rows={len(out)} -> {path}")


if __name__ == "__main__":
    main()
