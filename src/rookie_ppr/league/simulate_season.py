"""Simulate a fantasy season from projected rosters, then grade it against history.

Each drafted roster is turned into a weekly slot-legal lineup, every team-week
gets a mean and a standard deviation, and the real schedule is replayed many
times to produce win totals, points, seeds, playoff odds and title odds.

Two modelling notes worth stating outright:

*Weekly noise is calibrated, not assumed.* ESPN's own weekly projections are
stored alongside actual results in ``league_rosters``, so the spread of
"actual minus projected" tells us how wrong a projection of a given size
usually is. That relationship is fit on seasons strictly before the one being
simulated.

*Team scores are drawn as one normal per team-week* rather than one per player.
A team-week is the sum of nine independent player outcomes, and the sum of
independent normals is itself normal with the variances added, so this is
equivalent to drawing players individually while being far cheaper. It does drop
the mild right-skew of individual players, which matters little once nine are
added together.

The whole simulation runs on drafted rosters held all season. Real managers
stream and replace injuries, so projected points sit below actual points by a
systematic margin; the informative comparison is the ordering of teams, not the
absolute level.
"""
from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import pandas as pd

from rookie_ppr.ingest_nfl import load_schedules
from rookie_ppr.league.backtest_rosters import LEAGUE_BACKTEST_ROSTERS_CSV
from rookie_ppr.league.blend import (
    DEFAULT_BLEND_MODEL_WEIGHT,
    blend_weight_map,
    fit_blend_weights_walkforward,
)
from rookie_ppr.league.config import (
    CSV_OUTPUT_DIR,
    LEAGUE_BLEND_WEIGHTS_CSV,
    LEAGUE_MATCHUPS_CSV,
    LEAGUE_ROSTERS_CSV,
    LEAGUE_SEASONS,
    LEAGUE_SETTINGS_CSV,
    LEAGUE_SIM_GRADES_CSV,
    LEAGUE_SIM_TEAMS_CSV,
    LEAGUE_TEAMS_CSV,
)
from rookie_ppr.utils import normalize_team_abbr

# Starting lineup: one QB, two RB, two WR, one TE, one FLEX from the leftovers.
# D/ST and K are handled separately from league averages.
FLEX_POSITIONS = ("RB", "WR", "TE")
SLOT_PLAN = (("QB", 1), ("RB", 2), ("WR", 2), ("TE", 1))
N_FLEX = 1

DEFAULT_SIMS = 5000
PLAYOFF_TEAMS = 6
PLAYOFF_ROUNDS = 3

PROJECTION_SOURCES = ("model", "adp", "blend")


# ------------------------------------------------------- schedule / calibration


def team_games_and_byes(season: int) -> tuple[dict[str, int], dict[str, set[int]]]:
    """Games played and bye weeks per NFL team, from the real schedule."""
    sched = load_schedules([int(season)])
    games: dict[str, int] = {}
    weeks_played: dict[str, set[int]] = {}
    if sched.empty:
        return games, {}
    if "game_type" in sched.columns:
        sched = sched[sched["game_type"].astype(str).str.upper().isin(["REG", "REGULAR"])]
    for row in sched.itertuples(index=False):
        week = int(getattr(row, "week", 0) or 0)
        for side in ("home_team", "away_team"):
            team = normalize_team_abbr(getattr(row, side, None))
            if not team:
                continue
            games[team] = games.get(team, 0) + 1
            weeks_played.setdefault(team, set()).add(week)
    max_week = max((w for weeks in weeks_played.values() for w in weeks), default=0)
    byes = {
        team: set(range(1, max_week + 1)) - weeks
        for team, weeks in weeks_played.items()
    }
    return games, byes


def calibrate_weekly_noise(
    league_rosters: pd.DataFrame, before_season: int
) -> dict[str, tuple[float, float]]:
    """Per position, standard deviation of a weekly score as ``a + b * projection``.

    Fit on started players in seasons before ``before_season``, by binning ESPN's
    projection and measuring the spread of the miss inside each bin.
    """
    pool = league_rosters[
        (league_rosters["season"] < before_season)
        & league_rosters["is_starter"]
        & league_rosters["actual_points"].notna()
        & league_rosters["espn_projected_points"].notna()
    ]
    out: dict[str, tuple[float, float]] = {}
    for position, group in pool.groupby("position"):
        if len(group) < 200:
            continue
        proj = group["espn_projected_points"].to_numpy(dtype=float)
        resid = group["actual_points"].to_numpy(dtype=float) - proj
        # Ten equal-count bins; a flat fallback if the projection barely varies.
        try:
            bins = pd.qcut(proj, 10, duplicates="drop")
        except ValueError:
            out[str(position)] = (float(np.std(resid)), 0.0)
            continue
        frame = pd.DataFrame({"proj": proj, "resid": resid, "bin": bins})
        summary = frame.groupby("bin", observed=True).agg(
            mid=("proj", "mean"), sd=("resid", "std")
        ).dropna()
        if len(summary) < 3:
            out[str(position)] = (float(np.std(resid)), 0.0)
            continue
        slope, intercept = np.polyfit(summary["mid"], summary["sd"], 1)
        out[str(position)] = (float(intercept), float(slope))
    return out


def noise_sd(noise: dict[str, tuple[float, float]], position: str, mean: float) -> float:
    intercept, slope = noise.get(position, (6.0, 0.35))
    return float(max(1.0, intercept + slope * max(mean, 0.0)))


def slot_average_stats(
    league_rosters: pd.DataFrame, before_season: int
) -> dict[str, tuple[float, float]]:
    """Mean and SD of a started D/ST or K week, from seasons before ``before_season``."""
    pool = league_rosters[
        (league_rosters["season"] < before_season)
        & league_rosters["is_starter"]
        & league_rosters["actual_points"].notna()
        & league_rosters["position"].isin(["D/ST", "K"])
    ]
    out: dict[str, tuple[float, float]] = {}
    for position, group in pool.groupby("position"):
        points = group["actual_points"].to_numpy(dtype=float)
        if len(points) < 50:
            continue
        out[str(position)] = (float(points.mean()), float(points.std()))
    return out


# ------------------------------------------------------------------- lineups


def _projection_column(
    roster: pd.DataFrame, source: str, *, blend_weight: float = DEFAULT_BLEND_MODEL_WEIGHT
) -> pd.Series:
    model = pd.to_numeric(roster["proj_season_ppr"], errors="coerce")
    curve = pd.to_numeric(roster["adp_curve_ppr"], errors="coerce")
    if source == "model":
        return model
    if source == "adp":
        return curve.where(curve.notna(), model)
    if source == "blend":
        is_model = roster["proj_source"] == "model"
        blended = blend_weight * model + (1.0 - blend_weight) * curve
        return blended.where(is_model & curve.notna(), model.where(model.notna(), curve))
    raise ValueError(f"Unknown projection source: {source}")


def weekly_team_moments(
    roster: pd.DataFrame,
    source: str,
    *,
    weeks: list[int],
    games: dict[str, int],
    byes: dict[str, set[int]],
    noise: dict[str, tuple[float, float]],
    slot_stats: dict[str, tuple[float, float]],
    default_games: int,
    blend_weight: float = DEFAULT_BLEND_MODEL_WEIGHT,
) -> tuple[np.ndarray, np.ndarray]:
    """(mean, sd) per week for one team, using its best legal lineup each week."""
    players = roster[roster["position"].isin(FLEX_POSITIONS + ("QB",))].copy()
    players["season_proj"] = _projection_column(
        players, source, blend_weight=blend_weight
    )
    players = players[players["season_proj"].notna()]

    per_game: list[tuple[str, float, set[int]]] = []
    for row in players.itertuples(index=False):
        team = normalize_team_abbr(getattr(row, "pro_team", None))
        n_games = games.get(team, default_games) if team else default_games
        n_games = n_games or default_games
        per_game.append(
            (row.position, float(row.season_proj) / n_games, byes.get(team, set()) if team else set())
        )

    means = np.zeros(len(weeks), dtype=float)
    variances = np.zeros(len(weeks), dtype=float)

    for w_idx, week in enumerate(weeks):
        available: dict[str, list[float]] = {"QB": [], "RB": [], "WR": [], "TE": []}
        for position, value, bye_weeks in per_game:
            if week in bye_weeks:
                continue
            available[position].append(value)
        for values in available.values():
            values.sort(reverse=True)

        started: list[tuple[str, float]] = []
        leftovers: list[tuple[str, float]] = []
        for position, count in SLOT_PLAN:
            pool = available[position]
            started += [(position, v) for v in pool[:count]]
            leftovers += [(position, v) for v in pool[count:] if position in FLEX_POSITIONS]
        leftovers.sort(key=lambda pair: pair[1], reverse=True)
        started += leftovers[:N_FLEX]

        week_mean = 0.0
        week_var = 0.0
        for position, value in started:
            week_mean += value
            week_var += noise_sd(noise, position, value) ** 2
        for slot in ("D/ST", "K"):
            mean, sd = slot_stats.get(slot, (7.0, 5.0))
            week_mean += mean
            week_var += sd**2

        means[w_idx] = week_mean
        variances[w_idx] = week_var

    return means, np.sqrt(variances)


# ---------------------------------------------------------------- simulation


def _seed_order(wins: np.ndarray, points: np.ndarray, divisions: np.ndarray) -> np.ndarray:
    """Seeds under this league's rule: division winners first, then wins, then points.

    Returns team indices ordered best to worst, shaped (n_sims, n_teams).
    """
    score = wins * 1e6 + points
    is_div_winner = np.zeros_like(score, dtype=bool)
    for division in np.unique(divisions):
        cols = np.flatnonzero(divisions == division)
        best = cols[np.argmax(score[:, cols], axis=1)]
        is_div_winner[np.arange(score.shape[0]), best] = True
    ranking = is_div_winner * 1e12 + score
    return np.argsort(-ranking, axis=1)


def _run_bracket(seed_order: np.ndarray, playoff_scores: np.ndarray) -> np.ndarray:
    """Champion team index per sim. Six teams, top two on bye, no reseeding."""
    n_sims = seed_order.shape[0]
    rows = np.arange(n_sims)
    seeds = [seed_order[:, i] for i in range(PLAYOFF_TEAMS)]

    def winner(a: np.ndarray, b: np.ndarray, week: int) -> np.ndarray:
        a_pts = playoff_scores[rows, a, week]
        b_pts = playoff_scores[rows, b, week]
        return np.where(a_pts >= b_pts, a, b)

    w36 = winner(seeds[2], seeds[5], 0)
    w45 = winner(seeds[3], seeds[4], 0)
    semi1 = winner(seeds[0], w45, 1)
    semi2 = winner(seeds[1], w36, 1)
    return winner(semi1, semi2, 2)


def simulate_one_season(
    season: int,
    source: str,
    *,
    rosters: pd.DataFrame,
    matchups: pd.DataFrame,
    league_rosters: pd.DataFrame,
    teams: pd.DataFrame,
    reg_weeks: int,
    n_sims: int,
    rng: np.random.Generator,
    blend_weight: float = DEFAULT_BLEND_MODEL_WEIGHT,
) -> pd.DataFrame:
    season_teams = teams[teams["season"] == season].sort_values("team_id")
    team_ids = season_teams["team_id"].to_numpy()
    divisions = season_teams["division_id"].to_numpy()
    index_of = {int(t): i for i, t in enumerate(team_ids)}
    n_teams = len(team_ids)

    weeks = list(range(1, reg_weeks + PLAYOFF_ROUNDS + 1))
    games, byes = team_games_and_byes(season)
    default_games = 17 if season >= 2021 else 16
    noise = calibrate_weekly_noise(league_rosters, before_season=season)
    slot_stats = slot_average_stats(league_rosters, before_season=season)
    # The first simulated season has no earlier league data to calibrate from.
    if not noise or not slot_stats:
        noise = noise or {}
        slot_stats = slot_stats or {}

    means = np.zeros((n_teams, len(weeks)))
    sds = np.zeros((n_teams, len(weeks)))
    season_rosters = rosters[rosters["season"] == season]
    for team_id, group in season_rosters.groupby("team_id"):
        idx = index_of.get(int(team_id))
        if idx is None:
            continue
        m, s = weekly_team_moments(
            group,
            source,
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

    # (n_sims, n_teams, n_weeks)
    draws = rng.normal(loc=means, scale=np.maximum(sds, 1e-6), size=(n_sims, n_teams, len(weeks)))
    np.clip(draws, 0.0, None, out=draws)

    schedule = matchups[
        (matchups["season"] == season)
        & (matchups["week"] <= reg_weeks)
        & matchups["opp_team_id"].notna()
    ]
    wins = np.zeros((n_sims, n_teams))
    for row in schedule.itertuples(index=False):
        i = index_of.get(int(row.team_id))
        j = index_of.get(int(row.opp_team_id))
        if i is None or j is None:
            continue
        w_idx = int(row.week) - 1
        wins[:, i] += (draws[:, i, w_idx] > draws[:, j, w_idx]).astype(float)

    points_for = draws[:, :, :reg_weeks].sum(axis=2)
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
            "source": source,
            "blend_model_weight": blend_weight if source == "blend" else np.nan,
            "team_id": team_ids,
            "proj_points_for": means[:, :reg_weeks].sum(axis=1),
            "sim_wins_mean": wins.mean(axis=0),
            "sim_wins_p10": np.percentile(wins, 10, axis=0),
            "sim_wins_p90": np.percentile(wins, 90, axis=0),
            "sim_points_mean": points_for.mean(axis=0),
            "sim_seed_mean": seeds.mean(axis=0),
            "playoff_odds": (seeds <= PLAYOFF_TEAMS).mean(axis=0),
            "title_odds": champion_counts.mean(axis=0),
        }
    )


# ------------------------------------------------------------------- grading


def grade_simulations(sim: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    """Compare projected team seasons to what actually happened."""
    actual = teams[
        ["season", "team_id", "team_name", "is_my_team", "wins", "points_for", "playoff_seed", "final_rank"]
    ].copy()
    merged = sim.merge(actual, on=["season", "team_id"], how="inner")
    merged["made_playoffs"] = (merged["playoff_seed"] <= PLAYOFF_TEAMS).astype(float)
    merged["was_champion"] = (merged["final_rank"] == 1).astype(float)

    rows: list[dict[str, Any]] = []
    for source, group in merged.groupby("source"):
        block: dict[str, Any] = {"source": source, "n_team_seasons": len(group)}
        block["wins_pearson_r"] = round(
            float(np.corrcoef(group["sim_wins_mean"], group["wins"])[0, 1]), 4
        )
        block["wins_spearman_r"] = round(
            float(group["sim_wins_mean"].corr(group["wins"], method="spearman")), 4
        )
        block["wins_mae"] = round(float((group["sim_wins_mean"] - group["wins"]).abs().mean()), 3)
        block["points_pearson_r"] = round(
            float(np.corrcoef(group["sim_points_mean"], group["points_for"])[0, 1]), 4
        )
        block["points_spearman_r"] = round(
            float(group["sim_points_mean"].corr(group["points_for"], method="spearman")), 4
        )
        block["points_bias"] = round(
            float((group["sim_points_mean"] - group["points_for"]).mean()), 1
        )
        block["seed_spearman_r"] = round(
            float(group["sim_seed_mean"].corr(group["playoff_seed"], method="spearman")), 4
        )
        # Brier score: mean squared error of the probability forecast, lower is
        # better; 0.25 is what always guessing 50% would score.
        block["playoff_brier"] = round(
            float(((group["playoff_odds"] - group["made_playoffs"]) ** 2).mean()), 4
        )
        block["title_brier"] = round(
            float(((group["title_odds"] - group["was_champion"]) ** 2).mean()), 4
        )

        # How many actual playoff teams sat in our projected top six each season?
        hits: list[float] = []
        for _, season_group in group.groupby("season"):
            projected_top = set(
                season_group.nlargest(PLAYOFF_TEAMS, "playoff_odds")["team_id"]
            )
            actual_top = set(
                season_group[season_group["made_playoffs"] == 1.0]["team_id"]
            )
            if actual_top:
                hits.append(len(projected_top & actual_top) / len(actual_top))
        block["playoff_hit_rate"] = round(float(np.mean(hits)), 4) if hits else None
        rows.append(block)

    return pd.DataFrame(rows)


# -------------------------------------------------------------------- driver


def run_backtest(
    seasons: list[int] | None = None,
    sources: tuple[str, ...] = PROJECTION_SOURCES,
    *,
    n_sims: int = DEFAULT_SIMS,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seasons = list(seasons or LEAGUE_SEASONS)
    rosters = pd.read_csv(CSV_OUTPUT_DIR / LEAGUE_BACKTEST_ROSTERS_CSV)
    draft = pd.read_csv(CSV_OUTPUT_DIR / "league_draft.csv")[
        ["season", "overall_pick", "pro_team"]
    ]
    rosters = rosters.merge(draft, on=["season", "overall_pick"], how="left")
    matchups = pd.read_csv(CSV_OUTPUT_DIR / LEAGUE_MATCHUPS_CSV)
    league_rosters = pd.read_csv(CSV_OUTPUT_DIR / LEAGUE_ROSTERS_CSV)
    teams = pd.read_csv(CSV_OUTPUT_DIR / LEAGUE_TEAMS_CSV)
    settings = pd.read_csv(CSV_OUTPUT_DIR / LEAGUE_SETTINGS_CSV)
    reg_weeks = dict(zip(settings["season"], settings["reg_weeks"]))

    blend_weights = fit_blend_weights_walkforward(rosters, seasons)
    weights_by_season = blend_weight_map(blend_weights)
    print("walk-forward blend weights (model share):")
    for row in blend_weights.itertuples(index=False):
        print(
            f"  {int(row.season)}: w={row.blend_model_weight:.2f} "
            f"({row.source}, fit_n={row.fit_n}, fit_r={row.fit_r})"
        )

    rng = np.random.default_rng(seed)
    frames: list[pd.DataFrame] = []
    for season in seasons:
        w = weights_by_season.get(int(season), DEFAULT_BLEND_MODEL_WEIGHT)
        for source in sources:
            frames.append(
                simulate_one_season(
                    season,
                    source,
                    rosters=rosters,
                    matchups=matchups,
                    league_rosters=league_rosters,
                    teams=teams,
                    reg_weeks=int(reg_weeks[season]),
                    n_sims=n_sims,
                    rng=rng,
                    blend_weight=w,
                )
            )
        print(f"  simulated {season} ({', '.join(sources)})")

    sim = pd.concat(frames, ignore_index=True)
    grades = grade_simulations(sim, teams)
    return sim, grades, blend_weights


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulate and grade projected fantasy seasons against league history"
    )
    parser.add_argument("--seasons", type=int, nargs="+", default=list(LEAGUE_SEASONS))
    parser.add_argument("--sims", type=int, default=DEFAULT_SIMS)
    parser.add_argument(
        "--sources", nargs="+", default=list(PROJECTION_SOURCES), choices=list(PROJECTION_SOURCES)
    )
    args = parser.parse_args()

    print(f"simulating {args.sims} seasons per league-season:")
    sim, grades, blend_weights = run_backtest(
        args.seasons, tuple(args.sources), n_sims=args.sims
    )

    CSV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sim.to_csv(CSV_OUTPUT_DIR / LEAGUE_SIM_TEAMS_CSV, index=False)
    grades.to_csv(CSV_OUTPUT_DIR / LEAGUE_SIM_GRADES_CSV, index=False)
    blend_weights.to_csv(CSV_OUTPUT_DIR / LEAGUE_BLEND_WEIGHTS_CSV, index=False)

    print(f"\nteam-seasons={len(sim)} -> {CSV_OUTPUT_DIR / LEAGUE_SIM_TEAMS_CSV}")
    print(f"blend weights -> {CSV_OUTPUT_DIR / LEAGUE_BLEND_WEIGHTS_CSV}")
    print("\ngrades by projection source:")
    print(grades.to_string(index=False))


if __name__ == "__main__":
    main()
