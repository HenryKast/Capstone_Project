"""In-season remaining-pace updates for the weekly ForeKast.

Preseason ``proj_season_ppr`` is a strong prior, but it cannot see a role that
opens in week 4. This module blends that prior with evidence from games already
played, shrinking hard when the sample is small:

  * one spike barely moves the projection (a 50-point week against a 12-point
    prior is about a 10% bump after one game)
  * four to six weeks of the same usage/scoring is enough to re-rate the player
  * usage (targets, carries, dropbacks) is trusted more than raw fantasy points,
    because scoring variance is mostly noise at the weekly grain

When nflverse weekly box scores are not in yet (Monday morning, or a 404 on an
in-progress season file), we fall back to league ``actual_points`` mixed with
ESPN's remaining-week projections, which already embed a usage view.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from rookie_ppr.league.config import (
    LEAGUE_SEASON_MIN,
    LEAGUE_TARGET_SEASON,
    MODELED_POSITIONS,
)
from rookie_ppr.nflverse_http import try_read_release_csv
from rookie_ppr.utils import normalize_position

# Equivalent prior games in the empirical-Bayes weight n / (n + k). k=10 means
# week 1 is ~9% of the blend, week 4 is 29%, week 10 is 50%.
PRIOR_GAMES = 10.0
# When usage is missing, scoring is noisier, so we pretend the prior is stronger.
SCORING_ONLY_PRIOR_GAMES = 14.0
USAGE_MIX = 0.70
# One-week scoring outliers (50-burgers) should not set the mean.
SCORING_WEEK_CAP = 35.0
# Seasons of weekly box scores used to fit PPR-per-opportunity.
RATE_LOOKBACK = 4
# Safety rails on the multiplicative update of the preseason total.
MULT_FLOOR = 0.45
MULT_CEILING = 2.20

WEEKLY_STAT_CANDIDATES = [
    ("player_stats", "player_stats_{season}.csv"),
    ("player_stats", "player_stats_{season}.csv.gz"),
    ("stats_player", "stats_player_week_{season}.csv.gz"),
    ("stats_player", "stats_player_week_{season}.csv"),
]

_WEEKLY_CACHE: dict[tuple[int, bool], pd.DataFrame] = {}
_RATE_CACHE: dict[int, dict[str, dict[str, float]]] = {}

DEFAULT_RATES: dict[str, dict[str, float]] = {
    "QB": {"targets": 0.0, "carries": 0.70, "attempts": 0.44},
    "RB": {"targets": 1.55, "carries": 0.72, "attempts": 0.0},
    "WR": {"targets": 1.72, "carries": 0.55, "attempts": 0.0},
    "TE": {"targets": 1.55, "carries": 0.50, "attempts": 0.0},
}

# Trick plays are too rare to estimate. Fit only the opportunities that define
# the role, and leave the rest at zero.
RATE_FEATURES: dict[str, tuple[str, ...]] = {
    "QB": ("carries", "attempts"),
    "RB": ("targets", "carries"),
    "WR": ("targets", "carries"),
    "TE": ("targets", "carries"),
}


def _weekly_candidates(season: int) -> list[tuple[str, str]]:
    return [(rel, name.format(season=season)) for rel, name in WEEKLY_STAT_CANDIDATES]


def _col(frame: pd.DataFrame, *names: str) -> pd.Series:
    for name in names:
        if name in frame.columns:
            return pd.to_numeric(frame[name], errors="coerce")
    return pd.Series(np.nan, index=frame.index)


def load_weekly_player_stats(season: int, *, force: bool = False) -> pd.DataFrame:
    """Regular-season weekly box scores for ``season``, cached per process."""
    key = (int(season), bool(force))
    if not force and (int(season), False) in _WEEKLY_CACHE:
        return _WEEKLY_CACHE[(int(season), False)]
    if force and key in _WEEKLY_CACHE:
        return _WEEKLY_CACHE[key]

    raw = try_read_release_csv(_weekly_candidates(season), force=force)
    if raw.empty or "week" not in raw.columns:
        out = pd.DataFrame(
            columns=[
                "gsis_id",
                "season",
                "week",
                "position",
                "targets",
                "carries",
                "attempts",
                "ppr",
            ]
        )
        _WEEKLY_CACHE[key] = out
        if not force:
            _WEEKLY_CACHE[(int(season), False)] = out
        return out

    frame = raw.copy()
    if "season" not in frame.columns:
        frame["season"] = season
    if "season_type" in frame.columns:
        frame = frame[frame["season_type"].astype(str).str.upper().isin(["REG", "REGULAR"])]

    id_col = next((c for c in ("player_id", "gsis_id") if c in frame.columns), None)
    if id_col is None:
        out = pd.DataFrame()
        _WEEKLY_CACHE[key] = out
        return out

    out = pd.DataFrame(
        {
            "gsis_id": frame[id_col].astype(str),
            "season": pd.to_numeric(frame["season"], errors="coerce").astype("Int64"),
            "week": pd.to_numeric(frame["week"], errors="coerce"),
            "position": frame["position"].map(normalize_position)
            if "position" in frame.columns
            else None,
            "targets": _col(frame, "targets"),
            "carries": _col(frame, "carries", "rushing_attempts"),
            "attempts": _col(frame, "attempts", "passing_attempts"),
            "ppr": _col(frame, "fantasy_points_ppr"),
        }
    )
    out = out.dropna(subset=["gsis_id", "week"])
    out["week"] = out["week"].astype(int)
    out["opportunity"] = (
        out["targets"].fillna(0.0)
        + out["carries"].fillna(0.0)
        + out["attempts"].fillna(0.0)
    )
    out = out[out["opportunity"] >= 1].copy()
    _WEEKLY_CACHE[key] = out
    if not force:
        _WEEKLY_CACHE[(int(season), False)] = out
    elif int(season) >= LEAGUE_TARGET_SEASON:
        # A forced current-season pull replaces the stale cached copy too.
        _WEEKLY_CACHE[(int(season), False)] = out
    return out


def fit_usage_rates(
    weekly: pd.DataFrame, *, before_season: int
) -> dict[str, dict[str, float]]:
    """PPR per target / carry / dropback, fit on seasons strictly before ``before_season``."""
    if before_season in _RATE_CACHE:
        return _RATE_CACHE[before_season]
    if weekly.empty:
        _RATE_CACHE[before_season] = {k: dict(v) for k, v in DEFAULT_RATES.items()}
        return _RATE_CACHE[before_season]

    floor = max(LEAGUE_SEASON_MIN, int(before_season) - RATE_LOOKBACK)
    pool = weekly[
        (weekly["season"] >= floor)
        & (weekly["season"] < int(before_season))
        & weekly["position"].isin(MODELED_POSITIONS)
        & weekly["ppr"].notna()
    ].copy()
    rates = {k: dict(v) for k, v in DEFAULT_RATES.items()}
    for position, group in pool.groupby("position"):
        features = RATE_FEATURES.get(str(position))
        if not features or len(group) < 80:
            continue
        x = np.column_stack(
            [group[col].fillna(0.0).to_numpy(dtype=float) for col in features]
        )
        y = group["ppr"].clip(lower=0.0, upper=SCORING_WEEK_CAP).to_numpy(dtype=float)
        if float(x.sum()) <= 0:
            continue
        try:
            coef, *_ = np.linalg.lstsq(x, y, rcond=None)
        except np.linalg.LinAlgError:
            continue
        fitted = {name: 0.0 for name in ("targets", "carries", "attempts")}
        for name, value in zip(features, coef, strict=True):
            fitted[name] = float(max(value, 0.0))
        rates[str(position)] = fitted
    _RATE_CACHE[before_season] = rates
    return rates


def _nflverse_evidence(
    weekly: pd.DataFrame,
    *,
    season: int,
    as_of_week: int,
    rates: dict[str, dict[str, float]],
) -> pd.DataFrame:
    played = weekly[
        (weekly["season"] == season) & (weekly["week"] <= as_of_week)
    ].copy()
    if played.empty:
        return pd.DataFrame(columns=["gsis_id", "n_games", "observed_ppg", "k"])
    played["usage_ppr"] = 0.0
    for pos, coef in rates.items():
        mask = played["position"] == pos
        if not mask.any():
            continue
        played.loc[mask, "usage_ppr"] = (
            coef["targets"] * played.loc[mask, "targets"].fillna(0.0)
            + coef["carries"] * played.loc[mask, "carries"].fillna(0.0)
            + coef["attempts"] * played.loc[mask, "attempts"].fillna(0.0)
        )
    played["score_ppr"] = played["ppr"].clip(lower=0.0, upper=SCORING_WEEK_CAP)
    grouped = played.groupby("gsis_id", as_index=False).agg(
        n_games=("week", "count"),
        usage_ppg=("usage_ppr", "mean"),
        score_ppg=("score_ppr", "mean"),
    )
    both = grouped["usage_ppg"].notna() & grouped["score_ppg"].notna()
    grouped["observed_ppg"] = np.where(
        both,
        USAGE_MIX * grouped["usage_ppg"] + (1.0 - USAGE_MIX) * grouped["score_ppg"],
        grouped["usage_ppg"].where(grouped["usage_ppg"].notna(), grouped["score_ppg"]),
    )
    grouped["k"] = np.where(grouped["usage_ppg"].notna(), PRIOR_GAMES, SCORING_ONLY_PRIOR_GAMES)
    grouped["gsis_id"] = grouped["gsis_id"].astype(str)
    return grouped.dropna(subset=["observed_ppg"])[["gsis_id", "n_games", "observed_ppg", "k"]]


def _league_evidence(
    league_rosters: pd.DataFrame,
    *,
    season: int,
    as_of_week: int,
) -> pd.DataFrame:
    """Fallback when nflverse has not published this week's box scores yet."""
    skill = league_rosters[
        (league_rosters["season"] == season)
        & (league_rosters["week"] <= as_of_week)
        & league_rosters["position"].isin(MODELED_POSITIONS)
        & league_rosters["gsis_id"].notna()
    ].copy()
    if skill.empty:
        return pd.DataFrame(columns=["gsis_id", "n_games", "observed_ppg", "k"])
    skill["gsis_id"] = skill["gsis_id"].astype(str)
    skill["actual"] = pd.to_numeric(skill["actual_points"], errors="coerce")
    skill["espn"] = pd.to_numeric(skill["espn_projected_points"], errors="coerce")

    past = skill[skill["actual"].notna() & (skill["espn"].fillna(0) > 1.0)].copy()
    past["actual"] = past["actual"].clip(lower=0.0, upper=SCORING_WEEK_CAP)

    rest = league_rosters[
        (league_rosters["season"] == season)
        & (league_rosters["week"] > as_of_week)
        & league_rosters["position"].isin(MODELED_POSITIONS)
        & league_rosters["gsis_id"].notna()
    ].copy()
    espn_rest = pd.DataFrame(columns=["gsis_id", "espn_rest"])
    if not rest.empty:
        rest["gsis_id"] = rest["gsis_id"].astype(str)
        rest["espn"] = pd.to_numeric(rest["espn_projected_points"], errors="coerce")
        espn_rest = (
            rest.groupby("gsis_id")["espn"]
            .mean()
            .rename("espn_rest")
            .reset_index()
        )

    if past.empty and espn_rest.empty:
        return pd.DataFrame(columns=["gsis_id", "n_games", "observed_ppg", "k"])

    scoring = (
        past.groupby("gsis_id")
        .agg(n_games=("actual", "count"), score_ppg=("actual", "mean"))
        .reset_index()
        if not past.empty
        else pd.DataFrame(columns=["gsis_id", "n_games", "score_ppg"])
    )
    merged = scoring.merge(espn_rest, on="gsis_id", how="outer")
    merged["n_games"] = merged["n_games"].fillna(0)
    rows: list[dict[str, Any]] = []
    for row in merged.itertuples(index=False):
        score = float(row.score_ppg) if pd.notna(row.score_ppg) else float("nan")
        espn = float(row.espn_rest) if pd.notna(row.espn_rest) else float("nan")
        n = int(row.n_games)
        if np.isfinite(score) and np.isfinite(espn) and n > 0:
            observed = USAGE_MIX * espn + (1.0 - USAGE_MIX) * score
            k = PRIOR_GAMES
        elif np.isfinite(espn) and n == 0:
            # ESPN remaining-week view with no actuals yet: treat as a single
            # weak observation so it cannot swamp the prior.
            observed = espn
            n = 1
            k = SCORING_ONLY_PRIOR_GAMES
        elif np.isfinite(score):
            observed = score
            k = SCORING_ONLY_PRIOR_GAMES
        else:
            continue
        rows.append(
            {
                "gsis_id": str(row.gsis_id),
                "n_games": n,
                "observed_ppg": observed,
                "k": k,
            }
        )
    return pd.DataFrame(rows)


def player_evidence(
    *,
    season: int,
    as_of_week: int,
    league_rosters: pd.DataFrame,
    weekly_stats: pd.DataFrame | None,
    rates: dict[str, dict[str, float]] | None,
) -> pd.DataFrame:
    """Per-player observed remaining-pace signal through ``as_of_week``."""
    if as_of_week <= 0:
        return pd.DataFrame(columns=["gsis_id", "n_games", "observed_ppg", "k"])
    if weekly_stats is not None and not weekly_stats.empty:
        rates = rates or DEFAULT_RATES
        evidence = _nflverse_evidence(
            weekly_stats, season=season, as_of_week=as_of_week, rates=rates
        )
        if not evidence.empty:
            return evidence
    return _league_evidence(league_rosters, season=season, as_of_week=as_of_week)


def apply_in_season_update(
    roster: pd.DataFrame,
    *,
    season: int,
    as_of_week: int,
    league_rosters: pd.DataFrame,
    default_games: int,
    weekly_stats: pd.DataFrame | None = None,
    rates: dict[str, dict[str, float]] | None = None,
) -> pd.DataFrame:
    """Scale ``proj_season_ppr`` / ``adp_curve_ppr`` toward in-season evidence."""
    if as_of_week <= 0 or roster.empty or default_games <= 0:
        return roster

    evidence = player_evidence(
        season=season,
        as_of_week=as_of_week,
        league_rosters=league_rosters,
        weekly_stats=weekly_stats,
        rates=rates,
    )
    if evidence.empty:
        return roster

    out = roster.copy()
    out["gsis_id"] = out["gsis_id"].astype(str)
    out = out.merge(evidence, on="gsis_id", how="left")
    n = pd.to_numeric(out["n_games"], errors="coerce").fillna(0.0)
    k = pd.to_numeric(out["k"], errors="coerce").fillna(PRIOR_GAMES)
    prior_ppg = pd.to_numeric(out["proj_season_ppr"], errors="coerce") / float(default_games)
    observed = pd.to_numeric(out["observed_ppg"], errors="coerce")
    weight = n / (n + k)
    updated = (1.0 - weight) * prior_ppg + weight * observed
    use = n.gt(0) & prior_ppg.gt(0.5) & observed.notna() & np.isfinite(updated)
    mult = (updated / prior_ppg).where(use, 1.0).clip(MULT_FLOOR, MULT_CEILING)
    out["proj_season_ppr"] = pd.to_numeric(out["proj_season_ppr"], errors="coerce") * mult
    if "adp_curve_ppr" in out.columns:
        curve = pd.to_numeric(out["adp_curve_ppr"], errors="coerce")
        out["adp_curve_ppr"] = curve * mult
    return out.drop(columns=["n_games", "observed_ppg", "k"], errors="ignore")


def load_rates_for_season(season: int) -> dict[str, dict[str, float]]:
    """Fit opportunity rates from the lookback window before ``season``."""
    frames = [
        load_weekly_player_stats(s, force=False)
        for s in range(max(LEAGUE_SEASON_MIN, season - RATE_LOOKBACK), season)
    ]
    frames = [f for f in frames if not f.empty]
    weekly = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return fit_usage_rates(weekly, before_season=season)
