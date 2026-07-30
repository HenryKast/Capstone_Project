"""
Opponent-defense adjustment for week-by-week projections.

A week's projection is scaled by how good the opposing defense is against that
kind of production. The sensitivity is not a guess: it is fit from historical
weekly box scores by regressing each player-week's production, expressed as a
share of their own season average, on the opponent's defensive strength.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from rookie_ppr.nflverse_http import try_read_release_csv
from rookie_ppr.utils import normalize_position, normalize_team_abbr
from rookie_ppr.veteran.config import MODELS_DIR, SKILL_POSITIONS
from rookie_ppr.veteran.pbp_context import build_team_pbp_metrics

SENSITIVITY_FILE = "opponent_adjustment.json"

# Seasons used to fit the sensitivity; recent enough to reflect current usage
CALIBRATION_SEASON_MIN = 2015
# A player needs this many games in a season before their weekly ratios are usable
MIN_GAMES_FOR_RATIO = 8
# Guardrails on the resulting weekly multiplier
MULTIPLIER_FLOOR = 0.72
MULTIPLIER_CEILING = 1.28
# Defense strength blend for the upcoming season: most recent year carries more
DEFENSE_BLEND = ((0, 0.65), (1, 0.35))  # (seasons back, weight)

RUSH_STATS = ("carries", "rushing_yards", "rushing_tds")
PASS_STATS = (
    "targets",
    "receptions",
    "receiving_yards",
    "receiving_tds",
    "passing_yards",
    "passing_tds",
)
CATEGORY_STATS = {"rush": RUSH_STATS, "pass": PASS_STATS}
# Which stat drives each category's slope fit
CATEGORY_FIT_STAT = {
    "rush": "rushing_yards",
    "pass": {"QB": "passing_yards", "RB": "receiving_yards", "WR": "receiving_yards", "TE": "receiving_yards"},
}


def team_defense_z(seasons: list[int]) -> pd.DataFrame:
    """Per season × team defensive strength, z-scored within season.

    Positive z means the defense allows more than average, i.e. a soft matchup.
    """
    metrics = build_team_pbp_metrics(sorted(set(int(s) for s in seasons)))
    if metrics.empty:
        return pd.DataFrame(columns=["season", "team", "def_rush_z", "def_pass_z"])

    d = metrics[["season", "team", "def_rush_epa", "def_pass_epa"]].copy()
    d["team"] = d["team"].map(normalize_team_abbr)
    for src, dest in (("def_rush_epa", "def_rush_z"), ("def_pass_epa", "def_pass_z")):
        vals = pd.to_numeric(d[src], errors="coerce")
        grouped = vals.groupby(d["season"])
        d[dest] = (vals - grouped.transform("mean")) / grouped.transform("std").replace({0: np.nan})
    return d[["season", "team", "def_rush_z", "def_pass_z"]]


def _load_weekly_stats(seasons: list[int]) -> pd.DataFrame:
    """Weekly regular-season box scores with the opponent attached."""
    frames: list[pd.DataFrame] = []
    for season in sorted(set(int(s) for s in seasons)):
        df = try_read_release_csv(
            [
                ("player_stats", f"player_stats_{season}.csv"),
                ("player_stats", f"player_stats_{season}.csv.gz"),
                ("stats_player", f"stats_player_week_{season}.csv.gz"),
                ("stats_player", f"stats_player_week_{season}.csv"),
            ]
        )
        if df.empty or "week" not in df.columns:
            continue
        if "season" not in df.columns:
            df["season"] = season
        if "season_type" in df.columns:
            df = df[df["season_type"].astype(str).str.upper().isin(["REG", "REGULAR"])]
        id_col = next((c for c in ("player_id", "gsis_id") if c in df.columns), None)
        if id_col is None or "opponent_team" not in df.columns:
            continue
        keep = {
            id_col: "gsis_id",
            "position": "position",
            "season": "season",
            "week": "week",
            "opponent_team": "opponent",
        }
        cols = {k: v for k, v in keep.items() if k in df.columns}
        out = df[list(cols)].rename(columns=cols)
        for stat in (*RUSH_STATS, *PASS_STATS):
            out[stat] = pd.to_numeric(df[stat], errors="coerce") if stat in df.columns else np.nan
        frames.append(out)

    if not frames:
        return pd.DataFrame()
    weekly = pd.concat(frames, ignore_index=True)
    weekly["position"] = weekly["position"].map(normalize_position)
    weekly["opponent"] = weekly["opponent"].map(normalize_team_abbr)
    return weekly[weekly["position"].isin(SKILL_POSITIONS)]


def _fit_slope(ratios: np.ndarray, z: np.ndarray) -> tuple[float, int]:
    """Least-squares slope of production ratio on opponent z (intercept free)."""
    ok = np.isfinite(ratios) & np.isfinite(z)
    ratios, z = ratios[ok], z[ok]
    if len(z) < 200:
        return float("nan"), int(len(z))
    slope, _intercept = np.polyfit(z, ratios, 1)
    return float(slope), int(len(z))


def calibrate_opponent_sensitivity(
    season_min: int = CALIBRATION_SEASON_MIN,
    season_max: int | None = None,
) -> dict[str, Any]:
    """Fit and persist how much weekly production moves with opponent defense."""
    season_max = int(season_max or pd.Timestamp.today().year)
    seasons = list(range(int(season_min), season_max + 1))

    weekly = _load_weekly_stats(seasons)
    if weekly.empty:
        raise RuntimeError("No weekly stats available to calibrate opponent adjustment.")
    defense = team_defense_z(seasons)

    merged = weekly.merge(
        defense.rename(columns={"team": "opponent"}),
        how="left",
        on=["season", "opponent"],
    )

    slopes: dict[str, dict[str, float]] = {}
    diagnostics: dict[str, Any] = {}

    for position in SKILL_POSITIONS:
        pos_rows = merged[merged["position"] == position]
        if pos_rows.empty:
            continue
        pos_slopes: dict[str, float] = {}
        pos_diag: dict[str, Any] = {}
        for category in ("rush", "pass"):
            fit_stat = CATEGORY_FIT_STAT[category]
            if isinstance(fit_stat, dict):
                fit_stat = fit_stat.get(position)
            if not fit_stat or fit_stat not in pos_rows.columns:
                continue

            work = pos_rows[["gsis_id", "season", fit_stat, f"def_{category}_z"]].copy()
            work[fit_stat] = pd.to_numeric(work[fit_stat], errors="coerce")
            work = work.dropna(subset=[fit_stat, f"def_{category}_z"])
            if work.empty:
                continue

            grp = work.groupby(["gsis_id", "season"])[fit_stat]
            work["season_mean"] = grp.transform("mean")
            work["season_games"] = grp.transform("count")
            work = work[
                (work["season_games"] >= MIN_GAMES_FOR_RATIO) & (work["season_mean"] > 5.0)
            ].copy()
            if work.empty:
                continue

            work["ratio"] = work[fit_stat] / work["season_mean"]
            slope, n = _fit_slope(
                work["ratio"].to_numpy(dtype=float),
                work[f"def_{category}_z"].to_numpy(dtype=float),
            )
            if not np.isfinite(slope):
                continue
            pos_slopes[category] = round(slope, 4)
            pos_diag[category] = {
                "fit_stat": fit_stat,
                "n_player_weeks": n,
                "slope": round(slope, 4),
                "pct_swing_per_sd": round(100 * slope, 2),
            }

        if pos_slopes:
            slopes[position] = pos_slopes
            diagnostics[position] = pos_diag

    payload = {
        "slopes_by_position": slopes,
        "diagnostics": diagnostics,
        "calibration_seasons": [int(season_min), int(season_max)],
        "min_games_for_ratio": MIN_GAMES_FOR_RATIO,
        "multiplier_bounds": [MULTIPLIER_FLOOR, MULTIPLIER_CEILING],
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / SENSITIVITY_FILE).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_opponent_sensitivity() -> dict[str, dict[str, float]]:
    path = MODELS_DIR / SENSITIVITY_FILE
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return payload.get("slopes_by_position") or {}


def upcoming_defense_z(target_season: int) -> dict[str, dict[str, float]]:
    """{team: {'rush': z, 'pass': z}} for a season not yet played."""
    back = [target_season - 1 - offset for offset, _w in DEFENSE_BLEND]
    weights = {target_season - 1 - offset: w for offset, w in DEFENSE_BLEND}
    defense = team_defense_z(back)
    if defense.empty:
        return {}

    out: dict[str, dict[str, float]] = {}
    for team, g in defense.groupby("team"):
        acc = {"rush": 0.0, "pass": 0.0}
        wsum = 0.0
        for row in g.itertuples(index=False):
            w = weights.get(int(row.season), 0.0)
            rz, pz = row.def_rush_z, row.def_pass_z
            if w <= 0 or not np.isfinite(rz) or not np.isfinite(pz):
                continue
            acc["rush"] += w * float(rz)
            acc["pass"] += w * float(pz)
            wsum += w
        if wsum > 0:
            out[str(team)] = {"rush": acc["rush"] / wsum, "pass": acc["pass"] / wsum}
    return out


def matchup_multipliers(
    position: str,
    opponent_z: dict[str, float] | None,
    slopes: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Per-category multiplier for one matchup; 1.0 when data is missing."""
    pos_slopes = slopes.get(position) or {}
    out = {"rush": 1.0, "pass": 1.0}
    if not opponent_z:
        return out
    for category in ("rush", "pass"):
        slope = pos_slopes.get(category)
        z = opponent_z.get(category)
        if slope is None or z is None or not np.isfinite(z):
            continue
        out[category] = float(
            np.clip(1.0 + slope * z, MULTIPLIER_FLOOR, MULTIPLIER_CEILING)
        )
    return out


def matchup_label(multiplier: float) -> str:
    if multiplier >= 1.06:
        return "soft"
    if multiplier <= 0.94:
        return "tough"
    return "neutral"
