"""Preseason ForeKast for the upcoming season, once its schedule is final.

The odds simulation replays head-to-head matchups, so playoff and title odds are
only as stable as ``league_matchups.csv``. A schedule that is still being edited
moves preseason playoff odds by roughly 13 percentage points per team, which is
large enough to be visible on the league site. This module therefore refuses to
publish odds until the schedule is complete, and fingerprints the schedule it
used so a later edit is detectable.

Typical flow after a draft::

    python -m rookie_ppr.league.target_season --check
    # ...once the schedule stops changing...
    python -m rookie_ppr.league.ingest --seasons 2026 --force --skip-rosters
    python -m rookie_ppr.league.target_season --run

Team scoring strength does not depend on the schedule, so ``--check`` and the
roster/projection steps are safe to run at any point.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from typing import Any

import pandas as pd

from rookie_ppr.league.backtest_rosters import build_backtest_rosters
from rookie_ppr.league.config import (
    CSV_OUTPUT_DIR,
    LEAGUE_BACKTEST_ROSTERS_CSV,
    LEAGUE_DRAFT_CSV,
    LEAGUE_MATCHUPS_CSV,
    LEAGUE_SEASONS,
    LEAGUE_SETTINGS_CSV,
    LEAGUE_TARGET_SEASON,
    LEAGUE_TEAMS_CSV,
    LEAGUE_WEEKLY_ODDS_CSV,
    LEAGUE_WF_PROJECTIONS_CSV,
)
from rookie_ppr.league.weekly_odds import completed_weeks, run_weekly_odds

SCHEDULE_STATE_JSON = "league_schedule_state.json"


def _read(name: str) -> pd.DataFrame:
    path = CSV_OUTPUT_DIR / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def schedule_fingerprint(matchups: pd.DataFrame, season: int, reg_weeks: int) -> str:
    """Stable hash of the regular-season pairings, ignoring row order and sides."""
    games = matchups[
        (matchups["season"] == season)
        & (matchups["week"] <= reg_weeks)
        & matchups["opp_team_id"].notna()
    ]
    pairs = sorted(
        {
            (int(r.week), min(int(r.team_id), int(r.opp_team_id)), max(int(r.team_id), int(r.opp_team_id)))
            for r in games.itertuples(index=False)
        }
    )
    return hashlib.sha1(json.dumps(pairs).encode("utf-8")).hexdigest()[:12]


def schedule_status(season: int = LEAGUE_TARGET_SEASON) -> dict[str, Any]:
    """Everything the preseason run needs, and whether the schedule is usable."""
    settings = _read(LEAGUE_SETTINGS_CSV)
    teams = _read(LEAGUE_TEAMS_CSV)
    matchups = _read(LEAGUE_MATCHUPS_CSV)
    draft = _read(LEAGUE_DRAFT_CSV)
    projections = _read(LEAGUE_WF_PROJECTIONS_CSV)

    row = settings[settings["season"] == season] if not settings.empty else pd.DataFrame()
    reg_weeks = int(row["reg_weeks"].iloc[0]) if not row.empty else 0
    season_teams = teams[teams["season"] == season] if not teams.empty else pd.DataFrame()
    n_teams = len(season_teams)

    picks = len(draft[draft["season"] == season]) if not draft.empty else 0
    projected = (
        int((projections["target_season"] == season).sum())
        if not projections.empty and "target_season" in projections.columns
        else 0
    )

    games = pd.DataFrame()
    if not matchups.empty:
        games = matchups[
            (matchups["season"] == season)
            & (matchups["week"] <= reg_weeks)
            & matchups["opp_team_id"].notna()
        ]

    per_team = (
        games.groupby("team_id").size().to_dict() if not games.empty else {}
    )
    expected_per_team = reg_weeks
    teams_full = sum(1 for v in per_team.values() if v == expected_per_team)
    weeks_present = sorted({int(w) for w in games["week"].unique()}) if not games.empty else []

    schedule_complete = bool(
        reg_weeks
        and n_teams
        and len(weeks_present) == reg_weeks
        and teams_full == n_teams
    )

    played = (
        completed_weeks(matchups, season, reg_weeks)
        if not matchups.empty and reg_weeks
        else 0
    )

    return {
        "season": season,
        "regWeeks": reg_weeks,
        "teams": n_teams,
        "draftPicks": picks,
        "projectedPlayers": projected,
        "scheduledWeeks": len(weeks_present),
        "teamsWithFullSchedule": teams_full,
        "scheduleComplete": schedule_complete,
        "completedWeeks": played,
        "fingerprint": schedule_fingerprint(matchups, season, reg_weeks)
        if not matchups.empty and reg_weeks
        else None,
    }


def _state_path():
    return CSV_OUTPUT_DIR / SCHEDULE_STATE_JSON


def load_schedule_state() -> dict[str, Any]:
    path = _state_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def save_schedule_state(status: dict[str, Any]) -> None:
    state = load_schedule_state()
    state[str(status["season"])] = {
        "fingerprint": status["fingerprint"],
        "regWeeks": status["regWeeks"],
        "teams": status["teams"],
    }
    CSV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _state_path().write_text(json.dumps(state, indent=2), encoding="utf-8")


def schedule_changed(status: dict[str, Any]) -> bool | None:
    """True/False against the last published run, None if never published."""
    prior = load_schedule_state().get(str(status["season"]))
    if not prior or not prior.get("fingerprint"):
        return None
    return prior["fingerprint"] != status["fingerprint"]


def build_target_rosters(season: int = LEAGUE_TARGET_SEASON) -> pd.DataFrame:
    """Add the target season's drafted rosters to ``league_backtest_rosters.csv``."""
    fresh = build_backtest_rosters([season], projections_csv=LEAGUE_WF_PROJECTIONS_CSV)
    path = CSV_OUTPUT_DIR / LEAGUE_BACKTEST_ROSTERS_CSV
    out = fresh
    if path.exists():
        prior = pd.read_csv(path)
        if not prior.empty:
            kept = prior[prior["season"] != season]
            out = pd.concat([kept, fresh], ignore_index=True)
    out = out.sort_values(["season", "overall_pick"]).reset_index(drop=True)
    CSV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return fresh


def run_target_forekast(
    season: int = LEAGUE_TARGET_SEASON,
    *,
    n_sims: int = 4000,
    force: bool = False,
) -> pd.DataFrame:
    """Preseason odds for ``season``, merged into ``league_weekly_odds.csv``."""
    status = schedule_status(season)
    if not status["scheduleComplete"] and not force:
        raise RuntimeError(
            f"{season} schedule is incomplete "
            f"({status['scheduledWeeks']}/{status['regWeeks']} weeks, "
            f"{status['teamsWithFullSchedule']}/{status['teams']} teams full). "
            "Publishing odds now would change once the schedule settles. "
            "Re-ingest with --force, or pass --force to override."
        )
    if not status["draftPicks"]:
        raise RuntimeError(f"No {season} rows in {LEAGUE_DRAFT_CSV}; ingest the draft first.")

    build_target_rosters(season)
    odds = run_weekly_odds([season], n_sims=n_sims)

    path = CSV_OUTPUT_DIR / LEAGUE_WEEKLY_ODDS_CSV
    out = odds
    if path.exists():
        prior = pd.read_csv(path)
        if not prior.empty:
            kept = prior[prior["season"] != season]
            out = pd.concat([kept, odds], ignore_index=True) if not odds.empty else kept
    out = out.sort_values(["season", "as_of_week", "team_id"]).reset_index(drop=True)
    CSV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)

    save_schedule_state(status)
    return odds


def _print_status(status: dict[str, Any]) -> None:
    season = status["season"]
    print(f"=== {season} target-season readiness ===")
    checks = [
        ("teams ingested", status["teams"], status["teams"] > 0),
        ("draft picks", status["draftPicks"], status["draftPicks"] > 0),
        ("players projected", status["projectedPlayers"], status["projectedPlayers"] > 0),
        (
            "schedule weeks",
            f"{status['scheduledWeeks']}/{status['regWeeks']}",
            status["scheduledWeeks"] == status["regWeeks"] and status["regWeeks"] > 0,
        ),
        (
            "teams with full slate",
            f"{status['teamsWithFullSchedule']}/{status['teams']}",
            status["teams"] > 0 and status["teamsWithFullSchedule"] == status["teams"],
        ),
    ]
    for label, value, ok in checks:
        print(f"  [{'ok' if ok else '--'}] {label:<24} {value}")
    print(f"  weeks played: {status['completedWeeks']}")
    print(f"  schedule fingerprint: {status['fingerprint']}")

    changed = schedule_changed(status)
    if changed is None:
        print("  no previously published run for this season")
    elif changed:
        print("  ** schedule CHANGED since the last published odds - rerun --run **")
    else:
        print("  schedule matches the last published odds")

    if status["scheduleComplete"]:
        print(f"\nReady: python -m rookie_ppr.league.target_season --run --season {season}")
    else:
        print(
            "\nNot ready. Ingest the finalized schedule first:\n"
            f"  python -m rookie_ppr.league.ingest --seasons {season} --force --skip-rosters"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preflight and run the preseason ForeKast for the upcoming season"
    )
    parser.add_argument("--season", type=int, default=LEAGUE_TARGET_SEASON)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report readiness only (the default)",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Build rosters and publish preseason odds (default: check only)",
    )
    parser.add_argument("--sims", type=int, default=4000)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Publish odds even if the schedule looks incomplete",
    )
    args = parser.parse_args()

    if args.season in LEAGUE_SEASONS:
        print(
            f"note: {args.season} is a completed season; "
            "use refresh_forekast for those.\n"
        )

    status = schedule_status(args.season)
    _print_status(status)
    if args.check or not args.run:
        return

    print()
    try:
        odds = run_target_forekast(args.season, n_sims=args.sims, force=args.force)
    except RuntimeError as exc:
        raise SystemExit(f"refusing to publish: {exc}")
    if odds.empty:
        print("no odds produced")
        return
    print(f"\npublished {len(odds)} rows for {args.season} (as-of weeks "
          f"{sorted(odds['as_of_week'].unique().tolist())})")
    board = odds.sort_values("playoff_odds", ascending=False)[
        ["team_name", "sim_wins_mean", "playoff_odds", "title_odds"]
    ].copy()
    board["playoff_odds"] = (board["playoff_odds"] * 100).round(1)
    board["title_odds"] = (board["title_odds"] * 100).round(1)
    board["sim_wins_mean"] = board["sim_wins_mean"].round(2)
    print(board.to_string(index=False))


if __name__ == "__main__":
    main()
