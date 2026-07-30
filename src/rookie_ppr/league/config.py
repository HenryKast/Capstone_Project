"""ESPN fantasy league config: credentials, ESPN id maps, artifact paths."""
from __future__ import annotations

import os

# Importing rookie_ppr.config loads .env, so the ESPN vars below are populated.
from rookie_ppr.config import CSV_OUTPUT_DIR, RAW_DIR

ESPN_LEAGUE_ID = os.getenv("ESPN_LEAGUE_ID", "").strip()
ESPN_S2 = os.getenv("ESPN_S2", "").strip()
ESPN_SWID = os.getenv("ESPN_SWID", "").strip()

ESPN_CACHE_DIR = RAW_DIR / "espn_cache"

# Completed seasons in this league. 2026 exists but is undrafted.
LEAGUE_SEASON_MIN = 2018
LEAGUE_SEASON_MAX = 2025
LEAGUE_TARGET_SEASON = 2026
LEAGUE_SEASONS = tuple(range(LEAGUE_SEASON_MIN, LEAGUE_SEASON_MAX + 1))

# Weekly lineups and ESPN's weekly projections go back to the league's first
# season. The older leagueHistory endpoint returns one frozen end-of-season
# roster for every week, but the seasons endpoint answers for 2018-2019 too and
# serves real per-week box scores, so nothing here needs a floor.
LEAGUE_ROSTER_SEASON_MIN = LEAGUE_SEASON_MIN

# Scoring periods are NFL weeks; 14 regular + 3 playoff rounds in this league.
MAX_SCORING_PERIOD = 17

# This league's starting lineup, verified identical across 2018-2026.
LEAGUE_STARTER_SLOTS = {
    "QB": 1,
    "RB": 2,
    "WR": 2,
    "TE": 1,
    "FLEX": 1,
    "D/ST": 1,
    "K": 1,
}
LEAGUE_BENCH_SIZE = 7

# Positions our player models cover; D/ST and K are simulated from league
# averages instead, since those slots are mostly noise.
MODELED_POSITIONS = ("QB", "RB", "WR", "TE")
UNMODELED_SLOTS = ("D/ST", "K")

LINEUP_SLOT_NAMES: dict[int, str] = {
    0: "QB",
    1: "TQB",
    2: "RB",
    3: "RB/WR",
    4: "WR",
    5: "WR/TE",
    6: "TE",
    7: "OP",
    8: "DT",
    9: "DE",
    10: "LB",
    11: "DL",
    12: "CB",
    13: "S",
    14: "DB",
    15: "DP",
    16: "D/ST",
    17: "K",
    18: "P",
    19: "HC",
    20: "BE",
    21: "IR",
    23: "FLEX",
    24: "EDR",
}

# Bench and injured reserve never score, so anything else is a started slot.
BENCH_SLOT_IDS = frozenset({20, 21})

POSITION_BY_ESPN_ID: dict[int, str] = {
    1: "QB",
    2: "RB",
    3: "WR",
    4: "TE",
    5: "K",
    16: "D/ST",
}

NFL_TEAM_BY_ESPN_ID: dict[int, str] = {
    0: "FA",
    1: "ATL",
    2: "BUF",
    3: "CHI",
    4: "CIN",
    5: "CLE",
    6: "DAL",
    7: "DEN",
    8: "DET",
    9: "GB",
    10: "TEN",
    11: "IND",
    12: "KC",
    13: "LV",
    14: "LA",
    15: "MIA",
    16: "MIN",
    17: "NE",
    18: "NO",
    19: "NYG",
    20: "NYJ",
    21: "PHI",
    22: "ARI",
    23: "PIT",
    24: "LAC",
    25: "SF",
    26: "SEA",
    27: "TB",
    28: "WAS",
    29: "CAR",
    30: "JAX",
    33: "BAL",
    34: "HOU",
}

# Offensive scoring stat ids, enough to confirm the league's rules by name.
ESPN_STAT_NAMES: dict[int, str] = {
    3: "passing_yards",
    4: "passing_tds",
    19: "passing_2pt",
    20: "interceptions",
    24: "rushing_yards",
    25: "rushing_tds",
    26: "rushing_2pt",
    42: "receiving_yards",
    43: "receiving_tds",
    44: "receiving_2pt",
    53: "receptions",
    62: "pass_yards_300_399_bonus",
    63: "pass_yards_400_bonus",
    72: "fumbles_lost",
}

# statSourceId on a player's stat rows
STAT_SOURCE_ACTUAL = 0
STAT_SOURCE_PROJECTED = 1

# Artifacts (league_* prefix keeps them clear of the rookie/veteran tables)
LEAGUE_SETTINGS_CSV = "league_settings.csv"
LEAGUE_SCORING_CSV = "league_scoring.csv"
LEAGUE_TEAMS_CSV = "league_teams.csv"
LEAGUE_MATCHUPS_CSV = "league_matchups.csv"
LEAGUE_ROSTERS_CSV = "league_rosters.csv"
LEAGUE_DRAFT_CSV = "league_draft.csv"
LEAGUE_WF_PROJECTIONS_CSV = "league_wf_projections.csv"
LEAGUE_WF_METRICS_FILE = "model_metrics_league_walkforward.json"

# Production two-track setup (settled by the 2018-2025 backtest):
#   * Team simulation blends the BASE walk-forward model with the ADP curve.
#     The base model stays more independent of the market, so the blend keeps
#     its diversification edge at the team level.
#   * Player rankings use the ADP+opportunity walk-forward model, which is the
#     strongest standalone player forecast we have on drafted skill players.
LEAGUE_PLAYER_PROJECTIONS_CSV = "league_player_projections.csv"
LEAGUE_PLAYER_METRICS_FILE = "model_metrics_league_player.json"
LEAGUE_PLAYER_BOARD_CSV = "league_player_board.csv"
LEAGUE_BLEND_WEIGHTS_CSV = "league_blend_weights.csv"
LEAGUE_SIM_TEAMS_CSV = "league_sim_team_seasons.csv"
LEAGUE_SIM_GRADES_CSV = "league_sim_grades.csv"
LEAGUE_BACKTEST_ROSTERS_CSV = "league_backtest_rosters.csv"
LEAGUE_WEEKLY_ODDS_CSV = "league_weekly_odds.csv"
LEAGUE_INJURY_EVENTS_CSV = "league_injury_events.csv"
LEAGUE_TRADE_EVENTS_CSV = "league_trade_events.csv"
LEAGUE_DRAFT_REACHES_CSV = "league_draft_reaches.csv"
LEAGUE_FINISH_PROJ_VS_ACTUAL_CSV = "league_finish_proj_vs_actual_all.csv"

# Default blend when no prior league season exists to fit on (2018).
PRODUCTION_BLEND_DEFAULT = 0.30

__all__ = [
    "BENCH_SLOT_IDS",
    "CSV_OUTPUT_DIR",
    "ESPN_CACHE_DIR",
    "ESPN_LEAGUE_ID",
    "ESPN_S2",
    "ESPN_STAT_NAMES",
    "ESPN_SWID",
    "LEAGUE_BACKTEST_ROSTERS_CSV",
    "LEAGUE_BENCH_SIZE",
    "LEAGUE_BLEND_WEIGHTS_CSV",
    "LEAGUE_DRAFT_CSV",
    "LEAGUE_DRAFT_REACHES_CSV",
    "LEAGUE_FINISH_PROJ_VS_ACTUAL_CSV",
    "LEAGUE_INJURY_EVENTS_CSV",
    "LEAGUE_MATCHUPS_CSV",
    "LEAGUE_PLAYER_BOARD_CSV",
    "LEAGUE_PLAYER_METRICS_FILE",
    "LEAGUE_PLAYER_PROJECTIONS_CSV",
    "LEAGUE_ROSTERS_CSV",
    "LEAGUE_ROSTER_SEASON_MIN",
    "LEAGUE_SCORING_CSV",
    "LEAGUE_SEASONS",
    "LEAGUE_SEASON_MAX",
    "LEAGUE_SEASON_MIN",
    "LEAGUE_SETTINGS_CSV",
    "LEAGUE_SIM_GRADES_CSV",
    "LEAGUE_SIM_TEAMS_CSV",
    "LEAGUE_STARTER_SLOTS",
    "LEAGUE_TARGET_SEASON",
    "LEAGUE_TEAMS_CSV",
    "LEAGUE_TRADE_EVENTS_CSV",
    "LEAGUE_WEEKLY_ODDS_CSV",
    "LEAGUE_WF_METRICS_FILE",
    "LEAGUE_WF_PROJECTIONS_CSV",
    "LINEUP_SLOT_NAMES",
    "MAX_SCORING_PERIOD",
    "MODELED_POSITIONS",
    "NFL_TEAM_BY_ESPN_ID",
    "POSITION_BY_ESPN_ID",
    "PRODUCTION_BLEND_DEFAULT",
    "STAT_SOURCE_ACTUAL",
    "STAT_SOURCE_PROJECTED",
    "UNMODELED_SLOTS",
]
