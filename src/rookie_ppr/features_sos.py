from __future__ import annotations

import pandas as pd


def compute_team_sos(schedules: pd.DataFrame) -> pd.DataFrame:
    """
    Compute simple team strength-of-schedule for each season.

    SOS = average opponent winning percentage over the regular season.
    Also returns average opponent score differential if available.
    """
    if schedules.empty:
        return pd.DataFrame(columns=["season", "team", "sos_opp_win_pct", "games_scheduled"])

    s = schedules.copy()
    # Regular season only when game_type exists
    if "game_type" in s.columns:
        s = s[s["game_type"].astype(str).str.upper().isin(["REG", "REGULAR"])].copy()
    if "week" in s.columns:
        s = s[s["week"].astype("Float64") <= 18].copy()

    needed = {"season", "home_team", "away_team", "home_score", "away_score"}
    if not needed.issubset(set(s.columns)):
        # Try alternate column names
        alt = {
            "home_team": "home_team",
            "away_team": "away_team",
        }
        for a, b in alt.items():
            if a not in s.columns and b in s.columns:
                s[a] = s[b]
        if not needed.issubset(set(s.columns)):
            return pd.DataFrame(columns=["season", "team", "sos_opp_win_pct", "games_scheduled"])

    s = s.dropna(subset=["home_score", "away_score"])
    s["home_win"] = (s["home_score"] > s["away_score"]).astype(int)
    s["away_win"] = (s["away_score"] > s["home_score"]).astype(int)

    home = s[["season", "home_team", "home_win"]].rename(columns={"home_team": "team", "home_win": "win"})
    away = s[["season", "away_team", "away_win"]].rename(columns={"away_team": "team", "away_win": "win"})
    team_games = pd.concat([home, away], ignore_index=True)
    team_records = team_games.groupby(["season", "team"], as_index=False).agg(
        wins=("win", "sum"),
        games=("win", "count"),
    )
    team_records["win_pct"] = team_records["wins"] / team_records["games"].replace({0: pd.NA})

    win_map = team_records.set_index(["season", "team"])["win_pct"]

    rows = []
    for _, g in s.iterrows():
        season = g["season"]
        # home team faces away team's win_pct
        away_wp = win_map.get((season, g["away_team"]), pd.NA)
        home_wp = win_map.get((season, g["home_team"]), pd.NA)
        rows.append({"season": season, "team": g["home_team"], "opp_win_pct": away_wp})
        rows.append({"season": season, "team": g["away_team"], "opp_win_pct": home_wp})

    opp = pd.DataFrame(rows)
    sos = opp.groupby(["season", "team"], as_index=False).agg(
        sos_opp_win_pct=("opp_win_pct", "mean"),
        games_scheduled=("opp_win_pct", "count"),
    )
    return sos
