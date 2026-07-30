"""Detect mid-season fantasy trades from weekly rosters.

Only **player-for-player trades** are kept: at the same regular-season week,
team A must send at least one skill player to B **and** B must send at least
one skill player to A, with no multi-week roster gap (``gap_weeks == 0``).

One-way moves (drops / waiver pickups) are excluded — e.g. a player leaving
team A and later appearing on team B with no reverse traffic is not a trade.
"""
from __future__ import annotations

import argparse
import uuid

import pandas as pd

from rookie_ppr.league.config import (
    CSV_OUTPUT_DIR,
    LEAGUE_DRAFT_CSV,
    LEAGUE_ROSTERS_CSV,
    LEAGUE_SEASONS,
    LEAGUE_SETTINGS_CSV,
    LEAGUE_TEAMS_CSV,
    MODELED_POSITIONS,
)

LEAGUE_TRADE_EVENTS_CSV = "league_trade_events.csv"


def _pick_label(round_n: int | None, round_pick: int | None) -> str | None:
    if round_n is None or round_pick is None:
        return None
    return f"{int(round_n)}.{int(round_pick):02d}"


def _ascii(name: object) -> str:
    return str(name).encode("ascii", "ignore").decode("ascii").strip()


def _team_tag(name_map: dict[int, str], team_id: int) -> str:
    name = name_map.get(int(team_id))
    if name:
        return name
    return f"Team {int(team_id)}"


def detect_trade_events(
    seasons: list[int] | None = None,
) -> pd.DataFrame:
    seasons = list(seasons or LEAGUE_SEASONS)
    rosters = pd.read_csv(CSV_OUTPUT_DIR / LEAGUE_ROSTERS_CSV)
    draft = pd.read_csv(CSV_OUTPUT_DIR / LEAGUE_DRAFT_CSV)
    settings = pd.read_csv(CSV_OUTPUT_DIR / LEAGUE_SETTINGS_CSV)
    teams_df = pd.read_csv(CSV_OUTPUT_DIR / LEAGUE_TEAMS_CSV)
    reg_map = dict(zip(settings["season"].astype(int), settings["reg_weeks"].astype(int)))
    name_by_season: dict[int, dict[int, str]] = {}
    for season, g in teams_df.groupby("season"):
        name_by_season[int(season)] = {
            int(r.team_id): _ascii(r.team_name)
            for r in g.itertuples(index=False)
            if _ascii(r.team_name)
        }

    rosters = rosters[
        rosters["season"].isin(seasons) & rosters["position"].isin(MODELED_POSITIONS)
    ].copy()
    rosters["player_key"] = rosters["espn_player_id"].astype(str)
    missing = rosters["player_key"].isin(["", "nan", "None"]) | rosters[
        "espn_player_id"
    ].isna()
    rosters.loc[missing, "player_key"] = (
        "gsis:" + rosters.loc[missing, "gsis_id"].astype(str)
    )
    still = rosters["player_key"].isin(["gsis:", "gsis:nan", "gsis:None"])
    rosters.loc[still, "player_key"] = (
        "name:" + rosters.loc[still, "player_name"].astype(str)
    )

    draft = draft[draft["season"].isin(seasons)].copy()
    draft["gsis_key"] = draft["gsis_id"].astype(str)

    events: list[dict] = []
    for season in seasons:
        reg_weeks = int(reg_map[season])
        season_r = rosters[
            (rosters["season"] == season) & (rosters["week"] <= reg_weeks)
        ].copy()
        season_d = draft[draft["season"] == season]
        names = name_by_season.get(season, {})
        if season_r.empty:
            continue

        season_r = season_r.sort_values(
            ["player_key", "week", "is_starter"], ascending=[True, True, False]
        )
        season_r = season_r.drop_duplicates(subset=["player_key", "week"], keep="first")

        transfers: list[dict] = []
        for player_key, g in season_r.groupby("player_key"):
            g = g.sort_values("week")
            weeks = g["week"].astype(int).tolist()
            teams = g["team_id"].astype(int).tolist()
            if len(weeks) < 2:
                continue
            name = _ascii(g.iloc[0]["player_name"])
            gsis = g.iloc[0]["gsis_id"]
            gsis_key = str(gsis) if pd.notna(gsis) else ""
            drow = (
                season_d[season_d["gsis_key"] == gsis_key]
                if gsis_key
                else season_d.iloc[0:0]
            )
            overall = int(drow.iloc[0]["overall_pick"]) if not drow.empty else None
            round_n = int(drow.iloc[0]["round"]) if not drow.empty else None
            round_pick = int(drow.iloc[0]["round_pick"]) if not drow.empty else None
            pick = _pick_label(round_n, round_pick)

            for i in range(1, len(weeks)):
                if teams[i] == teams[i - 1]:
                    continue
                from_team = teams[i - 1]
                to_team = teams[i]
                last_from = weeks[i - 1]
                as_of = weeks[i]
                if last_from < 1 and as_of <= 1:
                    continue
                gap = as_of - last_from - 1
                # Waiver-style: missing weeks before reappearing on another team
                if gap > 0:
                    continue
                transfers.append(
                    {
                        "season": season,
                        "as_of_week": as_of,
                        "last_week_from": last_from,
                        "gap_weeks": gap,
                        "from_team_id": from_team,
                        "to_team_id": to_team,
                        "player_name": name,
                        "player_key": player_key,
                        "espn_player_id": g.iloc[0]["espn_player_id"],
                        "gsis_id": gsis if pd.notna(gsis) else None,
                        "overall_pick": overall,
                        "pick_label": pick,
                    }
                )

        # True trades only: both directions same week (player-for-player)
        package_ids: dict[tuple[int, int, int], str] = {}
        for t in transfers:
            a, b, w = t["from_team_id"], t["to_team_id"], t["as_of_week"]
            key = (w, min(a, b), max(a, b))
            has_reverse = any(
                u["as_of_week"] == w
                and u["from_team_id"] == b
                and u["to_team_id"] == a
                for u in transfers
            )
            if has_reverse:
                package_ids.setdefault(key, str(uuid.uuid4())[:8])

        for t in transfers:
            a, b, w = t["from_team_id"], t["to_team_id"], t["as_of_week"]
            key = (w, min(a, b), max(a, b))
            if key not in package_ids:
                continue  # one-way waiver / FA pickup
            package_id = package_ids[key]
            pick = t["pick_label"]
            pick_s = f" ({pick})" if pick else ""
            name = t["player_name"]
            from_tag = _team_tag(names, a)
            to_tag = _team_tag(names, b)
            away_label = f"{name} traded to {to_tag}{pick_s}"
            acq_label = f"{name} traded in from {from_tag}{pick_s}"

            base = {
                "season": t["season"],
                "as_of_week": w,
                "last_week_from": t["last_week_from"],
                "gap_weeks": t["gap_weeks"],
                "from_team_id": a,
                "to_team_id": b,
                "player_name": name,
                "player_key": t["player_key"],
                "espn_player_id": t["espn_player_id"],
                "gsis_id": t["gsis_id"],
                "overall_pick": t["overall_pick"],
                "pick_label": pick,
                "kind": "bilateral",
                "package_id": package_id,
            }
            events.append(
                {
                    **base,
                    "team_id": a,
                    "direction": "traded_away",
                    "counterparty_team_id": b,
                    "label": away_label,
                }
            )
            events.append(
                {
                    **base,
                    "team_id": b,
                    "direction": "acquired",
                    "counterparty_team_id": a,
                    "label": acq_label,
                }
            )

    out = pd.DataFrame(events)
    if out.empty:
        return out
    out = out.sort_values(
        ["season", "as_of_week", "team_id", "direction", "overall_pick"],
        ascending=[True, True, True, True, True],
    )
    out = out.drop_duplicates(
        subset=["season", "team_id", "as_of_week", "player_key", "direction"]
    )
    return out.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect fantasy player-for-player trade weeks"
    )
    parser.add_argument("--seasons", type=int, nargs="+", default=list(LEAGUE_SEASONS))
    args = parser.parse_args()
    out = detect_trade_events(args.seasons)
    path = CSV_OUTPUT_DIR / LEAGUE_TRADE_EVENTS_CSV
    CSV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print(f"events={len(out)} -> {path}")
    if not out.empty:
        print(out.groupby("kind").size().to_string())
        y = out[out["season"] == 2025]
        print(f"2025 rows={len(y)}")
        frank = y[y["player_name"].str.contains("Franklin", case=False, na=False)]
        print("Franklin rows:", len(frank))
        if not frank.empty:
            print(
                frank[["as_of_week", "team_id", "direction", "label"]].to_string(
                    index=False
                )
            )


if __name__ == "__main__":
    main()
