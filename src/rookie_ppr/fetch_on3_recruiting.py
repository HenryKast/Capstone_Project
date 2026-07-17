from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from rookie_ppr.config import HS_CLASS_MAX, HS_CLASS_MIN, RECRUITING_DIR, ensure_directories

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json,text/html",
}
SKILL = {"QB", "RB", "WR", "TE", "ATH", "HB", "FB"}


def _get_build_id(session: requests.Session) -> str:
    html = session.get(
        "https://www.on3.com/rivals/rankings/industry-comparison/football/2020/",
        timeout=60,
        headers=HEADERS,
    ).text
    m = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', html)
    if not m:
        raise RuntimeError("Could not find On3 __NEXT_DATA__ / buildId")
    return json.loads(m.group(1))["buildId"]


def _rating_by_type(ratings: list[dict[str, Any]] | None, typ: str) -> dict[str, Any]:
    if not ratings:
        return {}
    for row in ratings:
        if str(row.get("type", "")).lower() == typ.lower():
            return row
    return {}


def _parse_comparison_item(item: dict[str, Any], hs_class: int) -> dict[str, Any]:
    person = item.get("person") or {}
    ratings = item.get("ratings") or []
    ind = _rating_by_type(ratings, "Industry") or _rating_by_type(ratings, "Consensus")
    riv = _rating_by_type(ratings, "Rivals")
    pos = ((person.get("position") or {}).get("abbr")) or ind.get("positionAbbr") or riv.get("positionAbbr")
    hs = person.get("highSchoolName") or ((person.get("highSchool") or {}).get("name"))
    hometown = (person.get("hometown") or {}).get("abbr") or (person.get("hometown") or {}).get("name")
    state = (person.get("state") or {}).get("abbr")
    return {
        "hs_class": hs_class,
        "player_name": person.get("fullName")
        or f"{person.get('firstName', '')} {person.get('lastName', '')}".strip(),
        "position": pos,
        "industry_composite_rank": ind.get("overallRank"),
        "industry_composite_stars": ind.get("stars"),
        "industry_composite_rating": ind.get("rating"),
        "rivals_rank": riv.get("overallRank"),
        "rivals_stars": riv.get("stars"),
        "rivals_rating": riv.get("rating"),
        "hometown": hometown,
        "high_school": hs,
        "state": state,
        "on3_person_key": person.get("key"),
        "source_page": "industry_comparison",
    }


def _parse_player_item(item: dict[str, Any], hs_class: int) -> dict[str, Any]:
    person = item.get("person") or {}
    ratings = item.get("ratings") or []
    ind = _rating_by_type(ratings, "Industry") or _rating_by_type(ratings, "Consensus")
    riv = _rating_by_type(ratings, "Rivals")
    rating_obj = person.get("rating") or {}
    pos = (
        item.get("positionAbbreviation")
        or rating_obj.get("positionAbbr")
        or ind.get("positionAbbr")
        or riv.get("positionAbbr")
    )
    hs = person.get("highSchoolName") or ((person.get("highSchool") or {}).get("name"))
    hometown = (person.get("hometown") or {}).get("abbr") or (person.get("hometown") or {}).get("name")
    state = item.get("stateAbbreviation") or (person.get("state") or {}).get("abbr")
    ind_rank = (
        ind.get("overallRank")
        or item.get("consensusOverallRank")
        or rating_obj.get("consensusNationalRank")
    )
    ind_stars = ind.get("stars") or rating_obj.get("consensusStars")
    return {
        "hs_class": hs_class,
        "player_name": person.get("fullName")
        or f"{person.get('firstName', '')} {person.get('lastName', '')}".strip(),
        "position": pos,
        "industry_composite_rank": ind_rank,
        "industry_composite_stars": ind_stars,
        "industry_composite_rating": ind.get("rating") or rating_obj.get("consensusRating"),
        "rivals_rank": riv.get("overallRank") or item.get("overallRank") or rating_obj.get("nationalRank"),
        "rivals_stars": riv.get("stars") or rating_obj.get("stars"),
        "rivals_rating": riv.get("rating") or rating_obj.get("rating"),
        "hometown": hometown,
        "high_school": hs,
        "state": state,
        "on3_person_key": person.get("key") or item.get("key"),
        "source_page": "industry_player",
    }


def _fetch_paginated(
    session: requests.Session,
    build_id: str,
    kind: str,
    year: int,
    pause_s: float = 0.12,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    page = 1
    page_count = 1
    while page <= page_count:
        if kind == "industry-comparison":
            url = (
                f"https://www.on3.com/_next/data/{build_id}/rivals/rankings/"
                f"industry-comparison/football/{year}.json"
                f"?sport=football&year={year}&page={page}"
            )
            list_key = "industryComparisons"
            parse = _parse_comparison_item
        else:
            url = (
                f"https://www.on3.com/_next/data/{build_id}/rivals/rankings/"
                f"industry-player/football/{year}.json"
                f"?year={year}&page={page}"
            )
            list_key = "playerData"
            parse = _parse_player_item

        resp = session.get(url, timeout=90, headers=HEADERS)
        resp.raise_for_status()
        payload = resp.json()["pageProps"][list_key]
        page_count = int(payload["pagination"]["pageCount"])
        batch = payload.get("list") or []
        for item in batch:
            rows.append(parse(item, year))
        print(f"  {kind} {year} page {page}/{page_count} (+{len(batch)})")
        page += 1
        time.sleep(pause_s)
    return rows


def _merge_prefer_comparison(comparison: pd.DataFrame, player: pd.DataFrame) -> pd.DataFrame:
    if comparison.empty and player.empty:
        return pd.DataFrame()
    if comparison.empty:
        return player.copy()
    if player.empty:
        return comparison.copy()

    comp_keys = set(comparison["on3_person_key"].dropna().tolist())
    player_only = player[~player["on3_person_key"].isin(comp_keys)].copy()
    if not player_only.empty:
        player_only["source_page"] = "industry_player_only"
    return pd.concat([comparison, player_only], ignore_index=True)


def fetch_year(session: requests.Session, build_id: str, year: int) -> pd.DataFrame:
    print(f"Fetching On3 HS class {year} ...")
    comparison = pd.DataFrame(_fetch_paginated(session, build_id, "industry-comparison", year))
    player = pd.DataFrame(_fetch_paginated(session, build_id, "industry-player", year))
    merged = _merge_prefer_comparison(comparison, player)

    if not merged.empty:
        merged["position"] = merged["position"].astype(str).str.upper().str.strip()
        merged = merged[merged["position"].isin(SKILL)].copy()
        merged["position"] = merged["position"].replace({"HB": "RB", "FB": "RB"})
    return merged


def write_year_csv(df: pd.DataFrame, year: int) -> Path:
    RECRUITING_DIR.mkdir(parents=True, exist_ok=True)
    path = RECRUITING_DIR / f"hs_{year}.csv"
    cols = [
        "hs_class",
        "player_name",
        "position",
        "industry_composite_rank",
        "industry_composite_stars",
        "industry_composite_rating",
        "rivals_rank",
        "rivals_stars",
        "rivals_rating",
        "hometown",
        "high_school",
        "state",
        "on3_person_key",
        "source_page",
    ]
    out = df.reindex(columns=cols)
    out.to_csv(path, index=False)
    return path


def main(years: list[int] | None = None) -> int:
    ensure_directories()
    years = years or list(range(HS_CLASS_MIN, HS_CLASS_MAX + 1))
    session = requests.Session()
    build_id = _get_build_id(session)
    print(f"On3 buildId={build_id}")

    for year in years:
        df = fetch_year(session, build_id, year)
        path = write_year_csv(df, year)
        print(f"Wrote {path} rows={len(df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
