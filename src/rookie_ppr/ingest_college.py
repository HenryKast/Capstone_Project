from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import requests

from rookie_ppr.config import CFBD_API_KEY, CFBD_BASE_URL, CFBD_CACHE_DIR, DRAFT_YEAR_MAX, DRAFT_YEAR_MIN
from rookie_ppr.utils import normalize_name

CATEGORIES = ("passing", "rushing", "receiving")

EMPTY_COLS = [
    "player_name_norm",
    "position",
    "draft_year",
    "college",
    "cfb_final_season",
    "cfb_pass_yards",
    "cfb_pass_td",
    "cfb_rush_yards",
    "cfb_rush_td",
    "cfb_rec",
    "cfb_rec_yards",
    "cfb_rec_td",
]


def _cache_path(year: int, category: str) -> Path:
    CFBD_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CFBD_CACHE_DIR / f"player_season_{year}_{category}.json"


def fetch_cfbd_player_season(year: int, category: str) -> pd.DataFrame:
    """Fetch CFBD player season stats for one year/category. Requires CFBD_API_KEY."""
    if not CFBD_API_KEY:
        return pd.DataFrame()

    cache = _cache_path(year, category)
    if cache.exists():
        data = json.loads(cache.read_text(encoding="utf-8"))
        return pd.DataFrame(data)

    headers = {"Authorization": f"Bearer {CFBD_API_KEY}", "Accept": "application/json"}
    url = f"{CFBD_BASE_URL}/stats/player/season"
    try:
        resp = requests.get(
            url,
            headers=headers,
            params={"year": year, "category": category, "seasonType": "both"},
            timeout=120,
        )
        if resp.status_code != 200:
            print(f"  CFBD {year} {category}: HTTP {resp.status_code}")
            return pd.DataFrame()
        data = resp.json()
        cache.write_text(json.dumps(data), encoding="utf-8")
        return pd.DataFrame(data)
    except Exception as exc:  # noqa: BLE001
        print(f"  CFBD {year} {category}: error {exc}")
        return pd.DataFrame()


def _pivot_season_stats(long_df: pd.DataFrame) -> pd.DataFrame:
    """Convert CFBD long rows (category/statType/stat) into one row per player-season."""
    if long_df.empty or "player" not in long_df.columns:
        return pd.DataFrame()

    df = long_df.copy()
    df["player_name_norm"] = df["player"].map(normalize_name)
    df["statType"] = df.get("statType", pd.Series(dtype=str)).astype(str).str.upper().str.strip()
    df["category"] = df.get("category", pd.Series(dtype=str)).astype(str).str.lower().str.strip()
    df["stat"] = pd.to_numeric(df.get("stat"), errors="coerce")
    season_col = "cfb_season" if "cfb_season" in df.columns else "season"
    if season_col not in df.columns:
        return pd.DataFrame()

    targets = {
        ("passing", frozenset({"YDS", "YARDS"})): "cfb_pass_yards",
        ("passing", frozenset({"TD", "TDS"})): "cfb_pass_td",
        ("rushing", frozenset({"YDS", "YARDS"})): "cfb_rush_yards",
        ("rushing", frozenset({"TD", "TDS"})): "cfb_rush_td",
        ("receiving", frozenset({"REC", "RECEPTIONS"})): "cfb_rec",
        ("receiving", frozenset({"YDS", "YARDS"})): "cfb_rec_yards",
        ("receiving", frozenset({"TD", "TDS"})): "cfb_rec_td",
    }

    wide = df[["player_name_norm", season_col]].drop_duplicates().rename(columns={season_col: "cfb_season"})
    for (cat, types), out_col in targets.items():
        mask = (df["category"] == cat) & (df["statType"].isin(types))
        if not mask.any():
            wide[out_col] = pd.NA
            continue
        piece = (
            df.loc[mask]
            .groupby(["player_name_norm", season_col], as_index=False)["stat"]
            .sum()
            .rename(columns={season_col: "cfb_season", "stat": out_col})
        )
        wide = wide.merge(piece, how="left", on=["player_name_norm", "cfb_season"])
    return wide


def load_college_production(draft: pd.DataFrame) -> pd.DataFrame:
    """
    Build college production features for drafted players from CFBD.
    Uses each player's final CFB season before/at draft year.
    """
    if draft.empty or not CFBD_API_KEY:
        return pd.DataFrame(columns=EMPTY_COLS)

    years = list(range(DRAFT_YEAR_MIN - 4, DRAFT_YEAR_MAX + 1))
    frames: list[pd.DataFrame] = []
    for year in years:
        for category in CATEGORIES:
            print(f"  CFBD fetch {year} {category} ...")
            df = fetch_cfbd_player_season(year, category)
            if df.empty:
                continue
            if "season" not in df.columns:
                df["season"] = year
            df["cfb_season"] = year
            frames.append(df)

    if not frames:
        return pd.DataFrame(columns=EMPTY_COLS)

    long_df = pd.concat(frames, ignore_index=True)
    wide = _pivot_season_stats(long_df)
    if wide.empty:
        return pd.DataFrame(columns=EMPTY_COLS)

    out_rows = []
    for _, row in draft.iterrows():
        name = row.get("player_name_norm")
        pos = row.get("position")
        dyear = row.get("draft_year")
        college = row.get("college")
        subset = wide[wide["player_name_norm"] == name]
        if subset.empty:
            out_rows.append(
                {
                    "player_name_norm": name,
                    "position": pos,
                    "draft_year": dyear,
                    "college": college,
                    "cfb_final_season": pd.NA,
                    "cfb_pass_yards": pd.NA,
                    "cfb_pass_td": pd.NA,
                    "cfb_rush_yards": pd.NA,
                    "cfb_rush_td": pd.NA,
                    "cfb_rec": pd.NA,
                    "cfb_rec_yards": pd.NA,
                    "cfb_rec_td": pd.NA,
                }
            )
            continue

        # Prefer seasons at/before draft year
        if pd.notna(dyear):
            prior = subset[subset["cfb_season"] <= int(dyear)]
            if not prior.empty:
                subset = prior
        final_season = int(subset["cfb_season"].max())
        last = subset[subset["cfb_season"] == final_season].iloc[0]
        out_rows.append(
            {
                "player_name_norm": name,
                "position": pos,
                "draft_year": dyear,
                "college": college,
                "cfb_final_season": final_season,
                "cfb_pass_yards": last.get("cfb_pass_yards"),
                "cfb_pass_td": last.get("cfb_pass_td"),
                "cfb_rush_yards": last.get("cfb_rush_yards"),
                "cfb_rush_td": last.get("cfb_rush_td"),
                "cfb_rec": last.get("cfb_rec"),
                "cfb_rec_yards": last.get("cfb_rec_yards"),
                "cfb_rec_td": last.get("cfb_rec_td"),
            }
        )

    return pd.DataFrame(out_rows)
