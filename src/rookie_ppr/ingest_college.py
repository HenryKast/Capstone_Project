from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import requests

from rookie_ppr.config import (
    CFBD_API_KEY,
    CFBD_BASE_URL,
    CFBD_CACHE_DIR,
    DRAFT_YEAR_MAX,
    DRAFT_YEAR_MIN,
    MANUAL_DIR,
)
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

# Draft-name → CFBD display name (after normalize_name)
DEFAULT_NAME_ALIASES: dict[str, str] = {
    "mike washington": "michael washington",
    "kc concepcion": "kevin concepcion",
    "reggie virgil": "reginald virgil",
    "cj daniels": "c j daniels",
    "jam miller": "jamarion miller",
}

# nflverse / draft college labels → CFBD team names
COLLEGE_TO_CFBD_TEAM: dict[str, str] = {
    "miami (fl)": "Miami",
    "miami fl": "Miami",
    "miami": "Miami",
    "texas a&m": "Texas A&M",
    "texas am": "Texas A&M",
    "lsu": "LSU",
    "ole miss": "Ole Miss",
    "mississippi state": "Mississippi State",
    "southern cal": "USC",
    "usc": "USC",
    "ucla": "UCLA",
    "byu": "BYU",
    "tcu": "TCU",
    "smu": "SMU",
    "ucf": "UCF",
    "usf": "South Florida",
    "central florida": "UCF",
    "florida atlantic": "Florida Atlantic",
    "florida international": "Florida International",
    "middle tennessee": "Middle Tennessee",
    "western kentucky": "Western Kentucky",
    "new mexico state": "New Mexico State",
    "miami (oh)": "Miami (OH)",
    "miami oh": "Miami (OH)",
}


def _cache_path(year: int, category: str) -> Path:
    CFBD_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CFBD_CACHE_DIR / f"player_season_{year}_{category}.json"


def fetch_cfbd_player_season(year: int, category: str, *, force: bool = False) -> pd.DataFrame:
    """Fetch CFBD player season stats for one year/category. Requires CFBD_API_KEY."""
    if not CFBD_API_KEY:
        return pd.DataFrame()

    cache = _cache_path(year, category)
    if cache.exists() and not force:
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


def _load_name_aliases() -> dict[str, str]:
    aliases = dict(DEFAULT_NAME_ALIASES)
    path = MANUAL_DIR / "cfb_name_aliases.csv"
    if not path.exists():
        return aliases
    try:
        df = pd.read_csv(path)
    except Exception:  # noqa: BLE001
        return aliases
    for _, row in df.iterrows():
        src = normalize_name(row.get("draft_name") or row.get("player_name_norm"))
        dst = normalize_name(row.get("cfbd_name") or row.get("alias_name"))
        if src and dst:
            aliases[src] = dst
    return aliases


def _compact_initials(name_norm: str) -> str:
    """'c j daniels' / 'cj daniels' → 'cj daniels'."""
    parts = [p for p in str(name_norm).split() if p]
    if len(parts) < 2:
        return name_norm
    out: list[str] = []
    i = 0
    while i < len(parts):
        if len(parts[i]) == 1:
            initials = parts[i]
            j = i + 1
            while j < len(parts) and len(parts[j]) == 1:
                initials += parts[j]
                j += 1
            out.append(initials)
            i = j
        else:
            out.append(parts[i])
            i += 1
    return " ".join(out)


def _name_keys(name_norm: str, aliases: dict[str, str]) -> list[str]:
    keys: list[str] = []
    base = normalize_name(name_norm)
    for candidate in (base, aliases.get(base, ""), _compact_initials(base), aliases.get(_compact_initials(base), "")):
        c = normalize_name(candidate) if candidate else ""
        if c and c not in keys:
            keys.append(c)
        compact = _compact_initials(c) if c else ""
        if compact and compact not in keys:
            keys.append(compact)
    return keys


def _cfbd_team_for_college(college: str | None) -> str | None:
    if college is None or (isinstance(college, float) and pd.isna(college)):
        return None
    raw = str(college).strip()
    key = re.sub(r"[^a-z0-9\s&]", " ", raw.lower())
    key = re.sub(r"\s+", " ", key).strip()
    if key in COLLEGE_TO_CFBD_TEAM:
        return COLLEGE_TO_CFBD_TEAM[key]
    # Soft fallback: title-case original without parenthetical
    cleaned = re.sub(r"\([^)]*\)", "", raw).strip()
    return cleaned or raw


def _pivot_season_stats(long_df: pd.DataFrame) -> pd.DataFrame:
    """Convert CFBD long rows into one row per player-season-team."""
    if long_df.empty or "player" not in long_df.columns:
        return pd.DataFrame()

    df = long_df.copy()
    df["player_name_norm"] = df["player"].map(normalize_name)
    df["player_name_compact"] = df["player_name_norm"].map(_compact_initials)
    df["statType"] = df.get("statType", pd.Series(dtype=str)).astype(str).str.upper().str.strip()
    df["category"] = df.get("category", pd.Series(dtype=str)).astype(str).str.lower().str.strip()
    df["stat"] = pd.to_numeric(df.get("stat"), errors="coerce")
    if "team" not in df.columns:
        df["team"] = pd.NA
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

    keys = ["player_name_norm", "player_name_compact", "team", season_col]
    wide = df[keys].drop_duplicates().rename(columns={season_col: "cfb_season"})
    for (cat, types), out_col in targets.items():
        mask = (df["category"] == cat) & (df["statType"].isin(types))
        if not mask.any():
            wide[out_col] = pd.NA
            continue
        piece = (
            df.loc[mask]
            .groupby(["player_name_norm", "team", season_col], as_index=False)["stat"]
            .sum()
            .rename(columns={season_col: "cfb_season", "stat": out_col})
        )
        wide = wide.merge(piece, how="left", on=["player_name_norm", "team", "cfb_season"])
    wide["last_name"] = wide["player_name_norm"].map(lambda n: str(n).split()[-1] if str(n).strip() else "")
    wide["first_token"] = wide["player_name_norm"].map(lambda n: str(n).split()[0] if str(n).strip() else "")
    return wide


def _empty_row(name, pos, dyear, college) -> dict:
    return {
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


def _row_from_last(name, pos, dyear, college, last: pd.Series, final_season: int) -> dict:
    return {
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


def _match_player_seasons(
    wide: pd.DataFrame,
    name_norm: str,
    college: str | None,
    aliases: dict[str, str],
) -> pd.DataFrame:
    if wide.empty:
        return wide

    keys = _name_keys(str(name_norm or ""), aliases)
    subset = wide[wide["player_name_norm"].isin(keys) | wide["player_name_compact"].isin(keys)]
    team = _cfbd_team_for_college(college)

    if subset.empty and team:
        # Last-name + team (covers Mike→Michael, Jam→Jamarion, Reggie→Reginald)
        last = str(name_norm or "").split()[-1] if str(name_norm or "").strip() else ""
        if last:
            team_mask = wide["team"].astype(str).str.casefold() == str(team).casefold()
            subset = wide[team_mask & (wide["last_name"] == last)]
            # Prefer same first initial when multiple last-name matches
            if len(subset["player_name_norm"].unique()) > 1:
                first = str(name_norm or "").split()[0] if str(name_norm or "").strip() else ""
                if first:
                    init = first[0]
                    tighter = subset[subset["first_token"].str.startswith(init, na=False)]
                    if not tighter.empty:
                        subset = tighter

    if subset.empty and team and keys:
        # Compact initials on team (cj vs c j)
        team_mask = wide["team"].astype(str).str.casefold() == str(team).casefold()
        subset = wide[team_mask & wide["player_name_compact"].isin(keys)]

    return subset


def _stats_present(row: dict) -> bool:
    for c in (
        "cfb_pass_yards",
        "cfb_pass_td",
        "cfb_rush_yards",
        "cfb_rush_td",
        "cfb_rec",
        "cfb_rec_yards",
        "cfb_rec_td",
    ):
        v = row.get(c)
        if v is not None and not (isinstance(v, float) and pd.isna(v)):
            return True
    return False


def load_college_production(draft: pd.DataFrame) -> pd.DataFrame:
    """
    Build college production features for drafted players from CFBD,
    with athletics-site roster scraping as a fallback for unmatched names.
    """
    if draft.empty:
        return pd.DataFrame(columns=EMPTY_COLS)

    aliases = _load_name_aliases()
    wide = pd.DataFrame()

    if CFBD_API_KEY:
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
        if frames:
            long_df = pd.concat(frames, ignore_index=True)
            wide = _pivot_season_stats(long_df)

    out_rows: list[dict] = []
    display_by_key: dict[tuple, str] = {}

    for _, row in draft.iterrows():
        name = row.get("player_name_norm")
        pos = row.get("position")
        dyear = row.get("draft_year")
        college = row.get("college")
        display_name = row.get("player_name") or name
        display_by_key[(str(name), dyear)] = str(display_name)

        subset = _match_player_seasons(wide, str(name or ""), college, aliases) if not wide.empty else pd.DataFrame()
        if subset.empty:
            out_rows.append(_empty_row(name, pos, dyear, college))
            continue

        if pd.notna(dyear):
            prior = subset[subset["cfb_season"] <= int(dyear)]
            if not prior.empty:
                subset = prior
        final_season = int(subset["cfb_season"].max())
        last = subset[subset["cfb_season"] == final_season].iloc[0]
        out_rows.append(_row_from_last(name, pos, dyear, college, last, final_season))

    result = pd.DataFrame(out_rows)
    if result.empty:
        result = pd.DataFrame(columns=EMPTY_COLS)

    still_missing: list[dict] = []
    for _, r in result.iterrows():
        if _stats_present(r.to_dict()):
            continue
        key = (str(r.get("player_name_norm")), r.get("draft_year"))
        still_missing.append(
            {
                "player_name": display_by_key.get(key, r.get("player_name_norm")),
                "player_name_norm": r.get("player_name_norm"),
                "position": r.get("position"),
                "draft_year": r.get("draft_year"),
                "college": r.get("college"),
            }
        )

    if still_missing:
        from rookie_ppr.ingest_school_roster import fill_college_from_school_sites

        print(f"  School-site fallback for {len(still_missing)} players ...")
        scraped = fill_college_from_school_sites(still_missing)
        if not scraped.empty:
            for _, srow in scraped.iterrows():
                mask = (result["player_name_norm"] == srow["player_name_norm"]) & (
                    result["draft_year"] == srow["draft_year"]
                )
                if not mask.any():
                    result = pd.concat([result, pd.DataFrame([srow.to_dict()])], ignore_index=True)
                    continue
                for col in EMPTY_COLS:
                    if col in ("player_name_norm", "position", "draft_year", "college"):
                        continue
                    val = srow.get(col)
                    if val is not None and not (isinstance(val, float) and pd.isna(val)):
                        result.loc[mask, col] = val

    return result
