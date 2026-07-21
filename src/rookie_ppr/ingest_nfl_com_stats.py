"""Fetch NFL.com player season stats when nflverse releases lag (e.g. 2025 for 2026 drafts)."""
from __future__ import annotations

import re
import time
import urllib.error
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd

from rookie_ppr.config import RAW_DIR, SKILL_POSITIONS
from rookie_ppr.utils import compact_name_initials, normalize_name, normalize_position, normalize_team_abbr

NFL_COM_CACHE_DIR = RAW_DIR / "nfl_com"
BASE = "https://www.nfl.com"
UA = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml",
}

# category -> (sort key, column names after Player)
CATEGORY_SPECS: dict[str, tuple[str, list[str]]] = {
    "rushing": (
        "rushingyards",
        [
            "rushing_yards",
            "rushing_attempts",
            "rushing_tds",
            "rush_20",
            "rush_40",
            "rush_lng",
            "rush_1st",
            "rush_1st_pct",
            "rushing_fumbles",
        ],
    ),
    "receiving": (
        "receivingreceptions",
        [
            "receptions",
            "receiving_yards",
            "receiving_tds",
            "rec_20",
            "rec_40",
            "rec_lng",
            "rec_1st",
            "rec_1st_pct",
            "receiving_fumbles",
            "rec_yac",
            "targets",
        ],
    ),
    "passing": (
        "passingyards",
        [
            "passing_yards",
            "yds_per_att",
            "passing_attempts",
            "completions",
            "cmp_pct",
            "passing_tds",
            "interceptions",
            "passer_rating",
            "pass_1st",
            "pass_1st_pct",
            "pass_20",
            "pass_40",
            "pass_lng",
            "sacks",
            "sack_yards",
        ],
    ),
}


class _StatsTableParser(HTMLParser):
    """Extract player-stat table rows: name, slug, + numeric cells."""

    def __init__(self) -> None:
        super().__init__()
        self.in_table = False
        self.in_tbody = False
        self.in_row = False
        self.in_cell = False
        self.cell_bits: list[str] = []
        self.row_cells: list[str] = []
        self.current_name: str | None = None
        self.current_slug: str | None = None
        self.rows: list[tuple[str, str | None, list[float]]] = []
        self.next_href: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        ad = {k: (v or "") for k, v in attrs}
        classes = ad.get("class", "")
        if tag == "table" and "d3-o-table" in classes:
            self.in_table = True
        if not self.in_table:
            if tag == "a" and "nfl-o-table-pagination__next" in classes:
                self.next_href = ad.get("href") or None
            return
        if tag == "tbody":
            self.in_tbody = True
        if not self.in_tbody:
            return
        if tag == "tr":
            self.in_row = True
            self.row_cells = []
            self.current_name = None
            self.current_slug = None
        elif self.in_row and tag == "td":
            self.in_cell = True
            self.cell_bits = []
        elif self.in_row and tag == "a" and "d3-o-player-fullname" in classes:
            label = ad.get("aria-label", "")
            if label.endswith(" profile page"):
                self.current_name = label[: -len(" profile page")].strip()
            elif label:
                self.current_name = label.strip()
            href = ad.get("href", "")
            m = re.match(r"/players/([^/]+)/?", href)
            if m:
                self.current_slug = m.group(1)

    def handle_endtag(self, tag: str) -> None:
        if tag == "table" and self.in_table:
            self.in_table = False
            self.in_tbody = False
        if tag == "tbody":
            self.in_tbody = False
        if not self.in_row:
            return
        if tag == "td" and self.in_cell:
            text = re.sub(r"\s+", " ", "".join(self.cell_bits)).strip()
            self.row_cells.append(text)
            self.in_cell = False
            self.cell_bits = []
        elif tag == "tr":
            name = self.current_name
            if not name and self.row_cells:
                name = self.row_cells[0].strip() or None
            nums: list[float] = []
            for cell in self.row_cells[1:]:
                nums.append(_to_float(cell))
            if name and nums:
                self.rows.append((name, self.current_slug, nums))
            self.in_row = False

    def handle_data(self, data: str) -> None:
        if self.in_cell:
            self.cell_bits.append(data)


def _to_float(text: str) -> float:
    t = (text or "").strip().replace(",", "").replace("%", "")
    if not t or t in {".", "-", "—"}:
        return float("nan")
    try:
        return float(t)
    except ValueError:
        return float("nan")


def _fetch(url: str, *, retries: int = 3) -> str:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, TimeoutError) as exc:
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {url}: {last}")


def _absolute(href: str) -> str:
    if href.startswith("http"):
        return href
    return BASE + href


def scrape_category(
    season: int,
    category: str,
    *,
    sort_keys: list[str] | None = None,
    max_pages_per_sort: int = 40,
) -> pd.DataFrame:
    """
    Scrape one NFL.com leaderboard category for a season.

    NFL.com pagination is cursor-based and case-sensitive (REG/DESC). We also
    union several sort keys so dual-threat backs (rush + receiving) and deeper
    depth charts are less likely to be truncated when cursors stall.
    """
    if category not in CATEGORY_SPECS:
        raise ValueError(f"Unknown category: {category}")
    primary_sort, columns = CATEGORY_SPECS[category]
    sorts = sort_keys or {
        "rushing": ["rushingyards", "rushingattempts", "rushingtouchdowns"],
        "receiving": ["receivingreceptions", "receivingyards", "receivingtouchdowns"],
        "passing": ["passingyards", "passingattempts", "passingtouchdowns"],
    }.get(category, [primary_sort])

    records: list[dict] = []
    for sort_key in sorts:
        # First page uses lowercase path; next links use REG/DESC + aftercursor.
        start = f"{BASE}/stats/player-stats/category/{category}/{season}/reg/all/{sort_key}/desc"
        url: str | None = start
        seen_urls: set[str] = set()
        pages = 0
        prev_names: set[str] = set()

        while url and url not in seen_urls and pages < max_pages_per_sort:
            seen_urls.add(url)
            html = _fetch(url)
            parser = _StatsTableParser()
            parser.feed(html)
            page_names = {name for name, _slug, _nums in parser.rows}
            if page_names and page_names <= prev_names:
                break
            prev_names |= page_names
            for name, slug, nums in parser.rows:
                row = {"player_name": name, "nfl_slug": slug}
                for i, col in enumerate(columns):
                    row[col] = nums[i] if i < len(nums) else float("nan")
                records.append(row)
            pages += 1
            nxt = parser.next_href
            if not nxt:
                break
            url = _absolute(nxt)
            time.sleep(0.25)

    if not records:
        return pd.DataFrame(columns=["player_name", *columns])
    df = pd.DataFrame(records)
    sort_col = columns[0]
    df = df.sort_values(sort_col, ascending=False).drop_duplicates("player_name", keep="first")
    return df.reset_index(drop=True)


def _end_of_season_roster(rosters: pd.DataFrame, season: int) -> pd.DataFrame:
    if rosters is None or rosters.empty:
        return pd.DataFrame()
    r = rosters.copy()
    r["season"] = pd.to_numeric(r["season"], errors="coerce")
    r = r[r["season"] == season]
    if r.empty:
        return pd.DataFrame()
    name_col = next((c for c in ("full_name", "player_name", "display_name") if c in r.columns), None)
    if name_col is None or "gsis_id" not in r.columns:
        return pd.DataFrame()
    if "week" in r.columns:
        r = r.sort_values("week")
    r = r.groupby("gsis_id", as_index=False).last()
    r["position"] = r["position"].map(normalize_position) if "position" in r.columns else pd.NA
    r = r[r["position"].isin(SKILL_POSITIONS)].copy()
    r["player_name_norm"] = r[name_col].map(normalize_name)
    r["recent_team"] = r["team"].map(normalize_team_abbr) if "team" in r.columns else pd.NA
    r["player_display_name"] = r[name_col]
    return r[["gsis_id", "player_display_name", "player_name_norm", "position", "recent_team", "season"]]


def _compute_ppr(df: pd.DataFrame) -> pd.Series:
    def col(name: str) -> pd.Series:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").fillna(0.0)
        return pd.Series(0.0, index=df.index)

    return (
        col("receptions")
        + col("rushing_yards") / 10.0
        + col("receiving_yards") / 10.0
        + col("passing_yards") / 25.0
        + col("rushing_tds") * 6.0
        + col("receiving_tds") * 6.0
        + col("passing_tds") * 4.0
        - col("interceptions") * 2.0
    )


def _name_to_slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9\s-]", "", str(name).lower())
    return re.sub(r"\s+", "-", s.strip())


# NFL.com display name → roster full_name (after normalize_name)
NFL_COM_NAME_ALIASES: dict[str, str] = {
    "aj brown": "a j brown",
    "aj barner": "a j barner",
    "dk metcalf": "d k metcalf",
    "dj moore": "d j moore",
    "cj stroud": "c j stroud",
    "jk dobbins": "j k dobbins",
    "jj mccarthy": "j j mccarthy",
}


def _name_keys(name_norm: str) -> list[str]:
    keys: list[str] = []
    base = normalize_name(name_norm)
    for candidate in (
        base,
        compact_name_initials(base),
        NFL_COM_NAME_ALIASES.get(base, ""),
        NFL_COM_NAME_ALIASES.get(compact_name_initials(base), ""),
    ):
        c = normalize_name(candidate) if candidate else ""
        if c and c not in keys:
            keys.append(c)
        compact = compact_name_initials(c) if c else ""
        if compact and compact not in keys:
            keys.append(compact)
    return keys


def _match_leaderboard_to_roster(merged: pd.DataFrame, roster: pd.DataFrame) -> pd.DataFrame:
    """Join NFL.com names to roster with aliases / compact initials / last+initial fallback."""
    if merged.empty or roster.empty:
        return pd.DataFrame()

    m = merged.copy()
    r = roster.copy()
    m["player_name_norm"] = m["player_name"].map(normalize_name)
    m["name_compact"] = m["player_name_norm"].map(compact_name_initials)
    r["name_compact"] = r["player_name_norm"].map(compact_name_initials)

    r_keys = r.copy()
    r_keys["_join"] = r_keys["player_name_norm"]
    r_compact = r.copy()
    r_compact["_join"] = r_compact["name_compact"]
    roster_join = pd.concat([r_keys, r_compact], ignore_index=True).drop_duplicates(
        subset=["_join", "gsis_id"]
    )

    matched_parts: list[pd.DataFrame] = []
    remaining = m.copy()
    for key_fn in (
        lambda row: row["player_name_norm"],
        lambda row: row["name_compact"],
        lambda row: NFL_COM_NAME_ALIASES.get(row["player_name_norm"], ""),
        lambda row: NFL_COM_NAME_ALIASES.get(row["name_compact"], ""),
    ):
        if remaining.empty:
            break
        remaining = remaining.copy()
        remaining["_join"] = remaining.apply(key_fn, axis=1).map(normalize_name)
        hit = remaining.merge(roster_join, on="_join", how="inner", suffixes=("", "_roster"))
        if not hit.empty:
            matched_parts.append(hit)
            used = set(hit["player_name_norm"])
            remaining = remaining[~remaining["player_name_norm"].isin(used)]

    if not remaining.empty:
        r2 = roster.copy()
        r2["last"] = r2["player_name_norm"].map(lambda n: str(n).split()[-1] if str(n).strip() else "")
        r2["init"] = r2["player_name_norm"].map(lambda n: str(n).split()[0][:1] if str(n).strip() else "")
        counts = r2.groupby(["last", "init"]).size().reset_index(name="n")
        unique = counts[counts["n"] == 1][["last", "init"]]
        r2 = r2.merge(unique, on=["last", "init"], how="inner")
        remaining = remaining.copy()
        remaining["last"] = remaining["player_name_norm"].map(
            lambda n: str(n).split()[-1] if str(n).strip() else ""
        )
        remaining["init"] = remaining["player_name_norm"].map(
            lambda n: str(n).split()[0][:1] if str(n).strip() else ""
        )
        hit = remaining.merge(r2, on=["last", "init"], how="inner", suffixes=("", "_roster"))
        if not hit.empty:
            matched_parts.append(hit)

    if not matched_parts:
        return pd.DataFrame()
    return pd.concat(matched_parts, ignore_index=True).drop_duplicates(subset=["gsis_id"]).reset_index(
        drop=True
    )


def _parse_player_season_cells(cells: list[str], position: str) -> dict[str, float]:
    """
    Map NFL.com career-season row cells to fantasy-relevant fields.

    Cell layouts differ by position (RB vs WR/TE vs QB).
    """
    vals = [_to_float(c) for c in cells]
    # cells: year, team, G, GS, ...
    if len(vals) < 8:
        return {}
    pos = (position or "").upper()
    out: dict[str, float] = {}
    if pos == "QB":
        # G GS COMP ATT YDS AVG TD INT ...
        out["completions"] = vals[4]
        out["passing_attempts"] = vals[5]
        out["passing_yards"] = vals[6]
        out["passing_tds"] = vals[8]
        out["interceptions"] = vals[9]
    elif pos == "RB":
        # G GS ATT YDS AVG TD REC YDS AVG LNG TD FUM LOST
        out["rushing_attempts"] = vals[4]
        out["rushing_yards"] = vals[5]
        out["rushing_tds"] = vals[7]
        if len(vals) > 12:
            out["receptions"] = vals[8]
            out["receiving_yards"] = vals[9]
            out["receiving_tds"] = vals[12]
    else:
        # WR/TE: G GS REC YDS AVG LNG TD ATT YDS AVG TD ...
        out["receptions"] = vals[4]
        out["receiving_yards"] = vals[5]
        out["receiving_tds"] = vals[8]
        if len(vals) > 12:
            out["rushing_attempts"] = vals[9]
            out["rushing_yards"] = vals[10]
            out["rushing_tds"] = vals[12]
    return out


def fetch_player_season_stats(slug: str, season: int, position: str) -> dict[str, float]:
    """Pull one player's season line from https://www.nfl.com/players/{slug}/stats/."""
    if not slug:
        return {}
    cache_dir = NFL_COM_CACHE_DIR / "player_pages"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{slug}_{season}.html"
    if cache_path.exists():
        html = cache_path.read_text(encoding="utf-8", errors="replace")
    else:
        html = _fetch(f"{BASE}/players/{slug}/stats/")
        cache_path.write_text(html, encoding="utf-8")
        time.sleep(0.2)

    rows = re.findall(
        rf"<tr[^>]*>\s*<td[^>]*>\s*{season}\s*</td>.*?</tr>",
        html,
        flags=re.S | re.I,
    )
    if not rows:
        return {}
    cells = re.findall(r"<td[^>]*>(.*?)</td>", rows[0], flags=re.S | re.I)
    cells = [re.sub(r"<[^>]+>", "", c).strip() for c in cells]
    return _parse_player_season_cells(cells, position)


def build_nfl_com_season_stats(season: int, rosters: pd.DataFrame) -> pd.DataFrame:
    """
    Build nflverse-compatible season stats from NFL.com leaderboards + roster team/ids.

    Leaderboards are truncated (~50/sort); we union several sorts then enrich each
    matched player from their NFL.com player stats page for full rush+rec+pass lines.
    """
    rush = scrape_category(season, "rushing")
    recv = scrape_category(season, "receiving")
    pas = scrape_category(season, "passing")

    frames = []
    for part in (rush, recv, pas):
        if part.empty:
            continue
        p = part.copy()
        p["player_name_norm"] = p["player_name"].map(normalize_name)
        frames.append(p)

    if not frames:
        return pd.DataFrame()

    merged = frames[0]
    for other in frames[1:]:
        merged = merged.merge(other, on="player_name_norm", how="outer", suffixes=("", "_r"))
        if "player_name_r" in merged.columns:
            merged["player_name"] = merged["player_name"].fillna(merged["player_name_r"])
            merged = merged.drop(columns=["player_name_r"])
        drop_r = [c for c in merged.columns if c.endswith("_r")]
        for c in drop_r:
            base = c[:-2]
            if base in merged.columns:
                merged[base] = merged[base].combine_first(merged[c])
            else:
                merged[base] = merged[c]
            merged = merged.drop(columns=[c])

    roster = _end_of_season_roster(rosters, season)
    if roster.empty:
        return pd.DataFrame()

    out = _match_leaderboard_to_roster(merged, roster)
    if out.empty:
        print(f"  NFL.com {season}: 0 players matched to roster after name normalization")
        return pd.DataFrame()
    print(f"  NFL.com {season}: matched {len(out)} / {merged['player_name'].nunique()} leaderboard players to roster")
    out["season"] = season
    if "nfl_slug" not in out.columns:
        out["nfl_slug"] = pd.NA
    out["nfl_slug"] = out["nfl_slug"].fillna(out["player_name"].map(_name_to_slug))

    print(f"  Enriching {len(out)} players from NFL.com player pages ...")
    enriched_rows = []
    for _, row in out.iterrows():
        stats = fetch_player_season_stats(str(row["nfl_slug"]), season, str(row["position"]))
        if not stats:
            # fallback: keep leaderboard partial stats
            enriched_rows.append({})
            continue
        enriched_rows.append(stats)
        time.sleep(0.05)
    enriched = pd.DataFrame(enriched_rows, index=out.index)
    for col in enriched.columns:
        if col in out.columns:
            out[col] = enriched[col].combine_first(out[col])
        else:
            out[col] = enriched[col]

    out["fantasy_points_ppr"] = _compute_ppr(out)
    out["player_name"] = out["player_display_name"].fillna(out.get("player_name"))
    keep = [
        "gsis_id",
        "season",
        "player_name",
        "player_display_name",
        "position",
        "recent_team",
        "fantasy_points_ppr",
        "receptions",
        "rushing_yards",
        "receiving_yards",
        "passing_yards",
        "rushing_tds",
        "receiving_tds",
        "passing_tds",
        "interceptions",
        "rushing_attempts",
        "carries",
        "targets",
    ]
    out = out[[c for c in keep if c in out.columns]].copy()
    if "carries" not in out.columns and "rushing_attempts" in out.columns:
        out["carries"] = out["rushing_attempts"]
    out = out.dropna(subset=["recent_team", "position"])
    out = out[out["fantasy_points_ppr"].fillna(0) > 0]
    return out.drop_duplicates(subset=["gsis_id", "season"]).reset_index(drop=True)


def load_or_fetch_nfl_com_season_stats(
    season: int,
    rosters: pd.DataFrame,
    *,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Cache NFL.com season stats under data/raw/nfl_com/."""
    NFL_COM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = NFL_COM_CACHE_DIR / f"player_stats_{season}.csv"
    if path.exists() and not force_refresh:
        df = pd.read_csv(path)
        if not df.empty and "fantasy_points_ppr" in df.columns:
            return df
    print(f"  Fetching NFL.com {season} player stats (rushing/receiving/passing) ...")
    df = build_nfl_com_season_stats(season, rosters)
    if not df.empty:
        df.to_csv(path, index=False)
        print(f"  Wrote {path} ({len(df)} players)")
    else:
        print(f"  NFL.com {season} stats empty after roster match")
    return df


def fill_missing_season_stats(
    stats: pd.DataFrame,
    rosters: pd.DataFrame,
    seasons: list[int] | None = None,
) -> pd.DataFrame:
    """
    Append NFL.com season stats for any requested season missing from nflverse stats.
    """
    out = stats.copy() if stats is not None and not stats.empty else pd.DataFrame()
    have = set()
    if not out.empty and "season" in out.columns:
        have = set(pd.to_numeric(out["season"], errors="coerce").dropna().astype(int).unique())

    needed = seasons or []
    for season in needed:
        if season in have:
            continue
        extra = load_or_fetch_nfl_com_season_stats(season, rosters)
        if extra.empty:
            continue
        out = pd.concat([out, extra], ignore_index=True) if not out.empty else extra
        have.add(season)
    return out
