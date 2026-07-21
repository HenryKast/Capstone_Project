from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests
from bs4 import BeautifulSoup

from rookie_ppr.config import MANUAL_DIR, RAW_DIR
from rookie_ppr.utils import normalize_name

SCHOOL_ROSTER_CACHE = RAW_DIR / "school_rosters"

# college (lower) → athletics site base used for /roster/{slug}/
COLLEGE_ROSTER_BASE: dict[str, str] = {
    "arkansas": "https://arkansasrazorbacks.com",
    "alabama": "https://rolltide.com",
    "texas a&m": "https://12thman.com",
    "texas am": "https://12thman.com",
    "texas tech": "https://texastech.com",
    "miami (fl)": "https://miamihurricanes.com",
    "miami fl": "https://miamihurricanes.com",
    "miami": "https://miamihurricanes.com",
    "ohio state": "https://ohiostatebuckeyes.com",
    "georgia": "https://georgiadogs.com",
    "lsu": "https://lsusports.net",
    "florida": "https://floridagators.com",
    "tennessee": "https://utsports.com",
    "oregon": "https://goducks.com",
    "notre dame": "https://fightingirish.com",
    "penn state": "https://gopsusports.com",
    "michigan": "https://mgoblue.com",
    "oklahoma": "https://soonersports.com",
    "texas": "https://texassports.com",
    "usc": "https://usctrojans.com",
    "ucla": "https://uclabruins.com",
    "clemson": "https://clemsontigers.com",
    "florida state": "https://seminoles.com",
    "auburn": "https://auburntigers.com",
    "wisconsin": "https://uwbadgers.com",
    "iowa": "https://hawkeyesports.com",
    "nebraska": "https://huskers.com",
    "missouri": "https://mutigers.com",
    "kentucky": "https://ukathletics.com",
    "south carolina": "https://gamecocksonline.com",
    "mississippi state": "https://hailstate.com",
    "ole miss": "https://olemisssports.com",
    "vanderbilt": "https://vucommodores.com",
    "indiana": "https://iuhoosiers.com",
    "illinois": "https://fightingillini.com",
    "minnesota": "https://gophersports.com",
    "northwestern": "https://nusports.com",
    "purdue": "https://purduesports.com",
    "maryland": "https://umterps.com",
    "rutgers": "https://scarletknights.com",
    "washington": "https://gohuskies.com",
    "oregon state": "https://osubeavers.com",
    "arizona": "https://arizonawildcats.com",
    "arizona state": "https://thesundevils.com",
    "colorado": "https://cubuffs.com",
    "utah": "https://utahutes.com",
    "stanford": "https://gostanford.com",
    "california": "https://calbears.com",
    "nc state": "https://gopack.com",
    "north carolina": "https://goheels.com",
    "duke": "https://goduke.com",
    "virginia": "https://virginiasports.com",
    "virginia tech": "https://hokiesports.com",
    "louisville": "https://gocards.com",
    "pittsburgh": "https://pittsburghpanthers.com",
    "syracuse": "https://cuse.com",
    "boston college": "https://bceagles.com",
    "wake forest": "https://godeacs.com",
    "georgia tech": "https://ramblinwreck.com",
    "baylor": "https://baylorbears.com",
    "kansas": "https://kuathletics.com",
    "kansas state": "https://www.kstatesports.com",
    "kansas st.": "https://www.kstatesports.com",
    "florida st.": "https://seminoles.com",
    "arizona st.": "https://thesundevils.com",
    "oklahoma state": "https://okstate.com",
    "west virginia": "https://wvusports.com",
    "cincinnati": "https://gobearcats.com",
    "houston": "https://uhcougars.com",
    "ucf": "https://ucfknights.com",
    "central florida": "https://ucfknights.com",
    "byu": "https://byucougars.com",
    "boise state": "https://broncosports.com",
}

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


def _college_key(college: str | None) -> str:
    if college is None or (isinstance(college, float) and pd.isna(college)):
        return ""
    key = re.sub(r"[^a-z0-9\s&]", " ", str(college).lower())
    return re.sub(r"\s+", " ", key).strip()


def _roster_base(college: str | None) -> str | None:
    key = _college_key(college)
    if key in COLLEGE_ROSTER_BASE:
        return COLLEGE_ROSTER_BASE[key]
    # Manual overrides
    path = MANUAL_DIR / "college_roster_sites.csv"
    if path.exists():
        try:
            df = pd.read_csv(path)
            for _, row in df.iterrows():
                if _college_key(row.get("college")) == key and row.get("base_url"):
                    return str(row["base_url"]).rstrip("/")
        except Exception:  # noqa: BLE001
            pass
    return None


def _name_slugs(player_name: str) -> list[str]:
    raw = str(player_name or "").strip().lower()
    raw = raw.replace("'", "").replace(".", "")
    raw = re.sub(r"[^a-z0-9\s-]", " ", raw)
    raw = re.sub(r"\s+", " ", raw).strip()
    parts = raw.split()
    if not parts:
        return []
    slugs = [
        "-".join(parts),
        "-".join(p for p in parts if p not in {"jr", "sr", "ii", "iii", "iv"}),
    ]
    # Without middle names if 3+ tokens
    if len(parts) >= 3:
        slugs.append(f"{parts[0]}-{parts[-1]}")
        if parts[-1] in {"jr", "sr", "ii", "iii", "iv"} and len(parts) >= 3:
            slugs.append(f"{parts[0]}-{parts[-2]}-{parts[-1]}")
    # Dedupe preserving order
    out: list[str] = []
    for s in slugs:
        s = re.sub(r"-+", "-", s).strip("-")
        if s and s not in out:
            out.append(s)
    return out


def _cache_path(college: str, slug: str) -> Path:
    SCHOOL_ROSTER_CACHE.mkdir(parents=True, exist_ok=True)
    safe_college = re.sub(r"[^a-z0-9]+", "_", _college_key(college)) or "unknown"
    return SCHOOL_ROSTER_CACHE / f"{safe_college}_{slug}.html"


def fetch_roster_html(college: str, player_name: str) -> tuple[str | None, str | None]:
    """Return (html, url) for the first working roster slug."""
    base = _roster_base(college)
    if not base:
        return None, None
    headers = {"User-Agent": USER_AGENT, "Accept": "text/html"}
    for slug in _name_slugs(player_name):
        cache = _cache_path(college, slug)
        url = f"{base}/roster/{slug}/"
        if cache.exists():
            text = cache.read_text(encoding="utf-8", errors="ignore")
            if text and "404" not in text[:200].lower():
                return text, url
        try:
            resp = requests.get(url, headers=headers, timeout=45)
            if resp.status_code != 200:
                continue
            text = resp.text
            # Sidearm / modern sites often 200 with soft-404
            low = text.lower()
            if "page not found" in low or "404" in low[:500]:
                continue
            cache.write_text(text, encoding="utf-8")
            return text, url
        except Exception:  # noqa: BLE001
            continue
    return None, None


def _parse_int(match: re.Match[str] | None, group: int = 1) -> float:
    if not match:
        return float("nan")
    try:
        return float(str(match.group(group)).replace(",", ""))
    except Exception:  # noqa: BLE001
        return float("nan")


def parse_season_stats_from_bio(html: str, preferred_season: int | None = None) -> dict:
    """
    Extract final-season rushing/receiving/passing totals from athletics bio text.

    Handles patterns like:
      Rushed 167 times for 1,070 yards and eight touchdowns while catching 28 passes for 226 yards and a touchdown
    """
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text("\n", strip=True)
    # Prefer the preferred season block when present
    season = preferred_season
    if season is None:
        years = [int(y) for y in re.findall(r"\b(20\d{2})\b", text)]
        season = max(years) if years else None

    # Slice around the season header when possible
    block = text
    if season is not None:
        pat = re.compile(rf"{season}\s*\([^)]*\)\s*:(.*?)(?=\n20\d{{2}}\s*\(|\nAT |\nHIGH SCHOOL:|\Z)", re.S | re.I)
        m = pat.search(text)
        if m:
            block = m.group(1)
        else:
            # Fallback: from season year to next year heading
            pat2 = re.compile(rf"{season}\b(.*?)(?=\n20\d{{2}}\b|\nAT |\nHIGH SCHOOL:|\Z)", re.S | re.I)
            m2 = pat2.search(text)
            if m2:
                block = m2.group(1)

    word_nums = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "a": 1,
        "an": 1,
    }

    def _num(token: str) -> float:
        t = token.lower().replace(",", "")
        if t in word_nums:
            return float(word_nums[t])
        try:
            return float(t)
        except Exception:  # noqa: BLE001
            return float("nan")

    out = {
        "cfb_final_season": season if season is not None else pd.NA,
        "cfb_pass_yards": pd.NA,
        "cfb_pass_td": pd.NA,
        "cfb_rush_yards": pd.NA,
        "cfb_rush_td": pd.NA,
        "cfb_rec": pd.NA,
        "cfb_rec_yards": pd.NA,
        "cfb_rec_td": pd.NA,
    }

    # Rushing: "Rushed 167 times for 1,070 yards and eight touchdowns"
    rush = re.search(
        r"rushed\s+(\d[\d,]*)\s+times\s+for\s+(\d[\d,]*)\s+yards(?:\s+and\s+([a-z0-9]+)\s+touchdowns?)?",
        block,
        flags=re.I,
    )
    if rush:
        out["cfb_rush_yards"] = _num(rush.group(2))
        if rush.group(3):
            out["cfb_rush_td"] = _num(rush.group(3))

    # Receiving: "catching 28 passes for 226 yards and a touchdown"
    recv = re.search(
        r"catch(?:ing)?\s+(\d[\d,]*)\s+passes\s+for\s+(\d[\d,]*)\s+yards(?:\s+and\s+([a-z0-9]+)\s+touchdowns?)?",
        block,
        flags=re.I,
    )
    if not recv:
        recv = re.search(
            r"(\d[\d,]*)\s+receptions?\s+for\s+(\d[\d,]*)\s+yards(?:\s+and\s+([a-z0-9]+)\s+touchdowns?)?",
            block,
            flags=re.I,
        )
    if recv:
        out["cfb_rec"] = _num(recv.group(1))
        out["cfb_rec_yards"] = _num(recv.group(2))
        if recv.group(3):
            out["cfb_rec_td"] = _num(recv.group(3))

    # Passing
    pas = re.search(
        r"(?:threw|passed)\s+for\s+(\d[\d,]*)\s+yards(?:\s+and\s+([a-z0-9]+)\s+touchdowns?)?",
        block,
        flags=re.I,
    )
    if pas:
        out["cfb_pass_yards"] = _num(pas.group(1))
        if pas.group(2):
            out["cfb_pass_td"] = _num(pas.group(2))

    return out


def _is_missing_stat(v: object) -> bool:
    if v is None:
        return True
    try:
        return bool(pd.isna(v))
    except Exception:  # noqa: BLE001
        return False


def fill_college_from_school_sites(players: list[dict]) -> pd.DataFrame:
    """
    For each player dict with player_name/college/draft_year, try athletics roster pages.
    Returns rows shaped like college_production.
    """
    rows: list[dict] = []
    for p in players:
        name = p.get("player_name") or p.get("player_name_norm")
        college = p.get("college")
        if not name or not college:
            continue
        html, url = fetch_roster_html(str(college), str(name))
        if not html:
            print(f"    no roster page: {name} @ {college}")
            continue
        dyear = p.get("draft_year")
        preferred = int(dyear) - 1 if pd.notna(dyear) else None
        stats = parse_season_stats_from_bio(html, preferred_season=preferred)
        prod_keys = ("cfb_pass_yards", "cfb_rush_yards", "cfb_rec", "cfb_rec_yards")
        if all(_is_missing_stat(stats.get(c)) for c in prod_keys):
            stats = parse_season_stats_from_bio(html, preferred_season=None)
        if all(_is_missing_stat(stats.get(c)) for c in prod_keys):
            print(f"    parsed empty: {name} ({url})")
            continue
        print(
            f"    scraped {name}: rush={stats.get('cfb_rush_yards')} "
            f"rec={stats.get('cfb_rec')}/{stats.get('cfb_rec_yards')} ({url})"
        )
        rows.append(
            {
                "player_name_norm": normalize_name(p.get("player_name_norm") or name),
                "position": p.get("position"),
                "draft_year": p.get("draft_year"),
                "college": college,
                **stats,
            }
        )
    return pd.DataFrame(rows)
