from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")

DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RECRUITING_DIR = RAW_DIR / "recruiting"
NFLVERSE_CACHE_DIR = RAW_DIR / "nflverse_cache"
CFBD_CACHE_DIR = RAW_DIR / "cfbd_cache"
MANUAL_DIR = DATA_DIR / "manual"
OUTPUT_DIR = DATA_DIR / "output"
CSV_OUTPUT_DIR = OUTPUT_DIR / "csv"

SKILL_POSITIONS = ("QB", "RB", "WR", "TE")
HS_CLASS_MIN = 2010
HS_CLASS_MAX = 2020

# Draft / NFL seasons that can contain HS classes 2010-2020
DRAFT_YEAR_MIN = 2013
DRAFT_YEAR_MAX = 2026

CFBD_API_KEY = os.getenv("CFBD_API_KEY", "").strip()
CFBD_BASE_URL = "https://api.collegefootballdata.com"

# Template columns for On3 / Rivals recruiting exports
RECRUITING_REQUIRED_COLUMNS = (
    "hs_class",
    "player_name",
    "position",
    "industry_composite_rank",
    "industry_composite_stars",
    "rivals_rank",
    "rivals_stars",
    "hometown",
    "high_school",
    "state",
)


def ensure_directories() -> None:
    for path in (
        RECRUITING_DIR,
        NFLVERSE_CACHE_DIR,
        CFBD_CACHE_DIR,
        MANUAL_DIR,
        OUTPUT_DIR,
        CSV_OUTPUT_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
