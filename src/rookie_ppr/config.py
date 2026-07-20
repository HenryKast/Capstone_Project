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
MODELS_DIR = OUTPUT_DIR / "models"

SKILL_POSITIONS = ("QB", "RB", "WR", "TE")
HS_CLASS_MIN = 2010
# 2026 draft class spans HS ~2021–2023 (true juniors / early declarers)
HS_CLASS_MAX = 2023

# Draft / NFL seasons in the compile window
DRAFT_YEAR_MIN = 2013
DRAFT_YEAR_MAX = 2026

# Upcoming rookie class for pre-outcome prediction workbook tab
INCOMING_DRAFT_YEAR = 2026

# ML holdout: rookie NFL seasons by first_stat_season (fantasy season year)
# 2023 = 2023-24, 2024 = 2024-25, 2025 = 2025-26, 2026 = 2026-27
HOLDOUT_ROOKIE_SEASONS = (2023, 2024, 2025, 2026)
TUNING_ROOKIE_SEASON = 2022  # validation season for hyperparameter tuning (pre-holdout)

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
        MODELS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
