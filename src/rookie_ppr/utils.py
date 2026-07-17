from __future__ import annotations

import re
from typing import Any

import pandas as pd


def to_pandas(frame: Any) -> pd.DataFrame:
    """Convert nflreadpy/polars frames to pandas."""
    if frame is None:
        return pd.DataFrame()
    if isinstance(frame, pd.DataFrame):
        return frame.copy()
    if hasattr(frame, "to_pandas"):
        return frame.to_pandas()
    return pd.DataFrame(frame)


def normalize_name(name: str | None) -> str:
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return ""
    text = str(name).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Drop common suffixes for matching
    parts = [p for p in text.split(" ") if p not in {"jr", "sr", "ii", "iii", "iv", "v"}]
    return " ".join(parts)


def normalize_position(pos: str | None) -> str | None:
    if pos is None or (isinstance(pos, float) and pd.isna(pos)):
        return None
    p = str(pos).upper().strip()
    aliases = {
        "HB": "RB",
        "FB": "RB",
        "TB": "RB",
        "ATH": "ATH",
        "QB": "QB",
        "RB": "RB",
        "WR": "WR",
        "TE": "TE",
    }
    if p in aliases:
        return aliases[p]
    if p in {"QB", "RB", "WR", "TE", "ATH"}:
        return p
    return None


def estimate_hs_class_from_birth(birth_date: Any, draft_year: Any = None) -> float:
    """Rough HS graduation year estimate from birth date (birth year + 18)."""
    if birth_date is None or (isinstance(birth_date, float) and pd.isna(birth_date)):
        return pd.NA
    try:
        dt = pd.to_datetime(birth_date, errors="coerce")
    except Exception:
        return pd.NA
    if pd.isna(dt):
        return pd.NA
    return int(dt.year + 18)
