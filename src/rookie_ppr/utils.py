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


def compact_name_initials(name_norm: str) -> str:
    """'c j stroud' / 'cj stroud' → 'cj stroud' for matching."""
    parts = [p for p in str(name_norm or "").split() if p]
    if len(parts) < 2:
        return str(name_norm or "")
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


def normalize_team_abbr(team: str | None) -> str | None:
    """
    Map draft / schedule / stats team codes onto a common abbreviation set.

    nflverse player stats often use LV/LA/GB/KC/NE/NO/SF/TB while draft picks
    may use LVR/LAR/GNB/KAN/NWE/NOR/SFO/TAM.
    """
    if team is None or (isinstance(team, float) and pd.isna(team)):
        return None
    t = str(team).upper().strip()
    if not t or t in {"NAN", "NONE", "NAT"}:
        return None
    aliases = {
        "LVR": "LV",
        "RAI": "LV",
        "OAK": "LV",
        "LAR": "LA",
        "RAM": "LA",
        "GNB": "GB",
        "KAN": "KC",
        "NWE": "NE",
        "NOR": "NO",
        "SFO": "SF",
        "TAM": "TB",
        "WSH": "WAS",
        "JAC": "JAX",
        "STL": "LA",
        "SD": "LAC",
        "SDG": "LAC",
    }
    return aliases.get(t, t)


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
