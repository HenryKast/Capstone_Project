"""Cached CSV readers for the ForeKast API."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import math

import pandas as pd

from rookie_ppr.config import CSV_OUTPUT_DIR


def _ascii(name: object) -> str:
    return str(name).encode("ascii", "ignore").decode("ascii").strip()


def json_safe(value: Any) -> Any:
    """Convert pandas/numpy NaN/NA to None for JSON."""
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            pass
    return value


def records_safe(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        rows.append({k: json_safe(v) for k, v in row.items()})
    return rows


@lru_cache(maxsize=32)
def load_csv(name: str) -> pd.DataFrame:
    path = CSV_OUTPUT_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    return pd.read_csv(path)


def clear_cache() -> None:
    load_csv.cache_clear()


def csv_path(name: str) -> Path:
    return CSV_OUTPUT_DIR / name


def ascii_series(series: pd.Series) -> pd.Series:
    return series.map(lambda v: _ascii(v) if pd.notna(v) else "")
