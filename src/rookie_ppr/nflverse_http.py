from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from rookie_ppr.config import NFLVERSE_CACHE_DIR

BASE = "https://github.com/nflverse/nflverse-data/releases/download"


def _cache_file(name: str) -> Path:
    NFLVERSE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return NFLVERSE_CACHE_DIR / name


def read_release_csv(release: str, filename: str, force: bool = False) -> pd.DataFrame:
    """
    Download a nflverse-data release asset as CSV (or CSV.GZ) with local cache.
    Example: read_release_csv("draft_picks", "draft_picks.csv")
    """
    cache_name = f"{release}__{filename}".replace("/", "_")
    cache_path = _cache_file(cache_name)
    if cache_path.exists() and not force:
        return _read_table(cache_path)

    url = f"{BASE}/{release}/{filename}"
    resp = requests.get(url, timeout=120, allow_redirects=True)
    resp.raise_for_status()
    cache_path.write_bytes(resp.content)
    return _read_table(cache_path)


def _read_table(path: Path) -> pd.DataFrame:
    name = path.name.lower()
    if name.endswith(".gz"):
        return pd.read_csv(path, compression="gzip", low_memory=False)
    return pd.read_csv(path, low_memory=False)


def try_read_release_csv(candidates: list[tuple[str, str]]) -> pd.DataFrame:
    """Try multiple release/filename pairs; return first success or empty frame."""
    errors: list[str] = []
    for release, filename in candidates:
        try:
            df = read_release_csv(release, filename)
            if df is not None and not df.empty:
                return df
        except Exception as exc:  # noqa: BLE001 - best-effort multi-candidate load
            errors.append(f"{release}/{filename}: {exc}")
    return pd.DataFrame()
