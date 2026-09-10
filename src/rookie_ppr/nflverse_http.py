from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from rookie_ppr.config import NFLVERSE_CACHE_DIR

BASE = "https://github.com/nflverse/nflverse-data/releases/download"


def _cache_file(name: str) -> Path:
    NFLVERSE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return NFLVERSE_CACHE_DIR / name


def read_release_csv(
    release: str,
    filename: str,
    force: bool = False,
    *,
    usecols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Download a nflverse-data release asset as CSV (or CSV.GZ) with local cache.
    Example: read_release_csv("draft_picks", "draft_picks.csv")
    """
    cache_name = f"{release}__{filename}".replace("/", "_")
    cache_path = _cache_file(cache_name)
    if cache_path.exists() and not force:
        return _read_table(cache_path, usecols=usecols)

    url = f"{BASE}/{release}/{filename}"
    resp = requests.get(url, timeout=180, allow_redirects=True)
    resp.raise_for_status()
    cache_path.write_bytes(resp.content)
    return _read_table(cache_path, usecols=usecols)


def _read_table(path: Path, usecols: list[str] | None = None) -> pd.DataFrame:
    name = path.name.lower()
    kwargs: dict = {"low_memory": False}
    if usecols is not None:
        # Header-only pass to intersect available columns
        peek = pd.read_csv(
            path,
            compression="gzip" if name.endswith(".gz") else None,
            nrows=0,
        )
        cols = [c for c in usecols if c in peek.columns]
        if cols:
            kwargs["usecols"] = cols
    if name.endswith(".gz"):
        return pd.read_csv(path, compression="gzip", **kwargs)
    return pd.read_csv(path, **kwargs)


def try_read_release_csv(
    candidates: list[tuple[str, str]],
    *,
    usecols: list[str] | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Try multiple release/filename pairs; return first success or empty frame."""
    errors: list[str] = []
    for release, filename in candidates:
        try:
            df = read_release_csv(release, filename, force=force, usecols=usecols)
            if df is not None and not df.empty:
                return df
        except Exception as exc:  # noqa: BLE001 - best-effort multi-candidate load
            errors.append(f"{release}/{filename}: {exc}")
    return pd.DataFrame()