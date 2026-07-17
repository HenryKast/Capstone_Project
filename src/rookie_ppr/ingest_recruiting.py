from __future__ import annotations

from pathlib import Path

import pandas as pd

from rookie_ppr.config import (
    HS_CLASS_MAX,
    HS_CLASS_MIN,
    RECRUITING_DIR,
    RECRUITING_REQUIRED_COLUMNS,
)
from rookie_ppr.utils import normalize_name, normalize_position


def write_recruiting_template() -> Path:
    RECRUITING_DIR.mkdir(parents=True, exist_ok=True)
    path = RECRUITING_DIR / "hs_YYYY_template.csv"
    pd.DataFrame(columns=list(RECRUITING_REQUIRED_COLUMNS)).to_csv(path, index=False)
    readme = RECRUITING_DIR / "README.md"
    if not readme.exists():
        readme.write_text(
            """# Recruiting raw data

Place one CSV per high-school class year, named `hs_2010.csv` … `hs_2020.csv`.

## Required columns

- `hs_class` — high school class year (2010–2020)
- `player_name` — recruit name
- `position` — QB, RB, WR, or TE (ATH may be left blank / excluded)
- `industry_composite_rank` — On3 Rivals Industry Composite national rank (blank if unranked)
- `industry_composite_stars` — Industry Composite stars (blank if unranked)
- `rivals_rank` — Rivals-only national rank (used when Industry Composite is missing)
- `rivals_stars` — Rivals-only stars
- `hometown`, `high_school`, `state` — optional helpers for matching

## Ranking rule used by the compiler

1. If `industry_composite_rank` is present → use Industry Composite (source = `industry_composite`)
2. Else if `rivals_rank` is present → use Rivals-only (source = `rivals`)
3. Else → unranked (source = `unranked`)

Export from On3 Industry Comparison pages for football by class year, then map columns into this schema.
""",
            encoding="utf-8",
        )
    return path


def load_recruiting() -> pd.DataFrame:
    write_recruiting_template()
    files = sorted(RECRUITING_DIR.glob("hs_*.csv"))
    files = [f for f in files if "template" not in f.name.lower()]
    if not files:
        return pd.DataFrame(columns=list(RECRUITING_REQUIRED_COLUMNS) + ["recruiting_source", "recruiting_rank", "recruiting_stars", "player_name_norm"])

    frames = []
    for path in files:
        df = pd.read_csv(path)
        # Allow flexible headers
        rename = {
            "name": "player_name",
            "athlete": "player_name",
            "class": "hs_class",
            "year": "hs_class",
            "pos": "position",
            "industry_rank": "industry_composite_rank",
            "composite_rank": "industry_composite_rank",
            "industry_stars": "industry_composite_stars",
            "composite_stars": "industry_composite_stars",
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
        frames.append(df)
    rec = pd.concat(frames, ignore_index=True)

    for col in RECRUITING_REQUIRED_COLUMNS:
        if col not in rec.columns:
            rec[col] = pd.NA

    # Keep ATH for matching skill draftees who were high-school athletes
    raw_pos = rec["position"].astype(str).str.upper().str.strip()
    rec["position"] = raw_pos.map(lambda p: normalize_position(p) or (p if p == "ATH" else None))
    rec = rec[rec["position"].isin(["QB", "RB", "WR", "TE", "ATH"])].copy()
    rec["hs_class"] = pd.to_numeric(rec["hs_class"], errors="coerce")
    rec = rec[rec["hs_class"].between(HS_CLASS_MIN, HS_CLASS_MAX)]
    rec["player_name_norm"] = rec["player_name"].map(normalize_name)

    ind_rank = pd.to_numeric(rec["industry_composite_rank"], errors="coerce")
    riv_rank = pd.to_numeric(rec["rivals_rank"], errors="coerce")
    ind_stars = pd.to_numeric(rec["industry_composite_stars"], errors="coerce")
    riv_stars = pd.to_numeric(rec["rivals_stars"], errors="coerce")

    # Prefer Industry Composite; if unranked there, use Rivals-only
    use_industry = ind_rank.notna()
    use_rivals = (~use_industry) & riv_rank.notna()
    rec["recruiting_source"] = "unranked"
    rec.loc[use_industry, "recruiting_source"] = "industry_composite"
    rec.loc[use_rivals, "recruiting_source"] = "rivals"
    rec["recruiting_rank"] = ind_rank.where(use_industry, riv_rank.where(use_rivals, pd.NA))
    rec["recruiting_stars"] = ind_stars.where(use_industry, riv_stars.where(use_rivals, pd.NA))
    return rec.reset_index(drop=True)
