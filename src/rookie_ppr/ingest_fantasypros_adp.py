from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from rookie_ppr.config import MANUAL_DIR, SKILL_POSITIONS
from rookie_ppr.utils import normalize_name, normalize_position

_FILE_RE = re.compile(r"FantasyPros_(\d{4})_Overall_ADP_Rankings\.csv$", re.I)
# "Christian McCaffrey   SF (9)" or "Aaron Rodgers   PIT (5)" or plain "Doug Martin"
_PLAYER_RE = re.compile(
    r"^(?P<name>.+?)(?:\s{2,}[A-Z]{2,3}\s*\(\d{1,2}\))?$"
)


def _clean_player_name(raw: str) -> str:
    text = str(raw).strip()
    # Drop trailing "TEAM (bye)" blocks, e.g. "Christian McCaffrey   SF (9)"
    text = re.sub(r"\s+[A-Z]{2,3}\s*\(\d{1,2}\)\s*$", "", text)
    # Drop trailing team abbrev without bye, e.g. "Keenan Allen LAC"
    text = re.sub(r"\s+[A-Z]{2,3}$", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _pos_from_fp(pos: str) -> str | None:
    if pos is None or (isinstance(pos, float) and pd.isna(pos)):
        return None
    m = re.match(r"^([A-Za-z]+)", str(pos).strip())
    if not m:
        return None
    return normalize_position(m.group(1))


def load_fantasypros_adp(manual_dir: Path | None = None) -> pd.DataFrame:
    """
    Load FantasyPros Overall ADP CSVs from data/manual/.

    Returns one row per ranked player-year with:
      season, overall_rank, player_name, player_name_norm, position, adp_avg
    """
    root = manual_dir or MANUAL_DIR
    files = sorted(root.glob("FantasyPros_*_Overall_ADP_Rankings.csv"))
    if not files:
        return pd.DataFrame(
            columns=[
                "season",
                "overall_rank",
                "player_name",
                "player_name_norm",
                "position",
                "adp_avg",
            ]
        )

    frames: list[pd.DataFrame] = []
    for path in files:
        m = _FILE_RE.search(path.name)
        if not m:
            continue
        season = int(m.group(1))
        df = pd.read_csv(path)
        # Flexible headers
        rename = {c: c.strip() for c in df.columns}
        df = df.rename(columns=rename)
        player_col = next((c for c in df.columns if c.lower().startswith("player")), None)
        pos_col = next((c for c in df.columns if c.upper() == "POS" or c.lower() == "position"), None)
        rank_col = next((c for c in df.columns if c.lower() == "rank"), None)
        avg_col = next((c for c in df.columns if c.upper() == "AVG" or c.lower() in {"adp", "avg"}), None)
        if player_col is None:
            continue

        player_name = df[player_col].map(_clean_player_name)
        position = df[pos_col].map(_pos_from_fp) if pos_col else pd.Series([pd.NA] * len(df))
        overall_rank = pd.to_numeric(df[rank_col], errors="coerce") if rank_col else pd.Series([pd.NA] * len(df))
        if avg_col:
            adp_avg = (
                df[avg_col]
                .astype(str)
                .str.replace("—", "", regex=False)
                .str.replace("–", "", regex=False)
                .str.strip()
            )
            adp_avg = pd.to_numeric(adp_avg, errors="coerce")
        else:
            adp_avg = overall_rank

        out = pd.DataFrame(
            {
                "season": season,
                "player_name": player_name,
                "player_name_norm": player_name.map(normalize_name),
                "position": position,
                "overall_rank": overall_rank,
                "adp_avg": adp_avg,
            }
        )
        out = out[out["position"].isin(SKILL_POSITIONS)].copy()
        frames.append(out)

    if not frames:
        return pd.DataFrame(
            columns=[
                "season",
                "overall_rank",
                "player_name",
                "player_name_norm",
                "position",
                "adp_avg",
            ]
        )
    return pd.concat(frames, ignore_index=True)


def attach_rookie_adp(players: pd.DataFrame, adp: pd.DataFrame) -> pd.DataFrame:
    """
    Attach FantasyPros ADP only for a player's rookie fantasy season
    (ADP file year == draft_year).
    """
    base = players[["gsis_id", "player_name", "player_name_norm", "position", "draft_year"]].copy()
    base["ff_ecr"] = pd.NA
    base["ff_adp"] = pd.NA
    base["ff_adp_rank"] = pd.NA
    base["ff_rankings_note"] = "No FantasyPros ADP matched for rookie season."

    if players.empty or adp.empty:
        return base

    from rapidfuzz import fuzz, process

    matched_rows = []
    for _, row in base.iterrows():
        season = row.get("draft_year")
        name = row.get("player_name_norm") or ""
        pos = row.get("position")
        if pd.isna(season) or not name:
            matched_rows.append(row.to_dict())
            continue

        pool = adp[adp["season"] == int(season)]
        if pool.empty:
            matched_rows.append(row.to_dict())
            continue

        same = pool[pool["position"] == pos]
        search_pool = same if not same.empty else pool
        choices = search_pool["player_name_norm"].tolist()
        best = process.extractOne(name, choices, scorer=fuzz.token_sort_ratio)
        if best and best[1] >= 90:
            hit = search_pool[search_pool["player_name_norm"] == best[0]].iloc[0]
            d = row.to_dict()
            d["ff_ecr"] = hit.get("overall_rank")
            d["ff_adp"] = hit.get("adp_avg")
            d["ff_adp_rank"] = hit.get("overall_rank")
            d["ff_rankings_note"] = (
                f"FantasyPros Overall ADP for rookie season {int(season)} "
                f"(match_score={best[1]})"
            )
            matched_rows.append(d)
        else:
            d = row.to_dict()
            d["ff_rankings_note"] = (
                f"No FantasyPros ADP match in {int(season)} overall rankings "
                f"(rookies only; best_score={best[1] if best else 'n/a'})."
            )
            matched_rows.append(d)

    return pd.DataFrame(matched_rows)
