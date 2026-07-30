from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from rookie_ppr.config import MANUAL_DIR, SKILL_POSITIONS
from rookie_ppr.utils import normalize_name, normalize_position, normalize_team_abbr

_FILE_RE = re.compile(r"FantasyPros_(\d{4})_Overall_ADP_Rankings\.csv$", re.I)
# "Christian McCaffrey   SF (9)" or "Aaron Rodgers   PIT (5)" or plain "Doug Martin"
_PLAYER_RE = re.compile(
    r"^(?P<name>.+?)(?:\s{2,}[A-Z]{2,3}\s*\(\d{1,2}\))?$"
)
_TEAM_BYE_RE = re.compile(r"\s+([A-Z]{2,3})\s*\(\d{1,2}\)\s*$")
_TEAM_ONLY_RE = re.compile(r"\s+([A-Z]{2,3})$")


def _extract_adp_team(raw: str) -> str | None:
    """Parse FantasyPros team abbrev from 'Name   TEAM (bye)' / 'Name TEAM'."""
    text = str(raw).strip()
    m = _TEAM_BYE_RE.search(text)
    if m:
        return normalize_team_abbr(m.group(1))
    m = _TEAM_ONLY_RE.search(text)
    if m:
        return normalize_team_abbr(m.group(1))
    return None


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
      season, overall_rank, player_name, player_name_norm, position, adp_avg, adp_team
    """
    empty_cols = [
        "season",
        "overall_rank",
        "player_name",
        "player_name_norm",
        "position",
        "adp_avg",
        "adp_team",
    ]
    root = manual_dir or MANUAL_DIR
    files = sorted(root.glob("FantasyPros_*_Overall_ADP_Rankings.csv"))
    if not files:
        return pd.DataFrame(columns=empty_cols)

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

        raw_player = df[player_col].astype(str)
        player_name = raw_player.map(_clean_player_name)
        adp_team = raw_player.map(_extract_adp_team)
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
                "adp_team": adp_team,
            }
        )
        out = out[out["position"].isin(SKILL_POSITIONS)].copy()
        frames.append(out)

    if not frames:
        return pd.DataFrame(columns=empty_cols)
    return pd.concat(frames, ignore_index=True)


def attach_rookie_adp(players: pd.DataFrame, adp: pd.DataFrame) -> pd.DataFrame:
    """
    Attach FantasyPros ADP only for a player's rookie fantasy season
    (ADP file year == draft_year).
    """
    base = players[["gsis_id", "player_name", "player_name_norm", "position", "draft_year"]].copy()
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


def attach_incumbent_adp(
    players: pd.DataFrame,
    adp: pd.DataFrame,
    *,
    rookie_adp: pd.DataFrame | None = None,
    incumbent: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Best same-team / same-position FantasyPros ADP among non-rookie names in year Y.

    Features:
      incumbent_ff_adp  — min ADP among other ranked players at draft_team+position
                          (fallback: ADP of returning workhorse matched by name)
      adp_vs_incumbent  — incumbent_ff_adp - ff_adp (positive => market prefers rookie)
    """
    key_cols = ["gsis_id", "player_name", "position", "draft_year"]
    out_cols = key_cols + ["incumbent_ff_adp", "adp_vs_incumbent"]
    base = players[[c for c in (*key_cols, "draft_team", "player_name_norm") if c in players.columns]].copy()
    base["incumbent_ff_adp"] = pd.NA
    base["adp_vs_incumbent"] = pd.NA
    if base.empty or adp is None or adp.empty:
        return base[[c for c in out_cols if c in base.columns]]

    name_lookup: dict[tuple, str] = {}
    if incumbent is not None and not incumbent.empty and "incumbent_top_name_norm" in incumbent.columns:
        inc = incumbent.copy()
        inc["_team"] = inc["draft_team"].map(normalize_team_abbr) if "draft_team" in inc.columns else pd.NA
        for _, r in inc.iterrows():
            y, team, pos = r.get("draft_year"), r.get("_team"), r.get("position")
            nm = r.get("incumbent_top_name_norm")
            if pd.notna(y) and team and pd.notna(pos) and pd.notna(nm) and str(nm).strip():
                name_lookup[(int(y), str(team), str(pos))] = str(nm)

    ff_lookup: dict[str, float] = {}
    if rookie_adp is not None and not rookie_adp.empty and "gsis_id" in rookie_adp.columns:
        for _, r in rookie_adp.iterrows():
            gid = r.get("gsis_id")
            val = pd.to_numeric(r.get("ff_adp"), errors="coerce")
            if pd.notna(gid) and pd.notna(val):
                ff_lookup[str(gid)] = float(val)

    def _adp_for_name(season: int, pos: str, name_norm: str) -> float | None:
        from rapidfuzz import fuzz, process

        pool = adp[(adp["season"] == season) & (adp["position"] == pos)]
        if pool.empty or not name_norm:
            return None
        choices = pool["player_name_norm"].dropna().tolist()
        if not choices:
            return None
        best = process.extractOne(name_norm, choices, scorer=fuzz.token_sort_ratio)
        if not best or best[1] < 90:
            return None
        hit = pool[pool["player_name_norm"] == best[0]].iloc[0]
        val = pd.to_numeric(hit.get("adp_avg"), errors="coerce")
        return float(val) if pd.notna(val) else None

    rows: list[dict] = []
    has_team = "adp_team" in adp.columns and adp["adp_team"].notna().any()
    for _, row in base.iterrows():
        d = {c: row.get(c) for c in key_cols if c in row.index}
        d["incumbent_ff_adp"] = pd.NA
        d["adp_vs_incumbent"] = pd.NA
        season = row.get("draft_year")
        team = normalize_team_abbr(row.get("draft_team"))
        pos = row.get("position")
        name_norm = row.get("player_name_norm") or ""
        if pd.isna(season) or not team or pd.isna(pos):
            rows.append(d)
            continue
        season_i = int(season)

        incumbent_adp: float | None = None
        if has_team:
            pool = adp[
                (adp["season"] == season_i)
                & (adp["adp_team"] == team)
                & (adp["position"] == pos)
            ].copy()
            if not pool.empty and name_norm:
                exact = pool["player_name_norm"] == name_norm
                if exact.any():
                    pool = pool.loc[~exact]
                else:
                    from rapidfuzz import fuzz, process

                    choices = pool["player_name_norm"].dropna().tolist()
                    if choices:
                        best = process.extractOne(name_norm, choices, scorer=fuzz.token_sort_ratio)
                        if best and best[1] >= 90:
                            pool = pool.loc[pool["player_name_norm"] != best[0]]
            adp_vals = pd.to_numeric(pool["adp_avg"], errors="coerce").dropna()
            if not adp_vals.empty:
                incumbent_adp = float(adp_vals.min())

        if incumbent_adp is None:
            top_name = name_lookup.get((season_i, str(team), str(pos)))
            if top_name:
                incumbent_adp = _adp_for_name(season_i, str(pos), top_name)

        if incumbent_adp is None:
            rows.append(d)
            continue

        d["incumbent_ff_adp"] = incumbent_adp
        gid = row.get("gsis_id")
        rookie_ff = ff_lookup.get(str(gid)) if pd.notna(gid) else None
        if rookie_ff is None:
            rookie_ff = pd.to_numeric(row.get("ff_adp"), errors="coerce")
            rookie_ff = float(rookie_ff) if pd.notna(rookie_ff) else None
        if rookie_ff is not None:
            d["adp_vs_incumbent"] = incumbent_adp - float(rookie_ff)
        rows.append(d)

    return pd.DataFrame(rows)
