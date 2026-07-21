from __future__ import annotations

import pandas as pd
from rapidfuzz import fuzz, process

from rookie_ppr.config import HS_CLASS_MAX, HS_CLASS_MIN, INCOMING_DRAFT_YEAR, MANUAL_DIR


def load_overrides() -> pd.DataFrame:
    path = MANUAL_DIR / "player_id_overrides.csv"
    MANUAL_DIR.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        pd.DataFrame(
            columns=[
                "player_name_norm",
                "position",
                "draft_year",
                "recruiting_player_name_norm",
                "hs_class",
                "notes",
            ]
        ).to_csv(path, index=False)
        return pd.DataFrame()
    return pd.read_csv(path)


def _fuzzy_match_recruiting(players: pd.DataFrame, recruiting: pd.DataFrame) -> pd.DataFrame:
    if players.empty or recruiting.empty:
        players = players.copy()
        players["hs_class"] = players.get("hs_class_estimated", pd.NA)
        players["recruiting_rank"] = pd.NA
        players["recruiting_stars"] = pd.NA
        players["recruiting_source"] = pd.NA
        players["recruiting_match_score"] = pd.NA
        return players

    rec = recruiting.copy()
    overrides = load_overrides()

    matched_rows = []
    for _, row in players.iterrows():
        name = row.get("player_name_norm") or ""
        pos = row.get("position")
        draft_year = row.get("draft_year")

        # Manual override first
        if not overrides.empty:
            ov = overrides[
                (overrides.get("player_name_norm") == name)
                & (overrides.get("position") == pos)
            ]
            if "draft_year" in overrides.columns and pd.notna(draft_year):
                ov = ov[ov["draft_year"].isna() | (ov["draft_year"] == draft_year)]
            if not ov.empty:
                o = ov.iloc[0]
                rname = o.get("recruiting_player_name_norm")
                hit = rec[rec["player_name_norm"] == rname]
                if not hit.empty:
                    h = hit.iloc[0]
                    matched_rows.append(
                        {
                            **row.to_dict(),
                            "hs_class": h.get("hs_class", o.get("hs_class")),
                            "recruiting_rank": h.get("recruiting_rank"),
                            "recruiting_stars": h.get("recruiting_stars"),
                            "recruiting_source": h.get("recruiting_source"),
                            "recruiting_match_score": 100,
                        }
                    )
                    continue

        # Candidate pool: HS class near draft (draft_year - 6 .. draft_year - 2)
        cand = rec.copy()
        if pd.notna(draft_year):
            lo = int(draft_year) - 6
            hi = int(draft_year) - 2
            cand = cand[cand["hs_class"].between(lo, hi)]

        # Prefer same position, then ATH, then any skill position
        same = cand[cand["position"] == pos]
        ath = cand[cand["position"] == "ATH"]
        pools = [(same, 90), (ath, 92), (cand, 95)]

        best_hit = None
        best_score = None
        for pool, threshold in pools:
            if pool.empty:
                continue
            choices = pool["player_name_norm"].tolist()
            best = process.extractOne(name, choices, scorer=fuzz.token_sort_ratio)
            if best and best[1] >= threshold:
                best_hit = pool[pool["player_name_norm"] == best[0]].iloc[0]
                best_score = best[1]
                break

        if best_hit is not None:
            matched_rows.append(
                {
                    **row.to_dict(),
                    "hs_class": best_hit.get("hs_class"),
                    "recruiting_rank": best_hit.get("recruiting_rank"),
                    "recruiting_stars": best_hit.get("recruiting_stars"),
                    "recruiting_source": best_hit.get("recruiting_source"),
                    "recruiting_match_score": best_score,
                }
            )
        else:
            matched_rows.append(
                {
                    **row.to_dict(),
                    "hs_class": row.get("hs_class_estimated"),
                    "recruiting_rank": pd.NA,
                    "recruiting_stars": pd.NA,
                    "recruiting_source": pd.NA,
                    "recruiting_match_score": pd.NA,
                }
            )

    return pd.DataFrame(matched_rows)


def _on_season_adp(players: pd.DataFrame, ff_rankings: pd.DataFrame, season: int) -> pd.Series:
    """True when a player fuzzy-matches FantasyPros Overall ADP for the given season."""
    if players.empty or ff_rankings is None or ff_rankings.empty or "season" not in ff_rankings.columns:
        return pd.Series(False, index=players.index)

    pool = ff_rankings[ff_rankings["season"] == season].copy()
    if pool.empty or "player_name_norm" not in pool.columns:
        return pd.Series(False, index=players.index)

    hits: list[bool] = []
    for _, row in players.iterrows():
        name = row.get("player_name_norm") or ""
        pos = row.get("position")
        if not name:
            hits.append(False)
            continue
        same = pool[pool["position"] == pos] if "position" in pool.columns else pool.iloc[0:0]
        search = same if not same.empty else pool
        choices = search["player_name_norm"].dropna().astype(str).tolist()
        if not choices:
            hits.append(False)
            continue
        best = process.extractOne(name, choices, scorer=fuzz.token_sort_ratio)
        hits.append(bool(best and best[1] >= 90))
    return pd.Series(hits, index=players.index)


def build_tables(
    fantasy: pd.DataFrame,
    recruiting: pd.DataFrame,
    combine: pd.DataFrame,
    sos: pd.DataFrame,
    offense_env: pd.DataFrame,
    opportunity: pd.DataFrame,
    college: pd.DataFrame,
    ff_rankings: pd.DataFrame,
    incumbent: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    players = _fuzzy_match_recruiting(fantasy, recruiting)

    # Cohort filter: HS classes in configured window.
    # Also keep incoming draft-year players who appear on that season's FantasyPros ADP
    # (covers early declarers / missing birthdates that would otherwise drop).
    if recruiting.empty:
        est = pd.to_numeric(players.get("hs_class_estimated"), errors="coerce")
        players["hs_class"] = est
        in_cohort = est.isna() | est.between(HS_CLASS_MIN, HS_CLASS_MAX)
    else:
        hs = pd.to_numeric(players["hs_class"], errors="coerce")
        in_cohort = hs.between(HS_CLASS_MIN, HS_CLASS_MAX)

    draft_year_num = pd.to_numeric(players.get("draft_year"), errors="coerce")
    incoming = draft_year_num == INCOMING_DRAFT_YEAR
    on_adp = _on_season_adp(players, ff_rankings, INCOMING_DRAFT_YEAR)
    players = players[in_cohort | (incoming & on_adp)].copy()

    players_sheet = players[
        [
            c
            for c in [
                "gsis_id",
                "pfr_player_id",
                "player_name",
                "player_name_norm",
                "position",
                "hs_class",
                "birth_date",
                "draft_year",
                "college",
                "draft_team",
            ]
            if c in players.columns
        ]
    ].drop_duplicates()

    recruiting_sheet = players[
        [
            c
            for c in [
                "gsis_id",
                "player_name",
                "position",
                "hs_class",
                "recruiting_rank",
                "recruiting_stars",
                "recruiting_source",
                "recruiting_match_score",
            ]
            if c in players.columns
        ]
    ].copy()

    draft_sheet = players[
        [
            c
            for c in [
                "gsis_id",
                "player_name",
                "position",
                "college",
                "draft_year",
                "draft_round",
                "draft_overall",
                "draft_team",
            ]
            if c in players.columns
        ]
    ].copy()

    # Combine join on name/position/season~draft year
    combine_sheet = pd.DataFrame()
    if not combine.empty:
        c = combine.copy()
        year_col = next((x for x in ("season", "draft_year", "year") if x in c.columns), None)
        merge_cols = ["player_name_norm", "position"]
        left = players.copy()
        if year_col:
            c = c.rename(columns={year_col: "combine_year"})
            # nearest: combine year == draft year
            combine_sheet = left.merge(
                c,
                how="left",
                left_on=["player_name_norm", "position", "draft_year"],
                right_on=["player_name_norm", "position", "combine_year"],
            )
        else:
            combine_sheet = left.merge(c, how="left", on=[col for col in merge_cols if col in c.columns])
        keep = [
            "gsis_id",
            "player_name",
            "position",
            "draft_year",
            "ht",
            "wt",
            "forty",
            "bench",
            "vertical",
            "broad_jump",
            "cone",
            "shuttle",
        ]
        # nflverse may use different names
        rename_map = {
            "height": "ht",
            "weight": "wt",
            "forty_yd": "forty",
            "vertical_leap": "vertical",
            "broad": "broad_jump",
            "cone_drill": "cone",
            "twenty_yard_shuttle": "shuttle",
        }
        for old, new in rename_map.items():
            if old in combine_sheet.columns and new not in combine_sheet.columns:
                combine_sheet[new] = combine_sheet[old]
        combine_sheet = combine_sheet[[col for col in keep if col in combine_sheet.columns]].drop_duplicates()

    # Team context: SOS + offense + opportunity + incumbent (with prior-season fallback)
    from rookie_ppr.ingest_opportunity import attach_team_season_with_fallback
    from rookie_ppr.utils import normalize_team_abbr

    def _merge_team_pos_features(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
        if right is None or right.empty:
            return left
        l = left.copy()
        r = right.copy()
        l["_team_key"] = l["draft_team"].map(normalize_team_abbr)
        r["_team_key"] = r["draft_team"].map(normalize_team_abbr)
        feature_cols = [c for c in r.columns if c not in {"draft_year", "draft_team", "position", "_team_key"}]
        r = (
            r[["draft_year", "_team_key", "position"] + feature_cols]
            .drop_duplicates(subset=["draft_year", "_team_key", "position"])
        )
        out = l.merge(r, how="left", on=["draft_year", "_team_key", "position"])
        return out.drop(columns=["_team_key"])

    team_context = players[["gsis_id", "player_name", "position", "draft_year", "draft_team"]].copy()
    if not sos.empty:
        team_context = attach_team_season_with_fallback(
            team_context,
            sos,
            ["sos_opp_win_pct", "games_scheduled"],
        )
    if not offense_env.empty:
        team_context = attach_team_season_with_fallback(
            team_context,
            offense_env,
            ["off_pass_yards", "off_rush_yards", "off_pass_rate_proxy"],
        )
    team_context = _merge_team_pos_features(team_context, opportunity)
    team_context = _merge_team_pos_features(team_context, incumbent)

    college_sheet = pd.DataFrame()
    if not college.empty:
        college_sheet = players.merge(
            college,
            how="left",
            on=[c for c in ("player_name_norm", "position", "draft_year") if c in college.columns and c in players.columns],
        )
        cols = [
            "gsis_id",
            "player_name",
            "position",
            "college",
            "cfb_final_season",
            "cfb_pass_yards",
            "cfb_pass_td",
            "cfb_rush_yards",
            "cfb_rush_td",
            "cfb_rec",
            "cfb_rec_yards",
            "cfb_rec_td",
        ]
        college_sheet = college_sheet[[c for c in cols if c in college_sheet.columns]].drop_duplicates()

    # Pre-draft fantasy ADP: FantasyPros overall ADP for the player's rookie season only
    from rookie_ppr.ingest_fantasypros_adp import attach_rookie_adp

    if ff_rankings is not None and not ff_rankings.empty and "season" in ff_rankings.columns:
        pre_draft = attach_rookie_adp(players, ff_rankings)
    else:
        pre_draft = players[["gsis_id", "player_name", "player_name_norm", "position", "draft_year"]].copy()
        pre_draft["ff_adp"] = pd.NA
        pre_draft["ff_adp_rank"] = pd.NA
        pre_draft["ff_rankings_note"] = "No FantasyPros ADP files found in data/manual/."

    fantasy_sheet = players[
        [
            c
            for c in [
                "gsis_id",
                "player_name",
                "position",
                "draft_year",
                "first_stat_season",
                "rookie_ppr",
                "rookie_games",
                "rookie_ppr_per_game",
                "rookie_ppr_rank_pos",
            ]
            if c in players.columns
        ]
    ].copy()

    # Wide master
    master = players.copy()
    for frame, prefix_note in (
        (combine_sheet, "combine"),
        (team_context, "team"),
        (college_sheet, "college"),
        (pre_draft, "ff"),
    ):
        if frame is None or frame.empty:
            continue
        key = [c for c in ("gsis_id", "player_name", "position", "draft_year") if c in frame.columns and c in master.columns]
        if not key:
            continue
        extra = [c for c in frame.columns if c not in master.columns or c in key]
        # Avoid duplicate key-only merges
        add_cols = [c for c in frame.columns if c not in master.columns]
        if not add_cols:
            continue
        master = master.merge(frame[key + add_cols].drop_duplicates(key), how="left", on=key)

    drop_master_cols = [
        "draft_pick",
        "draft_category",
        "rookie_season",
        "hs_class_estimated",
        "opportunity_proxy",
        "prior_team_pos_ppr",
    ]
    master = master.drop(columns=[c for c in drop_master_cols if c in master.columns])

    data_dictionary = pd.DataFrame(
        [
            {"sheet": "players", "column": "hs_class", "description": "HS class year (recruiting match or estimate)", "source": "On3 ingest / birthdate estimate"},
            {"sheet": "recruiting", "column": "recruiting_rank", "description": "Industry Composite rank, else Rivals-only", "source": "On3/Rivals"},
            {"sheet": "recruiting", "column": "recruiting_source", "description": "industry_composite | rivals | unranked", "source": "derived"},
            {"sheet": "draft", "column": "college", "description": "College at time of NFL draft", "source": "nflverse draft picks"},
            {"sheet": "draft", "column": "draft_overall", "description": "Overall NFL draft pick", "source": "nflverse"},
            {"sheet": "draft", "column": "draft_team", "description": "Team that drafted the player", "source": "nflverse"},
            {"sheet": "team_context", "column": "sos_opp_win_pct", "description": "Rookie-season schedule SOS (avg opponent win%); prior season if unavailable", "source": "nflverse schedules"},
            {"sheet": "team_context", "column": "team_opportunity_ppr", "description": "Prior available season team positional fantasy points (vacated usage proxy)", "source": "nflverse / NFL.com player stats"},
            {"sheet": "team_context", "column": "team_pos_touches", "description": "Prior season team position touches (carries + receptions)", "source": "nflverse / NFL.com"},
            {"sheet": "team_context", "column": "team_pos_carries", "description": "Prior season team position carries", "source": "nflverse / NFL.com"},
            {"sheet": "team_context", "column": "team_pos_targets", "description": "Prior season team position targets", "source": "nflverse / NFL.com"},
            {"sheet": "team_context", "column": "incumbent_pos_ppr", "description": "Max prior-season fantasy points among returning same-team/position players", "source": "nflverse stats + rosters"},
            {"sheet": "team_context", "column": "incumbent_pos_ppr_sum", "description": "Sum of prior-season fantasy points among returning same-team/position players", "source": "nflverse stats + rosters"},
            {"sheet": "team_context", "column": "incumbent_pos_carries", "description": "Max prior-season carries among returning same-team/position players", "source": "nflverse stats + rosters"},
            {"sheet": "team_context", "column": "incumbent_pos_touches", "description": "Max prior-season touches among returning same-team/position players", "source": "nflverse stats + rosters"},
            {"sheet": "team_context", "column": "off_pass_rate_proxy", "description": "Team pass-rate proxy (latest available season before draft)", "source": "nflverse team stats"},
            {"sheet": "combine", "column": "forty", "description": "40-yard dash", "source": "nflverse combine"},
            {"sheet": "college_production", "column": "cfb_*", "description": "Final CFB season production (requires CFBD_API_KEY)", "source": "CollegeFootballData"},
            {"sheet": "fantasy_rookie", "column": "rookie_ppr", "description": "PPR fantasy points in first NFL season", "source": "nflverse player stats"},
            {"sheet": "pre_draft_fantasy", "column": "ff_adp", "description": "FantasyPros overall ADP (AVG) in the player's rookie season only", "source": "data/manual/FantasyPros_*_Overall_ADP_Rankings.csv"},
            {"sheet": "pre_draft_fantasy", "column": "ff_adp_rank", "description": "FantasyPros overall rank in that same rookie-season ADP file", "source": "FantasyPros"},
            {"sheet": "players_master", "column": "*", "description": "Wide joined modeling table", "source": "compiled"},
        ]
    )

    return {
        "players": players_sheet,
        "recruiting": recruiting_sheet,
        "draft": draft_sheet,
        "combine": combine_sheet if not combine_sheet.empty else pd.DataFrame({"note": ["No combine rows matched"]}),
        "college_production": college_sheet if not college_sheet.empty else pd.DataFrame({"note": ["College production empty — set CFBD_API_KEY to enable"]}),
        "team_context": team_context,
        "pre_draft_fantasy": pre_draft,
        "fantasy_rookie": fantasy_sheet,
        "players_master": master,
        "data_dictionary": data_dictionary,
    }
