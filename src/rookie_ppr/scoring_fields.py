from __future__ import annotations

from dataclasses import dataclass

# Factors with impact >= 25 in strongest_factors (full cohort correlation)
IMPORTANT_COLUMNS = frozenset(
    {
        "draft_overall",
        "draft_round",
        "ff_adp_rank",
        "ff_adp",
        "age_at_draft",
        "cfb_pass_td",
        "cfb_pass_yards",
        "recruiting_stars",
        "cone",
        "forty",
        "shuttle",
        "cfb_rush_td",
        "cfb_rec_yards",
        "cfb_rec",
        "cfb_rec_td",
        "incumbent_pos_ppr",
        "team_pos_touches",
        "incumbent_pos_carries",
    }
)


@dataclass(frozen=True)
class FieldSpec:
    column: str
    label: str
    hint: str
    group: str
    widget: str = "entry"  # entry | dropdown
    options: tuple[str, ...] = ()
    important: bool = False


FIELD_SPECS: list[FieldSpec] = [
    FieldSpec("position", "Position", "(QB, RB, WR, TE)", "Core", "dropdown", ("QB", "RB", "WR", "TE"), True),
    FieldSpec("draft_year", "Draft year", "(2013–2026)", "Core"),
    FieldSpec("draft_round", "Draft round", "(1–7)", "Core", important=True),
    FieldSpec("draft_overall", "Draft overall pick", "(1–262)", "Core", important=True),
    FieldSpec("draft_team", "NFL team", "(32 teams)", "Core", "dropdown"),
    FieldSpec("college", "College", "(school at draft)", "Core", "dropdown"),
    FieldSpec("recruiting_rank", "Recruiting rank", "(1–500+; lower is better)", "Recruiting"),
    FieldSpec("hs_class", "HS class", "(2010–2023)", "Recruiting", "dropdown", tuple(str(y) for y in range(2010, 2024))),
    FieldSpec(
        "recruiting_stars",
        "Recruiting stars",
        "(2–5)",
        "Recruiting",
        "dropdown",
        ("2", "3", "4", "5"),
        True,
    ),
    FieldSpec("age_at_draft", "Age at draft", "(20–27)", "Recruiting", "dropdown", tuple(str(a) for a in range(20, 28)), True),
    FieldSpec("ff_adp", "FantasyPros ADP (avg)", "(1–400 pick slot)", "Pre-draft fantasy", important=True),
    FieldSpec("ff_adp_rank", "FantasyPros ADP rank", "(1–400; lower is better)", "Pre-draft fantasy", important=True),
    FieldSpec("ht", "Height", "(e.g. 6-1)", "Combine"),
    FieldSpec("wt", "Weight (lbs)", "(150–350)", "Combine"),
    FieldSpec("forty", "40-yard dash (sec)", "(4.20–5.40)", "Combine", important=True),
    FieldSpec("cone", "3-cone (sec)", "(6.50–8.50)", "Combine", important=True),
    FieldSpec("shuttle", "20-yard shuttle (sec)", "(3.90–4.90)", "Combine", important=True),
    FieldSpec("vertical", "Vertical (in)", "(20–45)", "Combine"),
    FieldSpec("broad_jump", "Broad jump (in)", "(90–140)", "Combine"),
    FieldSpec("bench", "Bench reps", "(0–40)", "Combine"),
    FieldSpec("cfb_pass_yards", "CFB pass yards (final season)", "(0–6000)", "College", important=True),
    FieldSpec("cfb_pass_td", "CFB pass TD", "(0–60)", "College", important=True),
    FieldSpec("cfb_rush_yards", "CFB rush yards", "(0–2500)", "College", important=True),
    FieldSpec("cfb_rush_td", "CFB rush TD", "(0–30)", "College", important=True),
    FieldSpec("cfb_rec", "CFB receptions", "(0–120)", "College", important=True),
    FieldSpec("cfb_rec_yards", "CFB rec yards", "(0–2000)", "College", important=True),
    FieldSpec("cfb_rec_td", "CFB rec TD", "(0–25)", "College", important=True),
    FieldSpec("sos_opp_win_pct", "Schedule SOS (opp win %)", "(0.35–0.65)", "Team context"),
    FieldSpec(
        "team_opportunity_ppr",
        "Team opportunity points",
        "(prior-yr pos fantasy pts; 0–2500)",
        "Team context",
    ),
    FieldSpec(
        "team_pos_touches",
        "Team pos touches",
        "(prior-yr carries+receptions; 0–800)",
        "Team context",
        important=True,
    ),
    FieldSpec(
        "team_pos_carries",
        "Team pos carries",
        "(prior-yr team position carries; 0–600)",
        "Team context",
    ),
    FieldSpec(
        "team_pos_targets",
        "Team pos targets",
        "(prior-yr team position targets; 0–600)",
        "Team context",
    ),
    FieldSpec(
        "incumbent_pos_ppr",
        "Incumbent workhorse points",
        "(returning same-pos max prior pts; higher = more competition)",
        "Team context",
        important=True,
    ),
    FieldSpec(
        "incumbent_pos_ppr_sum",
        "Incumbent room points sum",
        "(returning same-pos prior fantasy pts sum)",
        "Team context",
    ),
    FieldSpec(
        "incumbent_pos_carries",
        "Incumbent workhorse carries",
        "(returning same-pos max prior carries)",
        "Team context",
        important=True,
    ),
    FieldSpec(
        "incumbent_pos_touches",
        "Incumbent workhorse touches",
        "(returning same-pos max prior touches)",
        "Team context",
    ),
    FieldSpec(
        "incumbent_ff_adp",
        "Incumbent FantasyPros ADP",
        "(best same-team/pos ADP; higher = weaker incumbent)",
        "Team context",
        important=True,
    ),
    FieldSpec(
        "adp_vs_incumbent",
        "ADP vs incumbent",
        "(incumbent ADP − your ADP; positive = market prefers you)",
        "Team context",
        important=True,
    ),
    FieldSpec(
        "vacated_touches",
        "Vacated touches",
        "(team touches − returning workhorse; higher = more open)",
        "Team context",
        important=True,
    ),
    FieldSpec(
        "vacated_carries",
        "Vacated carries",
        "(team carries − returning workhorse carries)",
        "Team context",
        important=True,
    ),
    FieldSpec("off_pass_rate_proxy", "Team pass rate", "(0.40–0.70)", "Team context"),
    FieldSpec("off_pass_yards", "Team pass yards", "(2500–5500)", "Team context"),
    FieldSpec("off_rush_yards", "Team rush yards", "(1200–2800)", "Team context"),
]

GROUP_ORDER = ("Core", "Recruiting", "Pre-draft fantasy", "Combine", "College", "Team context")
