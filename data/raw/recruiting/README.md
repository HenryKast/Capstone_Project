# Recruiting raw data

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

## Automated fetch

From the repo (venv active):

```powershell
python -m rookie_ppr.fetch_on3_recruiting
# or as part of compile:
python -m rookie_ppr.compile --fetch-on3
```

This pulls:
1. Industry Comparison (`/rivals/rankings/industry-comparison/football/{year}/`) — preferred
2. Industry Player (`/rivals/rankings/industry-player/football/{year}/`) — fills players missing from comparison

Ranking rule: Industry Composite when ranked; else Rivals-only.
