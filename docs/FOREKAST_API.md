# ForeKast API contract (Nerds United site)

Base URL (local): `http://127.0.0.1:8000`  
Interactive docs: `http://127.0.0.1:8000/docs`  
CORS allowed: `https://nerds-united.vercel.app`, `localhost:3000`, `localhost:5173`

## Primary: Takes / weekly ForeKast

### `GET /v1/forekast/snapshot?season=&week=`

One round-trip for the Takes page. Defaults to **latest season** and **latest `asOfWeek`**.

```json
{
  "season": 2025,
  "asOfWeek": 14,
  "odds": { "teams": [ /* see weekly */ ] },
  "injuries": { "events": [ /* … */ ] },
  "trades": { "events": [ /* … */ ] }
}
```

Suggested client:

```ts
const res = await fetch(`${API}/v1/forekast/snapshot`);
const data = await res.json();
// data.odds.teams[].playoffOddsPct, titleOddsPct, simulatedWins, ownerName, teamName
// data.injuries.events — major injury markers through asOfWeek
// data.trades.events — bilateral trade markers through asOfWeek
```

Refetch when `asOfWeek` increments (after each NFL week once Capstone CSVs are refreshed).

### `GET /v1/forekast/weekly?season=&week=`

Playoff / title odds only. Team objects use site field names:

| Field | Meaning |
| --- | --- |
| `teamId` | Franchise id (string) |
| `ownerName` | Primary manager (`league_managers.csv`) |
| `teamName` | Season team name |
| `playoffOddsPct` | 0–1 |
| `titleOddsPct` | 0–1 |
| `simulatedWins` | Mean remaining+locked wins |

Odds already bake in that week’s ESPN roster (**trades/waivers**) and a **soft injury discount** (ESPN ~0 proj + not started, non-bye). From week 1 onward they also **re-rate remaining-season player pace** from in-season usage and scoring, shrunk toward the preseason prior so a single spike cannot rewrite a projection. Injury/trade **event lists** are narrative overlays.

### `GET /v1/forekast/injuries?season=&week=&exact=`

Major starter injury events. Default: all events with `asOfWeek <= week`. `exact=true` → only that week.

### `GET /v1/forekast/trades?season=&week=&exact=`

Bilateral player-for-player trades (same filter semantics).

### `GET /v1/forekast/season?season=`

End-of-season / preseason finish board shaped like the bundled Henry ForeKast table (`espnBosRank`, `henryRegularSeasonRank`, `playoffOddsPct`, …).

### `GET /v1/forekast/draft-grades?season=`

Two grades per team, plus the per-pick highlights and the calibration that
qualifies them. Defaults to the latest graded season; `availableSeasons` lists
the rest.

- `valueGrade` / `valuePoints` / `valueZ` — did the team beat the ADP board?
  Each skill pick is compared to the player the board had at that point in the
  draft, in value-over-replacement terms so positions compare fairly. Exactly
  zero-sum across the league: `valuePoints` sums to 0 within a season.
- `rosterGrade` / `rosterPointsPerWeek` / `rosterZ` — how strong is the
  resulting team, from the same projection engine as the odds.
- `bestPick` / `biggestReach` — `{playerName, position, overallPick, adpRank,
  valueAdded}` for that team's largest gain and largest loss against the board.
- `playoffOddsPct` / `titleOddsPct` — **null until the season's odds are
  published**, because those depend on the schedule. Both grades do not.

```json
{
  "season": 2025,
  "availableSeasons": [2018, "…", 2025],
  "calibration": {
    "note": "What each letter has actually been worth …",
    "value":  [{ "grade": "A", "nTeamSeasons": 15, "avgWins": 7.07, "playoffRatePct": 0.6, "titleRatePct": 0.0 }],
    "roster": [{ "grade": "A", "nTeamSeasons": 13, "avgWins": 7.0,  "playoffRatePct": 0.538, "titleRatePct": 0.0 }]
  },
  "teams": [{ "teamId": "2", "ownerName": "…", "valueGrade": "A", "rosterGrade": "A" }]
}
```

Letters are banded on a fixed within-season z-score, not curved, so a season
where everyone drafted well does not manufacture a D. Ship `calibration`
alongside the letters: draft grades are weak predictors by construction, the
strongest draft-time signal explains roughly a tenth of the variance in wins,
and the calibration is non-monotonic at the top — an A has historically been
worth no more than a B. Presenting a letter without it overstates the grade.

## Meta

- `GET /health` — artifact presence + latest week
- `GET /v1/meta` — seasons + weeksBySeason
- `POST /v1/admin/reload` — clear CSV cache after regenerating files

## Preseason (`asOfWeek: 0`)

Before a season's first game, that season is served at `asOfWeek: 0`: odds come
from rosters as they stand right now, so post-draft waiver and trade activity is
reflected. `injuries` / `trades` are empty because those are derived from
week-over-week roster changes and no week has been played. Clients should treat
empty event lists as normal, not an error.

Once week 1 is played, `asOfWeek: 0` reverts to meaning the draft-day view and
week 1 onward carries the live rosters, which is what completed seasons publish.

`GET /v1/forekast/season` is the exception to "latest season wins": it compares
projected finish against actual, so with no `season` it returns the newest
*finished* season rather than one in progress.

Odds are only published once the league schedule is final — see
[`HOSTING.md`](HOSTING.md) — because playoff and title odds depend on the
head-to-head matchups.

## Weekly refresh (Capstone machine)

```powershell
# Needs ESPN cookies in .env only to pull new live weeks into league_rosters
python -m rookie_ppr.api.refresh_forekast --seasons 2025
# Then restart API or POST /v1/admin/reload
```

Shipped historical CSVs under `data/output/csv/` work without cookies.

## Secrets

Never send `ESPN_S2`, `ESPN_SWID`, or `CFBD_API_KEY` to the browser. This API only reads compiled CSVs.
