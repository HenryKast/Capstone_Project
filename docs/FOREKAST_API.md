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
| `ownerName` | Primary manager |
| `teamName` | Season team name |
| `playoffOddsPct` | 0–1 |
| `titleOddsPct` | 0–1 |
| `simulatedWins` | Mean remaining+locked wins |

Odds already bake in that week’s ESPN roster (**trades/waivers**) and a **soft injury discount** (ESPN ~0 proj + not started, non-bye). Injury/trade **event lists** are narrative overlays.

### `GET /v1/forekast/injuries?season=&week=&exact=`

Major starter injury events. Default: all events with `asOfWeek <= week`. `exact=true` → only that week.

### `GET /v1/forekast/trades?season=&week=&exact=`

Bilateral player-for-player trades (same filter semantics).

### `GET /v1/forekast/season?season=`

End-of-season / preseason finish board shaped like the bundled Henry ForeKast table (`espnBosRank`, `henryRegularSeasonRank`, `playoffOddsPct`, …).

## Meta

- `GET /health` — artifact presence + latest week
- `GET /v1/meta` — seasons + weeksBySeason
- `POST /v1/admin/reload` — clear CSV cache after regenerating files

## Weekly refresh (Capstone machine)

```powershell
# Needs ESPN cookies in .env only to pull new live weeks into league_rosters
python -m rookie_ppr.api.refresh_forekast --seasons 2025
# Then restart API or POST /v1/admin/reload
```

Shipped historical CSVs under `data/output/csv/` work without cookies.

## Secrets

Never send `ESPN_S2`, `ESPN_SWID`, or `CFBD_API_KEY` to the browser. This API only reads compiled CSVs.
