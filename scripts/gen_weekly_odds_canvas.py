"""Regenerate weekly-odds canvas with injury markers and team filter."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from rookie_ppr.league.config import CSV_OUTPUT_DIR
from rookie_ppr.league.injury_events import detect_injury_events

ODDS = CSV_OUTPUT_DIR / "league_weekly_odds.csv"
INJ = CSV_OUTPUT_DIR / "league_injury_events.csv"
CANVAS = Path(
    r"C:\Users\kasth\.cursor\projects\c-Users-kasth-School-Capstone-Project"
    r"\canvases\league-weekly-odds.canvas.tsx"
)

TEMPLATE = r'''import {
  Callout,
  H1,
  H2,
  Pill,
  Row,
  Select,
  Stack,
  Text,
  useCanvasState,
  useHostTheme,
  usageColorSequence,
} from "cursor/canvas";

const DATA = __DATA__ as const;

type Metric = "playoff" | "title";

function BandaidIcon({ x, y, size = 14 }: { x: number; y: number; size?: number }) {
  const theme = useHostTheme();
  const w = size;
  const h = size * 0.55;
  const pad = size * 0.22;
  return (
    <g transform={`translate(${x - w / 2}, ${y - h / 2})`}>
      <rect
        x={0}
        y={0}
        width={w}
        height={h}
        rx={h / 2}
        fill={theme.fill.primary}
        stroke={theme.stroke.primary}
        strokeWidth={1}
      />
      <rect
        x={(w - pad) / 2}
        y={0}
        width={pad}
        height={h}
        fill={theme.bg.editor}
        stroke={theme.stroke.primary}
        strokeWidth={0.75}
      />
      <line
        x1={w / 2}
        y1={h * 0.28}
        x2={w / 2}
        y2={h * 0.72}
        stroke={theme.text.tertiary}
        strokeWidth={1}
      />
      <line
        x1={w / 2 - pad * 0.35}
        y1={h / 2}
        x2={w / 2 + pad * 0.35}
        y2={h / 2}
        stroke={theme.text.tertiary}
        strokeWidth={1}
      />
    </g>
  );
}

function OddsChart({
  weekLabels,
  series,
  injuries,
  metric,
  hoverWeek,
  setHoverWeek,
  setInjuryTip,
  colorOffset = 0,
}: {
  weekLabels: string[];
  series: { id: number; name: string; data: number[] }[];
  injuries: {
    teamId: number;
    team: string;
    week: number;
    weekIndex: number;
    playoff: number;
    title: number;
    label: string;
  }[];
  metric: Metric;
  hoverWeek: number | null;
  setHoverWeek: (v: number | null) => void;
  setInjuryTip: (v: string | null) => void;
  colorOffset?: number;
}) {
  const theme = useHostTheme();
  const width = 1000;
  const height = 380;
  const pad = { top: 20, right: 24, bottom: 36, left: 44 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  const colors = usageColorSequence.map((key) => theme.category[key]);

  const xAt = (i: number) =>
    pad.left + (weekLabels.length <= 1 ? plotW / 2 : (i / (weekLabels.length - 1)) * plotW);
  const yAt = (v: number) => pad.top + plotH * (1 - Math.min(100, Math.max(0, v)) / 100);

  const nearestWeek = (svgX: number) => {
    let best = 0;
    let bestDist = Infinity;
    for (let i = 0; i < weekLabels.length; i++) {
      const d = Math.abs(xAt(i) - svgX);
      if (d < bestDist) {
        bestDist = d;
        best = i;
      }
    }
    return best;
  };

  const onMove = (e: { currentTarget: SVGSVGElement; clientX: number }) => {
    const rect = e.currentTarget.getBoundingClientRect();
    if (rect.width <= 0) return;
    const svgX = ((e.clientX - rect.left) / rect.width) * width;
    if (svgX < pad.left || svgX > pad.left + plotW) {
      setHoverWeek(null);
      return;
    }
    setHoverWeek(nearestWeek(svgX));
  };

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      width="100%"
      style={{ display: "block", maxWidth: width }}
      role="img"
      aria-label="Week-by-week odds with injury markers"
      onMouseMove={onMove}
      onMouseLeave={() => {
        setHoverWeek(null);
        setInjuryTip(null);
      }}
    >
      {[0, 25, 50, 75, 100].map((tick) => (
        <g key={tick}>
          <line
            x1={pad.left}
            x2={pad.left + plotW}
            y1={yAt(tick)}
            y2={yAt(tick)}
            stroke={theme.stroke.tertiary}
            strokeWidth={1}
          />
          <text
            x={pad.left - 8}
            y={yAt(tick) + 3}
            textAnchor="end"
            fill={theme.text.tertiary}
            fontSize={10}
          >
            {tick}%
          </text>
        </g>
      ))}
      {weekLabels.map((label, i) => (
        <text
          key={label}
          x={xAt(i)}
          y={height - 12}
          textAnchor="middle"
          fill={theme.text.tertiary}
          fontSize={10}
        >
          {label}
        </text>
      ))}

      {hoverWeek != null ? (
        <line
          x1={xAt(hoverWeek)}
          x2={xAt(hoverWeek)}
          y1={pad.top}
          y2={pad.top + plotH}
          stroke={theme.stroke.secondary}
          strokeWidth={1.5}
          strokeDasharray="4 3"
          pointerEvents="none"
        />
      ) : null}

      {series.map((s, si) => {
        const color = colors[(si + colorOffset) % colors.length];
        const pts = s.data.map((v, i) => `${xAt(i)},${yAt(v)}`).join(" ");
        return (
          <g key={s.id}>
            <polyline fill="none" stroke={color} strokeWidth={2.5} points={pts} />
            {s.data.map((v, i) => (
              <circle
                key={i}
                cx={xAt(i)}
                cy={yAt(v)}
                r={hoverWeek === i ? 4 : 2.5}
                fill={color}
                stroke={hoverWeek === i ? theme.stroke.primary : undefined}
                strokeWidth={hoverWeek === i ? 1 : undefined}
              />
            ))}
          </g>
        );
      })}

      {injuries.map((inj) => {
        const si = series.findIndex((s) => s.id === inj.teamId);
        if (si < 0) return null;
        const value = metric === "playoff" ? inj.playoff : inj.title;
        const cx = xAt(inj.weekIndex);
        const cy = yAt(value);
        const tip = `${inj.team}: ${inj.label}`;
        return (
          <g
            key={`${inj.teamId}-${inj.week}-${inj.label}`}
            style={{ cursor: "pointer" }}
            onMouseEnter={() => {
              setHoverWeek(inj.weekIndex);
              setInjuryTip(tip);
            }}
            onMouseLeave={() => setInjuryTip(null)}
          >
            <circle cx={cx} cy={cy} r={12} fill="transparent" />
            <BandaidIcon x={cx} y={cy} size={16} />
            <title>{tip}</title>
          </g>
        );
      })}
    </svg>
  );
}

export default function LeagueWeeklyOdds() {
  const theme = useHostTheme();
  const seasons = DATA.seasons.map(String);
  const [season, setSeason] = useCanvasState<string>(
    "season",
    String(DATA.seasons[DATA.seasons.length - 1]),
  );
  const [metric, setMetric] = useCanvasState<Metric>("metric", "playoff");
  const [teamFilter, setTeamFilter] = useCanvasState<string>("teamFilter", "all");
  const [hoverWeek, setHoverWeek] = useCanvasState<number | null>("hoverWeek", null);
  const [injuryTip, setInjuryTip] = useCanvasState<string | null>("injHover", null);

  const block = DATA.bySeason[season as keyof typeof DATA.bySeason];
  const seriesSrc = metric === "playoff" ? block.playoff : block.title;
  const teamId = teamFilter === "all" ? null : Number(teamFilter);
  const series = seriesSrc
    .filter((s) => teamId == null || s.id === teamId)
    .map((s) => ({
      id: s.id,
      name: s.name,
      data: [...s.data],
    }));
  const injuries = block.injuries.filter(
    (inj) => teamId == null || inj.teamId === teamId,
  );
  const colorOffset =
    teamId == null ? 0 : Math.max(0, seriesSrc.findIndex((s) => s.id === teamId));
  const colors = usageColorSequence.map((key) => theme.category[key]);
  const label = metric === "playoff" ? "Playoff odds (%)" : "Title odds (%)";
  const teamOptions = [
    { value: "all", label: "All teams" },
    ...seriesSrc.map((s) => ({ value: String(s.id), label: s.name })),
  ];
  const titleSuffix = teamId != null && series[0] ? ` · ${series[0].name}` : "";
  const weekBreakdown =
    hoverWeek == null
      ? []
      : [...series]
          .map((s, i) => ({
            id: s.id,
            name: s.name,
            value: s.data[hoverWeek] ?? 0,
            color: colors[(i + colorOffset) % colors.length],
          }))
          .sort((a, b) => b.value - a.value);

  return (
    <Stack gap={20} style={{ padding: 20, maxWidth: 1100 }}>
      <Stack gap={6}>
        <H1>Week-by-week playoff & title odds</H1>
        <Text tone="secondary">
          As-of each week: lock actual results so far, rebuild strength from that
          week&apos;s ESPN roster, then Monte Carlo the rest. Bandaid markers mark
          weeks when a drafted/starting player is lost for 3+ weeks or the season.
        </Text>
      </Stack>

      <Callout tone="neutral" title="Injury markers">
        A bandaid marks the first week a drafted or starting skill player is
        effectively out for 3+ consecutive weeks (ESPN proj ~0 / not started,
        byes excluded), or leaves the roster for the rest of the season. Hover
        for player, span, and draft slot — e.g. Malik Nabers out for season (1.09).
      </Callout>

      <Row gap={12} style={{ alignItems: "center", flexWrap: "wrap" }}>
        <Text weight="medium">Season</Text>
        <Select
          value={season}
          onChange={(v) => {
            setSeason(v);
            setTeamFilter("all");
          }}
          options={seasons.map((s) => ({ value: s, label: s }))}
        />
        <Text weight="medium">Team</Text>
        <Select
          value={teamFilter}
          onChange={setTeamFilter}
          options={teamOptions}
        />
        <Text weight="medium">Metric</Text>
        <Select
          value={metric}
          onChange={(v) => setMetric(v as Metric)}
          options={[
            { value: "playoff", label: "Playoff odds" },
            { value: "title", label: "Title odds" },
          ]}
        />
        <Pill tone="neutral">{block.weekLabels.length} checkpoints</Pill>
        <Pill tone="warning">{injuries.length} injuries</Pill>
      </Row>

      <Stack gap={8}>
        <H2>
          {season} · {label}
          {titleSuffix}
        </H2>
        <OddsChart
          weekLabels={[...block.weekLabels]}
          series={series}
          injuries={[...injuries]}
          metric={metric}
          hoverWeek={hoverWeek}
          setHoverWeek={setHoverWeek}
          setInjuryTip={setInjuryTip}
          colorOffset={colorOffset}
        />
        {injuryTip ? (
          <Text weight="medium">{injuryTip}</Text>
        ) : hoverWeek != null ? (
          <Stack gap={6}>
            <Text weight="medium">
              {block.weekLabels[hoverWeek]} · {label}
            </Text>
            <Stack gap={3}>
              {weekBreakdown.map((row) => (
                <Row
                  key={row.id}
                  gap={8}
                  style={{ alignItems: "center", justifyContent: "space-between" }}
                >
                  <Row gap={6} style={{ alignItems: "center", minWidth: 0 }}>
                    <span
                      style={{
                        width: 8,
                        height: 8,
                        borderRadius: 2,
                        background: row.color,
                        display: "inline-block",
                        flexShrink: 0,
                      }}
                    />
                    <Text style={{ fontSize: 13 }}>{row.name}</Text>
                  </Row>
                  <Text weight="medium" style={{ fontSize: 13 }}>
                    {row.value.toFixed(1)}%
                  </Text>
                </Row>
              ))}
            </Stack>
          </Stack>
        ) : (
          <Text tone="tertiary" style={{ fontSize: 12 }}>
            Hover the chart for week odds · hover a bandaid for injuries
          </Text>
        )}
      </Stack>

      <Stack gap={6}>
        <Text weight="medium">Legend</Text>
        <Row gap={10} style={{ flexWrap: "wrap" }}>
          {series.map((s, i) => (
            <Row key={s.id} gap={6} style={{ alignItems: "center" }}>
              <span
                style={{
                  width: 10,
                  height: 10,
                  borderRadius: 2,
                  background: colors[(i + colorOffset) % colors.length],
                  display: "inline-block",
                }}
              />
              <Text style={{ fontSize: 12 }}>{s.name}</Text>
            </Row>
          ))}
        </Row>
      </Stack>

      {injuries.length > 0 ? (
        <Stack gap={6}>
          <H2>Injury events · {season}</H2>
          <Stack gap={4}>
            {injuries.map((inj) => (
              <Text key={`${inj.teamId}-${inj.week}-${inj.label}`} style={{ fontSize: 13 }}>
                W{inj.week} · {inj.team}: {inj.label}
              </Text>
            ))}
          </Stack>
        </Stack>
      ) : null}
    </Stack>
  );
}
'''


def main() -> None:
    inj = detect_injury_events()
    inj.to_csv(INJ, index=False)
    print(f"injuries={len(inj)}")
    if not inj.empty:
        print(inj.groupby("kind").size().to_string())

    df = pd.read_csv(ODDS)
    df["team_name"] = (
        df["team_name"]
        .astype(str)
        .str.encode("ascii", "ignore")
        .str.decode("ascii")
        .str.strip()
    )
    df.loc[df["team_name"] == "", "team_name"] = "Team " + df["team_id"].astype(str)
    df.loc[(df["season"] == 2020) & (df["team_id"] == 8), "team_name"] = "(table flip)"

    if inj.empty:
        inj = pd.DataFrame(
            columns=[
                "season",
                "team_id",
                "as_of_week",
                "player_name",
                "label",
                "overall_pick",
            ]
        )

    payload: dict = {
        "seasons": sorted(int(s) for s in df["season"].unique()),
        "bySeason": {},
    }

    for season, sdf in df.groupby("season"):
        weeks = sorted(int(w) for w in sdf["as_of_week"].unique())
        week_labels = [f"W{w}" for w in weeks]
        teams = (
            sdf.sort_values("team_id")[["team_id", "team_name"]]
            .drop_duplicates("team_id")
            .itertuples(index=False)
        )
        playoff = []
        title = []
        for team_id, team_name in teams:
            tdf = sdf[sdf["team_id"] == team_id].set_index("as_of_week").sort_index()
            playoff.append(
                {
                    "id": int(team_id),
                    "name": str(team_name),
                    "data": [
                        round(float(tdf.loc[w, "playoff_odds"]) * 100, 1) for w in weeks
                    ],
                }
            )
            title.append(
                {
                    "id": int(team_id),
                    "name": str(team_name),
                    "data": [
                        round(float(tdf.loc[w, "title_odds"]) * 100, 1) for w in weeks
                    ],
                }
            )

        season_inj = inj[inj["season"] == season].copy()
        injuries = []
        if not season_inj.empty:
            name_by_id = {s["id"]: s["name"] for s in playoff}
            playoff_by_id = {s["id"]: s["data"] for s in playoff}
            title_by_id = {s["id"]: s["data"] for s in title}
            week_index = {w: i for i, w in enumerate(weeks)}
            season_inj = season_inj.sort_values(
                ["team_id", "as_of_week", "overall_pick"],
                ascending=[True, True, True],
            )
            grouped = season_inj.groupby(
                ["team_id", "as_of_week"], as_index=False
            ).agg(label=("label", lambda x: " · ".join(x.astype(str))))
            for row in grouped.itertuples(index=False):
                tid = int(row.team_id)
                week = int(row.as_of_week)
                if week not in week_index or tid not in playoff_by_id:
                    continue
                wi = week_index[week]
                injuries.append(
                    {
                        "teamId": tid,
                        "team": name_by_id[tid],
                        "week": week,
                        "weekIndex": wi,
                        "playoff": playoff_by_id[tid][wi],
                        "title": title_by_id[tid][wi],
                        "label": str(row.label),
                    }
                )

        payload["bySeason"][str(int(season))] = {
            "weeks": weeks,
            "weekLabels": week_labels,
            "playoff": playoff,
            "title": title,
            "injuries": injuries,
        }

    data_json = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    CANVAS.write_text(TEMPLATE.replace("__DATA__", data_json), encoding="utf-8")
    print(f"wrote {CANVAS}")
    print(f"2025 injuries: {len(payload['bySeason']['2025']['injuries'])}")
    for item in payload["bySeason"]["2025"]["injuries"]:
        print(f"  W{item['week']} t{item['teamId']}: {item['label']}")


if __name__ == "__main__":
    main()
