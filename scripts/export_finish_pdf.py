"""Export the full League Finish Proj vs Actual canvas layout to PDF.

Mirrors the canvas: title, summary stats, definitions, MAE charts, and the
full All-seasons table. Writes to the Capstone Project root.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.legends import Legend
from reportlab.graphics.shapes import Drawing
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from rookie_ppr.league.config import CSV_OUTPUT_DIR

ROOT = Path(__file__).resolve().parents[1]
CSV_SRC = CSV_OUTPUT_DIR / "league_finish_proj_vs_actual_all.csv"
JSON_SRC = CSV_OUTPUT_DIR / "league_finish_proj_vs_actual_all.json"
OUT = ROOT / "league_finish_proj_vs_actual.pdf"

INFO = colors.HexColor("#2563eb")
WARN = colors.HexColor("#d97706")
NEUTRAL = colors.HexColor("#6b7280")
HEADER_BG = colors.HexColor("#1f2937")
CALLOUT_BG = colors.HexColor("#f9fafb")
STAT_BORDER = colors.HexColor("#e5e7eb")


def _dash(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)) or pd.isna(v):
        return "—"
    if isinstance(v, float) and v == int(v):
        return str(int(v))
    return str(int(v)) if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v)


def _delta(proj, act) -> str:
    if pd.isna(proj) or pd.isna(act):
        return "—"
    d = int(act) - int(proj)
    if d == 0:
        return "exact"
    return f"+{d}" if d > 0 else str(d)


def _styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "T",
            parent=base["Heading1"],
            fontSize=16,
            leading=20,
            alignment=TA_CENTER,
            spaceAfter=4,
            textColor=HEADER_BG,
        ),
        "sub": ParagraphStyle(
            "S",
            parent=base["Normal"],
            fontSize=9,
            leading=12,
            alignment=TA_CENTER,
            textColor=NEUTRAL,
            spaceAfter=10,
        ),
        "h2": ParagraphStyle(
            "H2",
            parent=base["Heading2"],
            fontSize=11,
            leading=14,
            spaceBefore=8,
            spaceAfter=6,
            textColor=HEADER_BG,
        ),
        "body": ParagraphStyle(
            "B",
            parent=base["Normal"],
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#374151"),
        ),
        "note": ParagraphStyle(
            "N",
            parent=base["Normal"],
            fontSize=7.5,
            leading=10,
            textColor=NEUTRAL,
            spaceBefore=6,
        ),
        "stat_val": ParagraphStyle(
            "SV",
            parent=base["Normal"],
            fontSize=14,
            leading=16,
            alignment=TA_CENTER,
            textColor=HEADER_BG,
            fontName="Helvetica-Bold",
        ),
        "stat_lbl": ParagraphStyle(
            "SL",
            parent=base["Normal"],
            fontSize=7.5,
            leading=9,
            alignment=TA_CENTER,
            textColor=NEUTRAL,
        ),
        "cell": ParagraphStyle(
            "C",
            parent=base["Normal"],
            fontSize=7,
            leading=9,
        ),
        "hdr": ParagraphStyle(
            "Hdr",
            parent=base["Normal"],
            fontSize=7,
            leading=9,
            textColor=colors.white,
            alignment=TA_CENTER,
            fontName="Helvetica-Bold",
        ),
    }


def _stat_box(value: str, label: str, styles: dict, width: float) -> Table:
    inner = Table(
        [
            [Paragraph(value, styles["stat_val"])],
            [Paragraph(label, styles["stat_lbl"])],
        ],
        colWidths=[width],
    )
    inner.setStyle(
        TableStyle(
            [
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("BOX", (0, 0), (-1, -1), 0.6, STAT_BORDER),
                ("BACKGROUND", (0, 0), (-1, -1), colors.white),
            ]
        )
    )
    return inner


def _stats_row(items: list[tuple[str, str]], styles: dict, page_w: float) -> Table:
    n = len(items)
    gap = 8
    box_w = (page_w - gap * (n - 1)) / n
    cells = [_stat_box(v, lbl, styles, box_w) for v, lbl in items]
    t = Table([cells], colWidths=[box_w] * n)
    t.setStyle(
        TableStyle(
            [
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), gap),
                ("RIGHTPADDING", (-1, 0), (-1, -1), 0),
            ]
        )
    )
    return t


def _bar_chart(
    categories: list[str],
    series: list[tuple[str, list[float], colors.Color]],
    width: float,
    height: float,
) -> Drawing:
    d = Drawing(width, height)
    chart = VerticalBarChart()
    chart.x = 40
    chart.y = 35
    chart.height = height - 60
    chart.width = width - 60
    chart.data = [s[1] for s in series]
    chart.categoryAxis.categoryNames = categories
    chart.categoryAxis.labels.fontSize = 7
    chart.categoryAxis.labels.boxAnchor = "n"
    chart.valueAxis.valueMin = 0
    chart.valueAxis.valueMax = max(max(s[1]) for s in series) * 1.15
    chart.valueAxis.labels.fontSize = 7
    chart.barWidth = 8
    chart.groupSpacing = 12
    chart.barSpacing = 2
    for i, (_, _, color) in enumerate(series):
        chart.bars[i].fillColor = color
        chart.bars[i].strokeColor = color
    d.add(chart)

    legend = Legend()
    legend.x = 40
    legend.y = height - 14
    legend.fontSize = 7
    legend.boxAnchor = "nw"
    legend.columnMaximum = 1
    legend.colorNamePairs = [(s[2], s[0]) for s in series]
    d.add(legend)
    return d


def _callout(title: str, body: str, styles: dict, width: float) -> Table:
    content = Table(
        [
            [Paragraph(f"<b>{title}</b>", styles["body"])],
            [Paragraph(body, styles["body"])],
        ],
        colWidths=[width],
    )
    content.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), CALLOUT_BG),
                ("BOX", (0, 0), (-1, -1), 0.6, STAT_BORDER),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return content


def build() -> Path:
    df = pd.read_csv(CSV_SRC).sort_values(["season", "our_rs"]).reset_index(drop=True)
    summary = json.loads(JSON_SRC.read_text(encoding="utf-8"))["summary"]

    styles = _styles()
    page_w = landscape(letter)[0] - 0.8 * inch

    our_rs_mae = float(summary["overall_our_rs_mae"])
    our_po_mae = float(summary["overall_our_po_mae"])
    espn_overlap_our = summary["espn_overlap_our_rs_mae"]
    espn_overlap_espn = summary["espn_overlap_espn_rs_mae"]
    our_beats = summary["our_rs_beats_espn_pct"]
    espn_beats = summary["espn_rs_beats_our_pct"]
    tie_pct = summary["tie_vs_espn_pct"]

    espn_rows = df[df["espn_bos"].notna()]
    espn_rs_mae = float(espn_rows["err_espn_rs"].mean()) if len(espn_rows) else None

    years = [int(y) for y in summary["mae_our_rs_by_season"]]
    mae_our_rs = [float(summary["mae_our_rs_by_season"][str(y)]) for y in years]
    mae_our_po = [float(summary["mae_our_po_by_season"][str(y)]) for y in years]
    espn_years = [
        int(y)
        for y, v in summary["mae_espn_rs_by_season"].items()
        if v is not None
    ]
    mae_espn = [float(summary["mae_espn_rs_by_season"][str(y)]) for y in espn_years]
    mae_our_espn_years = [
        float(summary["mae_our_rs_by_season"][str(y)]) for y in espn_years
    ]

    # ---- full table ----
    headers = [
        "Year",
        "Team",
        "ESPN BOS",
        "Henry RS",
        "Act RS",
        "RS Δ",
        "Henry PO",
        "Act PO",
        "PO Δ",
        "Sim W",
        "Act W",
        "PO%",
        "Title%",
    ]
    data: list[list] = [[Paragraph(h, styles["hdr"]) for h in headers]]
    year_start_rows: list[int] = []
    prev_season: int | None = None

    for _, r in df.iterrows():
        season = int(r["season"])
        team = str(r["team"])
        row = [
            str(season),
            Paragraph(team[:38], styles["cell"]),
            _dash(r["espn_bos"]),
            str(int(r["our_rs"])),
            _dash(r["act_rs"]),
            _delta(r["our_rs"], r["act_rs"]),
            str(int(r["our_po"])),
            _dash(r["act_po"]),
            _delta(r["our_po"], r["act_po"]),
            f"{float(r['sim_wins_mean']):.1f}",
            str(int(r["wins"])),
            f"{100 * float(r['playoff_odds']):.0f}%",
            f"{100 * float(r['title_odds']):.0f}%",
        ]
        data.append(row)
        idx = len(data) - 1
        if prev_season is None or season != prev_season:
            year_start_rows.append(idx)
        prev_season = season

    col_widths = [
        0.45 * inch,
        2.25 * inch,
        0.55 * inch,
        0.5 * inch,
        0.5 * inch,
        0.5 * inch,
        0.5 * inch,
        0.5 * inch,
        0.5 * inch,
        0.45 * inch,
        0.45 * inch,
        0.45 * inch,
        0.5 * inch,
    ]
    table = Table(data, colWidths=col_widths, repeatRows=1)
    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), HEADER_BG),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 1), (-1, -1), 7),
        ("ALIGN", (0, 0), (0, -1), "CENTER"),
        ("ALIGN", (2, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d1d5db")),
        ("BACKGROUND", (0, 1), (-1, -1), colors.white),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2),
        # Thick black line under header
        ("LINEBELOW", (0, 0), (-1, 0), 2.0, colors.black),
    ]
    # Thick black border between year sections
    for r in year_start_rows:
        style_cmds.append(("LINEABOVE", (0, r), (-1, r), 2.5, colors.black))
    # Thick black line under the last data row
    style_cmds.append(("LINEBELOW", (0, -1), (-1, -1), 2.5, colors.black))
    table.setStyle(TableStyle(style_cmds))

    chart1 = _bar_chart(
        [str(y) for y in years],
        [
            ("Regular season", mae_our_rs, INFO),
            ("Playoff / final", mae_our_po, WARN),
        ],
        width=page_w,
        height=160,
    )
    chart2 = _bar_chart(
        [str(y) for y in espn_years],
        [
            ("Henry blend", mae_our_espn_years, INFO),
            ("ESPN draft-day", mae_espn, NEUTRAL),
        ],
        width=page_w * 0.7,
        height=160,
    )

    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=landscape(letter),
        leftMargin=0.4 * inch,
        rightMargin=0.4 * inch,
        topMargin=0.4 * inch,
        bottomMargin=0.35 * inch,
        title="Projected vs actual finishes (RS + playoffs)",
    )

    story = [
        Paragraph("Projected vs actual finishes (RS + playoffs)", styles["title"]),
        Paragraph(
            "Henry blend Monte Carlo vs ESPN draft-day projected rank vs actual "
            "regular-season seed and final playoff place. Seasons 2018–2025.",
            styles["sub"],
        ),
        _stats_row(
            [
                (f"{our_rs_mae:.2f}", "Henry RS MAE"),
                (
                    "—" if espn_rs_mae is None else f"{espn_rs_mae:.2f}",
                    "ESPN BOS RS MAE",
                ),
                (f"{our_po_mae:.2f}", "Henry playoff MAE"),
                ("8 yrs", "View"),
            ],
            styles,
            page_w,
        ),
        Spacer(1, 8),
        _stats_row(
            [
                (f"{espn_overlap_our}", "Henry RS MAE (2021–25)"),
                (f"{espn_overlap_espn}", "ESPN RS MAE (2021–25)"),
                (f"{our_beats}%", "Henry closer than ESPN"),
            ],
            styles,
            page_w,
        ),
        Spacer(1, 10),
        _callout(
            "Definitions",
            "ESPN BOS = draftDayProjectedRank (available 2021–2025 only). "
            "Henry RS = ranked mean simulated seed. Actual RS = ESPN playoff_seed. "
            "Henry playoff = ranked by title odds then playoff odds "
            "(sim does not store full placement dist). Actual playoff = "
            "rankCalculatedFinal / final_rank.",
            styles,
            page_w,
        ),
        Paragraph("Henry MAE by season (RS vs playoffs)", styles["h2"]),
        chart1,
        Paragraph(
            "Source: league_sim_team_seasons (blend) × league_teams · lower is better",
            styles["note"],
        ),
        Paragraph("RS finish MAE vs ESPN draft-day (2021–2025)", styles["h2"]),
        chart2,
        Paragraph(
            "ESPN draftDayProjectedRank missing for 2018–2020 in league cache",
            styles["note"],
        ),
        Paragraph(
            f"Season: <b>All</b> &nbsp;&nbsp; {len(df)} rows"
            + (f" &nbsp;&nbsp; {len(espn_rows)} with ESPN BOS" if len(espn_rows) else ""),
            styles["h2"],
        ),
        table,
        Paragraph(
            f"On overlapping seasons, ESPN is closer on {espn_beats}% of team-years, "
            f"Henry is closer on {our_beats}%, ties {tie_pct}%. "
            "CSV: data/output/csv/league_finish_proj_vs_actual_all.csv",
            styles["note"],
        ),
    ]
    doc.build(story)
    return OUT


if __name__ == "__main__":
    path = build()
    print(f"wrote {path}")
