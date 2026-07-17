from __future__ import annotations

from pathlib import Path

import pandas as pd

from rookie_ppr.config import CSV_OUTPUT_DIR, OUTPUT_DIR, ensure_directories


def export_workbook_and_csvs(tables: dict[str, pd.DataFrame], workbook_name: str = "rookie_ppr_master.xlsx") -> Path:
    ensure_directories()
    xlsx_path = OUTPUT_DIR / workbook_name

    # Stable sheet order
    preferred = [
        "players",
        "recruiting",
        "draft",
        "combine",
        "college_production",
        "team_context",
        "pre_draft_fantasy",
        "fantasy_rookie",
        "players_master",
        "data_dictionary",
    ]
    ordered = [k for k in preferred if k in tables] + [k for k in tables if k not in preferred]

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        for name in ordered:
            df = tables[name]
            if df is None:
                df = pd.DataFrame()
            # Excel sheet name limit 31 chars
            sheet = name[:31]
            df.to_excel(writer, sheet_name=sheet, index=False)

    for name in ordered:
        df = tables[name] if tables[name] is not None else pd.DataFrame()
        csv_path = CSV_OUTPUT_DIR / f"{name}.csv"
        df.to_csv(csv_path, index=False)

    # Convenience alias
    if "players_master" in tables:
        tables["players_master"].to_csv(OUTPUT_DIR / "players_master.csv", index=False)

    return xlsx_path
