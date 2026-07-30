"""Nerds United for Football (Crest) league board UI.

Opens from the crest control next to the helmet: a separate window with
Week-by-week odds and Finish proj vs actual tabs, reading league CSVs.
"""
from __future__ import annotations

import math
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Callable

import pandas as pd

from rookie_ppr.config import CSV_OUTPUT_DIR
from rookie_ppr.league.config import (
    LEAGUE_BACKTEST_ROSTERS_CSV,
    LEAGUE_DRAFT_CSV,
    LEAGUE_DRAFT_REACHES_CSV,
    LEAGUE_FINISH_PROJ_VS_ACTUAL_CSV,
    LEAGUE_INJURY_EVENTS_CSV,
    LEAGUE_TEAMS_CSV,
    LEAGUE_TRADE_EVENTS_CSV,
    LEAGUE_WEEKLY_ODDS_CSV,
)
from rookie_ppr.league.draft_reaches import _season_actuals, load_manager_map

CREST_NAME = "Nerds United for Football"
SERIES_COLORS = (
    "#FFE566",
    "#7CFFB2",
    "#FF8A8A",
    "#7EC8FF",
    "#FFB347",
    "#D4A5FF",
    "#FFFFFF",
    "#5CE1E6",
    "#F78FB3",
    "#A8E6CF",
)

# Fixed chart geometry so the odds plot never resizes with the window.
WEEKLY_CHART_W = 1000
WEEKLY_CHART_H = 380
FINISH_CHART_H = 160


def _ui():
    from rookie_ppr import ui as ui_mod

    return ui_mod


def _ascii(name: object) -> str:
    return str(name).encode("ascii", "ignore").decode("ascii").strip()


class LeagueScrollFrame(tk.Frame):
    """Vertical scroll container; put page content on ``.inner``."""

    def __init__(self, parent: tk.Misc) -> None:
        u = _ui()
        super().__init__(parent, bg=u.BG)
        self._canvas = tk.Canvas(self, bg=u.BG, highlightthickness=0, borderwidth=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self._canvas.yview)
        self.inner = tk.Frame(self._canvas, bg=u.BG)
        self._win = self._canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", self._on_inner_configure)
        self._canvas.bind("<Configure>", self._on_canvas_configure)
        self._canvas.configure(yscrollcommand=scrollbar.set)
        self._canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        def _on_mousewheel(event: tk.Event) -> None:
            self._canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        self._canvas.bind(
            "<Enter>", lambda _e: self._canvas.bind_all("<MouseWheel>", _on_mousewheel)
        )
        self._canvas.bind(
            "<Leave>", lambda _e: self._canvas.unbind_all("<MouseWheel>")
        )

    def _sync_window_width(self, viewport_w: int | None = None) -> None:
        if viewport_w is None:
            viewport_w = int(self._canvas.winfo_width())
        self.inner.update_idletasks()
        content_w = int(self.inner.winfo_reqwidth())
        # Never shrink below content — keeps fixed-size charts from being squeezed.
        self._canvas.itemconfigure(self._win, width=max(viewport_w, content_w, 1))

    def _on_inner_configure(self, _event: tk.Event | None = None) -> None:
        self._sync_window_width()
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _on_canvas_configure(self, event: tk.Event) -> None:
        self._sync_window_width(event.width)
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def scroll_to_top(self) -> None:
        self._canvas.yview_moveto(0)


def draw_crest(
    canvas: tk.Canvas,
    *,
    x: int = 2,
    y: int = 2,
    scale: float = 1.0,
    shell: str | None = None,
    accent: str | None = None,
    ink: str | None = None,
) -> None:
    """Shield crest with NUF monogram (for the tab-bar control)."""
    u = _ui()
    shell = shell or u.IMPORTANT_FG
    accent = accent or u.BG
    ink = ink or u.TAB_BAR_BG
    s = scale
    # Shield outline
    pts = [
        x + 10 * s,
        y + 1 * s,
        x + 22 * s,
        y + 1 * s,
        x + 28 * s,
        y + 8 * s,
        x + 28 * s,
        y + 18 * s,
        x + 16 * s,
        y + 28 * s,
        x + 4 * s,
        y + 18 * s,
        x + 4 * s,
        y + 8 * s,
    ]
    canvas.create_polygon(*pts, fill=shell, outline=ink, width=max(1, int(1.2 * s)))
    # Inner plate
    inner = [
        x + 12 * s,
        y + 4 * s,
        x + 20 * s,
        y + 4 * s,
        x + 24 * s,
        y + 9 * s,
        x + 24 * s,
        y + 16 * s,
        x + 16 * s,
        y + 24 * s,
        x + 8 * s,
        y + 16 * s,
        x + 8 * s,
        y + 9 * s,
    ]
    canvas.create_polygon(*inner, fill=accent, outline="")
    canvas.create_text(
        x + 16 * s,
        y + 13 * s,
        text="NUF",
        fill=shell,
        font=("Segoe UI", max(6, int(7 * s)), "bold"),
    )


class CrestButton(tk.Frame):
    """Crest control: shield + league name; opens the Crest window."""

    def __init__(
        self,
        master: tk.Misc,
        *,
        command: Callable[[], None],
        size: int = 30,
    ) -> None:
        u = _ui()
        super().__init__(master, bg=u.TAB_BAR_BG)
        self._command = command
        self._size = size
        self.canvas = tk.Canvas(
            self,
            width=size + 4,
            height=size + 2,
            bg=u.TAB_BAR_BG,
            highlightthickness=0,
            cursor="hand2",
        )
        self.canvas.pack(side="left")
        self.label = tk.Label(
            self,
            text="Nerds United\nfor Football",
            bg=u.TAB_BAR_BG,
            fg=u.IMPORTANT_FG,
            font=("Segoe UI", 7, "bold"),
            justify="left",
            cursor="hand2",
        )
        self.label.pack(side="left", padx=(4, 2))
        self.redraw()
        for w in (self.canvas, self.label, self):
            w.bind("<Button-1>", lambda _e: self._command())

    def redraw(self, *, bar_bg: str | None = None) -> None:
        u = _ui()
        bg = bar_bg or u.TAB_BAR_BG
        self.configure(bg=bg)
        self.canvas.configure(bg=bg)
        self.label.configure(bg=bg, fg=u.IMPORTANT_FG)
        self.canvas.delete("all")
        draw_crest(
            self.canvas,
            x=2,
            y=1,
            scale=self._size / 30,
            shell=u.IMPORTANT_FG,
            accent=u.BG,
            ink=u.TAB_BAR_BG,
        )


class NerdUnitedWindow(tk.Toplevel):
    """Separate league board: odds, finishes, draft reaches, draft history."""

    def __init__(self, parent: tk.Misc) -> None:
        u = _ui()
        super().__init__(parent)
        self.title(f"{CREST_NAME} — League board")
        self.configure(bg=u.BG)
        self.geometry("1120x720")
        self.minsize(1080, 560)
        self._panels: dict[str, tk.Frame] = {}
        self._tab_buttons: dict[str, tk.Button] = {}
        self._tab_accents: dict[str, tk.Frame] = {}
        self._active = "weekly"

        # Place to the right of the main window when possible
        try:
            self.update_idletasks()
            px = int(parent.winfo_rootx()) + int(parent.winfo_width()) + 12
            py = int(parent.winfo_rooty())
            self.geometry(f"1120x720+{px}+{py}")
        except tk.TclError:
            pass

        bar = tk.Frame(self, bg=u.TAB_BAR_BG, padx=8, pady=0)
        bar.pack(fill="x")
        left = tk.Frame(bar, bg=u.TAB_BAR_BG)
        left.pack(side="left", anchor="sw")
        crest_cv = tk.Canvas(
            left, width=36, height=32, bg=u.TAB_BAR_BG, highlightthickness=0
        )
        crest_cv.pack(side="left", padx=(2, 8), pady=4)
        draw_crest(crest_cv, x=2, y=1, scale=1.05)
        tk.Label(
            left,
            text=CREST_NAME,
            bg=u.TAB_BAR_BG,
            fg=u.IMPORTANT_FG,
            font=("Segoe UI", 11, "bold"),
        ).pack(side="left", pady=8)

        tabs_row = tk.Frame(bar, bg=u.TAB_BAR_BG)
        tabs_row.pack(side="right", anchor="sw")
        for key, label in (
            ("weekly", "Weekly ForeKast"),
            ("finish", "Season ForeKast"),
            ("reaches", "Draft Reaches / Falls"),
            ("draft", "Draft History"),
        ):
            col = tk.Frame(tabs_row, bg=u.TAB_BAR_BG)
            col.pack(side="left", padx=(0, 2))
            btn = tk.Button(
                col,
                text=label,
                command=lambda m=key: self._select_tab(m),
                relief="flat",
                bd=0,
                padx=14,
                pady=8,
                font=("Segoe UI", 10),
                cursor="hand2",
            )
            btn.pack(fill="x")
            accent = tk.Frame(col, bg=u.TAB_BAR_BG, height=3)
            accent.pack(fill="x")
            self._tab_buttons[key] = btn
            self._tab_accents[key] = accent

        content = tk.Frame(self, bg=u.BG)
        content.pack(fill="both", expand=True)
        self._panels["weekly"] = WeeklyOddsPanel(content)
        self._panels["finish"] = FinishVsActualPanel(content)
        self._panels["reaches"] = DraftReachesPanel(content)
        self._panels["draft"] = DraftHistoryPanel(content)
        self._select_tab("weekly")
        self.bind("<Escape>", lambda _e: self.destroy())

    def _select_tab(self, mode: str) -> None:
        u = _ui()
        if mode not in self._panels:
            return
        self._active = mode
        for key, panel in self._panels.items():
            if key == mode:
                panel.pack(fill="both", expand=True)
                scroll = getattr(panel, "_scroll", None)
                if scroll is not None:
                    scroll.scroll_to_top()
            else:
                panel.pack_forget()
        for key, btn in self._tab_buttons.items():
            active = key == mode
            btn.configure(
                bg=u.TAB_ACTIVE_BG if active else u.TAB_INACTIVE_BG,
                fg=u.TAB_ACTIVE_FG if active else u.TAB_INACTIVE_FG,
                activebackground=u.TAB_ACTIVE_BG if active else u.TAB_INACTIVE_BG,
                activeforeground=u.TAB_ACTIVE_FG if active else u.TAB_INACTIVE_FG,
                font=("Segoe UI", 10, "bold") if active else ("Segoe UI", 10),
            )
            self._tab_accents[key].configure(
                bg=u.TAB_ACCENT if active else u.TAB_BAR_BG
            )


class WeeklyOddsPanel(tk.Frame):
    """Week-by-week playoff/title odds with injury bandaids."""

    def __init__(self, master: tk.Misc) -> None:
        u = _ui()
        super().__init__(master, bg=u.BG)
        self._odds = pd.DataFrame()
        self._inj = pd.DataFrame()
        self._trades = pd.DataFrame()
        self._hover_week: int | None = None
        self._injury_tip: str | None = None
        self._plot_meta: dict = {}
        self._legend_pct: dict[int, tk.Label] = {}
        self._show_injuries = tk.BooleanVar(value=True)
        self._show_trades = tk.BooleanVar(value=True)

        scroll = LeagueScrollFrame(self)
        scroll.pack(fill="both", expand=True)
        page = scroll.inner
        self._scroll = scroll

        header = tk.Frame(page, bg=u.BG, padx=14, pady=10)
        header.pack(fill="x")
        tk.Label(
            header,
            text="Week-by-week playoff & title odds",
            bg=u.BG,
            fg=u.FG,
            font=("Segoe UI", 14, "bold"),
        ).pack(anchor="w")
        tk.Label(
            header,
            text=(
                "As-of each week: lock actual results, rebuild strength from that "
                "week's ESPN roster, then Monte Carlo the rest. Bandaids mark "
                "injuries; swap markers mark mid-season player-for-player trades."
            ),
            bg=u.BG,
            fg=u.MUTED_FG,
            font=("Segoe UI", 9),
            wraplength=1000,
            justify="left",
        ).pack(anchor="w", pady=(4, 0))

        controls = tk.Frame(page, bg=u.BG, padx=14, pady=4)
        controls.pack(fill="x")
        self._season_var = tk.StringVar(value="")
        self._team_var = tk.StringVar(value="All teams")
        self._metric_var = tk.StringVar(value="Playoff odds")

        self._mk_label(controls, "Season").pack(side="left")
        self._season_cb = ttk.Combobox(
            controls, textvariable=self._season_var, width=8, state="readonly"
        )
        self._season_cb.pack(side="left", padx=(4, 12))
        self._season_cb.bind("<<ComboboxSelected>>", lambda _e: self._on_season())

        self._mk_label(controls, "Team").pack(side="left")
        self._team_cb = ttk.Combobox(
            controls, textvariable=self._team_var, width=28, state="readonly"
        )
        self._team_cb.pack(side="left", padx=(4, 12))
        self._team_cb.bind("<<ComboboxSelected>>", lambda _e: self._redraw())

        self._mk_label(controls, "Metric").pack(side="left")
        self._metric_cb = ttk.Combobox(
            controls,
            textvariable=self._metric_var,
            values=["Playoff odds", "Title odds"],
            width=14,
            state="readonly",
        )
        self._metric_cb.pack(side="left", padx=(4, 12))
        self._metric_cb.bind("<<ComboboxSelected>>", lambda _e: self._redraw())

        u.FillCheckbox(
            controls,
            variable=self._show_injuries,
            text="Injuries",
            command=self._on_toggle_markers,
            bg=u.BG,
            fg=u.FG,
        ).pack(side="left", padx=(8, 8))
        u.FillCheckbox(
            controls,
            variable=self._show_trades,
            text="Trades",
            command=self._on_toggle_markers,
            bg=u.BG,
            fg=u.FG,
        ).pack(side="left", padx=(0, 8))

        self._counts = tk.Label(
            controls, text="", bg=u.BG, fg=u.MUTED_FG, font=("Segoe UI", 9)
        )
        self._counts.pack(side="left", padx=(8, 0))

        body = tk.Frame(page, bg=u.BG, padx=14, pady=6)
        body.pack(fill="x")
        self.canvas = tk.Canvas(
            body,
            width=WEEKLY_CHART_W,
            height=WEEKLY_CHART_H,
            bg=u.BG_FIELD,
            highlightthickness=1,
            highlightbackground=u.TAB_BAR_BG,
        )
        self.canvas.pack(anchor="w")
        self.canvas.bind("<Motion>", self._on_motion)
        self.canvas.bind("<Leave>", self._on_leave)

        # Injury tip (replaces the old week-odds hover list under the chart)
        self._tip_label = tk.Label(
            page,
            text="Hover a marker for injury/trade detail · hover the chart to update legend week",
            bg=u.BG,
            fg=u.MUTED_FG,
            font=("Segoe UI", 9),
            anchor="w",
            justify="left",
            padx=14,
        )
        self._tip_label.pack(fill="x", pady=(0, 2))

        bottom = tk.Frame(page, bg=u.BG, padx=14, pady=6)
        bottom.pack(fill="x")
        left = tk.Frame(bottom, bg=u.BG)
        left.pack(side="left", fill="both", expand=True)
        self._legend_title = tk.Label(
            left,
            text="Legend - Playoff Odds",
            bg=u.BG,
            fg=u.FG,
            font=("Segoe UI", 10, "bold"),
        )
        self._legend_title.pack(anchor="w")
        self._legend = tk.Frame(left, bg=u.BG)
        self._legend.pack(anchor="w", fill="x")

        right = tk.Frame(bottom, bg=u.BG)
        right.pack(side="right", fill="both", expand=True, padx=(12, 0))

        self._inj_block = tk.Frame(right, bg=u.BG)
        self._inj_title = tk.Label(
            self._inj_block,
            text="Injury events",
            bg=u.BG,
            fg=u.FG,
            font=("Segoe UI", 10, "bold"),
        )
        self._inj_title.pack(anchor="w")
        self._inj_list = tk.Listbox(
            self._inj_block,
            height=8,
            width=70,
            bg=u.BG_FIELD,
            fg=u.FG,
            selectbackground=u.ACCENT_BTN,
            font=("Segoe UI", 9),
            activestyle="none",
            highlightthickness=0,
            borderwidth=0,
        )
        self._inj_list.pack(fill="x", expand=False, pady=(4, 8))

        self._trade_block = tk.Frame(right, bg=u.BG)
        self._trade_title = tk.Label(
            self._trade_block,
            text="Trade events",
            bg=u.BG,
            fg=u.FG,
            font=("Segoe UI", 10, "bold"),
        )
        self._trade_title.pack(anchor="w")
        self._trade_list = tk.Listbox(
            self._trade_block,
            height=8,
            width=70,
            bg=u.BG_FIELD,
            fg=u.FG,
            selectbackground=u.ACCENT_BTN,
            font=("Segoe UI", 9),
            activestyle="none",
            highlightthickness=0,
            borderwidth=0,
        )
        self._trade_list.pack(fill="x", expand=False, pady=(4, 12))

        self._sync_event_blocks()
        self._load()

    def _sync_event_blocks(self) -> None:
        """Pack injury then trade lists when their toggles are on."""
        self._inj_block.pack_forget()
        self._trade_block.pack_forget()
        if self._show_injuries.get():
            self._inj_block.pack(fill="x", anchor="n")
        if self._show_trades.get():
            self._trade_block.pack(fill="x", anchor="n")

    def _on_toggle_markers(self) -> None:
        self._sync_event_blocks()
        self._redraw()

    def _mk_label(self, parent: tk.Misc, text: str) -> tk.Label:
        u = _ui()
        return tk.Label(
            parent, text=text, bg=u.BG, fg=u.FG, font=("Segoe UI", 9, "bold")
        )

    def _load(self) -> None:
        odds_path = CSV_OUTPUT_DIR / LEAGUE_WEEKLY_ODDS_CSV
        inj_path = CSV_OUTPUT_DIR / LEAGUE_INJURY_EVENTS_CSV
        trade_path = CSV_OUTPUT_DIR / LEAGUE_TRADE_EVENTS_CSV
        if not odds_path.exists():
            self._tip_label.configure(
                text=f"Missing {odds_path.name} — run weekly odds first.",
                fg=_ui().BUST_COLOR,
            )
            return
        self._odds = pd.read_csv(odds_path)
        self._odds["team_name"] = self._odds["team_name"].map(_ascii)
        self._odds.loc[self._odds["team_name"] == "", "team_name"] = (
            "Team " + self._odds["team_id"].astype(str)
        )
        self._odds.loc[
            (self._odds["season"] == 2020) & (self._odds["team_id"] == 8), "team_name"
        ] = "(table flip)"
        if inj_path.exists():
            self._inj = pd.read_csv(inj_path)
            if "label" in self._inj.columns:
                self._inj["label"] = self._inj["label"].map(_ascii)
        if trade_path.exists():
            self._trades = pd.read_csv(trade_path)
            if "label" in self._trades.columns:
                self._trades["label"] = self._trades["label"].map(_ascii)
        seasons = sorted(int(s) for s in self._odds["season"].unique())
        self._season_cb["values"] = [str(s) for s in seasons]
        if seasons:
            self._season_var.set(str(seasons[-1]))
            self._on_season()

    def _on_season(self) -> None:
        season = int(self._season_var.get())
        sdf = self._odds[self._odds["season"] == season]
        teams = (
            sdf.sort_values("team_id")[["team_id", "team_name"]]
            .drop_duplicates("team_id")
            .itertuples(index=False)
        )
        names = ["All teams"] + [f"{tid}: {_ascii(name)}" for tid, name in teams]
        self._team_cb["values"] = names
        self._team_var.set("All teams")
        self._redraw()

    def _filtered(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[tuple[int, str]]]:
        season = int(self._season_var.get())
        sdf = self._odds[self._odds["season"] == season].copy()
        teams = (
            sdf.sort_values("team_id")[["team_id", "team_name"]]
            .drop_duplicates("team_id")
            .itertuples(index=False)
        )
        team_list = [(int(tid), _ascii(name)) for tid, name in teams]
        sel = self._team_var.get()
        tid_filter: int | None = None
        if sel != "All teams" and ":" in sel:
            tid_filter = int(sel.split(":", 1)[0])
            sdf = sdf[sdf["team_id"] == tid_filter]
            team_list = [t for t in team_list if t[0] == tid_filter]
        inj = (
            self._inj[self._inj["season"] == season].copy()
            if not self._inj.empty
            else pd.DataFrame()
        )
        if not inj.empty and tid_filter is not None:
            inj = inj[inj["team_id"] == tid_filter]
        trades = (
            self._trades[self._trades["season"] == season].copy()
            if not self._trades.empty
            else pd.DataFrame()
        )
        if not trades.empty and tid_filter is not None:
            trades = trades[trades["team_id"] == tid_filter]
        return sdf, inj, trades, team_list

    def _redraw(self) -> None:
        if self._odds.empty or not self._season_var.get():
            return
        u = _ui()
        self.canvas.delete("all")
        sdf, inj, trades, team_list = self._filtered()
        metric_col = (
            "playoff_odds" if self._metric_var.get().startswith("Playoff") else "title_odds"
        )
        weeks = sorted(int(w) for w in sdf["as_of_week"].unique())
        if not weeks or not team_list:
            return

        w = WEEKLY_CHART_W
        h = WEEKLY_CHART_H
        pad_l, pad_r, pad_t, pad_b = 48, 16, 18, 34
        plot_w = w - pad_l - pad_r
        plot_h = h - pad_t - pad_b

        def x_at(i: int) -> float:
            if len(weeks) <= 1:
                return pad_l + plot_w / 2
            return pad_l + (i / (len(weeks) - 1)) * plot_w

        def y_at(v: float) -> float:
            return pad_t + plot_h * (1.0 - max(0.0, min(1.0, v)))

        for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
            y = y_at(tick)
            self.canvas.create_line(
                pad_l, y, pad_l + plot_w, y, fill=u.TAB_BAR_BG, width=1
            )
            self.canvas.create_text(
                pad_l - 8,
                y,
                text=f"{int(tick * 100)}%",
                fill=u.MUTED_FG,
                font=("Segoe UI", 8),
                anchor="e",
            )
        for i, week in enumerate(weeks):
            self.canvas.create_text(
                x_at(i),
                h - 12,
                text=f"W{week}",
                fill=u.MUTED_FG,
                font=("Segoe UI", 8),
            )

        series_pts: dict[int, list[float]] = {}
        name_by_id = {tid: name for tid, name in team_list}
        color_by_id: dict[int, str] = {}
        for si, (tid, _name) in enumerate(team_list):
            color = SERIES_COLORS[si % len(SERIES_COLORS)]
            color_by_id[tid] = color
            tdf = sdf[sdf["team_id"] == tid].set_index("as_of_week")
            vals = [float(tdf.loc[week, metric_col]) for week in weeks]
            series_pts[tid] = vals
            pts: list[float] = []
            for i, v in enumerate(vals):
                pts.extend([x_at(i), y_at(v)])
            if len(pts) >= 4:
                self.canvas.create_line(*pts, fill=color, width=2, smooth=False)
            for i, v in enumerate(vals):
                r = 2.5
                self.canvas.create_oval(
                    x_at(i) - r,
                    y_at(v) - r,
                    x_at(i) + r,
                    y_at(v) + r,
                    fill=color,
                    outline="",
                )

        injury_hits: list[tuple[float, float, float, str]] = []
        if self._show_injuries.get() and not inj.empty:
            week_index = {week: i for i, week in enumerate(weeks)}
            grouped = (
                inj.sort_values(
                    ["team_id", "as_of_week", "overall_pick"], na_position="last"
                )
                .groupby(["team_id", "as_of_week"], as_index=False)
                .agg(label=("label", lambda s: " · ".join(str(x) for x in s)))
            )
            for row in grouped.itertuples(index=False):
                tid = int(row.team_id)
                week = int(row.as_of_week)
                if tid not in series_pts or week not in week_index:
                    continue
                wi = week_index[week]
                val = series_pts[tid][wi]
                cx, cy = x_at(wi), y_at(val)
                tip = f"{name_by_id.get(tid, tid)}: {row.label}"
                self._draw_bandaid(cx, cy)
                injury_hits.append((cx, cy, 14.0, tip))

        trade_hits: list[tuple[float, float, float, str]] = []
        if self._show_trades.get() and not trades.empty:
            week_index = {week: i for i, week in enumerate(weeks)}
            # Offset trade markers slightly above injury bandaids when both share a week
            grouped = (
                trades.sort_values(
                    ["team_id", "as_of_week", "overall_pick"], na_position="last"
                )
                .groupby(["team_id", "as_of_week"], as_index=False)
                .agg(label=("label", lambda s: " · ".join(str(x) for x in s)))
            )
            for row in grouped.itertuples(index=False):
                tid = int(row.team_id)
                week = int(row.as_of_week)
                if tid not in series_pts or week not in week_index:
                    continue
                wi = week_index[week]
                val = series_pts[tid][wi]
                cx, cy = x_at(wi), y_at(val) - 14
                tip = f"{name_by_id.get(tid, tid)}: {row.label}"
                self._draw_trade_marker(cx, cy)
                trade_hits.append((cx, cy, 14.0, tip))

        self._plot_meta = {
            "weeks": weeks,
            "x_at": x_at,
            "pad_l": pad_l,
            "pad_t": pad_t,
            "plot_w": plot_w,
            "plot_h": plot_h,
            "series_pts": series_pts,
            "team_list": team_list,
            "name_by_id": name_by_id,
            "color_by_id": color_by_id,
            "injury_hits": injury_hits,
            "trade_hits": trade_hits,
            "metric_label": (
                "Playoff odds (%)"
                if metric_col == "playoff_odds"
                else "Title odds (%)"
            ),
        }
        n_inj = len(inj) if self._show_injuries.get() and not inj.empty else 0
        n_tr = len(trades) if self._show_trades.get() and not trades.empty else 0
        self._counts.configure(
            text=f"{len(weeks)} checkpoints · {n_inj} injuries · {n_tr} trades"
        )
        self._build_legend(team_list, color_by_id)
        self._fill_injuries(inj, name_by_id)
        self._fill_trades(trades, name_by_id)
        self._apply_hover()

    def _draw_bandaid(self, cx: float, cy: float) -> None:
        u = _ui()
        w, h = 16, 9
        x0, y0 = cx - w / 2, cy - h / 2
        self.canvas.create_rectangle(
            x0, y0, x0 + w, y0 + h, fill=u.IMPORTANT_FG, outline=u.FG, width=1
        )
        pad = 4
        self.canvas.create_rectangle(
            cx - pad / 2,
            y0,
            cx + pad / 2,
            y0 + h,
            fill=u.BG_FIELD,
            outline=u.FG,
            width=1,
        )

    def _draw_trade_marker(self, cx: float, cy: float) -> None:
        """Swap arrows icon for trade/transfer weeks."""
        u = _ui()
        # Circular badge
        r = 8
        self.canvas.create_oval(
            cx - r,
            cy - r,
            cx + r,
            cy + r,
            fill=u.BOOM_COLOR,
            outline=u.FG,
            width=1,
        )
        # Left→right arrow (top)
        self.canvas.create_line(
            cx - 5, cy - 3, cx + 5, cy - 3, fill=u.BG, width=1.5, arrow=tk.LAST
        )
        # Right→left arrow (bottom)
        self.canvas.create_line(
            cx + 5, cy + 3, cx - 5, cy + 3, fill=u.BG, width=1.5, arrow=tk.LAST
        )

    def _build_legend(
        self,
        team_list: list[tuple[int, str]],
        color_by_id: dict[int, str],
    ) -> None:
        """Create legend rows once; percentages are updated in place on hover."""
        u = _ui()
        for child in self._legend.winfo_children():
            child.destroy()
        self._legend_pct: dict[int, tk.Label] = {}
        for tid, name in team_list:
            line = tk.Frame(self._legend, bg=u.BG)
            line.pack(fill="x", pady=1)
            sw = tk.Canvas(line, width=10, height=10, bg=u.BG, highlightthickness=0)
            sw.pack(side="left", pady=2)
            sw.create_rectangle(0, 0, 10, 10, fill=color_by_id[tid], outline="")
            tk.Label(
                line,
                text=name,
                bg=u.BG,
                fg=u.FG,
                font=("Segoe UI", 9),
                anchor="w",
            ).pack(side="left", padx=(6, 8))
            pct = tk.Label(
                line,
                text="",
                bg=u.BG,
                fg=u.IMPORTANT_FG,
                font=("Segoe UI", 9, "bold"),
            )
            pct.pack(side="right")
            self._legend_pct[tid] = pct

    def _refresh_legend_values(self) -> None:
        meta = self._plot_meta
        if not meta:
            return
        weeks = meta["weeks"]
        series_pts = meta["series_pts"]
        playoff = self._metric_var.get().startswith("Playoff")
        week_i = self._hover_week
        if week_i is not None and 0 <= week_i < len(weeks):
            week_tag = f"W{weeks[week_i]}"
            i = week_i
        elif weeks:
            week_tag = f"W{weeks[-1]}"
            i = len(weeks) - 1
        else:
            week_tag = ""
            i = 0
        base = "Legend - Playoff Odds" if playoff else "Legend - Title Odds"
        self._legend_title.configure(
            text=f"{base} ({week_tag})" if week_tag else base
        )
        for tid, label in getattr(self, "_legend_pct", {}).items():
            vals = series_pts.get(tid) or []
            if not vals:
                label.configure(text="")
                continue
            idx = i if 0 <= i < len(vals) else len(vals) - 1
            label.configure(text=f"{vals[idx] * 100.0:.1f}%")

    def _set_hover_guide(self) -> None:
        u = _ui()
        meta = self._plot_meta
        self.canvas.delete("hover_guide")
        if not meta or self._hover_week is None:
            return
        weeks = meta["weeks"]
        if not (0 <= self._hover_week < len(weeks)):
            return
        x = meta["x_at"](self._hover_week)
        self.canvas.create_line(
            x,
            meta["pad_t"],
            x,
            meta["pad_t"] + meta["plot_h"],
            fill=u.MUTED_FG,
            width=1,
            dash=(4, 3),
            tags=("hover_guide",),
        )
        self.canvas.tag_raise("hover_guide")

    def _fill_injuries(
        self, inj: pd.DataFrame, name_by_id: dict[int, str]
    ) -> None:
        self._inj_list.delete(0, tk.END)
        if not self._show_injuries.get():
            return
        if inj.empty:
            self._inj_list.insert(tk.END, "No injury markers for this view")
            return
        rows = inj.sort_values(["as_of_week", "team_id"])
        for row in rows.itertuples(index=False):
            week = int(row.as_of_week)
            team = name_by_id.get(int(row.team_id), str(row.team_id))
            self._inj_list.insert(tk.END, f"W{week} · {team}: {row.label}")

    def _fill_trades(
        self, trades: pd.DataFrame, name_by_id: dict[int, str]
    ) -> None:
        self._trade_list.delete(0, tk.END)
        if not self._show_trades.get():
            return
        if trades.empty:
            self._trade_list.insert(tk.END, "No trade markers for this view")
            return
        # Labels use full fantasy team names from league_teams.csv.
        rows = trades.sort_values(["as_of_week", "team_id", "direction"])
        for row in rows.itertuples(index=False):
            week = int(row.as_of_week)
            team = name_by_id.get(int(row.team_id), str(row.team_id))
            label = str(row.label)
            self._trade_list.insert(tk.END, f"W{week} · {team}: {label}")

    def _apply_hover(self) -> None:
        self._set_hover_guide()
        self._refresh_legend_values()
        self._update_tip()

    def _on_motion(self, event: tk.Event) -> None:
        meta = self._plot_meta
        if not meta:
            return
        tip = None
        hits = list(meta.get("injury_hits", [])) + list(meta.get("trade_hits", []))
        for cx, cy, r, t in hits:
            if (event.x - cx) ** 2 + (event.y - cy) ** 2 <= r * r:
                tip = t
                break

        weeks = meta["weeks"]
        x_at = meta["x_at"]
        pad_l = meta["pad_l"]
        plot_w = meta["plot_w"]
        if event.x < pad_l or event.x > pad_l + plot_w:
            week = None
        else:
            best_i, best_d = 0, float("inf")
            for i in range(len(weeks)):
                d = abs(x_at(i) - event.x)
                if d < best_d:
                    best_d, best_i = d, i
            week = best_i

        if week == self._hover_week and tip == self._injury_tip:
            return
        self._hover_week = week
        self._injury_tip = tip
        self._apply_hover()

    def _on_leave(self, _event: tk.Event) -> None:
        if self._hover_week is None and self._injury_tip is None:
            return
        self._hover_week = None
        self._injury_tip = None
        self._apply_hover()

    def _update_tip(self) -> None:
        u = _ui()
        if self._injury_tip:
            self._tip_label.configure(
                text=self._injury_tip,
                fg=u.IMPORTANT_FG,
                font=("Segoe UI", 10, "bold"),
            )
        else:
            self._tip_label.configure(
                text="Hover a marker for injury/trade detail · hover the chart to update legend week",
                fg=u.MUTED_FG,
                font=("Segoe UI", 9),
            )


class FinishVsActualPanel(tk.Frame):
    """Projected vs actual finishes (Henry + ESPN BOS)."""

    def __init__(self, master: tk.Misc) -> None:
        u = _ui()
        super().__init__(master, bg=u.BG)
        self._df = pd.DataFrame()

        scroll = LeagueScrollFrame(self)
        scroll.pack(fill="both", expand=True)
        page = scroll.inner
        self._scroll = scroll

        header = tk.Frame(page, bg=u.BG, padx=14, pady=10)
        header.pack(fill="x")
        tk.Label(
            header,
            text="Projected vs actual finishes (RS + playoffs)",
            bg=u.BG,
            fg=u.FG,
            font=("Segoe UI", 14, "bold"),
        ).pack(anchor="w")
        tk.Label(
            header,
            text=(
                "Henry blend Monte Carlo vs ESPN draft-day projected rank vs actual "
                "regular-season seed and final playoff place. Seasons 2018–2025."
            ),
            bg=u.BG,
            fg=u.MUTED_FG,
            font=("Segoe UI", 9),
            wraplength=1000,
            justify="left",
        ).pack(anchor="w", pady=(4, 0))

        controls = tk.Frame(page, bg=u.BG, padx=14, pady=4)
        controls.pack(fill="x")
        tk.Label(
            controls, text="Season", bg=u.BG, fg=u.FG, font=("Segoe UI", 9, "bold")
        ).pack(side="left")
        self._season_var = tk.StringVar(value="All")
        self._season_cb = ttk.Combobox(
            controls, textvariable=self._season_var, width=8, state="readonly"
        )
        self._season_cb.pack(side="left", padx=(4, 12))
        self._season_cb.bind("<<ComboboxSelected>>", lambda _e: self._refresh())

        self._stats = tk.Frame(page, bg=u.BG, padx=14, pady=6)
        self._stats.pack(fill="x")

        note = tk.Label(
            page,
            text=(
                "ESPN BOS = draftDayProjectedRank (2021–2025). Henry RS = ranked mean "
                "simulated seed. Actual RS = playoff seed. Henry playoff = ranked by "
                "title then playoff odds. Actual playoff = final_rank."
            ),
            bg=u.BG,
            fg=u.MUTED_FG,
            font=("Segoe UI", 8),
            wraplength=1000,
            justify="left",
            padx=14,
        )
        note.pack(fill="x")

        chart_host = tk.Frame(page, bg=u.BG, padx=14)
        chart_host.pack(fill="x", pady=(8, 4))
        self._chart = tk.Canvas(
            chart_host,
            width=WEEKLY_CHART_W,
            height=FINISH_CHART_H,
            bg=u.BG_FIELD,
            highlightthickness=0,
        )
        self._chart.pack(anchor="w")

        table_host = tk.Frame(page, bg=u.BG, padx=14, pady=8)
        table_host.pack(fill="x")
        cols = (
            "year",
            "team",
            "espn",
            "henry_rs",
            "act_rs",
            "rs_d",
            "henry_po",
            "act_po",
            "po_d",
            "sim_w",
            "act_w",
            "po_pct",
            "title_pct",
        )
        self._tree = ttk.Treeview(
            table_host,
            columns=cols,
            show="headings",
            selectmode="browse",
            height=18,
        )
        headings = {
            "year": "Year",
            "team": "Team",
            "espn": "ESPN BOS",
            "henry_rs": "Henry RS",
            "act_rs": "Act RS",
            "rs_d": "RS Δ",
            "henry_po": "Henry PO",
            "act_po": "Act PO",
            "po_d": "PO Δ",
            "sim_w": "Sim W",
            "act_w": "Act W",
            "po_pct": "PO%",
            "title_pct": "Title%",
        }
        widths = {
            "year": 50,
            "team": 200,
            "espn": 70,
            "henry_rs": 70,
            "act_rs": 60,
            "rs_d": 50,
            "henry_po": 70,
            "act_po": 60,
            "po_d": 50,
            "sim_w": 55,
            "act_w": 55,
            "po_pct": 55,
            "title_pct": 60,
        }
        for c in cols:
            self._tree.heading(c, text=headings[c])
            self._tree.column(
                c,
                width=widths[c],
                anchor="e" if c != "team" else "w",
                stretch=c == "team",
            )
        scroll_y = ttk.Scrollbar(table_host, orient="vertical", command=self._tree.yview)
        scroll_x = ttk.Scrollbar(
            table_host, orient="horizontal", command=self._tree.xview
        )
        self._tree.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
        self._tree.grid(row=0, column=0, sticky="nsew")
        scroll_y.grid(row=0, column=1, sticky="ns")
        scroll_x.grid(row=1, column=0, sticky="ew")
        table_host.columnconfigure(0, weight=1)

        # Spacer so the last rows aren't flush against the window edge
        tk.Frame(page, bg=u.BG, height=16).pack(fill="x")

        self._load()

    def _load(self) -> None:
        path = CSV_OUTPUT_DIR / LEAGUE_FINISH_PROJ_VS_ACTUAL_CSV
        if not path.exists():
            tk.Label(
                self._stats,
                text=f"Missing {path.name}",
                bg=_ui().BG,
                fg=_ui().BUST_COLOR,
            ).pack(anchor="w")
            return
        self._df = pd.read_csv(path)
        self._df["team"] = self._df["team"].map(_ascii)
        seasons = sorted(int(s) for s in self._df["season"].unique())
        self._season_cb["values"] = ["All"] + [str(s) for s in seasons]
        self._season_var.set("All")
        self._refresh()

    @staticmethod
    def _delta(proj: object, act: object) -> str:
        try:
            if proj is None or act is None or (isinstance(proj, float) and math.isnan(proj)):
                return "—"
            d = int(act) - int(proj)
            if d == 0:
                return "exact"
            return f"+{d}" if d > 0 else str(d)
        except (TypeError, ValueError):
            return "—"

    def _refresh(self) -> None:
        u = _ui()
        for child in self._stats.winfo_children():
            child.destroy()
        if self._df.empty:
            return
        season = self._season_var.get()
        df = self._df if season == "All" else self._df[self._df["season"] == int(season)]
        our_rs = float(df["err_our_rs"].mean()) if len(df) else 0.0
        our_po = float(df["err_our_po"].mean()) if len(df) else 0.0
        with_espn = df[df["err_espn_rs"].notna()]
        espn_rs = float(with_espn["err_espn_rs"].mean()) if len(with_espn) else None

        overlap = self._df[self._df["err_espn_rs"].notna()]
        ov_our = float(overlap["err_our_rs"].mean()) if len(overlap) else float("nan")
        ov_espn = float(overlap["err_espn_rs"].mean()) if len(overlap) else float("nan")
        closer = 0.0
        if len(overlap):
            o = overlap.dropna(subset=["err_our_rs", "err_espn_rs"])
            if len(o):
                closer = 100.0 * float((o["err_our_rs"] < o["err_espn_rs"]).mean())

        stats = [
            (f"{our_rs:.2f}", "Henry RS MAE"),
            ("—" if espn_rs is None else f"{espn_rs:.2f}", "ESPN BOS RS MAE"),
            (f"{our_po:.2f}", "Henry playoff MAE"),
            (season if season != "All" else f"{df['season'].nunique()} yrs", "View"),
            (f"{ov_our:.2f}", "Henry RS MAE (2021–25)"),
            (f"{ov_espn:.2f}", "ESPN RS MAE (2021–25)"),
            (f"{closer:.0f}%", "Henry closer than ESPN"),
        ]
        for i, (val, lab) in enumerate(stats):
            cell = tk.Frame(self._stats, bg=u.BG_FIELD, padx=10, pady=6)
            cell.grid(row=i // 4, column=i % 4, sticky="ew", padx=4, pady=4)
            tk.Label(
                cell, text=val, bg=u.BG_FIELD, fg=u.IMPORTANT_FG, font=("Segoe UI", 14, "bold")
            ).pack(anchor="w")
            tk.Label(
                cell, text=lab, bg=u.BG_FIELD, fg=u.MUTED_FG, font=("Segoe UI", 8)
            ).pack(anchor="w")
        for c in range(4):
            self._stats.columnconfigure(c, weight=1)

        for item in self._tree.get_children():
            self._tree.delete(item)
        for row in df.sort_values(["season", "our_rs"]).itertuples(index=False):
            espn = row.espn_bos
            espn_s = "—" if espn is None or (isinstance(espn, float) and math.isnan(espn)) else str(int(espn))
            self._tree.insert(
                "",
                "end",
                values=(
                    int(row.season),
                    _ascii(row.team),
                    espn_s,
                    int(row.our_rs),
                    int(row.act_rs),
                    self._delta(row.our_rs, row.act_rs),
                    int(row.our_po),
                    int(row.act_po),
                    self._delta(row.our_po, row.act_po),
                    f"{float(row.sim_wins_mean):.1f}",
                    int(row.wins),
                    f"{100 * float(row.playoff_odds):.0f}%",
                    f"{100 * float(row.title_odds):.0f}%",
                ),
            )
        self._draw_mae_chart()

    def _draw_mae_chart(self) -> None:
        u = _ui()
        cv = self._chart
        cv.delete("all")
        if self._df.empty:
            return
        by = (
            self._df.groupby("season")[["err_our_rs", "err_our_po"]]
            .mean()
            .sort_index()
        )
        years = [int(y) for y in by.index]
        rs = [float(v) for v in by["err_our_rs"]]
        po = [float(v) for v in by["err_our_po"]]
        w = WEEKLY_CHART_W
        h = FINISH_CHART_H
        pad_l, pad_r, pad_t, pad_b = 40, 16, 28, 28
        plot_w = w - pad_l - pad_r
        plot_h = h - pad_t - pad_b
        ymax = max(rs + po + [1.0]) * 1.15
        n = len(years)
        group_w = plot_w / max(n, 1)
        bar_w = group_w * 0.32
        cv.create_text(
            pad_l,
            12,
            text="Henry MAE by season (RS vs playoffs)",
            fill=u.FG,
            font=("Segoe UI", 10, "bold"),
            anchor="w",
        )
        cv.create_rectangle(w - 160, 8, w - 148, 20, fill="#7EC8FF", outline="")
        cv.create_text(w - 144, 14, text="RS", fill=u.MUTED_FG, font=("Segoe UI", 8), anchor="w")
        cv.create_rectangle(w - 110, 8, w - 98, 20, fill=u.IMPORTANT_FG, outline="")
        cv.create_text(w - 94, 14, text="Playoff", fill=u.MUTED_FG, font=("Segoe UI", 8), anchor="w")
        for i, year in enumerate(years):
            x0 = pad_l + i * group_w + group_w * 0.18
            h_rs = (rs[i] / ymax) * plot_h
            h_po = (po[i] / ymax) * plot_h
            y_base = pad_t + plot_h
            cv.create_rectangle(
                x0, y_base - h_rs, x0 + bar_w, y_base, fill="#7EC8FF", outline=""
            )
            cv.create_rectangle(
                x0 + bar_w + 2,
                y_base - h_po,
                x0 + 2 * bar_w + 2,
                y_base,
                fill=u.IMPORTANT_FG,
                outline="",
            )
            cv.create_text(
                x0 + bar_w,
                h - 10,
                text=str(year),
                fill=u.MUTED_FG,
                font=("Segoe UI", 8),
            )


class DraftReachesPanel(tk.Frame):
    """Biggest ADP reaches / falls (top 20 abs gaps per season)."""

    def __init__(self, master: tk.Misc) -> None:
        u = _ui()
        super().__init__(master, bg=u.BG)
        self._df = pd.DataFrame()

        scroll = LeagueScrollFrame(self)
        scroll.pack(fill="both", expand=True)
        page = scroll.inner
        self._scroll = scroll

        header = tk.Frame(page, bg=u.BG, padx=14, pady=10)
        header.pack(fill="x")
        tk.Label(
            header,
            text="Biggest Draft Reaches / Falls",
            bg=u.BG,
            fg=u.FG,
            font=("Segoe UI", 14, "bold"),
        ).pack(anchor="w")
        tk.Label(
            header,
            text=(
                "Per season: the 20 skill picks with the largest |FantasyPros ADP − "
                "overall pick| among players with ADP ≤ 122 and drafted overall pick "
                "≤ 99. Sorted greatest reach (drafted earlier than ADP) → greatest "
                "fall. Default view stacks all seasons (160 rows)."
            ),
            bg=u.BG,
            fg=u.MUTED_FG,
            font=("Segoe UI", 9),
            wraplength=1000,
            justify="left",
        ).pack(anchor="w", pady=(4, 0))

        controls = tk.Frame(page, bg=u.BG, padx=14, pady=4)
        controls.pack(fill="x")
        tk.Label(
            controls, text="Season", bg=u.BG, fg=u.FG, font=("Segoe UI", 9, "bold")
        ).pack(side="left")
        self._season_var = tk.StringVar(value="All")
        self._season_cb = ttk.Combobox(
            controls, textvariable=self._season_var, width=8, state="readonly"
        )
        self._season_cb.pack(side="left", padx=(4, 16))
        self._season_cb.bind("<<ComboboxSelected>>", lambda _e: self._refresh())

        tk.Label(
            controls, text="Manager", bg=u.BG, fg=u.FG, font=("Segoe UI", 9, "bold")
        ).pack(side="left")
        self._mgr_var = tk.StringVar(value="All managers")
        self._mgr_cb = ttk.Combobox(
            controls, textvariable=self._mgr_var, width=22, state="readonly"
        )
        self._mgr_cb.pack(side="left", padx=(4, 12))
        self._mgr_cb.bind("<<ComboboxSelected>>", lambda _e: self._refresh())

        self._count_label = tk.Label(
            controls, text="", bg=u.BG, fg=u.MUTED_FG, font=("Segoe UI", 9)
        )
        self._count_label.pack(side="left", padx=(8, 0))

        note = tk.Label(
            page,
            text=(
                "Delta = ADP rank − drafted overall pick. Positive = reach; "
                "negative = fall. Finish = season PPR rank at position (WR3). "
                "Avg PPG = season PPR / games."
            ),
            bg=u.BG,
            fg=u.MUTED_FG,
            font=("Segoe UI", 8),
            wraplength=1000,
            justify="left",
            padx=14,
        )
        note.pack(fill="x", pady=(4, 0))

        table_host = tk.Frame(page, bg=u.BG, padx=14, pady=8)
        table_host.pack(fill="x")
        cols = (
            "season",
            "player",
            "pos",
            "manager",
            "adp",
            "drafted",
            "delta",
            "finish",
            "ppg",
        )
        self._tree = ttk.Treeview(
            table_host,
            columns=cols,
            show="headings",
            selectmode="browse",
            height=22,
        )
        headings = {
            "season": "Year",
            "player": "Player",
            "pos": "Pos",
            "manager": "Manager",
            "adp": "ADP",
            "drafted": "Drafted",
            "delta": "Δ Draft",
            "finish": "Finish",
            "ppg": "Avg PPG",
        }
        widths = {
            "season": 56,
            "player": 160,
            "pos": 44,
            "manager": 130,
            "adp": 56,
            "drafted": 70,
            "delta": 70,
            "finish": 64,
            "ppg": 72,
        }
        for c in cols:
            self._tree.heading(c, text=headings[c])
            self._tree.column(
                c,
                width=widths[c],
                anchor="w" if c in ("player", "manager", "finish") else "e",
                stretch=c == "player",
            )
        scroll_y = ttk.Scrollbar(table_host, orient="vertical", command=self._tree.yview)
        scroll_x = ttk.Scrollbar(
            table_host, orient="horizontal", command=self._tree.xview
        )
        self._tree.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
        self._tree.grid(row=0, column=0, sticky="nsew")
        scroll_y.grid(row=0, column=1, sticky="ns")
        scroll_x.grid(row=1, column=0, sticky="ew")
        table_host.columnconfigure(0, weight=1)

        tk.Frame(page, bg=u.BG, height=16).pack(fill="x")
        self._load()

    def _load(self) -> None:
        path = CSV_OUTPUT_DIR / LEAGUE_DRAFT_REACHES_CSV
        if not path.exists():
            self._count_label.configure(
                text=f"Missing {path.name} — run: python -m rookie_ppr.league.draft_reaches",
                fg=_ui().BUST_COLOR,
            )
            return
        self._df = pd.read_csv(path)
        for col in ("player_name", "manager_name", "team_name", "finish_label"):
            if col in self._df.columns:
                self._df[col] = self._df[col].map(
                    lambda v: _ascii(v) if pd.notna(v) else ""
                )
        seasons = sorted(int(s) for s in self._df["season"].unique())
        self._season_cb["values"] = ["All"] + [str(s) for s in seasons]
        self._season_var.set("All")
        managers = sorted(
            {
                _ascii(m)
                for m in self._df["manager_name"].dropna().unique()
                if _ascii(m)
            }
        )
        self._mgr_cb["values"] = ["All managers"] + managers
        self._mgr_var.set("All managers")
        self._refresh()

    def _refresh(self) -> None:
        for item in self._tree.get_children():
            self._tree.delete(item)
        if self._df.empty:
            self._count_label.configure(text="No rows", fg=_ui().MUTED_FG)
            return

        df = self._df
        season = self._season_var.get()
        if season != "All":
            df = df[df["season"] == int(season)]
        mgr = self._mgr_var.get()
        if mgr != "All managers":
            df = df[df["manager_name"] == mgr]

        df = df.sort_values(
            ["draft_delta", "abs_delta", "season", "overall_pick"],
            ascending=[False, False, True, True],
        )
        self._count_label.configure(
            text=f"{len(df)} picks",
            fg=_ui().MUTED_FG,
        )

        for row in df.itertuples(index=False):
            adp = float(row.adp_rank)
            pick = int(row.overall_pick)
            delta = float(row.draft_delta)
            delta_s = f"+{delta:.0f}" if delta > 0 else f"{delta:.0f}"
            ppg = row.avg_ppg
            if ppg is None or (isinstance(ppg, float) and math.isnan(ppg)):
                ppg_s = "—"
            else:
                ppg_s = f"{float(ppg):.1f}"
            finish = row.finish_label if row.finish_label and row.finish_label != "nan" else "—"
            self._tree.insert(
                "",
                "end",
                values=(
                    int(row.season),
                    _ascii(row.player_name),
                    str(row.position),
                    _ascii(row.manager_name),
                    f"{adp:.0f}",
                    pick,
                    delta_s,
                    finish,
                    ppg_s,
                ),
            )


class DraftHistoryPanel(tk.Frame):
    """Full league draft board by season, filterable by manager."""

    # A complete snake draft for this league is 10 teams x 16 rounds.
    FULL_DRAFT_PICKS = 160

    def __init__(self, master: tk.Misc) -> None:
        u = _ui()
        super().__init__(master, bg=u.BG)
        self._df = pd.DataFrame()
        self._full_seasons: list[int] = []

        scroll = LeagueScrollFrame(self)
        scroll.pack(fill="both", expand=True)
        page = scroll.inner
        self._scroll = scroll

        header = tk.Frame(page, bg=u.BG, padx=14, pady=10)
        header.pack(fill="x")
        tk.Label(
            header,
            text="Draft History",
            bg=u.BG,
            fg=u.FG,
            font=("Segoe UI", 14, "bold"),
        ).pack(anchor="w")
        tk.Label(
            header,
            text=(
                "Every pick from a full league draft. Defaults to the most recent "
                "completed season. Filter by manager to see one franchise's board."
            ),
            bg=u.BG,
            fg=u.MUTED_FG,
            font=("Segoe UI", 9),
            wraplength=1000,
            justify="left",
        ).pack(anchor="w", pady=(4, 0))

        controls = tk.Frame(page, bg=u.BG, padx=14, pady=4)
        controls.pack(fill="x")
        tk.Label(
            controls, text="Season", bg=u.BG, fg=u.FG, font=("Segoe UI", 9, "bold")
        ).pack(side="left")
        self._season_var = tk.StringVar(value="")
        self._season_cb = ttk.Combobox(
            controls, textvariable=self._season_var, width=8, state="readonly"
        )
        self._season_cb.pack(side="left", padx=(4, 16))
        self._season_cb.bind("<<ComboboxSelected>>", lambda _e: self._on_season())

        tk.Label(
            controls, text="Manager", bg=u.BG, fg=u.FG, font=("Segoe UI", 9, "bold")
        ).pack(side="left")
        self._mgr_var = tk.StringVar(value="All managers")
        self._mgr_cb = ttk.Combobox(
            controls, textvariable=self._mgr_var, width=22, state="readonly"
        )
        self._mgr_cb.pack(side="left", padx=(4, 12))
        self._mgr_cb.bind("<<ComboboxSelected>>", lambda _e: self._refresh())

        self._count_label = tk.Label(
            controls, text="", bg=u.BG, fg=u.MUTED_FG, font=("Segoe UI", 9)
        )
        self._count_label.pack(side="left", padx=(8, 0))

        table_host = tk.Frame(page, bg=u.BG, padx=14, pady=8)
        table_host.pack(fill="x")
        cols = (
            "pick",
            "round",
            "player",
            "pos",
            "nfl",
            "manager",
            "team",
            "adp",
            "finish",
        )
        self._tree = ttk.Treeview(
            table_host,
            columns=cols,
            show="headings",
            selectmode="browse",
            height=22,
        )
        headings = {
            "pick": "Pick",
            "round": "Rnd",
            "player": "Player",
            "pos": "Pos",
            "nfl": "NFL",
            "manager": "Manager",
            "team": "Team",
            "adp": "ADP",
            "finish": "Finish",
        }
        widths = {
            "pick": 52,
            "round": 44,
            "player": 160,
            "pos": 44,
            "nfl": 48,
            "manager": 130,
            "team": 160,
            "adp": 52,
            "finish": 64,
        }
        for c in cols:
            self._tree.heading(c, text=headings[c])
            self._tree.column(
                c,
                width=widths[c],
                anchor="w" if c in ("player", "manager", "team", "finish") else "e",
                stretch=c in ("player", "team"),
            )
        scroll_y = ttk.Scrollbar(table_host, orient="vertical", command=self._tree.yview)
        scroll_x = ttk.Scrollbar(
            table_host, orient="horizontal", command=self._tree.xview
        )
        self._tree.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
        self._tree.grid(row=0, column=0, sticky="nsew")
        scroll_y.grid(row=0, column=1, sticky="ns")
        scroll_x.grid(row=1, column=0, sticky="ew")
        table_host.columnconfigure(0, weight=1)

        tk.Frame(page, bg=u.BG, height=16).pack(fill="x")
        self._load()

    def _load(self) -> None:
        path = CSV_OUTPUT_DIR / LEAGUE_DRAFT_CSV
        if not path.exists():
            self._count_label.configure(
                text=f"Missing {path.name}",
                fg=_ui().BUST_COLOR,
            )
            return
        draft = pd.read_csv(path)
        draft["player_name"] = draft["player_name"].map(_ascii)
        draft["position"] = draft["position"].astype(str)
        draft["pro_team"] = draft.get("pro_team", pd.Series(dtype=str)).fillna("").map(
            lambda v: _ascii(v) if pd.notna(v) else ""
        )

        managers = load_manager_map(
            sorted(int(s) for s in draft["season"].dropna().unique())
        )
        teams = pd.read_csv(CSV_OUTPUT_DIR / LEAGUE_TEAMS_CSV)
        teams = teams[["season", "team_id", "team_name", "team_abbrev"]].copy()
        teams["team_name"] = teams["team_name"].map(_ascii)
        teams["team_abbrev"] = teams["team_abbrev"].fillna("").map(
            lambda v: _ascii(v) if pd.notna(v) else ""
        )

        adp_path = CSV_OUTPUT_DIR / LEAGUE_BACKTEST_ROSTERS_CSV
        if adp_path.exists():
            adp = pd.read_csv(
                adp_path,
                usecols=["season", "team_id", "overall_pick", "adp_rank"],
            )
        else:
            adp = pd.DataFrame(
                columns=["season", "team_id", "overall_pick", "adp_rank"]
            )

        out = draft.merge(managers, on=["season", "team_id"], how="left")
        out = out.merge(teams, on=["season", "team_id"], how="left")
        out = out.merge(adp, on=["season", "team_id", "overall_pick"], how="left")
        finishes = _season_actuals()[["gsis_id", "season", "finish_label"]]
        out = out.merge(finishes, on=["gsis_id", "season"], how="left")
        out["manager_name"] = out["manager_name"].fillna(
            "Team " + out["team_id"].astype(str)
        )
        out["team_name"] = out["team_name"].fillna("").map(
            lambda v: _ascii(v) if pd.notna(v) else ""
        )
        out["finish_label"] = out["finish_label"].fillna("-").map(
            lambda v: _ascii(v) if pd.notna(v) and str(v) not in ("", "nan") else "-"
        )
        self._df = out

        counts = self._df.groupby("season").size()
        self._full_seasons = sorted(
            int(s)
            for s, n in counts.items()
            if int(n) >= self.FULL_DRAFT_PICKS
        )
        if not self._full_seasons:
            self._full_seasons = sorted(int(s) for s in self._df["season"].unique())
        self._season_cb["values"] = [str(s) for s in self._full_seasons]
        if self._full_seasons:
            self._season_var.set(str(self._full_seasons[-1]))
            self._on_season()

    def _on_season(self) -> None:
        if not self._season_var.get() or self._df.empty:
            return
        season = int(self._season_var.get())
        sdf = self._df[self._df["season"] == season]
        managers = sorted(
            {
                _ascii(m)
                for m in sdf["manager_name"].dropna().unique()
                if _ascii(m)
            }
        )
        self._mgr_cb["values"] = ["All managers"] + managers
        # Keep selection if still valid for this season
        if self._mgr_var.get() not in self._mgr_cb["values"]:
            self._mgr_var.set("All managers")
        self._refresh()

    def _refresh(self) -> None:
        for item in self._tree.get_children():
            self._tree.delete(item)
        if self._df.empty or not self._season_var.get():
            self._count_label.configure(text="No rows", fg=_ui().MUTED_FG)
            return

        season = int(self._season_var.get())
        df = self._df[self._df["season"] == season]
        mgr = self._mgr_var.get()
        if mgr != "All managers":
            df = df[df["manager_name"] == mgr]
        df = df.sort_values(["overall_pick", "round", "round_pick"])

        self._count_label.configure(
            text=f"{len(df)} picks",
            fg=_ui().MUTED_FG,
        )
        for row in df.itertuples(index=False):
            adp = getattr(row, "adp_rank", None)
            if adp is None or (isinstance(adp, float) and math.isnan(adp)):
                adp_s = "—"
            else:
                adp_s = f"{float(adp):.0f}"
            finish = getattr(row, "finish_label", None) or "-"
            if str(finish) in ("", "nan", "None"):
                finish = "-"
            self._tree.insert(
                "",
                "end",
                values=(
                    int(row.overall_pick),
                    f"{int(row.round)}.{int(row.round_pick):02d}",
                    _ascii(row.player_name),
                    str(row.position),
                    _ascii(getattr(row, "pro_team", "") or ""),
                    _ascii(row.manager_name),
                    _ascii(getattr(row, "team_name", "") or ""),
                    adp_s,
                    finish,
                ),
            )


def open_nerd_united_window(parent: tk.Misc) -> NerdUnitedWindow:
    """Open (or focus) the Crest league board window."""
    existing = getattr(parent, "_nerd_united_window", None)
    if existing is not None:
        try:
            if existing.winfo_exists():
                existing.lift()
                existing.focus_force()
                return existing
        except tk.TclError:
            pass
    win = NerdUnitedWindow(parent)
    setattr(parent, "_nerd_united_window", win)

    def _clear(_e=None):
        setattr(parent, "_nerd_united_window", None)

    win.bind("<Destroy>", _clear)
    return win
