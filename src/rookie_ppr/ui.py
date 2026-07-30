from __future__ import annotations

import math
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Callable

import numpy as np

from rookie_ppr.analysis_config import AnalysisMode, get_mode
from rookie_ppr.config import INCOMING_DRAFT_YEAR
from rookie_ppr.mascot_assets import MASCOT_DIR
from rookie_ppr.score_runner import load_players_master, player_lookup_labels, row_from_player, score_player
from rookie_ppr.league.ui_panel import CrestButton, open_nerd_united_window
from rookie_ppr.veteran.ui_panel import VeteranScorerPanel
from rookie_ppr.scoring_fields import FIELD_SPECS, GROUP_ORDER

# Chinese / cobalt blue
BG = "#0B3D91"
BG_FIELD = "#134AAD"
FG = "#FFFFFF"
IMPORTANT_FG = "#FFE566"
ACCENT_BTN = "#1E5AA8"
POPUP_BG = "#0A3580"
CHECK_FILL = "#FFFFFF"
CHECK_EMPTY = "#0B3D91"
BOOM_COLOR = "#7CFFB2"
BUST_COLOR = "#FF8A8A"
MEDIAN_COLOR = "#FFE566"
PRED_COLOR = "#FFFFFF"
BOX_COLOR = "#4A90D9"

# ESPN-style boom/bust gauge (light card on dark popup)
GAUGE_CARD = "#FFFFFF"
GAUGE_TRACK = "#E6E6E6"
GAUGE_BUST = "#E31C3D"
GAUGE_BOOM = "#0B3D91"
GAUGE_MID = "#00C853"
GAUGE_TEXT = "#1A1A1A"
GAUGE_MUTED = "#6B6B6B"

NAV_KEYS = frozenset({"Up", "Down", "Return", "Escape", "Tab"})

# Compare side popups: up to 4 in a 2×2 grid beside the main picker
MAX_COMPARE_WINDOWS = 4
COMPARE_POPUP_WIDTH = 400
COMPARE_POPUP_EST_HEIGHT = 320  # collapsed content ~315; drivers expand further
COMPARE_GRID_GAP = 8
COMPARE_POPUP_MIN_WIDTH = 280
COMPARE_POPUP_MIN_HEIGHT = 160

# Browser-style format tabs
TAB_BAR_BG = "#072A6B"
TAB_INACTIVE_BG = "#0A3578"
TAB_INACTIVE_FG = "#9BB8E8"
TAB_ACTIVE_BG = BG
TAB_ACTIVE_FG = FG
TAB_ACCENT = IMPORTANT_FG
MUTED_FG = "#B8D4FF"

BOOM_BUST_LABELS = {
    "boom": "Boom (>75 success score)",
    "bust": "Bust (<25 success score)",
    "neutral": "Neutral (25–75)",
    "unknown": "—",
}


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    h = color.lstrip("#")
    if len(h) != 6:
        return (11, 61, 145)
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _rgb_to_hex(r: float, g: float, b: float) -> str:
    return (
        f"#{int(max(0, min(255, round(r)))):02X}"
        f"{int(max(0, min(255, round(g)))):02X}"
        f"{int(max(0, min(255, round(b)))):02X}"
    )


def _blend_hex(a: str, b: str, t: float) -> str:
    ar, ag, ab = _hex_to_rgb(a)
    br, bg_, bb = _hex_to_rgb(b)
    return _rgb_to_hex(ar + (br - ar) * t, ag + (bg_ - ag) * t, ab + (bb - ab) * t)


def _luminance(color: str) -> float:
    r, g, b = _hex_to_rgb(color)
    return 0.299 * r + 0.587 * g + 0.114 * b


def _theme_colors(primary: str, secondary: str) -> dict[str, str]:
    """Build a full UI palette from a team primary / secondary pair."""
    dark_shell = _luminance(primary) < 150
    bg = primary
    fg = "#FFFFFF" if dark_shell else "#101820"
    tab_bar = _blend_hex(primary, "#000000", 0.28)
    inactive = _blend_hex(primary, "#000000" if dark_shell else "#FFFFFF", 0.12)
    field = _blend_hex(primary, "#FFFFFF" if dark_shell else "#000000", 0.14)
    popup = _blend_hex(primary, "#000000", 0.18)
    accent = _blend_hex(primary, secondary, 0.35) if dark_shell else secondary
    muted = _blend_hex(fg, primary, 0.35) if dark_shell else _blend_hex("#1A1A1A", primary, 0.25)
    important = secondary if _luminance(secondary) > 120 else "#FFE566"
    box = _blend_hex(primary, "#FFFFFF", 0.35)
    return {
        "BG": bg,
        "BG_FIELD": field,
        "FG": fg,
        "IMPORTANT_FG": important,
        "ACCENT_BTN": accent,
        "POPUP_BG": popup,
        "CHECK_FILL": "#FFFFFF",
        "CHECK_EMPTY": bg,
        "BOX_COLOR": box,
        "GAUGE_BOOM": bg,
        "TAB_BAR_BG": tab_bar,
        "TAB_INACTIVE_BG": inactive,
        "TAB_INACTIVE_FG": muted,
        "TAB_ACTIVE_BG": bg,
        "TAB_ACTIVE_FG": fg,
        "TAB_ACCENT": important,
        "MUTED_FG": muted,
    }


# Default cobalt UI + 32 NFL teams (8 divisions × 4) for the helmet picker
DEFAULT_THEME_ID = "default"
NFL_TEAM_THEMES: list[tuple[str, str, str, str]] = [
    # (id, label, primary, secondary) — AFC then NFC, 8 rows × 4
    ("buf", "Buffalo Bills", "#00338D", "#C60C30"),
    ("mia", "Miami Dolphins", "#008E97", "#FC4C02"),
    ("ne", "New England Patriots", "#002244", "#C60C30"),
    ("nyj", "New York Jets", "#125740", "#FFFFFF"),
    ("bal", "Baltimore Ravens", "#241773", "#9E7C0C"),
    ("cin", "Cincinnati Bengals", "#FB4F14", "#000000"),
    ("cle", "Cleveland Browns", "#311D00", "#FF3C00"),
    ("pit", "Pittsburgh Steelers", "#101820", "#FFB612"),
    ("hou", "Houston Texans", "#03202F", "#A71930"),
    ("ind", "Indianapolis Colts", "#002C5F", "#A2AAAD"),
    ("jax", "Jacksonville Jaguars", "#006778", "#9F792C"),
    ("ten", "Tennessee Titans", "#0C2340", "#4B92DB"),
    ("den", "Denver Broncos", "#FB4F14", "#002244"),
    ("kc", "Kansas City Chiefs", "#E31837", "#FFB81C"),
    ("lv", "Las Vegas Raiders", "#000000", "#A5ACAF"),
    ("lac", "Los Angeles Chargers", "#0080C6", "#FFC20E"),
    ("dal", "Dallas Cowboys", "#003594", "#869397"),
    ("nyg", "New York Giants", "#0B2265", "#A71930"),
    ("phi", "Philadelphia Eagles", "#004C54", "#A5ACAF"),
    ("was", "Washington Commanders", "#5A1414", "#FFB612"),
    ("chi", "Chicago Bears", "#0B162A", "#C83803"),
    ("det", "Detroit Lions", "#0076B6", "#B0B7BC"),
    ("gb", "Green Bay Packers", "#203731", "#FFB612"),
    ("min", "Minnesota Vikings", "#4F2683", "#FFC62F"),
    ("atl", "Atlanta Falcons", "#A71930", "#000000"),
    ("car", "Carolina Panthers", "#0085CA", "#101820"),
    ("no", "New Orleans Saints", "#101820", "#D3BC8D"),
    ("tb", "Tampa Bay Buccaneers", "#D50A0A", "#FF7900"),
    ("ari", "Arizona Cardinals", "#97233F", "#000000"),
    ("lar", "Los Angeles Rams", "#003594", "#FFA300"),
    ("sf", "San Francisco 49ers", "#AA0000", "#B3995D"),
    ("sea", "Seattle Seahawks", "#002244", "#69BE28"),
]

THEME_CATALOG: dict[str, dict[str, str]] = {
    DEFAULT_THEME_ID: {
        "label": "Default blue",
        "primary": "#0B3D91",
        "secondary": "#FFE566",
        **_theme_colors("#0B3D91", "#FFE566"),
    }
}
for _tid, _label, _pri, _sec in NFL_TEAM_THEMES:
    THEME_CATALOG[_tid] = {
        "label": _label,
        "primary": _pri,
        "secondary": _sec,
        **_theme_colors(_pri, _sec),
    }


def _snapshot_theme_globals() -> dict[str, str]:
    return {
        "BG": BG,
        "BG_FIELD": BG_FIELD,
        "FG": FG,
        "IMPORTANT_FG": IMPORTANT_FG,
        "ACCENT_BTN": ACCENT_BTN,
        "POPUP_BG": POPUP_BG,
        "CHECK_FILL": CHECK_FILL,
        "CHECK_EMPTY": CHECK_EMPTY,
        "BOX_COLOR": BOX_COLOR,
        "GAUGE_BOOM": GAUGE_BOOM,
        "TAB_BAR_BG": TAB_BAR_BG,
        "TAB_INACTIVE_BG": TAB_INACTIVE_BG,
        "TAB_INACTIVE_FG": TAB_INACTIVE_FG,
        "TAB_ACTIVE_BG": TAB_ACTIVE_BG,
        "TAB_ACTIVE_FG": TAB_ACTIVE_FG,
        "TAB_ACCENT": TAB_ACCENT,
        "MUTED_FG": MUTED_FG,
    }


def _apply_theme_globals(theme: dict[str, str]) -> None:
    global BG, BG_FIELD, FG, IMPORTANT_FG, ACCENT_BTN, POPUP_BG
    global CHECK_FILL, CHECK_EMPTY, BOX_COLOR, GAUGE_BOOM
    global TAB_BAR_BG, TAB_INACTIVE_BG, TAB_INACTIVE_FG, TAB_ACTIVE_BG, TAB_ACTIVE_FG, TAB_ACCENT
    global MUTED_FG
    BG = theme["BG"]
    BG_FIELD = theme["BG_FIELD"]
    FG = theme["FG"]
    IMPORTANT_FG = theme["IMPORTANT_FG"]
    ACCENT_BTN = theme["ACCENT_BTN"]
    POPUP_BG = theme["POPUP_BG"]
    CHECK_FILL = theme["CHECK_FILL"]
    CHECK_EMPTY = theme["CHECK_EMPTY"]
    BOX_COLOR = theme["BOX_COLOR"]
    GAUGE_BOOM = theme["GAUGE_BOOM"]
    TAB_BAR_BG = theme["TAB_BAR_BG"]
    TAB_INACTIVE_BG = theme["TAB_INACTIVE_BG"]
    TAB_INACTIVE_FG = theme["TAB_INACTIVE_FG"]
    TAB_ACTIVE_BG = theme["TAB_ACTIVE_BG"]
    TAB_ACTIVE_FG = theme["TAB_ACTIVE_FG"]
    TAB_ACCENT = theme["TAB_ACCENT"]
    MUTED_FG = theme["MUTED_FG"]


def _restyle_widget_tree(widget: tk.Misc, color_map: dict[str, str]) -> None:
    """Remap known theme colors on an existing widget tree (case-insensitive hex)."""
    norm = {k.upper(): v for k, v in color_map.items()}

    def _map(val: str) -> str | None:
        if not isinstance(val, str):
            return None
        key = val.upper() if val.startswith("#") else val
        return norm.get(key)

    opts = (
        "bg",
        "fg",
        "activebackground",
        "activeforeground",
        "highlightbackground",
        "highlightcolor",
        "insertbackground",
        "selectbackground",
        "selectforeground",
        "disabledforeground",
    )
    for opt in opts:
        try:
            cur = str(widget.cget(opt))
        except tk.TclError:
            continue
        mapped = _map(cur)
        if mapped is not None:
            try:
                widget.configure(**{opt: mapped})
            except tk.TclError:
                pass

    # Canvas fill/outline items
    if isinstance(widget, tk.Canvas):
        for item in widget.find_all():
            for opt in ("fill", "outline"):
                try:
                    cur = widget.itemcget(item, opt)
                except tk.TclError:
                    continue
                mapped = _map(cur)
                if mapped is not None and cur not in ("", "none"):
                    try:
                        widget.itemconfigure(item, **{opt: mapped})
                    except tk.TclError:
                        pass

    for child in widget.winfo_children():
        _restyle_widget_tree(child, color_map)


def draw_helmet(
    canvas: tk.Canvas,
    *,
    shell: str,
    x: int = 2,
    y: int = 3,
    scale: float = 1.0,
    outline: str = "#FFFFFF",
    facemask: str = "#FFFFFF",
) -> None:
    """Simple side-view helmet: colored shell, white outline + facemask."""
    s = scale
    # Shell
    canvas.create_oval(
        x + 2 * s,
        y + 2 * s,
        x + 26 * s,
        y + 22 * s,
        fill=shell,
        outline=outline,
        width=max(1, int(1.5 * s)),
    )
    # Ear / rear bump
    canvas.create_oval(
        x + 1 * s,
        y + 8 * s,
        x + 10 * s,
        y + 18 * s,
        fill=shell,
        outline=outline,
        width=1,
    )
    # Facemask bars
    canvas.create_arc(
        x + 14 * s,
        y + 8 * s,
        x + 30 * s,
        y + 22 * s,
        start=200,
        extent=100,
        style="arc",
        outline=facemask,
        width=max(1, int(1.5 * s)),
    )
    canvas.create_line(
        x + 18 * s,
        y + 12 * s,
        x + 28 * s,
        y + 12 * s,
        fill=facemask,
        width=max(1, int(1.2 * s)),
    )
    canvas.create_line(
        x + 17 * s,
        y + 15 * s,
        x + 27 * s,
        y + 15 * s,
        fill=facemask,
        width=max(1, int(1.2 * s)),
    )
    canvas.create_line(
        x + 17 * s,
        y + 18 * s,
        x + 25 * s,
        y + 18 * s,
        fill=facemask,
        width=max(1, int(1.2 * s)),
    )
    # Facemask posts
    canvas.create_line(
        x + 16 * s,
        y + 10 * s,
        x + 18 * s,
        y + 20 * s,
        fill=facemask,
        width=max(1, int(1.2 * s)),
    )


class HelmetButton(tk.Canvas):
    """Small helmet control that opens the team-color theme menu."""

    def __init__(
        self,
        master: tk.Misc,
        *,
        theme_id: str = DEFAULT_THEME_ID,
        command: Callable[[], None] | None = None,
        size: int = 28,
    ) -> None:
        theme = THEME_CATALOG[theme_id]
        super().__init__(
            master,
            width=size,
            height=size,
            bg=TAB_BAR_BG,
            highlightthickness=0,
            cursor="hand2",
        )
        self._command = command
        self._theme_id = theme_id
        self._size = size
        self.bind("<Button-1>", lambda _e: self._command() if self._command else None)
        self.redraw(theme_id)

    def redraw(self, theme_id: str, *, bar_bg: str | None = None) -> None:
        self._theme_id = theme_id
        theme = THEME_CATALOG.get(theme_id) or THEME_CATALOG[DEFAULT_THEME_ID]
        bg = bar_bg if bar_bg is not None else TAB_BAR_BG
        self.configure(bg=bg)
        self.delete("all")
        scale = self._size / 30.0
        draw_helmet(
            self,
            shell=theme["primary"],
            x=1,
            y=2,
            scale=scale,
            outline="#FFFFFF",
            facemask="#FFFFFF",
        )


class ThemeHelmetMenu(tk.Toplevel):
    """Dropdown: default helmet + 8×4 NFL team helmets."""

    COLS = 4
    ROWS = 8

    def __init__(
        self,
        parent: tk.Misc,
        *,
        current_id: str,
        on_pick: Callable[[str], None],
        anchor_widget: tk.Misc,
    ) -> None:
        super().__init__(parent)
        self.overrideredirect(True)
        self.configure(bg=TAB_BAR_BG)
        self._on_pick = on_pick
        self._current_id = current_id

        wrap = tk.Frame(self, bg=TAB_BAR_BG, padx=8, pady=8)
        wrap.pack(fill="both", expand=True)
        tk.Label(
            wrap,
            text="UI color (helmet)",
            bg=TAB_BAR_BG,
            fg=TAB_INACTIVE_FG,
            font=("Segoe UI", 9, "bold"),
        ).pack(anchor="w", pady=(0, 6))

        # Default row
        default_row = tk.Frame(wrap, bg=TAB_BAR_BG)
        default_row.pack(fill="x", pady=(0, 6))
        self._add_helmet_cell(default_row, DEFAULT_THEME_ID, tip="Default blue")

        grid = tk.Frame(wrap, bg=TAB_BAR_BG)
        grid.pack()
        for i, (tid, label, _pri, _sec) in enumerate(NFL_TEAM_THEMES):
            r, c = divmod(i, self.COLS)
            cell = tk.Frame(grid, bg=TAB_BAR_BG)
            cell.grid(row=r, column=c, padx=3, pady=3)
            self._add_helmet_cell(cell, tid, tip=label)

        self.update_idletasks()
        ax = anchor_widget.winfo_rootx()
        ay = anchor_widget.winfo_rooty() + anchor_widget.winfo_height() + 2
        w = self.winfo_reqwidth()
        # Right-align under the helmet button
        x = ax + anchor_widget.winfo_width() - w
        self.geometry(f"+{x}+{ay}")
        self.bind("<Escape>", lambda _e: self.destroy())
        self.after(10, self.focus_force)

    def _add_helmet_cell(self, parent: tk.Misc, theme_id: str, *, tip: str) -> None:
        theme = THEME_CATALOG[theme_id]
        selected = theme_id == self._current_id
        border = tk.Frame(
            parent,
            bg=IMPORTANT_FG if selected else TAB_BAR_BG,
            padx=1,
            pady=1,
        )
        border.pack()
        cv = tk.Canvas(
            border,
            width=34,
            height=30,
            bg=TAB_INACTIVE_BG,
            highlightthickness=0,
            cursor="hand2",
        )
        cv.pack()
        draw_helmet(cv, shell=theme["primary"], x=2, y=2, scale=1.05)
        cv.bind("<Button-1>", lambda _e, t=theme_id: self._pick(t))
        border.bind("<Button-1>", lambda _e, t=theme_id: self._pick(t))

        def _enter(_e, c=cv, b=border):
            c.configure(bg=_blend_hex(TAB_INACTIVE_BG, "#FFFFFF", 0.12))
            if theme_id != self._current_id:
                b.configure(bg=TAB_INACTIVE_FG)

        def _leave(_e, c=cv, b=border):
            c.configure(bg=TAB_INACTIVE_BG)
            b.configure(bg=IMPORTANT_FG if theme_id == self._current_id else TAB_BAR_BG)

        cv.bind("<Enter>", _enter)
        cv.bind("<Leave>", _leave)

    def _pick(self, theme_id: str) -> None:
        self._on_pick(theme_id)
        self.destroy()

class FillCheckbox(tk.Frame):
    """Small checkbox: solid white fill when selected, empty box when unselected."""

    SIZE = 14

    def __init__(
        self,
        master: tk.Misc,
        variable: tk.BooleanVar,
        text: str = "",
        command: Callable[[], None] | None = None,
        bg: str = BG,
        fg: str = FG,
    ) -> None:
        super().__init__(master, bg=bg)
        self.variable = variable
        self._command = command
        self._bg = bg
        self._fg = fg

        self.canvas = tk.Canvas(
            self,
            width=self.SIZE,
            height=self.SIZE,
            bg=bg,
            highlightthickness=0,
            bd=0,
            cursor="hand2",
        )
        self.canvas.pack(side="left")
        self.canvas.bind("<Button-1>", self._toggle)

        if text:
            label = tk.Label(
                self,
                text=text,
                bg=bg,
                fg=fg,
                font=("Segoe UI", 9),
                cursor="hand2",
            )
            label.pack(side="left", padx=(6, 0))
            label.bind("<Button-1>", self._toggle)

        self.variable.trace_add("write", lambda *_: self._draw())
        self._draw()

    def _toggle(self, _event: tk.Event | None = None) -> None:
        self.variable.set(not self.variable.get())
        if self._command:
            self._command()

    def _draw(self) -> None:
        self.canvas.delete("all")
        pad = 1
        s = self.SIZE - 1
        if self.variable.get():
            self.canvas.create_rectangle(
                pad,
                pad,
                s,
                s,
                outline=self._fg,
                width=1,
                fill=CHECK_FILL,
            )
        else:
            self.canvas.create_rectangle(
                pad,
                pad,
                s,
                s,
                outline=self._fg,
                width=1,
                fill=CHECK_EMPTY,
            )

class SearchableDropdown(tk.Frame):
    """
    Entry + floating Listbox dropdown.
    Typing filters options by case-insensitive substring (e.g. 'Car' keeps Carson/Carter).
    More reliable on Windows than filtering ttk.Combobox values.
    """

    def __init__(
        self,
        master: tk.Misc,
        full_values: list[str],
        textvariable: tk.StringVar | None = None,
        on_select: Callable[[], None] | None = None,
        width: int = 54,
    ) -> None:
        super().__init__(master, bg=BG)
        self._all_values = [v for v in full_values if v != ""]
        self._on_select = on_select
        self._var = textvariable if textvariable is not None else tk.StringVar()
        self._popup: tk.Toplevel | None = None
        self._listbox: tk.Listbox | None = None
        self._hide_job: str | None = None

        self.entry = tk.Entry(
            self,
            textvariable=self._var,
            bg=BG_FIELD,
            fg=FG,
            insertbackground=FG,
            relief="flat",
            font=("Segoe UI", 10),
            width=width,
        )
        self.entry.pack(side="left", fill="x", expand=True, ipady=3)

        self._btn = tk.Button(
            self,
            text="▾",
            command=self._toggle_popup,
            bg=ACCENT_BTN,
            fg=FG,
            relief="flat",
            width=2,
            font=("Segoe UI", 9),
            activebackground="#2563B8",
            activeforeground=FG,
        )
        self._btn.pack(side="right", padx=(4, 0), ipady=1)

        self.entry.bind("<KeyRelease>", self._on_keyrelease)
        self.entry.bind("<FocusOut>", self._schedule_hide)
        self.entry.bind("<Down>", self._on_arrow_open)
        self.entry.bind("<Up>", self._move_selection)
        self.entry.bind("<Return>", self._confirm_selection)
        self.entry.bind("<Escape>", lambda _e: self._hide_popup())
        self._suppress_select = False

    def get(self) -> str:
        return self._var.get()

    def set(self, value: str) -> None:
        self._var.set(value)

    def set_full_values(self, values: list[str]) -> None:
        self._all_values = [v for v in values if v != ""]
        if self._popup is not None and self._popup.winfo_exists():
            self._show_popup()

    def _filtered(self, text: str) -> list[str]:
        needle = text.strip().lower()
        if not needle:
            return list(self._all_values)
        return [v for v in self._all_values if needle in v.lower()]

    def _on_keyrelease(self, event: tk.Event) -> None:
        if event.keysym in NAV_KEYS:
            return
        # Only open/filter while typing — never auto-apply a list value
        self._show_popup()

    def _on_arrow_open(self, event: tk.Event) -> str:
        if self._popup is None or not self._popup.winfo_exists():
            self._show_popup()
            return "break"
        return self._move_selection(event)

    def _toggle_popup(self) -> None:
        if self._popup is not None and self._popup.winfo_exists():
            self._hide_popup()
        else:
            self.entry.focus_set()
            self._show_popup(force_all=not self._var.get().strip())

    def _show_popup(self, force_all: bool = False) -> None:
        self._cancel_hide()
        typed = "" if force_all else self._var.get()
        matches = self._filtered(typed)
        if not matches:
            self._hide_popup()
            return

        if self._popup is None or not self._popup.winfo_exists():
            self._popup = tk.Toplevel(self)
            self._popup.wm_overrideredirect(True)
            self._popup.configure(bg=BG_FIELD)
            self._popup.attributes("-topmost", True)
            frame = tk.Frame(self._popup, bg=BG_FIELD, bd=1, relief="solid")
            frame.pack(fill="both", expand=True)
            scroll = tk.Scrollbar(frame, orient="vertical")
            self._listbox = tk.Listbox(
                frame,
                height=min(12, max(4, len(matches))),
                bg=BG_FIELD,
                fg=FG,
                selectbackground=ACCENT_BTN,
                selectforeground=FG,
                activestyle="none",
                font=("Segoe UI", 10),
                highlightthickness=0,
                relief="flat",
                yscrollcommand=scroll.set,
                exportselection=False,
            )
            scroll.config(command=self._listbox.yview)
            scroll.pack(side="right", fill="y")
            self._listbox.pack(side="left", fill="both", expand=True)
            # Apply only on explicit click / Return — not on programmatic selection
            self._listbox.bind("<ButtonRelease-1>", self._on_listbox_click)
            self._listbox.bind("<Double-Button-1>", self._on_listbox_click)
            self._listbox.bind("<Return>", self._confirm_selection)
            self._listbox.bind("<Escape>", lambda _e: self._hide_popup())
            self._listbox.bind("<FocusOut>", self._schedule_hide)
        else:
            assert self._listbox is not None
            self._listbox.configure(height=min(12, max(4, len(matches))))

        assert self._listbox is not None and self._popup is not None
        self._suppress_select = True
        self._listbox.delete(0, tk.END)
        for item in matches:
            self._listbox.insert(tk.END, item)
        if matches:
            self._listbox.selection_clear(0, tk.END)
            self._listbox.selection_set(0)
            self._listbox.activate(0)
            self._listbox.see(0)
        self._suppress_select = False

        self.update_idletasks()
        x = self.winfo_rootx()
        y = self.winfo_rooty() + self.winfo_height()
        width = max(self.winfo_width(), 280)
        self._popup.geometry(f"{width}x{min(280, 22 * min(12, len(matches)) + 8)}+{x}+{y}")
        self._popup.deiconify()
        self._popup.lift()

    def _schedule_hide(self, _event: tk.Event | None = None) -> None:
        self._cancel_hide()
        self._hide_job = self.after(180, self._hide_if_unfocused)

    def _cancel_hide(self) -> None:
        if self._hide_job is not None:
            try:
                self.after_cancel(self._hide_job)
            except tk.TclError:
                pass
            self._hide_job = None

    def _hide_if_unfocused(self) -> None:
        self._hide_job = None
        focus = self.focus_get()
        if focus in (self.entry, self._btn, self._listbox):
            return
        if self._popup is not None and focus is not None:
            try:
                if str(focus).startswith(str(self._popup)):
                    return
            except tk.TclError:
                pass
        self._hide_popup()

    def _hide_popup(self) -> None:
        self._cancel_hide()
        if self._popup is not None:
            try:
                self._popup.destroy()
            except tk.TclError:
                pass
        self._popup = None
        self._listbox = None

    def _move_selection(self, event: tk.Event) -> str:
        if self._listbox is None or self._popup is None or not self._popup.winfo_exists():
            self._show_popup()
            return "break"
        size = self._listbox.size()
        if size == 0:
            return "break"
        current = self._listbox.curselection()
        idx = int(current[0]) if current else 0
        if event.keysym == "Down":
            idx = min(idx + 1, size - 1)
        else:
            idx = max(idx - 1, 0)
        self._listbox.selection_clear(0, tk.END)
        self._listbox.selection_set(idx)
        self._listbox.activate(idx)
        self._listbox.see(idx)
        return "break"

    def _on_listbox_click(self, _event: tk.Event | None = None) -> None:
        if self._suppress_select:
            return
        self.after_idle(self._apply_listbox_selection)

    def _apply_listbox_selection(self) -> None:
        if self._suppress_select or self._listbox is None:
            return
        sel = self._listbox.curselection()
        if not sel:
            return
        value = self._listbox.get(sel[0])
        self._var.set(value)
        self._hide_popup()
        self.entry.icursor(tk.END)
        if self._on_select:
            self._on_select()

    def _confirm_selection(self, _event: tk.Event | None = None) -> str:
        if self._popup is None or not self._popup.winfo_exists():
            return "break"
        self._apply_listbox_selection()
        return "break"


class ScrollFrame(tk.Frame):
    def __init__(self, parent: tk.Misc, **kwargs) -> None:
        super().__init__(parent, bg=BG, **kwargs)
        canvas = tk.Canvas(self, bg=BG, highlightthickness=0, borderwidth=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.inner = tk.Frame(canvas, bg=BG)
        self.inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self._canvas = canvas

        def _on_mousewheel(event) -> None:
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind("<Enter>", lambda _e: canvas.bind_all("<MouseWheel>", _on_mousewheel))
        canvas.bind("<Leave>", lambda _e: canvas.unbind_all("<MouseWheel>"))


class BoxWhiskerPlot(tk.Canvas):
    """Horizontal box-and-whisker with configurable zone labels."""

    def __init__(
        self,
        master: tk.Misc,
        dist: dict,
        marker: float,
        title: str = "",
        low_label: str = "Bust",
        mid_label: str = "IQR",
        high_label: str = "Boom",
        footer: str = "",
        width: int = 300,
        height: int = 110,
        **kwargs,
    ) -> None:
        super().__init__(
            master,
            width=width,
            height=height,
            bg=POPUP_BG,
            highlightthickness=0,
            **kwargs,
        )
        self._dist = dist
        self._marker = marker
        self._title = title
        self._low_label = low_label
        self._mid_label = mid_label
        self._high_label = high_label
        self._footer = footer
        self._width = width
        self._height = height
        self.bind("<Configure>", self._on_configure)
        self.after_idle(self._draw)

    def _on_configure(self, _event: tk.Event | None = None) -> None:
        self._width = max(self.winfo_width(), 200)
        self._draw()

    def _x(self, value: float, lo: float, hi: float, left: int, right: int) -> float:
        if hi <= lo:
            return (left + right) / 2
        return left + (value - lo) / (hi - lo) * (right - left)

    def _draw(self) -> None:
        self.delete("all")
        dist = self._dist
        marker = self._marker
        vals = [dist.get(k) for k in ("min", "q1", "median", "q3", "max")]
        if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in vals):
            self.create_text(
                self._width // 2,
                self._height // 2,
                text="Not enough data for plot",
                fill="#B8D4FF",
                font=("Segoe UI", 9),
            )
            return

        lo = float(dist["min"])
        hi = float(dist["max"])
        if not (isinstance(marker, float) and math.isnan(marker)):
            lo = min(lo, float(marker))
            hi = max(hi, float(marker))
        pad = (hi - lo) * 0.05 if hi > lo else 1.0
        lo -= pad
        hi += pad

        left, right = 36, self._width - 16
        y = 48
        box_h = 22

        q1 = float(dist["q1"])
        med = float(dist["median"])
        q3 = float(dist["q3"])
        dmin = float(dist["min"])
        dmax = float(dist["max"])

        x_min = self._x(dmin, lo, hi, left, right)
        x_q1 = self._x(q1, lo, hi, left, right)
        x_med = self._x(med, lo, hi, left, right)
        x_q3 = self._x(q3, lo, hi, left, right)
        x_max = self._x(dmax, lo, hi, left, right)

        self.create_rectangle(x_min, y - box_h // 2, x_q1, y + box_h // 2, fill="#5A2030", outline="")
        self.create_rectangle(x_q3, y - box_h // 2, x_max, y + box_h // 2, fill="#1E5A3A", outline="")

        self.create_line(x_min, y, x_q1, y, fill=FG, width=2)
        self.create_line(x_q3, y, x_max, y, fill=FG, width=2)
        self.create_line(x_min, y - 8, x_min, y + 8, fill=FG, width=2)
        self.create_line(x_max, y - 8, x_max, y + 8, fill=FG, width=2)

        self.create_rectangle(
            x_q1,
            y - box_h // 2,
            x_q3,
            y + box_h // 2,
            fill=BOX_COLOR,
            outline=FG,
            width=1,
        )
        self.create_line(x_med, y - box_h // 2, x_med, y + box_h // 2, fill=MEDIAN_COLOR, width=2)

        if not (isinstance(marker, float) and math.isnan(marker)):
            x_pred = self._x(float(marker), lo, hi, left, right)
            self.create_line(x_pred, y - box_h // 2 - 10, x_pred, y + box_h // 2 + 10, fill=PRED_COLOR, width=2)
            self.create_oval(x_pred - 5, y - 5, x_pred + 5, y + 5, fill=IMPORTANT_FG, outline=FG)

        self.create_text(x_min, y + 28, text=self._low_label, fill=BUST_COLOR, font=("Segoe UI", 8, "bold"), anchor="n")
        self.create_text(
            (x_q1 + x_q3) / 2,
            y + 28,
            text=self._mid_label,
            fill="#B8D4FF",
            font=("Segoe UI", 8),
            anchor="n",
        )
        self.create_text(
            x_max,
            y + 28,
            text=self._high_label,
            fill=BOOM_COLOR,
            font=("Segoe UI", 8, "bold"),
            anchor="n",
        )
        if self._title:
            self.create_text(left, 12, text=self._title, fill="#B8D4FF", font=("Segoe UI", 8), anchor="w")
        if self._footer:
            self.create_text(
                left,
                self._height - 8,
                text=self._footer,
                fill="#B8D4FF",
                font=("Segoe UI", 8),
                anchor="sw",
            )


class BoomBustGauge(tk.Canvas):
    """
    Prediction-error boom/bust as a horizontal box-and-whisker
    (same visual language as the historical peers plot).
    """

    def __init__(
        self,
        master: tk.Misc,
        player_name: str,
        predicted: float,
        ppr_low: float,
        ppr_high: float,
        boom_chance_pct: float = 50.0,
        bust_chance_pct: float = 50.0,
        width: int = 350,
        height: int = 130,
        **kwargs,
    ) -> None:
        super().__init__(
            master,
            width=width,
            height=height,
            bg=POPUP_BG,
            highlightthickness=0,
            **kwargs,
        )
        self._name = player_name
        self._predicted = float(predicted)
        self._low = float(ppr_low)
        self._high = float(ppr_high)
        self._boom_pct = float(boom_chance_pct)
        self._bust_pct = float(bust_chance_pct)
        self._width = width
        self._height = height
        self.bind("<Configure>", self._on_configure)
        self.after_idle(self._draw)

    def _on_configure(self, _event: tk.Event | None = None) -> None:
        self._width = max(self.winfo_width(), 200)
        self._draw()

    def _x(self, value: float, lo: float, hi: float, left: int, right: int) -> float:
        if hi <= lo:
            return (left + right) / 2
        return left + (value - lo) / (hi - lo) * (right - left)

    def _draw(self) -> None:
        self.delete("all")
        pred = self._predicted
        q1 = min(self._low, self._high)
        q3 = max(self._low, self._high)
        bust_pct = max(0.0, min(100.0, self._bust_pct))
        boom_pct = max(0.0, min(100.0, self._boom_pct))
        iqr = max(q3 - q1, 1.0)
        # Visual whiskers beyond predictive Q1/Q3 (error-model IQR wings)
        dmin = q1 - 0.45 * iqr
        dmax = q3 + 0.45 * iqr
        dmin = min(dmin, pred)
        dmax = max(dmax, pred)

        pad = (dmax - dmin) * 0.05 if dmax > dmin else 1.0
        lo, hi = dmin - pad, dmax + pad
        left, right = 36, self._width - 16
        y = 44
        box_h = 22

        x_min = self._x(dmin, lo, hi, left, right)
        x_q1 = self._x(q1, lo, hi, left, right)
        x_med = self._x(pred, lo, hi, left, right)
        x_q3 = self._x(q3, lo, hi, left, right)
        x_max = self._x(dmax, lo, hi, left, right)

        # Bust / boom outer zones
        self.create_rectangle(x_min, y - box_h // 2, x_q1, y + box_h // 2, fill="#5A2030", outline="")
        self.create_rectangle(x_q3, y - box_h // 2, x_max, y + box_h // 2, fill="#1E5A3A", outline="")

        # Whiskers
        self.create_line(x_min, y, x_q1, y, fill=FG, width=2)
        self.create_line(x_q3, y, x_max, y, fill=FG, width=2)
        self.create_line(x_min, y - 8, x_min, y + 8, fill=FG, width=2)
        self.create_line(x_max, y - 8, x_max, y + 8, fill=FG, width=2)

        # IQR box
        self.create_rectangle(
            x_q1,
            y - box_h // 2,
            x_q3,
            y + box_h // 2,
            fill=BOX_COLOR,
            outline=FG,
            width=1,
        )
        # Median line = projected points
        self.create_line(x_med, y - box_h // 2, x_med, y + box_h // 2, fill=MEDIAN_COLOR, width=2)
        self.create_line(x_med, y - box_h // 2 - 10, x_med, y + box_h // 2 + 10, fill=PRED_COLOR, width=2)
        self.create_oval(x_med - 5, y - 5, x_med + 5, y + 5, fill=IMPORTANT_FG, outline=FG)

        self.create_text(
            x_med,
            y - box_h // 2 - 14,
            text=f"Points {pred:.1f}",
            fill=IMPORTANT_FG,
            font=("Segoe UI", 9, "bold"),
            anchor="s",
        )

        # Quartile threshold values under Q1 / Q3 lines
        self.create_text(
            x_q1,
            y + 26,
            text=f"{q1:.0f}",
            fill=BUST_COLOR,
            font=("Segoe UI", 9, "bold"),
            anchor="n",
        )
        self.create_text(
            x_q3,
            y + 26,
            text=f"{q3:.0f}",
            fill=BOOM_COLOR,
            font=("Segoe UI", 9, "bold"),
            anchor="n",
        )

        self.create_text(
            x_min,
            y + 42,
            text=f"Bust (~{bust_pct:.0f}%)",
            fill=BUST_COLOR,
            font=("Segoe UI", 8, "bold"),
            anchor="n",
        )
        self.create_text(
            (x_q1 + x_q3) / 2,
            y + 42,
            text="IQR",
            fill="#B8D4FF",
            font=("Segoe UI", 8),
            anchor="n",
        )
        self.create_text(
            x_max,
            y + 42,
            text=f"Boom (~{boom_pct:.0f}%)",
            fill=BOOM_COLOR,
            font=("Segoe UI", 8, "bold"),
            anchor="n",
        )

        first = self._name.split()[0] if self._name.strip() else "This player"
        footer = (
            f"{first}: {bust_pct:.0f}% chance < {q1:.0f}, {boom_pct:.0f}% chance > {q3:.0f}"
        )
        self.create_text(
            left,
            self._height - 8,
            text=footer,
            fill="#B8D4FF",
            font=("Segoe UI", 8),
            anchor="sw",
            width=self._width - left - 8,
        )





def _result_mode(result: dict) -> str:
    return str(result.get("mode") or "redraft")


def _result_predicted_points(result: dict) -> float:
    if "predicted_points" in result and result["predicted_points"] is not None:
        return float(result["predicted_points"])
    return float(result.get("predicted_rookie_ppr", 0.0))


def _result_success_score(result: dict) -> float:
    if "success_score" in result and result["success_score"] is not None:
        return float(result["success_score"])
    return float(result.get("success_score_0_100", 0.0))


def _predicted_points_label(mode: str | AnalysisMode | None) -> str:
    name = mode.name if isinstance(mode, AnalysisMode) else str(mode or "redraft")
    if name == "dynasty":
        return "Predicted Y1–Y3 total points"
    return "Predicted points"


def _pack_year_by_year(parent: tk.Misc, result: dict, *, wraplength: int = 360) -> None:
    """Show dynasty Y1/Y2/Y3 predicted (and actual if present) as a text summary."""
    years = result.get("year_by_year")
    if not years or _result_mode(result) != "dynasty":
        return

    frame = tk.Frame(parent, bg=POPUP_BG)
    frame.pack(fill="x", pady=(6, 0))
    tk.Label(
        frame,
        text="Year-by-year points",
        bg=POPUP_BG,
        fg=FG,
        font=("Segoe UI", 9, "bold"),
        justify="left",
    ).pack(anchor="w")

    parts: list[str] = []
    for i in (1, 2, 3):
        block = years.get(f"y{i}") or {}
        pred = block.get("predicted")
        actual = block.get("actual")
        if pred is None and actual is None:
            parts.append(f"Y{i}: —")
            continue
        if pred is not None and actual is not None:
            parts.append(f"Y{i}: {float(pred):.0f} pred / {float(actual):.0f} actual")
        elif pred is not None:
            parts.append(f"Y{i}: {float(pred):.0f}")
        else:
            parts.append(f"Y{i}: {float(actual):.0f} actual")

    tk.Label(
        frame,
        text="  ·  ".join(parts),
        bg=POPUP_BG,
        fg="#B8D4FF",
        font=("Segoe UI", 9),
        justify="left",
        wraplength=wraplength,
    ).pack(anchor="w", pady=(2, 0))


def _horizon_plot_args(block: dict | None, fallback_pred: float | None = None) -> tuple[float, float, float, float, float] | None:
    """Extract BoomBustGauge args from a year_by_year block or total confidence."""
    if not block:
        return None
    pred = block.get("predicted")
    if pred is None:
        pred = fallback_pred
    if pred is None:
        return None
    conf = block.get("confidence") or {}
    low = conf.get("ppr_low")
    high = conf.get("ppr_high")
    if low is None or high is None:
        return None
    boom = float(conf.get("boom_chance_pct", 25.0))
    bust = float(conf.get("bust_chance_pct", 25.0))
    return float(pred), float(low), float(high), boom, bust


def _pack_dynasty_horizon_gauges(
    parent: tk.Misc,
    name: str,
    result: dict,
    *,
    width: int = 370,
) -> None:
    """One boom/bust gauge per season (Y1–Y3) plus the Y1–Y3 total."""
    if _result_mode(result) != "dynasty":
        return

    years = result.get("year_by_year") or {}
    total_conf = result.get("confidence") or {}
    total_block = {
        "predicted": _result_predicted_points(result),
        "confidence": total_conf,
    }

    sections: list[tuple[str, dict | None]] = [
        ("Year 1", years.get("y1")),
        ("Year 2", years.get("y2")),
        ("Year 3", years.get("y3")),
        ("Y1–Y3 total", total_block),
    ]

    wrap = tk.Frame(parent, bg=POPUP_BG)
    wrap.pack(fill="x", padx=8, pady=(2, 0))

    for title, block in sections:
        args = _horizon_plot_args(block)
        if args is None:
            continue
        pred, low, high, boom_pct, bust_pct = args
        section = tk.Frame(wrap, bg=POPUP_BG)
        section.pack(fill="x", pady=(6, 0))
        subtitle = title
        if block and block.get("actual") is not None:
            subtitle = f"{title}  (actual {float(block['actual']):.0f})"
        elif block and block.get("success_score") is not None and title != "Y1–Y3 total":
            subtitle = f"{title}  ·  peer score {float(block['success_score']):.0f}"
        tk.Label(
            section,
            text=subtitle,
            bg=POPUP_BG,
            fg=FG,
            font=("Segoe UI", 9, "bold"),
            anchor="w",
        ).pack(anchor="w", padx=4)
        BoomBustGauge(
            section,
            player_name=name,
            predicted=pred,
            ppr_low=low,
            ppr_high=high,
            boom_chance_pct=boom_pct,
            bust_chance_pct=bust_pct,
            width=width,
            height=120,
        ).pack(fill="x")


def _historical_dist_title(result: dict, dist: dict) -> str:
    pos = result.get("position", "")
    n = dist.get("n", 0)
    if _result_mode(result) == "dynasty":
        return f"Historical {pos} Y1–Y3 totals (n={n})"
    return f"Historical {pos} rookie points (n={n})"


def _prediction_plot_args(result: dict) -> tuple[float, float, float, float, float]:
    pred = _result_predicted_points(result)
    conf = result.get("confidence") or {}
    half = float(conf.get("half_width") or 0.0)
    low = float(conf.get("ppr_low", pred - half))
    high = float(conf.get("ppr_high", pred + half))
    # Symmetric error model around the prediction → equal chance above/below
    boom_pct = float(conf.get("boom_chance_pct", 50.0))
    bust_pct = float(conf.get("bust_chance_pct", 50.0))
    return pred, low, high, boom_pct, bust_pct


def _popup_content_height(win: tk.Toplevel) -> int:
    """Best-effort content height (reqheight, floored by mapped child bottoms)."""
    win.update_idletasks()
    content_h = int(win.winfo_reqheight())
    for child in win.winfo_children():
        try:
            child.update_idletasks()
            bottom = int(child.winfo_y()) + max(int(child.winfo_reqheight()), int(child.winfo_height()))
            if bottom > content_h:
                content_h = bottom
        except tk.TclError:
            pass
    return content_h


def _fit_popup_to_content(win: tk.Toplevel, width: int = 400, bottom_pad: int = 8) -> int:
    """Shrink/grow a popup so the bottom sits just under the last widget.

    Returns the applied client height. Always ends with update_idletasks so
    subsequent winfo_height / geometry reads see the new size (important when
    compare tiling runs in the same idle queue).
    """
    content_h = _popup_content_height(win)
    h = max(content_h + bottom_pad, COMPARE_POPUP_MIN_HEIGHT)
    try:
        x, y = win.winfo_x(), win.winfo_y()
        win.geometry(f"{width}x{h}+{x}+{y}")
    except tk.TclError:
        win.geometry(f"{width}x{h}")
    win.update_idletasks()
    # Reflow can change wrapping; grow once more if content still needs room.
    content_h = _popup_content_height(win)
    h2 = max(content_h + bottom_pad, COMPARE_POPUP_MIN_HEIGHT)
    if h2 != h:
        h = h2
        try:
            x, y = win.winfo_x(), win.winfo_y()
            win.geometry(f"{width}x{h}+{x}+{y}")
        except tk.TclError:
            win.geometry(f"{width}x{h}")
        win.update_idletasks()
    return h


class PredictionPopup(tk.Toplevel):
    """Side popup: boom/bust box plot, optional historical points plot."""

    def __init__(self, parent: tk.Misc, name: str, result: dict, x: int, y: int) -> None:
        super().__init__(parent)
        self.title(f"Prediction — {name}")
        self.configure(bg=POPUP_BG)
        self.minsize(360, 160)
        self._result = result
        self._ppr_plot_holder: tk.Frame | None = None
        self._popup_x = x
        self._popup_y = y

        pred = _result_predicted_points(result)
        score = _result_success_score(result)
        conf = result.get("confidence") or {}
        dist = result.get("ppr_distribution") or {}
        p, low, high, boom_pct, bust_pct = _prediction_plot_args(result)
        pred_label = _predicted_points_label(result.get("mode"))

        header = tk.Frame(self, bg=POPUP_BG, padx=12, pady=8)
        header.pack(fill="x")
        tk.Label(
            header,
            text=name,
            bg=POPUP_BG,
            fg=FG,
            font=("Segoe UI", 12, "bold"),
            wraplength=360,
            justify="left",
        ).pack(anchor="w")
        tk.Label(
            header,
            text=f"{pred_label}: {pred:.1f}",
            bg=POPUP_BG,
            fg=IMPORTANT_FG,
            font=("Segoe UI", 12, "bold"),
            justify="left",
        ).pack(anchor="w", pady=(4, 0))
        tk.Label(
            header,
            text=f"Peer success score: {score:.1f}",
            bg=POPUP_BG,
            fg="#B8D4FF",
            font=("Segoe UI", 9),
            justify="left",
        ).pack(anchor="w", pady=(2, 0))
        _pack_year_by_year(header, result, wraplength=360)
        is_dynasty = _result_mode(result) == "dynasty"
        if conf and not is_dynasty:
            method = conf.get("method")
            if method == "predictive_quartile":
                tk.Label(
                    header,
                    text=(
                        f"Error-model quartiles: {conf.get('bust_chance_pct'):.0f}% chance "
                        f"< {conf.get('ppr_low')}  ·  {conf.get('boom_chance_pct'):.0f}% chance "
                        f"> {conf.get('ppr_high')}"
                    ),
                    bg=POPUP_BG,
                    fg="#B8D4FF",
                    font=("Segoe UI", 8),
                    justify="left",
                    wraplength=360,
                ).pack(anchor="w", pady=(2, 0))
                if conf.get("cqr_low") is not None:
                    tk.Label(
                        header,
                        text=(
                            f"Model 80% CQR band: {conf.get('cqr_low')} – {conf.get('cqr_high')}"
                        ),
                        bg=POPUP_BG,
                        fg="#B8D4FF",
                        font=("Segoe UI", 8),
                        justify="left",
                    ).pack(anchor="w", pady=(1, 0))
            elif method == "peer_quartile":
                tk.Label(
                    header,
                    text=(
                        f"Bust = bottom quartile (≤Q1)  ·  Boom = top quartile (≥Q3)  ·  "
                        f"~{conf.get('bust_chance_pct')}%/{conf.get('boom_chance_pct')}%"
                    ),
                    bg=POPUP_BG,
                    fg="#B8D4FF",
                    font=("Segoe UI", 8),
                    justify="left",
                    wraplength=360,
                ).pack(anchor="w", pady=(2, 0))
            elif method == "cqr":
                cov = conf.get("nominal_coverage_pct", 80)
                tk.Label(
                    header,
                    text=(
                        f"{cov:.0f}% conformal interval (CQR)  ·  "
                        f"bust/boom ~{conf.get('bust_chance_pct')}%/"
                        f"{conf.get('boom_chance_pct')}% (asymmetric)"
                    ),
                    bg=POPUP_BG,
                    fg="#B8D4FF",
                    font=("Segoe UI", 8),
                    justify="left",
                    wraplength=360,
                ).pack(anchor="w", pady=(2, 0))
            else:
                tk.Label(
                    header,
                    text=(
                        f"Error band from holdout MAE × missing composites "
                        f"(×{conf.get('multiplier', 1):.2f})"
                    ),
                    bg=POPUP_BG,
                    fg="#B8D4FF",
                    font=("Segoe UI", 8),
                    justify="left",
                    wraplength=360,
                ).pack(anchor="w", pady=(2, 0))

        plot_frame = tk.Frame(self, bg=POPUP_BG, padx=8, pady=2)
        plot_frame.pack(fill="x")
        if is_dynasty:
            _pack_dynasty_horizon_gauges(plot_frame, name, result, width=370)
        else:
            BoomBustGauge(
                plot_frame,
                player_name=name,
                predicted=p,
                ppr_low=low,
                ppr_high=high,
                boom_chance_pct=boom_pct,
                bust_chance_pct=bust_pct,
                width=370,
                height=130,
            ).pack(fill="x")

        self._show_ppr_var = tk.BooleanVar(value=False)
        FillCheckbox(
            self,
            self._show_ppr_var,
            text="Show historical points distribution",
            command=self._toggle_ppr_plot,
            bg=POPUP_BG,
            fg=FG,
        ).pack(anchor="w", padx=12, pady=(4, 0))

        self._ppr_plot_holder = tk.Frame(self, bg=POPUP_BG, padx=8, pady=0)
        self._ppr_plot_holder.pack(fill="x")
        self._ppr_dist = dist
        self._pred = pred

        pop = len(result.get("composite_populated", []))
        tk.Label(
            self,
            text=f"Composites used: {pop}/11",
            bg=POPUP_BG,
            fg="#B8D4FF",
            font=("Segoe UI", 9),
        ).pack(anchor="w", padx=12, pady=(4, 6))

        self.geometry(f"400x200+{x}+{y}")
        self.after_idle(lambda: _fit_popup_to_content(self, width=400, bottom_pad=6))

    def _toggle_ppr_plot(self) -> None:
        if self._ppr_plot_holder is None:
            return
        for child in self._ppr_plot_holder.winfo_children():
            child.destroy()
        if not self._show_ppr_var.get():
            _fit_popup_to_content(self, width=400, bottom_pad=6)
            return
        BoxWhiskerPlot(
            self._ppr_plot_holder,
            self._ppr_dist,
            float(self._pred),
            title=_historical_dist_title(self._result, self._ppr_dist),
            low_label="Low",
            mid_label="IQR",
            high_label="High",
            footer=(
                f"Q1 {self._ppr_dist.get('q1', float('nan')):.0f}   "
                f"Med {self._ppr_dist.get('median', float('nan')):.0f}   "
                f"Q3 {self._ppr_dist.get('q3', float('nan')):.0f}"
            ),
            width=370,
            height=120,
        ).pack(fill="x")
        _fit_popup_to_content(self, width=400, bottom_pad=6)


class ComparePlayerPopup(tk.Toplevel):
    """Side popup matching Calculate score style; checkbox reveals score drivers."""

    def __init__(self, parent: tk.Misc, entry: dict, x: int, y: int) -> None:
        super().__init__(parent)
        self.title(f"Compare — {entry['name']}")
        self.configure(bg=POPUP_BG)
        self.minsize(360, 160)
        self._result = entry["result"]
        self._drivers_holder: tk.Frame | None = None
        self._fitted_height = COMPARE_POPUP_EST_HEIGHT

        result = entry["result"]
        pred = _result_predicted_points(result)
        score = _result_success_score(result)
        conf = result.get("confidence") or {}
        p, low, high, boom_pct, bust_pct = _prediction_plot_args(result)
        pred_label = _predicted_points_label(result.get("mode"))

        header = tk.Frame(self, bg=POPUP_BG, padx=12, pady=8)
        header.pack(fill="x")
        tk.Label(
            header,
            text=entry["name"],
            bg=POPUP_BG,
            fg=FG,
            font=("Segoe UI", 12, "bold"),
            wraplength=360,
            justify="left",
        ).pack(anchor="w")
        tk.Label(
            header,
            text=f"{pred_label}: {pred:.1f}",
            bg=POPUP_BG,
            fg=IMPORTANT_FG,
            font=("Segoe UI", 12, "bold"),
            justify="left",
        ).pack(anchor="w", pady=(4, 0))
        tk.Label(
            header,
            text=f"Peer success score: {score:.1f}",
            bg=POPUP_BG,
            fg="#B8D4FF",
            font=("Segoe UI", 9),
            justify="left",
        ).pack(anchor="w", pady=(2, 0))
        _pack_year_by_year(header, result, wraplength=360)
        is_dynasty = _result_mode(result) == "dynasty"
        if conf and not is_dynasty:
            method = conf.get("method")
            if method == "predictive_quartile":
                tk.Label(
                    header,
                    text=(
                        f"Error-model quartiles: {conf.get('bust_chance_pct'):.0f}% chance "
                        f"< {conf.get('ppr_low')}  ·  {conf.get('boom_chance_pct'):.0f}% chance "
                        f"> {conf.get('ppr_high')}"
                    ),
                    bg=POPUP_BG,
                    fg="#B8D4FF",
                    font=("Segoe UI", 8),
                    justify="left",
                    wraplength=360,
                ).pack(anchor="w", pady=(2, 0))
            elif method == "peer_quartile":
                tk.Label(
                    header,
                    text=(
                        f"Bust = bottom quartile (≤Q1)  ·  Boom = top quartile (≥Q3)  ·  "
                        f"~{conf.get('bust_chance_pct')}%/{conf.get('boom_chance_pct')}%"
                    ),
                    bg=POPUP_BG,
                    fg="#B8D4FF",
                    font=("Segoe UI", 8),
                    justify="left",
                    wraplength=360,
                ).pack(anchor="w", pady=(2, 0))
            elif method == "cqr":
                cov = conf.get("nominal_coverage_pct", 80)
                tk.Label(
                    header,
                    text=(
                        f"{cov:.0f}% conformal interval (CQR)  ·  "
                        f"bust/boom ~{conf.get('bust_chance_pct')}%/"
                        f"{conf.get('boom_chance_pct')}% (asymmetric)"
                    ),
                    bg=POPUP_BG,
                    fg="#B8D4FF",
                    font=("Segoe UI", 8),
                    justify="left",
                    wraplength=360,
                ).pack(anchor="w", pady=(2, 0))
            else:
                tk.Label(
                    header,
                    text=(
                        f"Error band from holdout MAE × missing composites "
                        f"(×{conf.get('multiplier', 1):.2f})"
                    ),
                    bg=POPUP_BG,
                    fg="#B8D4FF",
                    font=("Segoe UI", 8),
                    justify="left",
                    wraplength=360,
                ).pack(anchor="w", pady=(2, 0))

        plot_frame = tk.Frame(self, bg=POPUP_BG, padx=8, pady=2)
        plot_frame.pack(fill="x")
        if is_dynasty:
            _pack_dynasty_horizon_gauges(plot_frame, entry["name"], result, width=370)
        else:
            BoomBustGauge(
                plot_frame,
                player_name=entry["name"],
                predicted=p,
                ppr_low=low,
                ppr_high=high,
                boom_chance_pct=boom_pct,
                bust_chance_pct=bust_pct,
                width=370,
                height=130,
            ).pack(fill="x")

        self._show_drivers_var = tk.BooleanVar(value=False)
        FillCheckbox(
            self,
            self._show_drivers_var,
            text="Show what influences this score",
            command=self._toggle_drivers,
            bg=POPUP_BG,
            fg=FG,
        ).pack(anchor="w", padx=12, pady=(4, 0))

        self._drivers_holder = tk.Frame(self, bg=POPUP_BG, padx=12, pady=0)
        self._drivers_holder.pack(fill="x")

        pop = len(result.get("composite_populated", []))
        tk.Label(
            self,
            text=f"Composites used: {pop}/11",
            bg=POPUP_BG,
            fg="#B8D4FF",
            font=("Segoe UI", 9),
        ).pack(anchor="w", padx=12, pady=(4, 6))

        self.geometry(f"400x200+{x}+{y}")
        self.after_idle(self._fit_and_relayout_compare)

    def _fit_and_relayout_compare(self) -> None:
        self._fitted_height = _fit_popup_to_content(
            self, width=COMPARE_POPUP_WIDTH, bottom_pad=4
        )
        parent = self.master
        layout = getattr(parent, "_layout_compare_popups", None)
        if callable(layout):
            parent.after_idle(layout)

    def _toggle_drivers(self) -> None:
        if self._drivers_holder is None:
            return
        for child in self._drivers_holder.winfo_children():
            child.destroy()
        if not self._show_drivers_var.get():
            self._fit_and_relayout_compare()
            return

        drivers = list(self._result.get("score_drivers") or [])
        if not drivers:
            tk.Label(
                self._drivers_holder,
                text="No score drivers available for this player.",
                bg=POPUP_BG,
                fg="#B8D4FF",
                font=("Segoe UI", 8),
                wraplength=360,
                justify="left",
            ).pack(anchor="w", pady=(4, 2))
            self._fit_and_relayout_compare()
            return

        tk.Label(
            self._drivers_holder,
            text="Top drivers (Δ points vs typical / average):",
            bg=POPUP_BG,
            fg="#B8D4FF",
            font=("Segoe UI", 8),
            justify="left",
        ).pack(anchor="w", pady=(4, 2))

        for item in drivers:
            delta = float(item.get("delta_ppr") or 0.0)
            sign = "+" if delta >= 0 else ""
            color = BOOM_COLOR if delta >= 0 else BUST_COLOR
            row = tk.Frame(self._drivers_holder, bg=POPUP_BG)
            row.pack(fill="x", pady=1)
            tk.Label(
                row,
                text=str(item.get("label") or item.get("feature") or "—"),
                bg=POPUP_BG,
                fg=FG,
                font=("Segoe UI", 9),
                anchor="w",
            ).pack(side="left", fill="x", expand=True)
            tk.Label(
                row,
                text=f"{sign}{delta:.1f} points",
                bg=POPUP_BG,
                fg=color,
                font=("Segoe UI", 9, "bold"),
                anchor="e",
            ).pack(side="right")

        self._fit_and_relayout_compare()


class RookieScorerPanel(tk.Frame):
    """One format's scorer UI (redraft or dynasty) as an embeddable panel."""

    def __init__(
        self,
        master: tk.Misc,
        mode: str | AnalysisMode = "redraft",
        *,
        master_df=None,
        show_header: bool = True,
    ) -> None:
        super().__init__(master, bg=BG)
        self.analysis_mode = mode if isinstance(mode, AnalysisMode) else get_mode(mode)
        self.mode_name = self.analysis_mode.name
        is_dynasty = self.mode_name == "dynasty"
        title = "Dynasty Points Scorer" if is_dynasty else "Rookie Points Scorer"
        subtitle = (
            "Fantasy scoring format: PPR — predicted Y1–Y3 total points"
            if is_dynasty
            else "Fantasy scoring format: PPR"
        )

        try:
            self.master_df = master_df if master_df is not None else load_players_master()
            self.dropdowns = self._build_dropdowns()
            self.lookup_items = player_lookup_labels(self.master_df)
        except FileNotFoundError as exc:
            messagebox.showerror("Data missing", str(exc))
            self._init_failed = True
            return
        self._init_failed = False

        self.vars: dict[str, tk.StringVar] = {}
        self._field_widgets: dict[str, tk.Misc] = {}
        self._dropdown_widgets: list[SearchableDropdown] = []
        self._all_lookup_items = list(self.lookup_items)
        self._label_to_index = {label: idx for label, idx in self._all_lookup_items}
        self._compare_entries: list[dict] = []
        self._compare_vars: list[tk.BooleanVar] = []
        self._compare_popups: list[ComparePlayerPopup] = []
        self._prediction_popup: PredictionPopup | None = None

        if show_header:
            header = tk.Frame(self, bg=BG, padx=12, pady=10)
            header.pack(fill="x")
            tk.Label(
                header,
                text=title,
                bg=BG,
                fg=FG,
                font=("Segoe UI", 14, "bold"),
            ).pack(anchor="w")
            tk.Label(
                header,
                text=subtitle,
                bg=BG,
                fg="#B8D4FF",
                font=("Segoe UI", 9),
            ).pack(anchor="w", pady=(2, 0))
            tk.Label(
                header,
                text=(
                    f"Default lookup: {INCOMING_DRAFT_YEAR}–{INCOMING_DRAFT_YEAR + 1} rookies. "
                    "Type to filter. Add up to 4 players, then open a 2×2 grid of side popups."
                ),
                bg=BG,
                fg=FG,
                font=("Segoe UI", 9),
                wraplength=480,
                justify="left",
            ).pack(anchor="w", pady=(4, 0))
        else:
            # Compact subtitle under the shared tab bar
            sub = tk.Frame(self, bg=BG)
            sub.pack(fill="x", padx=12, pady=(8, 0))
            tk.Label(
                sub,
                text=subtitle,
                bg=BG,
                fg="#B8D4FF",
                font=("Segoe UI", 9),
            ).pack(anchor="w")
            tk.Label(
                sub,
                text=(
                    f"Default lookup: {INCOMING_DRAFT_YEAR}–{INCOMING_DRAFT_YEAR + 1} rookies. "
                    "Type to filter. Add up to 4 players, then open a 2×2 grid of side popups."
                ),
                bg=BG,
                fg=FG,
                font=("Segoe UI", 9),
                wraplength=480,
                justify="left",
            ).pack(anchor="w", pady=(2, 0))

        lookup_frame = tk.Frame(self, bg=BG, padx=12, pady=6)
        lookup_frame.pack(fill="x")
        tk.Label(
            lookup_frame,
            text=f"Player lookup ({INCOMING_DRAFT_YEAR} rookies)",
            bg=BG,
            fg=FG,
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor="w")

        self.include_past_var = tk.BooleanVar(value=False)
        FillCheckbox(
            lookup_frame,
            self.include_past_var,
            text="Include past players",
            command=self._on_include_past_toggled,
            bg=BG,
            fg=FG,
        ).pack(anchor="w", pady=(4, 2))

        self.player_var = tk.StringVar()
        self.player_combo = SearchableDropdown(
            lookup_frame,
            self._lookup_labels_for_mode(),
            textvariable=self.player_var,
            on_select=self._on_player_selected,
            width=58,
        )
        self.player_combo.pack(fill="x", pady=(4, 0))
        self._dropdown_widgets.append(self.player_combo)
        self._rebuild_lookup_index()

        compare_frame = tk.LabelFrame(
            self,
            text="  Compare players  ",
            bg=BG,
            fg=FG,
            font=("Segoe UI", 10, "bold"),
            padx=10,
            pady=8,
        )
        compare_frame.pack(fill="x", padx=12, pady=(0, 4))

        compare_btns = tk.Frame(compare_frame, bg=BG)
        compare_btns.pack(fill="x")
        tk.Button(
            compare_btns,
            text="Add current form",
            command=self._add_current_to_compare,
            bg=ACCENT_BTN,
            fg=FG,
            relief="flat",
            padx=8,
            pady=4,
        ).pack(side="left")
        tk.Button(
            compare_btns,
            text="Add lookup player",
            command=self._add_lookup_to_compare,
            bg=BG_FIELD,
            fg=FG,
            relief="flat",
            padx=8,
            pady=4,
        ).pack(side="left", padx=(6, 0))
        tk.Button(
            compare_btns,
            text="Open compare windows",
            command=self._open_compare_windows,
            bg=ACCENT_BTN,
            fg=FG,
            relief="flat",
            padx=8,
            pady=4,
        ).pack(side="left", padx=(6, 0))
        tk.Button(
            compare_btns,
            text="Clear compare",
            command=self._clear_compare,
            bg=BG_FIELD,
            fg=FG,
            relief="flat",
            padx=8,
            pady=4,
        ).pack(side="left", padx=(6, 0))

        self.compare_list_frame = tk.Frame(compare_frame, bg=BG)
        self.compare_list_frame.pack(fill="x", pady=(8, 0))
        self._refresh_compare_list()

        scroll = ScrollFrame(self)
        scroll.pack(fill="both", expand=True, padx=8, pady=4)

        for group in GROUP_ORDER:
            self._add_group(scroll.inner, group)

        btn_frame = tk.Frame(self, bg=BG, padx=12, pady=8)
        btn_frame.pack(fill="x")
        tk.Button(
            btn_frame,
            text="Calculate score",
            command=self._run_score,
            bg=ACCENT_BTN,
            fg=FG,
            activebackground="#2563B8",
            activeforeground=FG,
            font=("Segoe UI", 10, "bold"),
            relief="flat",
            padx=12,
            pady=6,
        ).pack(side="left")
        tk.Button(
            btn_frame,
            text="Clear all",
            command=self._clear_fields,
            bg=BG_FIELD,
            fg=FG,
            activebackground=ACCENT_BTN,
            activeforeground=FG,
            font=("Segoe UI", 10),
            relief="flat",
            padx=12,
            pady=6,
        ).pack(side="left", padx=(8, 0))

        self.result_var = tk.StringVar(value="Enter stats and click Calculate score. Prediction opens in a side window.")
        tk.Label(
            self,
            textvariable=self.result_var,
            bg=BG,
            fg=IMPORTANT_FG,
            font=("Segoe UI", 10),
            wraplength=480,
            justify="left",
            padx=12,
            pady=8,
        ).pack(fill="x")

    def _app_window(self) -> tk.Misc:
        return self.winfo_toplevel()

    def _lookup_labels_for_mode(self) -> list[str]:
        return [label for label, _ in self._filtered_lookup_items()]

    def _filtered_lookup_items(self) -> list[tuple[str, int]]:
        if self.include_past_var.get():
            return list(self._all_lookup_items)
        items: list[tuple[str, int]] = []
        for label, idx in self._all_lookup_items:
            year = self.master_df.loc[idx].get("draft_year")
            try:
                if int(year) == INCOMING_DRAFT_YEAR:
                    items.append((label, idx))
            except (TypeError, ValueError):
                continue
        return items

    def _rebuild_lookup_index(self) -> None:
        self._label_to_index = {label: idx for label, idx in self._filtered_lookup_items()}

    def _on_include_past_toggled(self) -> None:
        labels = self._lookup_labels_for_mode()
        self.player_combo.set_full_values(labels)
        self._rebuild_lookup_index()
        current = self.player_var.get().strip()
        if current and current not in self._label_to_index:
            self.player_var.set("")

    def _build_dropdowns(self) -> dict[str, list[str]]:
        from rookie_ppr.score_runner import dropdown_options

        opts = dropdown_options(self.master_df)
        for spec in FIELD_SPECS:
            if spec.widget == "dropdown" and spec.options:
                opts[spec.column] = list(spec.options)
        return opts

    def _add_group(self, parent: tk.Frame, group: str) -> None:
        specs = [s for s in FIELD_SPECS if s.group == group]
        if not specs:
            return
        box = tk.LabelFrame(
            parent,
            text=f"  {group}  ",
            bg=BG,
            fg=FG,
            font=("Segoe UI", 10, "bold"),
            labelanchor="nw",
            padx=8,
            pady=6,
        )
        box.pack(fill="x", padx=4, pady=6)

        for spec in specs:
            row = tk.Frame(box, bg=BG)
            row.pack(fill="x", pady=3)

            label_text = spec.label
            if spec.important:
                label_text += "  (Important)"

            tk.Label(
                row,
                text=label_text,
                bg=BG,
                fg=IMPORTANT_FG if spec.important else FG,
                font=("Segoe UI", 9, "bold" if spec.important else "normal"),
                anchor="w",
            ).pack(anchor="w")

            tk.Label(
                row,
                text=spec.hint,
                bg=BG,
                fg="#B8D4FF",
                font=("Segoe UI", 8),
                anchor="w",
            ).pack(anchor="w")

            var = tk.StringVar()
            self.vars[spec.column] = var

            if spec.widget == "dropdown":
                values = self.dropdowns.get(spec.column, list(spec.options))
                widget = SearchableDropdown(row, values, textvariable=var, width=54)
                self._dropdown_widgets.append(widget)
            else:
                widget = tk.Entry(
                    row,
                    textvariable=var,
                    bg=BG_FIELD,
                    fg=FG,
                    insertbackground=FG,
                    relief="flat",
                    font=("Segoe UI", 10),
                )
            self._field_widgets[spec.column] = widget
            widget.pack(fill="x", pady=(2, 0), ipady=3)

    def _on_player_selected(self) -> None:
        label = self.player_var.get().strip()
        if not label or label not in self._label_to_index:
            return
        # New search closes the prediction popup unless compare windows are open
        if not self._comparing_active():
            self._close_prediction_popup()
        idx = self._label_to_index[label]
        data = row_from_player(self.master_df, idx)
        for col, var in self.vars.items():
            if col in data:
                var.set(str(data[col]))
            else:
                var.set("")

    def _comparing_active(self) -> bool:
        alive = []
        for popup in self._compare_popups:
            try:
                if popup.winfo_exists():
                    alive.append(popup)
            except tk.TclError:
                pass
        self._compare_popups = alive
        return bool(alive)

    def _close_prediction_popup(self) -> None:
        if self._prediction_popup is not None:
            try:
                self._prediction_popup.destroy()
            except tk.TclError:
                pass
            self._prediction_popup = None

    def _open_prediction_popup(self, name: str, result: dict) -> None:
        self._close_prediction_popup()
        self.update_idletasks()
        top = self._app_window()
        x = top.winfo_x() + top.winfo_width() + 12
        y = top.winfo_y()
        self._prediction_popup = PredictionPopup(top, name, result, x, y)

    def _clear_fields(self) -> None:
        self._hide_all_dropdowns()
        if not self._comparing_active():
            self._close_prediction_popup()
        self.player_var.set("")
        for var in self.vars.values():
            var.set("")
        self.result_var.set("Fields cleared.")

    def _hide_all_dropdowns(self) -> None:
        for dd in self._dropdown_widgets:
            try:
                dd._hide_popup()
            except tk.TclError:
                pass

    def _collect_row(self) -> dict:
        """Read current form values (live widget text + StringVars)."""
        self._hide_all_dropdowns()
        self.update_idletasks()
        row: dict = {}
        for col, var in self.vars.items():
            widget = self._field_widgets.get(col)
            if isinstance(widget, SearchableDropdown):
                text = widget.entry.get().strip()
            elif isinstance(widget, tk.Entry):
                text = widget.get().strip()
            else:
                text = var.get().strip()
            if text:
                row[col] = text
                # Keep StringVar in sync for later edits / compare
                if var.get() != text:
                    var.set(text)
        return row

    def _display_name(self, row: dict) -> str:
        lookup = self.player_var.get().strip()
        if lookup:
            return lookup.split(" (")[0]
        pos = row.get("position", "?")
        pick = row.get("draft_overall", "?")
        return f"Custom ({pos}, pick {pick})"

    def _score_row(self, row: dict) -> dict | None:
        if not row.get("position"):
            messagebox.showwarning("Missing position", "Select or enter a position (QB, RB, WR, TE).")
            return None
        try:
            return score_player(row, mode=self.analysis_mode)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Scoring error", str(exc))
            return None

    def _add_to_compare(self, name: str, row: dict, result: dict) -> None:
        if any(entry["name"] == name for entry in self._compare_entries):
            messagebox.showinfo("Compare", f"{name} is already in the compare list.")
            return
        if len(self._compare_entries) >= MAX_COMPARE_WINDOWS:
            messagebox.showinfo(
                "Compare",
                f"You can compare up to {MAX_COMPARE_WINDOWS} players (2×2 grid). "
                "Clear compare or remove a player first.",
            )
            return
        self._compare_entries.append({"name": name, "row": row.copy(), "result": result})
        self._compare_vars.append(tk.BooleanVar(value=True))
        self._refresh_compare_list()

    def _refresh_compare_list(self) -> None:
        for child in self.compare_list_frame.winfo_children():
            child.destroy()
        if not self._compare_entries:
            tk.Label(
                self.compare_list_frame,
                text=(
                    f"No players added yet. Add up to {MAX_COMPARE_WINDOWS}, "
                    "check them, then open compare windows."
                ),
                bg=BG,
                fg="#B8D4FF",
                font=("Segoe UI", 8),
                anchor="w",
            ).pack(anchor="w")
            return
        for i, entry in enumerate(self._compare_entries):
            row = tk.Frame(self.compare_list_frame, bg=BG)
            row.pack(fill="x", pady=2)
            FillCheckbox(
                row,
                self._compare_vars[i],
                text="",
                bg=BG,
                fg=FG,
            ).pack(side="left", padx=(0, 4))
            result = entry["result"]
            pts = _result_predicted_points(result)
            sc = _result_success_score(result)
            tk.Label(
                row,
                text=(
                    f"{entry['name']}  —  {pts:.1f} points, "
                    f"Score {sc:.1f}"
                ),
                bg=BG,
                fg=FG,
                font=("Segoe UI", 9),
                anchor="w",
            ).pack(side="left", fill="x", expand=True)

    def _add_current_to_compare(self) -> None:
        row = self._collect_row()
        result = self._score_row(row)
        if result is None:
            return
        self._add_to_compare(self._display_name(row), row, result)

    def _add_lookup_to_compare(self) -> None:
        label = self.player_var.get().strip()
        if not label or label not in self._label_to_index:
            messagebox.showwarning("Lookup", "Select a player from the lookup dropdown first.")
            return
        idx = self._label_to_index[label]
        row = row_from_player(self.master_df, idx)
        if not row.get("position"):
            messagebox.showwarning("Lookup", "Selected player is missing a position.")
            return
        result = self._score_row(row)
        if result is None:
            return
        self._add_to_compare(label.split(" (")[0], row, result)

    def _compare_anchor(self) -> tuple[int, int]:
        """Top-left for the compare grid: beside the main picker (past prediction if open)."""
        self.update_idletasks()
        top = self._app_window()
        base_x = top.winfo_x() + top.winfo_width() + 12
        base_y = top.winfo_y()
        if self._prediction_popup is not None:
            try:
                if self._prediction_popup.winfo_exists():
                    pred_right = (
                        self._prediction_popup.winfo_x()
                        + self._prediction_popup.winfo_width()
                        + COMPARE_GRID_GAP
                    )
                    base_x = max(base_x, pred_right)
            except tk.TclError:
                pass
        return base_x, base_y

    def _compare_window_positions(
        self,
        count: int,
        *,
        cell_w: int | None = None,
        cell_h: int | None = None,
        row_heights: list[int] | None = None,
    ) -> tuple[list[tuple[int, int]], int, list[int]]:
        """2×2 slot positions beside the main popup; clamps/shrinks if screen is tight.

        Returns (positions, cell_width, row_heights) after any screen-fit shrink.
        """
        if count <= 0:
            return [], cell_w or COMPARE_POPUP_WIDTH, []

        gap = COMPARE_GRID_GAP
        width = cell_w or COMPARE_POPUP_WIDTH
        est_h = cell_h or COMPARE_POPUP_EST_HEIGHT
        base_x, base_y = self._compare_anchor()

        cols = min(2, count)
        rows = (count + 1) // 2
        if row_heights is None:
            row_heights = [est_h] * rows
        while len(row_heights) < rows:
            row_heights.append(est_h)

        grid_w = cols * width + max(0, cols - 1) * gap
        grid_h = sum(row_heights[:rows]) + max(0, rows - 1) * gap

        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        available_w = max(COMPARE_POPUP_MIN_WIDTH, screen_w - base_x - 12)
        available_h = max(COMPARE_POPUP_MIN_HEIGHT, screen_h - base_y - 12)

        # Shrink cell width if the 2-column grid cannot fit to the right
        if grid_w > available_w and cols > 0:
            width = max(
                COMPARE_POPUP_MIN_WIDTH,
                (available_w - max(0, cols - 1) * gap) // cols,
            )
            grid_w = cols * width + max(0, cols - 1) * gap

        if grid_h > available_h and rows > 0:
            scale = available_h / grid_h
            row_heights = [
                max(COMPARE_POPUP_MIN_HEIGHT, int(h * scale)) for h in row_heights[:rows]
            ]
            grid_h = sum(row_heights) + max(0, rows - 1) * gap

        if base_x + grid_w > screen_w - 8:
            base_x = max(0, screen_w - grid_w - 8)
        if base_y + grid_h > screen_h - 8:
            base_y = max(0, screen_h - grid_h - 8)

        positions: list[tuple[int, int]] = []
        for i in range(count):
            col = i % 2
            row = i // 2
            x = base_x + col * (width + gap)
            y = base_y + sum(row_heights[:row]) + row * gap
            positions.append((x, y))
        return positions, width, row_heights[:rows]

    def _layout_compare_popups(self) -> None:
        """Reposition open compare windows into a non-overlapping 2×2 grid."""
        alive: list[ComparePlayerPopup] = []
        for popup in self._compare_popups:
            try:
                if popup.winfo_exists():
                    alive.append(popup)
            except tk.TclError:
                pass
        self._compare_popups = alive
        if not alive:
            return

        # Fit each window to content first so tiling never reads a stale
        # pre-fit height (e.g. initial 400x200) and shrinks them again.
        natural_heights: list[int] = []
        for popup in alive:
            natural_heights.append(
                _fit_popup_to_content(popup, width=COMPARE_POPUP_WIDTH, bottom_pad=4)
            )

        rows = (len(alive) + 1) // 2
        row_heights: list[int] = []
        for row in range(rows):
            idxs = [i for i in range(len(alive)) if i // 2 == row]
            if idxs:
                row_heights.append(max(natural_heights[i] for i in idxs))
            else:
                row_heights.append(COMPARE_POPUP_EST_HEIGHT)

        positions, cell_w, row_heights = self._compare_window_positions(
            len(alive),
            cell_w=COMPARE_POPUP_WIDTH,
            row_heights=row_heights,
        )

        for i, (popup, (x, y)) in enumerate(zip(alive, positions)):
            row = i // 2
            slot_h = row_heights[row] if row < len(row_heights) else natural_heights[i]
            natural_h = natural_heights[i]
            try:
                if cell_w != COMPARE_POPUP_WIDTH:
                    natural_h = _fit_popup_to_content(popup, width=cell_w, bottom_pad=4)
                    natural_heights[i] = natural_h
                # Prefer natural content height; only modestly shrink when the
                # screen cannot fit the 2×2 grid (avoids overlap / off-screen clip).
                h = natural_h if natural_h <= slot_h else slot_h
                popup.geometry(f"{cell_w}x{h}+{x}+{y}")
                popup._fitted_height = h
            except tk.TclError:
                pass
        # Commit geometries before any follow-up idle callbacks read sizes.
        self.update_idletasks()

    def _open_compare_windows(self) -> None:
        selected = [
            self._compare_entries[i]
            for i, var in enumerate(self._compare_vars)
            if i < len(self._compare_entries) and var.get()
        ]
        if not selected:
            messagebox.showinfo("Compare", "Check at least one player in the compare list.")
            return
        if len(selected) > MAX_COMPARE_WINDOWS:
            messagebox.showinfo(
                "Compare",
                f"You can open at most {MAX_COMPARE_WINDOWS} compare windows (2×2 grid). "
                "Uncheck extras, then try again.",
            )
            return

        # Compare mode: keep prediction popup if open; place compare windows in a 2×2 grid beside picker
        for popup in self._compare_popups:
            try:
                popup.destroy()
            except tk.TclError:
                pass
        self._compare_popups.clear()

        positions, _, _ = self._compare_window_positions(len(selected))
        for entry, (x, y) in zip(selected, positions):
            popup = ComparePlayerPopup(self, entry, x, y)
            self._compare_popups.append(popup)

        # After content fit settles, re-tile so taller windows do not overlap the row below
        self.after_idle(self._layout_compare_popups)

    def _clear_compare(self) -> None:
        for popup in self._compare_popups:
            try:
                popup.destroy()
            except tk.TclError:
                pass
        self._compare_popups.clear()
        self._compare_entries.clear()
        self._compare_vars.clear()
        self._refresh_compare_list()

    def _run_score(self) -> None:
        # Leave the active entry so the latest typed value is committed, then score.
        self.focus_set()
        row = self._collect_row()
        result = self._score_row(row)
        if result is None:
            return

        pop = len(result["composite_populated"])
        miss = len(result["composite_missing"])
        # Main window keeps missing / coverage info; prediction opens beside
        detail = f"Composites filled: {pop}/11"
        if miss:
            detail += f"  |  Missing: {', '.join(result['composite_missing'])}"
        else:
            detail += "  |  All composite groups populated"
        self.result_var.set(detail)

        self._open_prediction_popup(self._display_name(row), result)


def main(mode: str | None = None) -> int:
    """
    Launch the combined scorer UI with Redraft / Dynasty / Veterans tabs.

    CLI: ``python -m rookie_ppr.ui`` or ``--mode redraft|dynasty|veteran``.
    """
    import argparse

    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--mode",
        choices=("redraft", "dynasty", "veteran"),
        default=mode or "redraft",
        help="Initial tab. All available tabs stay in the UI.",
    )
    args, _unknown = parser.parse_known_args()

    app = TabbedScorerApp(initial_mode=args.mode)
    if getattr(app, "_init_failed", False) or not app.winfo_exists():
        return 1
    app.mainloop()
    return 0


class TabbedScorerApp(tk.Tk):
    """Single window with browser-style Redraft / Dynasty tabs."""

    def __init__(self, initial_mode: str = "redraft") -> None:
        super().__init__()
        self.title("Fantasy Points Scorer")
        self.configure(bg=BG)
        self.geometry("520x800")
        self.minsize(480, 640)
        self._init_failed = False
        self._active_mode = "redraft"
        self._theme_id = DEFAULT_THEME_ID
        self._theme_menu: ThemeHelmetMenu | None = None
        self._panels: dict[str, tk.Frame] = {}
        self._tab_buttons: dict[str, tk.Button] = {}
        self._tab_accents: dict[str, tk.Frame] = {}

        try:
            master_df = load_players_master()
        except FileNotFoundError as exc:
            messagebox.showerror("Data missing", str(exc))
            self._init_failed = True
            self.destroy()
            return

        # Tab strip
        self._tab_bar = tk.Frame(self, bg=TAB_BAR_BG, padx=8, pady=0)
        self._tab_bar.pack(fill="x")

        tabs_row = tk.Frame(self._tab_bar, bg=TAB_BAR_BG)
        tabs_row.pack(side="left", anchor="sw")

        for key, label in (("redraft", "Redraft"), ("dynasty", "Dynasty"), ("veteran", "Veterans")):
            col = tk.Frame(tabs_row, bg=TAB_BAR_BG)
            col.pack(side="left", padx=(0, 2))
            btn = tk.Button(
                col,
                text=label,
                command=lambda m=key: self._select_tab(m),
                relief="flat",
                bd=0,
                padx=18,
                pady=8,
                font=("Segoe UI", 11),
                cursor="hand2",
            )
            btn.pack(fill="x")
            accent = tk.Frame(col, bg=TAB_BAR_BG, height=3)
            accent.pack(fill="x")
            self._tab_buttons[key] = btn
            self._tab_accents[key] = accent

        right = tk.Frame(self._tab_bar, bg=TAB_BAR_BG)
        right.pack(side="right", padx=(4, 2), pady=4)
        self._helmet_btn = HelmetButton(
            right,
            theme_id=self._theme_id,
            command=self._toggle_theme_menu,
            size=28,
        )
        self._helmet_btn.pack(side="right")
        self._crest_btn = CrestButton(
            right,
            command=self._open_nerd_united,
            size=28,
        )
        self._crest_btn.pack(side="right", padx=(0, 8))
        self._ppr_label = tk.Label(
            right,
            text="PPR",
            bg=TAB_BAR_BG,
            fg=TAB_INACTIVE_FG,
            font=("Segoe UI", 9),
        )
        self._ppr_label.pack(side="right", padx=(0, 8))

        # Content host
        self._content = tk.Frame(self, bg=BG)
        self._content.pack(fill="both", expand=True)

        self._panels["redraft"] = RookieScorerPanel(
            self._content, mode="redraft", master_df=master_df, show_header=False
        )
        self._panels["dynasty"] = RookieScorerPanel(
            self._content, mode="dynasty", master_df=master_df, show_header=False
        )
        self._panels["veteran"] = VeteranScorerPanel(self._content)
        if any(getattr(p, "_init_failed", False) for p in self._panels.values()):
            # Veterans can fail if compile not run; keep redraft/dynasty usable
            if getattr(self._panels["veteran"], "_init_failed", False):
                self._panels.pop("veteran", None)
                btn = self._tab_buttons.pop("veteran", None)
                self._tab_accents.pop("veteran", None)
                if btn is not None:
                    try:
                        btn.master.destroy()
                    except Exception:  # noqa: BLE001
                        pass
            if any(getattr(self._panels.get(k), "_init_failed", False) for k in ("redraft", "dynasty")):
                self._init_failed = True
                self.destroy()
                return

        # Team mascot overlay (bottom-right); hidden on default theme
        self._mascot_photo: object | None = None
        self._mascot_label = tk.Label(self._content, bg=BG, bd=0, highlightthickness=0)
        self._mascot_label.place(relx=1.0, rely=1.0, x=-10, y=-8, anchor="se")
        self._update_mascot_overlay()

        start = initial_mode if initial_mode in self._panels else "redraft"
        self._select_tab(start)

    def _mascot_path(self, theme_id: str):
        if theme_id == DEFAULT_THEME_ID:
            return None
        path = MASCOT_DIR / f"{theme_id}.png"
        return path if path.exists() else None

    def _update_mascot_overlay(self) -> None:
        path = self._mascot_path(self._theme_id)
        if path is None:
            self._mascot_photo = None
            self._mascot_label.configure(image="", bg=BG)
            self._mascot_label.place_forget()
            return
        try:
            from PIL import Image, ImageTk

            img = Image.open(path).convert("RGBA")
            # Corner accent — high-res assets downscaled for display (~1/3 of prior 80px)
            max_h = 27
            if img.height > max_h:
                w = max(1, int(img.width * (max_h / img.height)))
                img = img.resize((w, max_h), Image.Resampling.LANCZOS)
            self._mascot_photo = ImageTk.PhotoImage(img)
            self._mascot_label.configure(image=self._mascot_photo, bg=BG)
            self._mascot_label.place(relx=1.0, rely=1.0, x=-10, y=-8, anchor="se")
            self._mascot_label.lift()
        except Exception:  # noqa: BLE001
            self._mascot_photo = None
            self._mascot_label.place_forget()

    def _toggle_theme_menu(self) -> None:
        if self._theme_menu is not None:
            try:
                if self._theme_menu.winfo_exists():
                    self._theme_menu.destroy()
                    self._theme_menu = None
                    return
            except tk.TclError:
                self._theme_menu = None
        self._theme_menu = ThemeHelmetMenu(
            self,
            current_id=self._theme_id,
            on_pick=self._set_theme,
            anchor_widget=self._helmet_btn,
        )

    def _set_theme(self, theme_id: str) -> None:
        theme = THEME_CATALOG.get(theme_id)
        if theme is None:
            return
        old = _snapshot_theme_globals()
        _apply_theme_globals(theme)
        new = _snapshot_theme_globals()
        color_map = {old[k]: new[k] for k in old if old[k] != new[k]}
        # Also remap common hardcoded muted blues from the default palette
        color_map.setdefault("#B8D4FF", new["MUTED_FG"])
        color_map.setdefault("#B8D0F0", new["MUTED_FG"])
        color_map.setdefault("#9BB8E8", new["TAB_INACTIVE_FG"])

        self.configure(bg=BG)
        _restyle_widget_tree(self, color_map)
        self._theme_id = theme_id
        self._helmet_btn.redraw(theme_id, bar_bg=TAB_BAR_BG)
        self._crest_btn.redraw(bar_bg=TAB_BAR_BG)
        self._select_tab(self._active_mode)
        self._update_mascot_overlay()
        self._theme_menu = None

    def _open_nerd_united(self) -> None:
        open_nerd_united_window(self)

    def _select_tab(self, mode: str) -> None:
        if mode not in self._panels:
            return
        self._active_mode = mode
        for key, panel in self._panels.items():
            if key == mode:
                panel.pack(fill="both", expand=True)
            else:
                panel.pack_forget()
                # Hide open dropdowns on the inactive tab
                try:
                    panel._hide_all_dropdowns()
                except Exception:  # noqa: BLE001
                    pass

        for key, btn in self._tab_buttons.items():
            active = key == mode
            btn.configure(
                bg=TAB_ACTIVE_BG if active else TAB_INACTIVE_BG,
                fg=TAB_ACTIVE_FG if active else TAB_INACTIVE_FG,
                activebackground=TAB_ACTIVE_BG if active else TAB_INACTIVE_BG,
                activeforeground=TAB_ACTIVE_FG if active else TAB_INACTIVE_FG,
                font=("Segoe UI", 11, "bold") if active else ("Segoe UI", 11),
            )
            self._tab_accents[key].configure(bg=TAB_ACCENT if active else TAB_BAR_BG)

        if mode == "dynasty":
            subtitle = "Y1–Y3 dynasty"
        elif mode == "veteran":
            subtitle = "veteran next-season"
        else:
            subtitle = "rookie season (redraft)"
        self.title(f"Fantasy Points Scorer — {subtitle}")
        try:
            self._mascot_label.lift()
        except Exception:  # noqa: BLE001
            pass


# Back-compat aliases
class RookieScorerApp(TabbedScorerApp):
    """Legacy name — launches the tabbed app (optional initial mode)."""

    def __init__(self, mode: str | AnalysisMode = "redraft") -> None:
        name = mode.name if isinstance(mode, AnalysisMode) else str(mode or "redraft")
        super().__init__(
            initial_mode=name if name in ("redraft", "dynasty", "veteran") else "redraft"
        )


class DynastyScorerApp(TabbedScorerApp):
    """Legacy name — opens on the Dynasty tab."""

    def __init__(self) -> None:
        super().__init__(initial_mode="dynasty")


if __name__ == "__main__":
    raise SystemExit(main())
