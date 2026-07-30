"""Veterans tab: rookie-style lookup form + projection popup.

Matches the Rookies layout: search fills grouped prior-season stats; a side
popup shows season projection, injury/availability score, elite score, the
boom/bust gauge, and week-by-week projections underneath.
"""
from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass
from tkinter import messagebox, ttk
from typing import Any

import numpy as np
import pandas as pd

from rookie_ppr.config import CSV_OUTPUT_DIR
from rookie_ppr.veteran.config import (
    VET_DRAFT_BOARD_CSV,
    VET_FEATURES_CSV,
    VET_WEEKLY_COMPARE_CSV,
    VET_WEEKLY_CSV,
)


@dataclass(frozen=True)
class VetFieldSpec:
    column: str
    label: str
    hint: str
    group: str
    important: bool = False
    digits: int | None = 1  # None = raw string


VET_GROUP_ORDER = (
    "Core",
    "Prior production",
    "Role",
    "Trajectory",
    "Market",
)

VET_FIELD_SPECS: list[VetFieldSpec] = [
    VetFieldSpec("position", "Position", "(QB, RB, WR, TE)", "Core", True, None),
    VetFieldSpec("team", "NFL team", "(current / latest)", "Core", True, None),
    VetFieldSpec("season", "Prior season", "(feature season)", "Core", False, 0),
    VetFieldSpec("target_season", "Forecast season", "(season being projected)", "Core", True, 0),
    VetFieldSpec("age", "Age", "(as of prior season)", "Core", False, 0),
    VetFieldSpec("career_seasons", "Career seasons", "(NFL seasons played)", "Core", False, 0),
    VetFieldSpec("team_changed", "Changed team", "(1 = yes)", "Core", False, 0),
    VetFieldSpec("ppr", "Prior PPR", "(full prior season)", "Prior production", True),
    VetFieldSpec("games", "Games played", "(prior season)", "Prior production", True, 0),
    VetFieldSpec("ppr_per_game", "PPR / game", "(prior season rate)", "Prior production", True),
    VetFieldSpec("touches", "Touches", "(carries + receptions)", "Prior production"),
    VetFieldSpec("carries", "Carries", "", "Prior production", False, 0),
    VetFieldSpec("targets", "Targets", "", "Prior production", False, 0),
    VetFieldSpec("receptions", "Receptions", "", "Prior production", False, 0),
    VetFieldSpec("rushing_yards", "Rushing yards", "", "Prior production", False, 0),
    VetFieldSpec("receiving_yards", "Receiving yards", "", "Prior production", False, 0),
    VetFieldSpec("passing_yards", "Passing yards", "", "Prior production", False, 0),
    VetFieldSpec("rushing_tds", "Rushing TDs", "", "Prior production"),
    VetFieldSpec("receiving_tds", "Receiving TDs", "", "Prior production"),
    VetFieldSpec("passing_tds", "Passing TDs", "", "Prior production"),
    VetFieldSpec("off_snap_pct", "Snap %", "(offense snap share)", "Role", True),
    VetFieldSpec("depth_rank", "Depth rank", "(1 = starter)", "Role", True, 0),
    VetFieldSpec("target_share", "Target share", "(of team targets)", "Role"),
    VetFieldSpec("carry_share", "Carry share", "(of team carries)", "Role"),
    VetFieldSpec("touch_share", "Touch share", "(of team touches)", "Role"),
    VetFieldSpec("lag1_ppr", "Lag-1 PPR", "(two seasons ago)", "Trajectory"),
    VetFieldSpec("lag2_ppr", "Lag-2 PPR", "(three seasons ago)", "Trajectory"),
    VetFieldSpec("delta_ppr", "Δ PPR", "(prior − lag1)", "Trajectory", True),
    VetFieldSpec("ppr_trail3_mean", "Trail-3 mean PPR", "(last 3 seasons)", "Trajectory"),
    VetFieldSpec("adp_rank", "FantasyPros ADP", "(pre-season overall)", "Market", True, 0),
    VetFieldSpec("adp_pos_rank", "ADP position rank", "(within position)", "Market", False, 0),
]


def _ui():
    """Late import — ui.py imports this module for the Veterans tab."""
    from rookie_ppr import ui as ui_mod

    return ui_mod


def load_veteran_features() -> pd.DataFrame:
    path = CSV_OUTPUT_DIR / VET_FEATURES_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run: python -m rookie_ppr.veteran.compile"
        )
    return pd.read_csv(path, low_memory=False)


def _fmt(val: Any, digits: int | None = 1) -> str:
    try:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return ""
        if digits is None:
            return str(val)
        return f"{float(val):.{digits}f}"
    except (TypeError, ValueError):
        return "" if val is None else str(val)


def _injury_score(games: float, *, full_season: float = 17.0) -> float:
    """0–100 availability score from prior-season games (higher = healthier)."""
    if not np.isfinite(games) or full_season <= 0:
        return float("nan")
    return float(np.clip(100.0 * float(games) / full_season, 0.0, 100.0))


def _elite_score(pred: float, peer_preds: np.ndarray) -> float:
    """0–100 peer percentile of next-season projection within position."""
    if not np.isfinite(pred) or peer_preds.size == 0:
        return float("nan")
    peers = peer_preds[np.isfinite(peer_preds)]
    if peers.size == 0:
        return float("nan")
    return float(100.0 * (np.sum(peers <= pred) / peers.size))


class VeteranPredictionPopup(tk.Toplevel):
    """Side popup: season projection, injury/elite scores, weekly lines."""

    def __init__(
        self,
        parent: tk.Misc,
        *,
        name: str,
        result: dict[str, Any],
        weekly: pd.DataFrame,
        x: int,
        y: int,
    ) -> None:
        super().__init__(parent)
        ui = _ui()
        self.title(f"Prediction — {name}")
        self.configure(bg=ui.POPUP_BG)
        self.minsize(380, 200)

        pred = float(result["predicted_ppr"])
        target = result.get("target_season")
        tgt_s = int(target) if target is not None and pd.notna(target) else "next"
        injury = result.get("injury_score")
        elite = result.get("elite_score")
        q1 = result.get("q25")
        q3 = result.get("q75")
        low = result.get("ppr_low", q1)
        high = result.get("ppr_high", q3)
        games = result.get("games")

        header = tk.Frame(self, bg=ui.POPUP_BG, padx=12, pady=8)
        header.pack(fill="x")
        tk.Label(
            header,
            text=name,
            bg=ui.POPUP_BG,
            fg=ui.FG,
            font=("Segoe UI", 12, "bold"),
            wraplength=380,
            justify="left",
        ).pack(anchor="w")
        tk.Label(
            header,
            text=f"Season projection ({tgt_s}): {pred:.1f} PPR",
            bg=ui.POPUP_BG,
            fg=ui.IMPORTANT_FG,
            font=("Segoe UI", 12, "bold"),
            justify="left",
        ).pack(anchor="w", pady=(4, 0))
        pace_17 = result.get("projected_ppr_17")
        if pace_17 is not None and np.isfinite(pace_17):
            tk.Label(
                header,
                text=f"17-game pace: {float(pace_17):.1f}",
                bg=ui.POPUP_BG,
                fg="#B8D4FF",
                font=("Segoe UI", 9),
            ).pack(anchor="w", pady=(1, 0))

        scores = tk.Frame(header, bg=ui.POPUP_BG)
        scores.pack(fill="x", pady=(6, 0))
        inj_txt = (
            f"Injury score: {injury:.0f}/100"
            if injury is not None and np.isfinite(injury)
            else "Injury score: —"
        )
        elite_txt = (
            f"Elite score: {elite:.0f}/100"
            if elite is not None and np.isfinite(elite)
            else "Elite score: —"
        )
        tk.Label(
            scores,
            text=inj_txt,
            bg=ui.POPUP_BG,
            fg=ui.FG,
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor="w")
        tk.Label(
            scores,
            text=(
                f"Prior-season availability ({_fmt(games, 0) or '—'}/17 games). "
                "Higher = healthier / more available."
            ),
            bg=ui.POPUP_BG,
            fg="#B8D4FF",
            font=("Segoe UI", 8),
            wraplength=380,
            justify="left",
        ).pack(anchor="w")
        tk.Label(
            scores,
            text=elite_txt,
            bg=ui.POPUP_BG,
            fg=ui.FG,
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor="w", pady=(4, 0))
        tk.Label(
            scores,
            text="Peer percentile of this projection within the same position.",
            bg=ui.POPUP_BG,
            fg="#B8D4FF",
            font=("Segoe UI", 8),
            wraplength=380,
            justify="left",
        ).pack(anchor="w")

        q1_f = float(q1) if q1 is not None and pd.notna(q1) else float("nan")
        q3_f = float(q3) if q3 is not None and pd.notna(q3) else float("nan")
        low_f = float(low) if low is not None and pd.notna(low) else q1_f
        high_f = float(high) if high is not None and pd.notna(high) else q3_f
        if np.isfinite(q1_f) and np.isfinite(q3_f):
            tk.Label(
                header,
                text=(
                    f"Predictive IQR: {q1_f:.0f}–{q3_f:.0f}  ·  "
                    f"80% band: {_fmt(low_f, 0) or '—'}–{_fmt(high_f, 0) or '—'}"
                ),
                bg=ui.POPUP_BG,
                fg="#B8D4FF",
                font=("Segoe UI", 8),
            ).pack(anchor="w", pady=(6, 0))

        plot_frame = tk.Frame(self, bg=ui.POPUP_BG, padx=8, pady=2)
        plot_frame.pack(fill="x")
        band_lo = low_f if np.isfinite(low_f) else pred * 0.7
        band_hi = high_f if np.isfinite(high_f) else pred * 1.3
        if np.isfinite(q1_f) and np.isfinite(q3_f):
            bust_pct = float(np.clip(25.0 * (q1_f / max(pred, 1.0)), 5.0, 40.0))
            boom_pct = float(np.clip(25.0 * (max(pred, 1.0) / max(q3_f, 1.0)), 5.0, 55.0))
        else:
            bust_pct, boom_pct = 25.0, 25.0
        ui.BoomBustGauge(
            plot_frame,
            player_name=name,
            predicted=pred,
            ppr_low=band_lo,
            ppr_high=band_hi,
            boom_chance_pct=boom_pct,
            bust_chance_pct=bust_pct,
            width=390,
            height=120,
        ).pack(fill="x")

        weekly_box = tk.LabelFrame(
            self,
            text="  Weekly projections  ",
            bg=ui.POPUP_BG,
            fg=ui.FG,
            font=("Segoe UI", 10, "bold"),
            padx=8,
            pady=6,
        )
        weekly_box.pack(fill="both", expand=True, padx=10, pady=(6, 8))

        if weekly is None or weekly.empty:
            tk.Label(
                weekly_box,
                text="No weekly projections for this player.",
                bg=ui.POPUP_BG,
                fg="#B8D4FF",
                font=("Segoe UI", 9),
            ).pack(anchor="w")
        else:
            pts = pd.to_numeric(weekly.get("week_points"), errors="coerce").fillna(0)
            total = float(pts.sum())
            if "is_bye" in weekly.columns:
                n_games = int((~weekly["is_bye"].astype(bool)).sum())
            else:
                n_games = int(len(weekly))
            tk.Label(
                weekly_box,
                text=f"{total:.1f} pts over {n_games} scheduled games (bye = 0)",
                bg=ui.POPUP_BG,
                fg=ui.IMPORTANT_FG,
                font=("Segoe UI", 9),
            ).pack(anchor="w", pady=(0, 4))

            cols = ("week", "opp", "matchup", "pts")
            tree = ttk.Treeview(
                weekly_box, columns=cols, show="headings", height=min(12, len(weekly))
            )
            for c, w in (("week", 44), ("opp", 50), ("matchup", 70), ("pts", 52)):
                tree.heading(c, text=c)
                tree.column(c, width=w, anchor="center")
            sb = ttk.Scrollbar(weekly_box, orient="vertical", command=tree.yview)
            tree.configure(yscrollcommand=sb.set)
            tree.pack(side="left", fill="both", expand=True)
            sb.pack(side="right", fill="y")
            for r in weekly.sort_values("week").itertuples(index=False):
                bye = bool(getattr(r, "is_bye", False))
                tree.insert(
                    "",
                    "end",
                    values=(
                        int(r.week),
                        "BYE" if bye else (getattr(r, "opponent", "") or "—"),
                        "—" if bye else (getattr(r, "matchup", "") or "—"),
                        _fmt(getattr(r, "week_points", None)) or "—",
                    ),
                )

        self.geometry(f"420x520+{x}+{y}")
        self.after_idle(lambda: ui._fit_popup_to_content(self, width=420, bottom_pad=8))


class VeteranScorerPanel(tk.Frame):
    """Rookie-style lookup: fill prior-season stats, open projection popup."""

    def __init__(self, master: tk.Misc) -> None:
        ui = _ui()
        super().__init__(master, bg=ui.BG)
        self._init_failed = False
        self._prediction_popup: VeteranPredictionPopup | None = None
        self._dropdown_widgets: list[Any] = []

        try:
            self.df = load_veteran_features()
        except FileNotFoundError as exc:
            messagebox.showerror("Veteran data missing", str(exc))
            self._init_failed = True
            return

        self.df["season"] = pd.to_numeric(self.df.get("season"), errors="coerce")
        self.df["ppr"] = pd.to_numeric(self.df.get("ppr"), errors="coerce")
        self.df["predicted_ppr_next"] = pd.to_numeric(
            self.df.get("predicted_ppr_next"), errors="coerce"
        )
        self.df["games"] = pd.to_numeric(self.df.get("games"), errors="coerce")

        ordered = self.df.sort_values(["gsis_id", "season"])
        latest = ordered.groupby("gsis_id", as_index=False).tail(1)
        latest = latest.sort_values("player_name")
        self._latest = latest
        self._labels = [
            f"{r.player_name} ({r.position}, {r.team})"
            for r in latest.itertuples(index=False)
        ]
        self._label_to_id = {
            f"{r.player_name} ({r.position}, {r.team})": r.gsis_id
            for r in latest.itertuples(index=False)
        }

        self._peer_preds: dict[str, np.ndarray] = {}
        for pos, g in latest.groupby("position"):
            vals = pd.to_numeric(g["predicted_ppr_next"], errors="coerce").to_numpy()
            self._peer_preds[str(pos)] = vals

        board_path = CSV_OUTPUT_DIR / VET_DRAFT_BOARD_CSV
        self._board = (
            pd.read_csv(board_path, low_memory=False) if board_path.exists() else pd.DataFrame()
        )
        weekly_path = CSV_OUTPUT_DIR / VET_WEEKLY_CSV
        self._weekly = (
            pd.read_csv(weekly_path, low_memory=False) if weekly_path.exists() else pd.DataFrame()
        )

        header = tk.Frame(self, bg=ui.BG, padx=12, pady=10)
        header.pack(fill="x")
        tk.Label(
            header,
            text="Veteran Next-Season Scorer",
            bg=ui.BG,
            fg=ui.FG,
            font=("Segoe UI", 14, "bold"),
        ).pack(anchor="w")
        tk.Label(
            header,
            text="Fantasy scoring format: PPR — prior NFL seasons → next-season projection",
            bg=ui.BG,
            fg="#B8D4FF",
            font=("Segoe UI", 9),
        ).pack(anchor="w", pady=(2, 0))
        tk.Label(
            header,
            text=(
                "Type to filter. Selecting a player fills prior-season stats "
                "and opens the projection popup."
            ),
            bg=ui.BG,
            fg=ui.FG,
            font=("Segoe UI", 9),
            wraplength=480,
            justify="left",
        ).pack(anchor="w", pady=(4, 0))

        lookup_frame = tk.Frame(self, bg=ui.BG, padx=12, pady=6)
        lookup_frame.pack(fill="x")
        tk.Label(
            lookup_frame,
            text="Player lookup",
            bg=ui.BG,
            fg=ui.FG,
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor="w")

        self._var = tk.StringVar()
        self._combo = ui.SearchableDropdown(
            lookup_frame,
            self._labels,
            textvariable=self._var,
            on_select=self._on_player_selected,
            width=58,
        )
        self._combo.pack(fill="x", pady=(4, 0))
        self._dropdown_widgets.append(self._combo)

        btn_frame = tk.Frame(self, bg=ui.BG, padx=12, pady=6)
        btn_frame.pack(fill="x")
        tk.Button(
            btn_frame,
            text="Show projection",
            command=self._on_player_selected,
            bg=ui.ACCENT_BTN,
            fg=ui.FG,
            activebackground="#2563B8",
            activeforeground=ui.FG,
            font=("Segoe UI", 10, "bold"),
            relief="flat",
            padx=12,
            pady=6,
        ).pack(side="left")
        tk.Button(
            btn_frame,
            text="Clear",
            command=self._clear_fields,
            bg=ui.BG_FIELD,
            fg=ui.FG,
            relief="flat",
            padx=12,
            pady=6,
        ).pack(side="left", padx=(8, 0))

        self.result_var = tk.StringVar(
            value="Select a player to fill prior-season stats and open the projection window."
        )
        tk.Label(
            self,
            textvariable=self.result_var,
            bg=ui.BG,
            fg=ui.IMPORTANT_FG,
            font=("Segoe UI", 10),
            wraplength=480,
            justify="left",
            padx=12,
            pady=4,
        ).pack(fill="x")

        scroll = ui.ScrollFrame(self)
        scroll.pack(fill="both", expand=True, padx=8, pady=4)
        self.vars: dict[str, tk.StringVar] = {}
        self._field_widgets: dict[str, tk.Misc] = {}
        for group in VET_GROUP_ORDER:
            self._add_group(scroll.inner, group)

        hist_box = tk.LabelFrame(
            scroll.inner,
            text="  Season history  ",
            bg=ui.BG,
            fg=ui.FG,
            font=("Segoe UI", 10, "bold"),
            labelanchor="nw",
            padx=8,
            pady=6,
        )
        hist_box.pack(fill="x", padx=4, pady=6)
        cols = ("season", "team", "ppr", "games", "touches", "snap%", "depth")
        self._tree = ttk.Treeview(hist_box, columns=cols, show="headings", height=8)
        widths = {
            "season": 56,
            "team": 44,
            "ppr": 56,
            "games": 52,
            "touches": 58,
            "snap%": 54,
            "depth": 48,
        }
        for c in cols:
            self._tree.heading(c, text=c)
            self._tree.column(c, width=widths[c], anchor="center")
        sb = ttk.Scrollbar(hist_box, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=sb.set)
        self._tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

    def _add_group(self, parent: tk.Frame, group: str) -> None:
        ui = _ui()
        specs = [s for s in VET_FIELD_SPECS if s.group == group]
        if not specs:
            return
        box = tk.LabelFrame(
            parent,
            text=f"  {group}  ",
            bg=ui.BG,
            fg=ui.FG,
            font=("Segoe UI", 10, "bold"),
            labelanchor="nw",
            padx=8,
            pady=6,
        )
        box.pack(fill="x", padx=4, pady=6)

        for spec in specs:
            row = tk.Frame(box, bg=ui.BG)
            row.pack(fill="x", pady=3)
            label_text = spec.label + ("  (Important)" if spec.important else "")
            tk.Label(
                row,
                text=label_text,
                bg=ui.BG,
                fg=ui.IMPORTANT_FG if spec.important else ui.FG,
                font=("Segoe UI", 9, "bold" if spec.important else "normal"),
                anchor="w",
            ).pack(anchor="w")
            if spec.hint:
                tk.Label(
                    row,
                    text=spec.hint,
                    bg=ui.BG,
                    fg="#B8D4FF",
                    font=("Segoe UI", 8),
                    anchor="w",
                ).pack(anchor="w")
            var = tk.StringVar()
            self.vars[spec.column] = var
            entry = tk.Entry(
                row,
                textvariable=var,
                bg=ui.BG_FIELD,
                fg=ui.FG,
                insertbackground=ui.FG,
                relief="flat",
                font=("Segoe UI", 10),
                state="readonly",
                readonlybackground=ui.BG_FIELD,
            )
            self._field_widgets[spec.column] = entry
            entry.pack(fill="x", pady=(2, 0), ipady=3)

    def _clear_fields(self) -> None:
        for var in self.vars.values():
            var.set("")
        for item in self._tree.get_children():
            self._tree.delete(item)
        self.result_var.set(
            "Select a player to fill prior-season stats and open the projection window."
        )
        self._close_prediction_popup()

    def _hide_all_dropdowns(self) -> None:
        for dd in self._dropdown_widgets:
            try:
                dd._hide_popup()
            except Exception:  # noqa: BLE001
                pass

    def _close_prediction_popup(self) -> None:
        if self._prediction_popup is not None:
            try:
                self._prediction_popup.destroy()
            except tk.TclError:
                pass
            self._prediction_popup = None

    def _resolve_gsis(self) -> str | None:
        label = (self._var.get() or "").strip()
        gsis = self._label_to_id.get(label)
        if gsis is not None:
            return gsis
        hits = [k for k in self._labels if label.lower() in k.lower()]
        if len(hits) == 1:
            self._var.set(hits[0])
            return self._label_to_id[hits[0]]
        return None

    def _on_player_selected(self) -> None:
        gsis = self._resolve_gsis()
        if gsis is None:
            messagebox.showinfo("Player", "Pick a player from the list.")
            return

        hist = self.df[self.df["gsis_id"] == gsis].sort_values("season")
        if hist.empty:
            return
        latest = hist.iloc[-1]
        self._fill_fields(latest)
        self._fill_history(hist)
        self._open_prediction_popup(latest)

    def _fill_fields(self, row: pd.Series) -> None:
        for spec in VET_FIELD_SPECS:
            val = row[spec.column] if spec.column in row.index else None
            if spec.column == "off_snap_pct" and val is not None and pd.notna(val):
                try:
                    val = 100.0 * float(val)
                except (TypeError, ValueError):
                    pass
            text = _fmt(val, spec.digits)
            if spec.column == "off_snap_pct" and text:
                text = f"{text}%"
            self.vars[spec.column].set(text)

    def _fill_history(self, hist: pd.DataFrame) -> None:
        for item in self._tree.get_children():
            self._tree.delete(item)
        for r in hist.itertuples(index=False):
            snap = getattr(r, "off_snap_pct", None)
            snap_s = (
                _fmt(100 * float(snap), 0)
                if snap is not None and pd.notna(snap)
                else "—"
            )
            self._tree.insert(
                "",
                "end",
                values=(
                    int(r.season) if pd.notna(r.season) else "—",
                    getattr(r, "team", None) or "—",
                    _fmt(getattr(r, "ppr", None)) or "—",
                    _fmt(getattr(r, "games", None), 0) or "—",
                    _fmt(getattr(r, "touches", None), 0) or "—",
                    snap_s if snap_s else "—",
                    _fmt(getattr(r, "depth_rank", None), 0) or "—",
                ),
            )

    def _build_result(self, latest: pd.Series) -> dict[str, Any]:
        pred = pd.to_numeric(latest.get("predicted_ppr_next"), errors="coerce")
        pred_f = float(pred) if pd.notna(pred) else float("nan")
        games = pd.to_numeric(latest.get("games"), errors="coerce")
        games_f = float(games) if pd.notna(games) else float("nan")
        pos = str(latest.get("position") or "")
        peers = self._peer_preds.get(pos, np.array([]))

        projected_17 = float("nan")
        gsis = latest.get("gsis_id")
        if not self._board.empty and gsis is not None and "gsis_id" in self._board.columns:
            match = self._board[self._board["gsis_id"] == gsis]
            if not match.empty and "projected_ppr_17" in match.columns:
                projected_17 = float(
                    pd.to_numeric(match.iloc[0]["projected_ppr_17"], errors="coerce")
                )

        return {
            "predicted_ppr": pred_f,
            "projected_ppr_17": projected_17,
            "target_season": latest.get("target_season"),
            "q25": pd.to_numeric(latest.get("predictive_q25"), errors="coerce"),
            "q75": pd.to_numeric(latest.get("predictive_q75"), errors="coerce"),
            "ppr_low": pd.to_numeric(latest.get("ppr_low"), errors="coerce"),
            "ppr_high": pd.to_numeric(latest.get("ppr_high"), errors="coerce"),
            "games": games_f,
            "injury_score": _injury_score(games_f),
            "elite_score": _elite_score(pred_f, peers),
            "position": pos,
            "gsis_id": gsis,
        }

    def _open_prediction_popup(self, latest: pd.Series) -> None:
        self._close_prediction_popup()
        name = str(latest.get("player_name") or "Player")
        result = self._build_result(latest)
        if not np.isfinite(result["predicted_ppr"]):
            self.result_var.set(f"{name}: no next-season forecast on the latest row.")
            return

        gsis = latest.get("gsis_id")
        weekly = pd.DataFrame()
        if not self._weekly.empty and gsis is not None:
            weekly = self._weekly[self._weekly["gsis_id"] == gsis].copy()

        top = self.winfo_toplevel()
        top.update_idletasks()
        x = top.winfo_x() + top.winfo_width() + 12
        y = top.winfo_y() + 40
        self._prediction_popup = VeteranPredictionPopup(
            top,
            name=name,
            result=result,
            weekly=weekly,
            x=x,
            y=y,
        )
        inj = result["injury_score"]
        elite = result["elite_score"]
        if np.isfinite(inj) and np.isfinite(elite):
            self.result_var.set(
                f"{name}: {result['predicted_ppr']:.1f} PPR  ·  "
                f"injury {inj:.0f}  ·  elite {elite:.0f}"
            )
        else:
            self.result_var.set(f"{name}: {result['predicted_ppr']:.1f} PPR projected")
