from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Callable

from rookie_ppr.config import INCOMING_DRAFT_YEAR
from rookie_ppr.score_runner import load_players_master, player_lookup_labels, row_from_player, score_player
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

NAV_KEYS = frozenset({"Up", "Down", "Return", "Escape", "Tab"})


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


class ComparePlayerPopup(tk.Toplevel):
    def __init__(self, parent: tk.Misc, entry: dict, x: int, y: int) -> None:
        super().__init__(parent)
        self.title(f"Compare — {entry['name']}")
        self.configure(bg=POPUP_BG)
        self.geometry(f"340x520+{x}+{y}")
        self.minsize(300, 420)

        result = entry["result"]
        header = tk.Frame(self, bg=POPUP_BG, padx=12, pady=10)
        header.pack(fill="x")
        tk.Label(
            header,
            text=entry["name"],
            bg=POPUP_BG,
            fg=FG,
            font=("Segoe UI", 12, "bold"),
            wraplength=300,
            justify="left",
        ).pack(anchor="w")
        tk.Label(
            header,
            text=(
                f"Predicted PPR: {result['predicted_rookie_ppr']:.1f}\n"
                f"Success score: {result['success_score_0_100']:.1f}"
            ),
            bg=POPUP_BG,
            fg=IMPORTANT_FG,
            font=("Segoe UI", 11, "bold"),
            justify="left",
        ).pack(anchor="w", pady=(6, 0))

        body = tk.Frame(self, bg=POPUP_BG, padx=12, pady=4)
        body.pack(fill="both", expand=True)

        important_cols = {s.column for s in FIELD_SPECS if s.important}
        shown = 0
        for spec in FIELD_SPECS:
            val = entry["row"].get(spec.column)
            if val is None or str(val).strip() == "":
                continue
            shown += 1
            row = tk.Frame(body, bg=POPUP_BG)
            row.pack(fill="x", pady=2)
            label = spec.label
            if spec.column in important_cols:
                label += " (Important)"
            tk.Label(
                row,
                text=f"{label}:",
                bg=POPUP_BG,
                fg=IMPORTANT_FG if spec.column in important_cols else "#B8D4FF",
                font=("Segoe UI", 8),
                anchor="w",
            ).pack(anchor="w")
            tk.Label(
                row,
                text=str(val),
                bg=POPUP_BG,
                fg=FG,
                font=("Segoe UI", 10),
                anchor="w",
            ).pack(anchor="w")

        if shown == 0:
            tk.Label(body, text="No stats entered.", bg=POPUP_BG, fg=FG).pack(anchor="w")

        pop = len(result["composite_populated"])
        tk.Label(
            self,
            text=f"Composites filled: {pop}/11",
            bg=POPUP_BG,
            fg="#B8D4FF",
            font=("Segoe UI", 9),
            padx=12,
            pady=8,
        ).pack(fill="x")


class RookieScorerApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Rookie PPR Success Scorer")
        self.configure(bg=BG)
        self.geometry("520x760")
        self.minsize(480, 620)

        try:
            self.master_df = load_players_master()
            self.dropdowns = self._build_dropdowns()
            self.lookup_items = player_lookup_labels(self.master_df)
        except FileNotFoundError as exc:
            messagebox.showerror("Data missing", str(exc))
            self.destroy()
            return

        self.vars: dict[str, tk.StringVar] = {}
        self._field_widgets: dict[str, tk.Misc] = {}
        self._dropdown_widgets: list[SearchableDropdown] = []
        self._all_lookup_items = list(self.lookup_items)
        self._label_to_index = {label: idx for label, idx in self._all_lookup_items}
        self._compare_entries: list[dict] = []
        self._compare_vars: list[tk.BooleanVar] = []
        self._compare_popups: list[ComparePlayerPopup] = []

        header = tk.Frame(self, bg=BG, padx=12, pady=10)
        header.pack(fill="x")
        tk.Label(
            header,
            text="Rookie PPR ML Scorer",
            bg=BG,
            fg=FG,
            font=("Segoe UI", 14, "bold"),
        ).pack(anchor="w")
        tk.Label(
            header,
            text=(
                f"Default lookup: {INCOMING_DRAFT_YEAR}–{INCOMING_DRAFT_YEAR + 1} rookies. "
                "Type to filter. Add players to compare and open side popups."
            ),
            bg=BG,
            fg=FG,
            font=("Segoe UI", 9),
            wraplength=480,
            justify="left",
        ).pack(anchor="w", pady=(4, 0))

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

        self.result_var = tk.StringVar(value="Enter stats and click Calculate score.")
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
        idx = self._label_to_index[label]
        data = row_from_player(self.master_df, idx)
        for col, var in self.vars.items():
            if col in data:
                var.set(str(data[col]))
            else:
                var.set("")

    def _clear_fields(self) -> None:
        self._hide_all_dropdowns()
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
            return score_player(row)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Scoring error", str(exc))
            return None

    def _add_to_compare(self, name: str, row: dict, result: dict) -> None:
        if any(entry["name"] == name for entry in self._compare_entries):
            messagebox.showinfo("Compare", f"{name} is already in the compare list.")
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
                text="No players added yet. Check players below, then open compare windows.",
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
            tk.Label(
                row,
                text=(
                    f"{entry['name']}  —  PPR {result['predicted_rookie_ppr']:.1f}, "
                    f"Score {result['success_score_0_100']:.1f}"
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

    def _open_compare_windows(self) -> None:
        selected = [
            self._compare_entries[i]
            for i, var in enumerate(self._compare_vars)
            if i < len(self._compare_entries) and var.get()
        ]
        if not selected:
            messagebox.showinfo("Compare", "Check at least one player in the compare list.")
            return

        for popup in self._compare_popups:
            try:
                popup.destroy()
            except tk.TclError:
                pass
        self._compare_popups.clear()

        self.update_idletasks()
        base_x = self.winfo_x() + self.winfo_width() + 12
        base_y = self.winfo_y()
        popup_width = 340
        gap = 8

        for i, entry in enumerate(selected):
            x = base_x + i * (popup_width + gap)
            y = base_y
            popup = ComparePlayerPopup(self, entry, x, y)
            self._compare_popups.append(popup)

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
        detail = (
            f"Predicted rookie PPR: {result['predicted_rookie_ppr']:.1f}  |  "
            f"Success score (0–100): {result['success_score_0_100']:.1f}  |  "
            f"Composites filled: {pop}/11"
        )
        if miss:
            detail += f"  |  Missing: {', '.join(result['composite_missing'])}"
        self.result_var.set(detail)


def main() -> int:
    app = RookieScorerApp()
    if not app.winfo_exists():
        return 1
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
