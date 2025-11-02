# -*- coding: utf-8 -*-
"""
Activity Stretch Browser — Main + Reference (wider ref panel)
-------------------------------------------------------------
- Left: main plot (narrower)
- Middle: main previews (prev/current/next)
- Right: reference previews (prev/current/next) with MORE width
"""

import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from dataclasses import dataclass

REQUIRED_COLS = {"timestamp", "label"}

PLOT_CHOICES = [
    "acc_x", "acc_y", "acc_z",
    "lacc_x", "lacc_y", "lacc_z",
    "acc_e", "lacc_e",
    "gyro_x", "gyro_y", "gyro_z",
    "mag_x", "mag_y", "mag_z",
    "gravity_x", "gravity_y", "gravity_z",
    "quat_w", "quat_x", "quat_y", "quat_z",
    "acc_xyz (3 lines)", "gyro_xyz (3 lines)"
]

COLUMN_TO_UNIT = {
    "gyro_x": r'Angular Velocity ($^\circ$/s)',
    "gyro_y": r'Angular Velocity ($^\circ$/s)',
    "gyro_z": r'Angular Velocity ($^\circ$/s)',
    "acc_x": r'Acceleration (m/s$^2$)',
    "acc_y": r'Acceleration (m/s$^2$)',
    "acc_z": r'Acceleration (m/s$^2$)',
    "lacc_x": r'Acceleration (m/s$^2$)',
    "lacc_y": r'Acceleration (m/s$^2$)',
    "lacc_z": r'Acceleration (m/s$^2$)',
    "acc_e": r'Acceleration (m/s$^2$)',
    "lacc_e": r'Acceleration (m/s$^2$)',
    "gravity_x": r'Acceleration (m/s$^2$)',
    "gravity_y": r'Acceleration (m/s$^2$)',
    "gravity_z": r'Acceleration (m/s$^2$)',
    "mag_x": r'Magnetic Field ($\\mu$T)',
    "mag_y": r'Magnetic Field ($\\mu$T)',
    "mag_z": r'Magnetic Field ($\\mu$T)',
    "quat_w": r'$q_w$',
    "quat_x": r'$q_x$',
    "quat_y": r'$q_y$',
    "quat_z": r'$q_z$',
    "calib_accel": "",
    "calib_gyro": "",
    "calib_mag": "",
    "calib_sys": ""
}

@dataclass
class Stretch:
    label: str
    start_idx: int
    end_idx: int
    start_time: float
    end_time: float
    person: str | int | None
    stretch_id: int
    idx_within_label: int
    n_rows: int

class ActivityStretchBrowser(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Activity Stretch Browser — Main + Reference")
        self.geometry("1620x900")

        # Main CSV state
        self.df: pd.DataFrame | None = None
        self.stretches: list[Stretch] = []
        self.stretches_by_label: dict[str, list[Stretch]] = {}
        self.current_label: str | None = None
        self.current_list: list[Stretch] = []
        self.current_pos: int = 0

        # Reference CSV state
        self.df_ref: pd.DataFrame | None = None
        self.stretches_ref: list[Stretch] = []
        self.stretches_by_label_ref: dict[str, list[Stretch]] = {}
        self.current_list_ref: list[Stretch] = []

        # Person filter state
        self.filter_person_enabled = tk.BooleanVar(value=False)

        self._build_ui()

    # ---------------------- UI ----------------------
    def _build_ui(self):
        # Top controls
        top = ttk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        self.btn_open = ttk.Button(top, text="Open CSV…", command=self.on_open)
        self.btn_open.pack(side=tk.LEFT)

        self.btn_open_ref = ttk.Button(top, text="Open Ref CSV…", command=self.on_open_ref)
        self.btn_open_ref.pack(side=tk.LEFT, padx=(6, 0))

        ttk.Label(top, text="Category:").pack(side=tk.LEFT, padx=(12, 4))
        self.label_var = tk.StringVar(value="")
        self.cmb_label = ttk.Combobox(top, textvariable=self.label_var, values=[], state="readonly", width=22)
        self.cmb_label.bind("<<ComboboxSelected>>", self.on_label_change)
        self.cmb_label.pack(side=tk.LEFT)

        ttk.Label(top, text=" Plot:").pack(side=tk.LEFT, padx=(12, 4))
        self.plot_var = tk.StringVar(value="acc_x")
        self.cmb_plot = ttk.Combobox(top, textvariable=self.plot_var, values=PLOT_CHOICES, state="readonly", width=20)
        self.cmb_plot.bind("<<ComboboxSelected>>", lambda e: self._redraw_all())
        self.cmb_plot.pack(side=tk.LEFT)

        self.chk_person = ttk.Checkbutton(top, text="Filter by person", variable=self.filter_person_enabled, command=self.on_person_toggle)
        self.chk_person.pack(side=tk.LEFT, padx=(12, 4))

        ttk.Label(top, text=" Person:").pack(side=tk.LEFT, padx=(6, 4))
        self.person_var = tk.StringVar(value="")
        self.cmb_person = ttk.Combobox(top, textvariable=self.person_var, values=[], state="disabled", width=12)
        self.cmb_person.bind("<<ComboboxSelected>>", self.on_person_change)
        self.cmb_person.pack(side=tk.LEFT)

        ttk.Label(top, text="  Jump to:").pack(side=tk.LEFT, padx=(12, 4))
        self.pos_var = tk.StringVar(value="1")
        self.spn_pos = ttk.Spinbox(top, from_=1, to=1, textvariable=self.pos_var, width=6, command=self.on_jump_spin)
        self.spn_pos.pack(side=tk.LEFT)

        self.btn_prev = ttk.Button(top, text="◀ Previous", command=self.on_prev)
        self.btn_prev.pack(side=tk.LEFT, padx=(12, 4))
        self.btn_next = ttk.Button(top, text="Next ▶", command=self.on_next)
        self.btn_next.pack(side=tk.LEFT, padx=(4, 4))

        self.btn_savefig = ttk.Button(top, text="Save figure (300 dpi)", command=self.on_save_fig)
        self.btn_savefig.pack(side=tk.RIGHT, padx=(4, 0))
        self.btn_export = ttk.Button(top, text="Export current stretch (CSV)", command=self.on_export_current)
        self.btn_export.pack(side=tk.RIGHT, padx=(0, 8))

        self.info_var = tk.StringVar(value="Open a CSV to begin.")
        self.lbl_info = ttk.Label(self, textvariable=self.info_var, anchor="w")
        self.lbl_info.pack(side=tk.TOP, fill=tk.X, padx=10, pady=4)

        # -------- Center area: grid layout (gives ref panel more width) --------
        center = ttk.Frame(self)
        center.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=6, pady=6)

        # 3 columns: 0 = main plot (narrow), 1 = main previews, 2 = REF previews (wide)
        center.grid_columnconfigure(0, weight=3, minsize=500)   # main plot
        center.grid_columnconfigure(1, weight=2, minsize=300)   # main preview column
        center.grid_columnconfigure(2, weight=6, minsize=300)   # REF preview column (widest)
        center.grid_rowconfigure(0, weight=1)

        # Main plot (col 0)
        left = ttk.Frame(center)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        # Narrower figure width than before
        self.fig, self.ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=left)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Main previews (col 1)
        colA = ttk.Frame(center)
        colA.grid(row=0, column=1, sticky="ns", padx=(0, 8))
        ttk.Label(colA, text="Main: Preview (previous)").pack(anchor="w", padx=4)
        self.preview_prev_fig, self.preview_prev_ax = plt.subplots(figsize=(3, 1.5), constrained_layout=True)
        self.preview_prev_canvas = FigureCanvasTkAgg(self.preview_prev_fig, master=colA)
        self.preview_prev_canvas.get_tk_widget().pack(fill=tk.X, padx=4, pady=(2, 8))
        self.preview_prev_canvas.get_tk_widget().bind("<Button-1>", lambda e: self._jump_relative(-1))

        ttk.Label(colA, text="Main: Preview (current)").pack(anchor="w", padx=4)
        self.preview_cur_fig, self.preview_cur_ax = plt.subplots(figsize=(3, 1.5), constrained_layout=True)
        self.preview_cur_canvas = FigureCanvasTkAgg(self.preview_cur_fig, master=colA)
        self.preview_cur_canvas.get_tk_widget().pack(fill=tk.X, padx=4, pady=(2, 8))

        ttk.Label(colA, text="Main: Preview (next)").pack(anchor="w", padx=4)
        self.preview_next_fig, self.preview_next_ax = plt.subplots(figsize=(3, 1.5), constrained_layout=True)
        self.preview_next_canvas = FigureCanvasTkAgg(self.preview_next_fig, master=colA)
        self.preview_next_canvas.get_tk_widget().pack(fill=tk.X, padx=4, pady=(2, 2))
        self.preview_next_canvas.get_tk_widget().bind("<Button-1>", lambda e: self._jump_relative(1))

        # Reference previews (col 2, *much wider*)
        colB = ttk.Frame(center)
        colB.grid(row=0, column=2, sticky="ns")
        ttk.Label(colB, text="Ref: Preview (previous)").pack(anchor="w", padx=4)
        # Slightly wider figures so they benefit from the extra column width
        self.preview_prev_ref_fig, self.preview_prev_ref_ax = plt.subplots(figsize=(3, 1.5), constrained_layout=True)
        self.preview_prev_ref_canvas = FigureCanvasTkAgg(self.preview_prev_ref_fig, master=colB)
        self.preview_prev_ref_canvas.get_tk_widget().pack(fill=tk.X, padx=4, pady=(2, 8))

        ttk.Label(colB, text="Ref: Preview (current)").pack(anchor="w", padx=4)
        self.preview_cur_ref_fig, self.preview_cur_ref_ax = plt.subplots(figsize=(3, 1.5), constrained_layout=True)
        self.preview_cur_ref_canvas = FigureCanvasTkAgg(self.preview_cur_ref_fig, master=colB)
        self.preview_cur_ref_canvas.get_tk_widget().pack(fill=tk.X, padx=4, pady=(2, 8))

        ttk.Label(colB, text="Ref: Preview (next)").pack(anchor="w", padx=4)
        self.preview_next_ref_fig, self.preview_next_ref_ax = plt.subplots(figsize=(3, 1.5), constrained_layout=True)
        self.preview_next_ref_canvas = FigureCanvasTkAgg(self.preview_next_ref_fig, master=colB)
        self.preview_next_ref_canvas.get_tk_widget().pack(fill=tk.X, padx=4, pady=(2, 2))

        # Bottom list
        bottom = ttk.Frame(self)
        bottom.pack(side=tk.BOTTOM, fill=tk.BOTH, padx=6, pady=6)
        ttk.Label(bottom, text="Stretches:").pack(anchor="w")
        self.lst = tk.Listbox(bottom, height=7)
        self.lst.pack(fill=tk.BOTH, expand=True)
        self.lst.bind("<<ListboxSelect>>", self.on_list_select)

    # ---------------------- File loading ----------------------
    def _read_csv_build_stretches(self, path: str):
        df = pd.read_csv(path)
        missing = REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")
        try:
            df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        except Exception:
            pass
        df = df.dropna(subset=["timestamp"]).reset_index(drop=True)
        if "person" in df.columns:
            df["person"] = df["person"].astype(str)
        else:
            df["person"] = "(none)"
        df = df.sort_values(by=["person", "timestamp"]).reset_index(drop=True)
        changes = (df["label"].ne(df["label"].shift(1))) | (df["person"].ne(df["person"].shift(1)))
        df["stretch_id"] = changes.cumsum()
        stretches: list[Stretch] = []
        for sid, g in df.groupby("stretch_id", sort=True):
            label = g["label"].iloc[0]
            person = g["person"].iloc[0]
            start_idx = int(g.index.min())
            end_idx = int(g.index.max())
            start_time = float(g["timestamp"].iloc[0])
            end_time = float(g["timestamp"].iloc[-1])
            stretches.append(Stretch(
                label=label,
                start_idx=start_idx,
                end_idx=end_idx,
                start_time=start_time,
                end_time=end_time,
                person=person,
                stretch_id=int(sid),
                idx_within_label=0,
                n_rows=int(len(g)),
            ))
        stretches_by_label: dict[str, list[Stretch]] = {}
        for st in stretches:
            stretches_by_label.setdefault(st.label, []).append(st)
        for label, lst in stretches_by_label.items():
            lst.sort(key=lambda s: (s.person, s.start_time, s.start_idx))
            for k, st in enumerate(lst, start=1):
                st.idx_within_label = k
        return df, stretches, stretches_by_label

    def on_open(self):
        path = filedialog.askopenfilename(title="Open CSV", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not path:
            return
        try:
            df, stretches, stretches_by_label = self._read_csv_build_stretches(path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read CSV:\n{e}")
            return
        self.df = df
        self.stretches = stretches
        self.stretches_by_label = stretches_by_label

        labels = ["(all)"] + sorted(stretches_by_label.keys())
        self.cmb_label["values"] = labels
        if labels:
            self.label_var.set("(all)")
            self.on_label_change()

        persons = sorted(df["person"].unique().tolist())
        self.cmb_person["values"] = persons
        if persons:
            self.person_var.set(persons[0])
        self.cmb_person.config(state="disabled")
        self.filter_person_enabled.set(False)

        self.info_var.set(f"Loaded MAIN: {len(df)} rows, {len(stretches)} stretches, {len(labels)-1} categories (+ all).")

    def on_open_ref(self):
        path = filedialog.askopenfilename(title="Open Reference CSV", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not path:
            return
        try:
            df, stretches, stretches_by_label = self._read_csv_build_stretches(path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read Reference CSV:\n{e}")
            return
        self.df_ref = df
        self.stretches_ref = stretches
        self.stretches_by_label_ref = stretches_by_label
        self._redraw_all()
        messagebox.showinfo("Reference", f"Loaded REFERENCE: {len(df)} rows, {len(stretches)} stretches.")

    # ---------------------- Filtering / nav ----------------------
    def _filtered_list_for_label(self, label: str, use_ref: bool = False) -> list[Stretch]:
        has_df = (self.df_ref is not None) if use_ref else (self.df is not None)
        if not has_df:
            return []
        use_person = self.filter_person_enabled.get()
        selected_person = self.person_var.get()
        if use_ref:
            src_all = self.stretches_ref; src_by_label = self.stretches_by_label_ref
        else:
            src_all = self.stretches;     src_by_label = self.stretches_by_label
        if label == "(all)":
            lst = sorted(src_all, key=lambda s: s.stretch_id)
        else:
            lst = src_by_label.get(label, [])
        if use_person and selected_person:
            lst = [s for s in lst if str(s.person) == str(selected_person)]
        return lst

    def on_person_toggle(self):
        self.cmb_person.config(state="readonly" if self.filter_person_enabled.get() else "disabled")
        if self.current_label:
            self.current_list = self._filtered_list_for_label(self.current_label, use_ref=False)
            self.current_list_ref = self._filtered_list_for_label(self.current_label, use_ref=True)
            self.current_pos = 0
            self.refresh_listbox_and_nav()
            self._redraw_all()

    def on_person_change(self, event=None):
        if not self.current_label or not self.filter_person_enabled.get():
            return
        self.current_list = self._filtered_list_for_label(self.current_label, use_ref=False)
        self.current_list_ref = self._filtered_list_for_label(self.current_label, use_ref=True)
        self.current_pos = 0
        self.refresh_listbox_and_nav()
        self._redraw_all()

    def on_label_change(self, event=None):
        label = self.label_var.get()
        self.current_label = label
        self.current_list = self._filtered_list_for_label(label, use_ref=False)
        self.current_list_ref = self._filtered_list_for_label(label, use_ref=True)
        self.current_pos = 0
        self.refresh_listbox_and_nav()
        self._redraw_all()

    def refresh_listbox_and_nav(self):
        self.lst.delete(0, tk.END)
        for i, st in enumerate(self.current_list, start=1):
            dur = st.end_time - st.start_time
            self.lst.insert(
                tk.END,
                f"[{i}] label={st.label} person={st.person} rows={st.n_rows} start_row={st.start_idx} end_row={st.end_idx} "
                f"t={st.start_time:.2f}→{st.end_time:.2f} (Δ={dur:.2f}s)"
            )
        n = len(self.current_list)
        if n == 0:
            self.spn_pos.config(from_=0, to=0); self.pos_var.set("0")
        else:
            self.spn_pos.config(from_=1, to=n); self.pos_var.set(str(self.current_pos+1))
            self.lst.select_clear(0, tk.END); self.lst.select_set(self.current_pos); self.lst.see(self.current_pos)
        self.btn_prev.config(state=tk.NORMAL if self.current_pos > 0 else tk.DISABLED)
        self.btn_next.config(state=tk.NORMAL if self.current_pos < n-1 else tk.DISABLED)

    def on_list_select(self, event=None):
        sel = self.lst.curselection()
        if sel:
            self.current_pos = int(sel[0]); self.pos_var.set(str(self.current_pos+1))
            self._redraw_all()

    def on_jump_spin(self):
        try:
            v = int(self.pos_var.get()) - 1
        except ValueError:
            return
        if 0 <= v < len(self.current_list):
            self.current_pos = v
            self.lst.select_clear(0, tk.END); self.lst.select_set(self.current_pos); self.lst.see(self.current_pos)
            self._redraw_all()

    def on_prev(self):
        if self.current_pos > 0:
            self.current_pos -= 1
            self.refresh_listbox_and_nav(); self._redraw_all()

    def on_next(self):
        if self.current_pos < len(self.current_list) - 1:
            self.current_pos += 1
            self.refresh_listbox_and_nav(); self._redraw_all()

    def _jump_relative(self, delta: int):
        new_pos = self.current_pos + delta
        if 0 <= new_pos < len(self.current_list):
            self.current_pos = new_pos
            self.refresh_listbox_and_nav(); self._redraw_all()

    # ---------------------- Plotting ----------------------
    def _ylabel_for_choice(self, choice: str, present_cols: list[str]) -> str:
        if choice == "acc_xyz (3 lines)":
            return r'Acceleration (m/s$^2$)'
        if choice == "gyro_xyz (3 lines)":
            return r'Angular Velocity ($^\circ$/s)'
        for c in present_cols:
            if c in COLUMN_TO_UNIT and COLUMN_TO_UNIT[c]:
                return COLUMN_TO_UNIT[c]
        return "value"

    def _plot_columns(self, df_slice: pd.DataFrame, ax: plt.Axes, show_legend: bool = True):
        choice = self.plot_var.get()
        ax.clear()
        x = df_slice["timestamp"].values
        if choice == "acc_xyz (3 lines)":
            cols = [c for c in ["acc_x", "acc_y", "acc_z"] if c in df_slice.columns]
            if not cols:
                ax.text(0.5, 0.5, "No acc_x/acc_y/acc_z found.", ha="center", va="center"); ylab = "value"
            else:
                for c in cols: ax.plot(x, df_slice[c].values, label=c)
                if show_legend: ax.legend()
                ylab = self._ylabel_for_choice(choice, cols)
        elif choice == "gyro_xyz (3 lines)":
            cols = [c for c in ["gyro_x", "gyro_y", "gyro_z"] if c in df_slice.columns]
            if not cols:
                ax.text(0.5, 0.5, "No gyro_x/gyro_y/gyro_z found.", ha="center", va="center"); ylab = "value"
            else:
                for c in cols: ax.plot(x, df_slice[c].values, label=c)
                if show_legend: ax.legend()
                ylab = self._ylabel_for_choice(choice, cols)
        else:
            if choice not in df_slice.columns:
                ax.text(0.5, 0.5, f"Column '{choice}' not in CSV.", ha="center", va="center"); ylab = "value"
            else:
                ax.plot(x, df_slice[choice].values, label=choice)
                if show_legend: ax.legend()
                ylab = self._ylabel_for_choice(choice, [choice])
        ax.set_xlabel("timestamp (s)"); ax.set_ylabel(ylab); ax.grid(True)

    def _redraw_all(self):
        self.redraw_plot()
        self.redraw_previews_main()
        self.redraw_previews_ref()

    def redraw_plot(self):
        if self.df is None or not self.current_list:
            self.ax.clear(); self.ax.text(0.5, 0.5, "Open a CSV and select a category.", ha="center", va="center")
            self.canvas.draw_idle(); return
        st = self.current_list[self.current_pos]
        dsub = self.df.iloc[st.start_idx:st.end_idx+1]
        dur = st.end_time - st.start_time
        self.info_var.set(
            f"Label={st.label} | Person={st.person} | Rows={st.n_rows} "
            f"(start_row={st.start_idx}, end_row={st.end_idx}) | "
            f"Time {st.start_time:.2f}→{st.end_time:.2f} (Δ={dur:.2f}s) | "
            f"StretchID={st.stretch_id}"
        )
        self._plot_columns(dsub, self.ax, show_legend=True)
        self.ax.set_title(f"{self.current_label} — stretch {self.current_pos+1}/{len(self.current_list)}")
        self.canvas.draw_idle()
        self.btn_prev.config(state=tk.NORMAL if self.current_pos > 0 else tk.DISABLED)
        self.btn_next.config(state=tk.NORMAL if self.current_pos < len(self.current_list) - 1 else tk.DISABLED)

    def _draw_preview_tile(self, ax, data_ok: bool, title: str, df_slice: pd.DataFrame | None):
        ax.clear()
        if not data_ok or df_slice is None or df_slice.empty:
            ax.text(0.5, 0.5, title, ha="center", va="center")
        else:
            self._plot_columns(df_slice, ax, show_legend=False)
            ax.set_title(title, fontsize=9)

    def _get_slice_for(self, lst: list[Stretch], pos: int):
        if not lst: return False, None, "No data"
        if pos < 0 or pos >= len(lst): return False, None, "Out of range"
        st = lst[pos]
        df = self.df if lst is self.current_list else self.df_ref
        dsub = df.iloc[st.start_idx:st.end_idx+1]
        title = f"{st.label} ({st.start_time:.2f}→{st.end_time:.2f})"
        return True, dsub, title

    def redraw_previews_main(self):
        ok, dsub, title = self._get_slice_for(self.current_list, self.current_pos - 1)
        self._draw_preview_tile(self.preview_prev_ax, ok, title if ok else "No previous stretch", dsub)
        self.preview_prev_canvas.draw_idle()

        ok, dsub, title = self._get_slice_for(self.current_list, self.current_pos)
        self._draw_preview_tile(self.preview_cur_ax, ok, title if ok else "No data", dsub)
        self.preview_cur_canvas.draw_idle()

        ok, dsub, title = self._get_slice_for(self.current_list, self.current_pos + 1)
        self._draw_preview_tile(self.preview_next_ax, ok, title if ok else "No next stretch", dsub)
        self.preview_next_canvas.draw_idle()

    def redraw_previews_ref(self):
        if self.current_label is None:
            self.current_list_ref = []
        else:
            self.current_list_ref = self._filtered_list_for_label(self.current_label, use_ref=True)

        ok, dsub, title = self._get_slice_for(self.current_list_ref, self.current_pos - 1)
        self._draw_preview_tile(self.preview_prev_ref_ax, ok, title if ok else "No previous (ref)", dsub)
        self.preview_prev_ref_canvas.draw_idle()

        ok, dsub, title = self._get_slice_for(self.current_list_ref, self.current_pos)
        self._draw_preview_tile(self.preview_cur_ref_ax, ok, title if ok else "No current (ref)", dsub)
        self.preview_cur_ref_canvas.draw_idle()

        ok, dsub, title = self._get_slice_for(self.current_list_ref, self.current_pos + 1)
        self._draw_preview_tile(self.preview_next_ref_ax, ok, title if ok else "No next (ref)", dsub)
        self.preview_next_ref_canvas.draw_idle()

    # ---------------------- Export / save ----------------------
    def on_export_current(self):
        if self.df is None or not self.current_list:
            messagebox.showinfo("Export", "No stretch selected."); return
        st = self.current_list[self.current_pos]
        dsub = self.df.iloc[st.start_idx:st.end_idx+1]
        save_path = filedialog.asksaveasfilename(
            title="Save current stretch as CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=f"{st.label}_person{st.person}_rows{st.start_idx}-{st.end_idx}.csv"
        )
        if not save_path: return
        try:
            dsub.to_csv(save_path, index=False)
            messagebox.showinfo("Export", f"Saved {len(dsub)} rows to:\n{save_path}")
        except Exception as e:
            messagebox.showerror("Export error", f"Failed to save:\n{e}")

    def on_save_fig(self):
        if self.df is None or not self.current_list:
            messagebox.showinfo("Save figure", "Nothing to save yet."); return
        st = self.current_list[self.current_pos]
        default_name = f"stretch_{st.stretch_id}_{st.label}_rows{st.start_idx}-{st.end_idx}.png"
        save_path = filedialog.asksaveasfilename(
            title="Save figure (300 dpi PNG)",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("All files", "*.*")],
            initialfile=default_name
        )
        if not save_path: return
        try:
            self.fig.savefig(save_path, dpi=300, bbox_inches="tight")
            messagebox.showinfo("Save figure", f"Saved figure:\n{save_path}")
        except Exception as e:
            messagebox.showerror("Save error", f"Failed to save figure:\n{e}")

if __name__ == "__main__":
    app = ActivityStretchBrowser()
    app.mainloop()
