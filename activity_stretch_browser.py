# -*- coding: utf-8 -*-
"""
Activity Stretch Browser
------------------------
Browse contiguous stretches of the same activity label in a CSV.
- Open a CSV with columns including at least: timestamp, label
- Choose a category (label) to browse
- Navigate through contiguous stretches with Previous/Next
- Choose which sensor column to plot (e.g., acc_x, acc_y, acc_z, gyro_x/y/z, etc.)

Notes:
- One chart per view (no subplots). Multiple series are supported if "acc_xyz" or "gyro_xyz" is chosen.
- Does not set custom colors or styles.
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
    "gyro_x", "gyro_y", "gyro_z",
    "mag_x", "mag_y", "mag_z",
    "gravity_x", "gravity_y", "gravity_z",
    "quat_w", "quat_x", "quat_y", "quat_z",
    "acc_xyz (3 lines)", "gyro_xyz (3 lines)"
]

@dataclass
class Stretch:
    """Represents a contiguous stretch of the same label (and person, if present)."""
    label: str
    start_idx: int
    end_idx: int
    start_time: float
    end_time: float
    person: str | int | None
    stretch_id: int
    idx_within_label: int  # order within this label group
    n_rows: int

class ActivityStretchBrowser(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Activity Stretch Browser")
        self.geometry("1100x700")

        # Data / state
        self.df: pd.DataFrame | None = None
        self.stretches: list[Stretch] = []
        self.stretches_by_label: dict[str, list[Stretch]] = {}
        self.current_label: str | None = None
        self.current_list: list[Stretch] = []
        self.current_pos: int = 0

        # UI
        self._build_ui()

    def _build_ui(self):
        # Top controls frame
        top = ttk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        self.btn_open = ttk.Button(top, text="Open CSV…", command=self.on_open)
        self.btn_open.pack(side=tk.LEFT)

        ttk.Label(top, text="Category:").pack(side=tk.LEFT, padx=(12, 4))
        self.label_var = tk.StringVar(value="")
        self.cmb_label = ttk.Combobox(top, textvariable=self.label_var, values=[], state="readonly", width=20)
        self.cmb_label.bind("<<ComboboxSelected>>", self.on_label_change)
        self.cmb_label.pack(side=tk.LEFT)

        ttk.Label(top, text=" Plot:").pack(side=tk.LEFT, padx=(12, 4))
        self.plot_var = tk.StringVar(value="acc_x")
        self.cmb_plot = ttk.Combobox(top, textvariable=self.plot_var, values=PLOT_CHOICES, state="readonly", width=20)
        self.cmb_plot.bind("<<ComboboxSelected>>", lambda e: self.redraw_plot())
        self.cmb_plot.pack(side=tk.LEFT)

        ttk.Label(top, text=" Person:").pack(side=tk.LEFT, padx=(12, 4))
        self.person_var = tk.StringVar(value="(all)")
        self.cmb_person = ttk.Combobox(top, textvariable=self.person_var, values=["(all)"], state="readonly", width=10)
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

        self.btn_export = ttk.Button(top, text="Export current stretch (CSV)", command=self.on_export_current)
        self.btn_export.pack(side=tk.RIGHT)

        # Info label
        self.info_var = tk.StringVar(value="Open a CSV to begin.")
        self.lbl_info = ttk.Label(self, textvariable=self.info_var, anchor="w")
        self.lbl_info.pack(side=tk.TOP, fill=tk.X, padx=10, pady=4)

        # Matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=6, pady=6)

        # Bottom stretch list
        bottom = ttk.Frame(self)
        bottom.pack(side=tk.BOTTOM, fill=tk.BOTH, padx=6, pady=6)
        ttk.Label(bottom, text="Stretches in selected category:").pack(anchor="w")
        self.lst = tk.Listbox(bottom, height=8)
        self.lst.pack(fill=tk.BOTH, expand=True)
        self.lst.bind("<<ListboxSelect>>", self.on_list_select)

    def on_open(self):
        path = filedialog.askopenfilename(
            title="Open CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            df = pd.read_csv(path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read CSV:\n{e}")
            return

        missing = REQUIRED_COLS - set(df.columns)
        if missing:
            messagebox.showerror("Missing columns", f"Your CSV is missing required columns: {missing}")
            return

        # Ensure timestamp is numeric
        try:
            df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        except Exception:
            pass
        df = df.dropna(subset=["timestamp"]).reset_index(drop=True)

        # Normalize person column (optional)
        if "person" in df.columns:
            df["person"] = df["person"].astype(str)
        else:
            df["person"] = "(none)"

        # Sort by time just in case
        df = df.sort_values(by=["person", "timestamp"]).reset_index(drop=True)

        # Build stretches (contiguous regions with same label AND same person)
        # A break occurs when label changes OR person changes OR time goes backward/large gap (optional rule here is just label/person change).
        changes = (df["label"].ne(df["label"].shift(1))) | (df["person"].ne(df["person"].shift(1)))
        df["stretch_id"] = changes.cumsum()

        # Build list of Stretch objects
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
                idx_within_label=0,  # temp
                n_rows=int(len(g)),
            ))

        # Order within label
        stretches_by_label: dict[str, list[Stretch]] = {}
        for st in stretches:
            stretches_by_label.setdefault(st.label, []).append(st)
        for label, lst in stretches_by_label.items():
            lst.sort(key=lambda s: (s.person, s.start_time, s.start_idx))
            for k, st in enumerate(lst, start=1):
                st.idx_within_label = k

        self.df = df
        self.stretches = stretches
        self.stretches_by_label = stretches_by_label

        # Populate label combobox
        labels = sorted(stretches_by_label.keys())
        self.cmb_label["values"] = labels
        if labels:
            self.label_var.set(labels[0])
            self.on_label_change()

        # Populate person filter
        persons = ["(all)"] + sorted(df["person"].unique().tolist())
        self.cmb_person["values"] = persons
        self.person_var.set("(all)")

        self.info_var.set(f"Loaded {len(df)} rows, {len(stretches)} stretches, {len(labels)} categories.")

    def _filtered_list_for_label(self, label: str) -> list[Stretch]:
        if not label:
            return []
        all_list = self.stretches_by_label.get(label, [])
        psel = self.person_var.get()
        if psel and psel != "(all)":
            return [s for s in all_list if str(s.person) == str(psel)]
        return all_list

    def on_person_change(self, event=None):
        # Recompute list for current label
        if not self.current_label:
            return
        self.current_list = self._filtered_list_for_label(self.current_label)
        self.current_pos = 0
        self.refresh_listbox_and_nav()
        self.redraw_plot()

    def on_label_change(self, event=None):
        label = self.label_var.get()
        self.current_label = label
        self.current_list = self._filtered_list_for_label(label)
        self.current_pos = 0
        self.refresh_listbox_and_nav()
        self.redraw_plot()

    def refresh_listbox_and_nav(self):
        self.lst.delete(0, tk.END)
        for i, st in enumerate(self.current_list, start=1):
            dur = st.end_time - st.start_time
            # Now include start_row and end_row
            self.lst.insert(
                tk.END,
                f"[{i}] person={st.person}  rows={st.n_rows}  start_row={st.start_idx}  end_row={st.end_idx}  "
                f"t={st.start_time:.2f}→{st.end_time:.2f} (Δ={dur:.2f}s)"
            )
        n = len(self.current_list)
        if n == 0:
            self.spn_pos.config(from_=0, to=0)
            self.pos_var.set("0")
        else:
            self.spn_pos.config(from_=1, to=n)
            self.pos_var.set(str(self.current_pos+1))
            self.lst.select_clear(0, tk.END)
            self.lst.select_set(self.current_pos)
            self.lst.see(self.current_pos)

        self.btn_prev.config(state=tk.NORMAL if self.current_pos > 0 else tk.DISABLED)
        self.btn_next.config(state=tk.NORMAL if self.current_pos < n-1 else tk.DISABLED)

    def on_list_select(self, event=None):
        sel = self.lst.curselection()
        if sel:
            self.current_pos = int(sel[0])
            self.pos_var.set(str(self.current_pos+1))
            self.redraw_plot()

    def on_jump_spin(self):
        try:
            v = int(self.pos_var.get()) - 1
        except ValueError:
            return
        if 0 <= v < len(self.current_list):
            self.current_pos = v
            self.lst.select_clear(0, tk.END)
            self.lst.select_set(self.current_pos)
            self.lst.see(self.current_pos)
            self.redraw_plot()

    def on_prev(self):
        if self.current_pos > 0:
            self.current_pos -= 1
            self.refresh_listbox_and_nav()
            self.redraw_plot()

    def on_next(self):
        if self.current_pos < len(self.current_list) - 1:
            self.current_pos += 1
            self.refresh_listbox_and_nav()
            self.redraw_plot()

    def _plot_columns(self, df_slice: pd.DataFrame):
        """Plot selected column(s) in a single axes."""
        choice = self.plot_var.get()
        self.ax.clear()

        x = df_slice["timestamp"].values

        if choice == "acc_xyz (3 lines)":
            cols = ["acc_x", "acc_y", "acc_z"]
            cols = [c for c in cols if c in df_slice.columns]
            if not cols:
                self.ax.text(0.5, 0.5, "No acc_x/acc_y/acc_z found.", ha="center", va="center")
            else:
                for c in cols:
                    self.ax.plot(x, df_slice[c].values, label=c)
                self.ax.legend()
        elif choice == "gyro_xyz (3 lines)":
            cols = ["gyro_x", "gyro_y", "gyro_z"]
            cols = [c for c in cols if c in df_slice.columns]
            if not cols:
                self.ax.text(0.5, 0.5, "No gyro_x/gyro_y/gyro_z found.", ha="center", va="center")
            else:
                for c in cols:
                    self.ax.plot(x, df_slice[c].values, label=c)
                self.ax.legend()
        else:
            if choice not in df_slice.columns:
                self.ax.text(0.5, 0.5, f"Column '{choice}' not in CSV.", ha="center", va="center")
            else:
                self.ax.plot(x, df_slice[choice].values, label=choice)
                self.ax.legend()

        self.ax.set_xlabel("timestamp (s)")
        self.ax.set_ylabel("value")
        self.ax.set_title(f"{self.current_label} — stretch {self.current_pos+1}/{len(self.current_list)}")
        self.ax.grid(True)

    def redraw_plot(self):
        if self.df is None or not self.current_list:
            self.ax.clear()
            self.ax.text(0.5, 0.5, "Open a CSV and select a category.", ha="center", va="center")
            self.canvas.draw_idle()
            return

        st = self.current_list[self.current_pos]
        dsub = self.df.iloc[st.start_idx:st.end_idx+1]

        # Update info (now includes start/end row indices)
        dur = st.end_time - st.start_time
        self.info_var.set(
            f"Label={st.label} | Person={st.person} | Rows={st.n_rows} (start_row={st.start_idx}, end_row={st.end_idx}) "
            f"| Time {st.start_time:.2f}→{st.end_time:.2f} (Δ={dur:.2f}s) | StretchID={st.stretch_id}"
        )

        self._plot_columns(dsub)
        self.canvas.draw_idle()

        # Update nav buttons
        self.btn_prev.config(state=tk.NORMAL if self.current_pos > 0 else tk.DISABLED)
        self.btn_next.config(state=tk.NORMAL if self.current_pos < len(self.current_list) - 1 else tk.DISABLED)

    def on_export_current(self):
        if self.df is None or not self.current_list:
            messagebox.showinfo("Export", "No stretch selected.")
            return
        st = self.current_list[self.current_pos]
        dsub = self.df.iloc[st.start_idx:st.end_idx+1]
        # Include index range in default filename for clarity
        save_path = filedialog.asksaveasfilename(
            title="Save current stretch as CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=f"{st.label}_person{st.person}_rows{st.start_idx}-{st.end_idx}.csv"
        )
        if not save_path:
            return
        try:
            dsub.to_csv(save_path, index=False)
            messagebox.showinfo("Export", f"Saved {len(dsub)} rows to:\n{save_path}")
        except Exception as e:
            messagebox.showerror("Export error", f"Failed to save:\n{e}")

if __name__ == "__main__":
    app = ActivityStretchBrowser()
    app.mainloop()
