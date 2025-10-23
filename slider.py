# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import Tk, Frame, filedialog, Radiobutton, StringVar, Button, OptionMenu, Listbox, Label, Entry, messagebox
import tkinter as tk
from load_imu_data import formats
import os
from intervals import all_intervals
from utils import *
from extract_with_label import extract_labeled_data
from collections import defaultdict
import time
from datetime import datetime

column_to_unit = {
    "gyro_x":r'Angular Velocity ($^\circ$/s)',
    "gyro_y":r'Angular Velocity ($^\circ$/s)',
    "gyro_z":r'Angular Velocity ($^\circ$/s)',
    "acc_x":r'Acceleration (m/s$^2$)',
    "acc_y":r'Acceleration (m/s$^2$)',
    "acc_z":r'Acceleration (m/s$^2$)',
    "lacc_x":r'Acceleration (m/s$^2$)',
    "lacc_y":r'Acceleration (m/s$^2$)',
    "lacc_z":r'Acceleration (m/s$^2$)',
    "gravity_x":r'Acceleration (m/s$^2$)',
    "gravity_y":r'Acceleration (m/s$^2$)',
    "gravity_z":r'Acceleration (m/s$^2$)',
    "mag_x":r'Magnetic Field ($\mu$T)',
    "mag_y":r'Magnetic Field ($\mu$T)',
    "mag_z":r'Magnetic Field ($\mu$T)',
    "quat_w":r'$q_w$',
    "quat_x":r'$q_x$',
    "quat_y":r'$q_y$',
    "quat_z":r'$q_z$',
    "calib_accel":"",
    "calib_gyro":"",
    "calib_mag":"",
    "calib_sys":""
}

# Matplotlib sizing
dpi = 300
base_dpi = 200
base_font_size = 6
base_linewidth = 0.2
font_size_scale = dpi / base_dpi
scaled_font_size = base_font_size / font_size_scale
linewidth_scale = dpi / base_dpi
labelpad_scale = font_size_scale

plt.rcParams.update({
    'font.size': scaled_font_size,
    'axes.titlesize': scaled_font_size * 1.1,
    'axes.labelsize': scaled_font_size * 0.8,
    'xtick.labelsize': scaled_font_size * 0.7,
    'ytick.labelsize': scaled_font_size * 0.7,
    'legend.fontsize': scaled_font_size * 0.7,
    'lines.linewidth': base_linewidth / linewidth_scale,
    'axes.linewidth': 0.15 * (dpi / base_dpi),
    'figure.dpi':600,
    'savefig.dpi':600
})

class MockEvent:
    pass

class WrappedLabel(tk.Frame):
    def __init__(self, master, text="", width=200, height=30, bg='white', fg='black', relief='flat', borderwidth=1, command=None, **kwargs):
        super().__init__(master, height=height, **kwargs)
        self.command = command
        self.canvas = tk.Canvas(self, height=height, width=width, bg=bg, highlightthickness=0, bd=0)
        self.canvas.pack(side='left', fill='both', expand=True)
        self.label = tk.Label(self.canvas, text=text, anchor='w', bg=bg, fg=fg, relief=relief, borderwidth=borderwidth, padx=5)
        self.label_id = self.canvas.create_window((0, 0), window=self.label, anchor='nw')
        self.label.bind("<Button-1>", self._on_click)
        self.scroll_x = tk.Scrollbar(self, orient='horizontal', command=self.canvas.xview)
        self.scroll_x.pack(side='bottom', fill='x')
        self.canvas.configure(xscrollcommand=self.scroll_x.set)
        self.label.bind("<Configure>", self._resize)
        self.canvas.bind("<Configure>", self._resize)

    def _resize(self, event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_click(self, event):
        if self.command:
            self.command()

class DraggableIntervals:
    def __init__(self, master, csv=None):
        self.master = master
        self.master.bind_all("<ButtonRelease>", self.on_any_click)

        # drag state
        self.press = None          # main-ax drag anchor
        self.zoom_press = None     # zoom-ax drag anchor (xdata)
        self.zoom_dragging = False # only drag if press was on a mask in zoom

        self.init_components()
        self.set_new_file(csv)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

    # ---------- UI ----------
    def init_components(self):
        self.fig, (self.ax, self.zoom_ax) = plt.subplots(
            2, 1, gridspec_kw={'height_ratios': [3, 2]}, figsize=(8, 6), dpi=300
        )
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        plt.subplots_adjust(top=0.92, bottom=0.1, hspace=0.5, right=0.8)
        self.set_font_sizes(dpi=300)

        # Slider state + UI
        self.zoom_window_width = 15.0
        self.user_set_slider = False
        self._suspend_slider_callback = False
        self.slider_frame = Frame(self.master)
        self.slider_frame.pack(side='bottom', fill='x')
        Label(self.slider_frame, text="Zoom position").pack(side='left', padx=6)
        self.zoom_slider = None  # created after data load

        # Buttons
        self.button_frame = Frame(self.master); self.button_frame.pack(side='bottom', fill='x')
        Button(self.button_frame, text='Export Labeled Data', command=self.on_export_button_press).pack(side='left', padx=5)
        Button(self.button_frame, text='Load File', command=self.load_file).pack(side='left', padx=5)
        Button(self.button_frame, text='Save Figure', command=self.save_figure).pack(side='left', padx=5)

        # Right controls (radio + list)
        self.radio_frame = Frame(self.master); self.radio_frame.pack(side='right', fill='y')
        self.radio_var = StringVar(); self.radio_buttons = []

        self.alignments_listbox_frame = Frame(self.radio_frame); self.alignments_listbox_frame.pack(side='bottom', fill='x', pady=5)
        self.dropdown_var = StringVar(); self.dropdown_var.set(list(all_intervals.keys())[0])
        self.dropdown_menu = OptionMenu(self.alignments_listbox_frame, self.dropdown_var, *[key for key in all_intervals])
        self.dropdown_menu.pack(pady=5, fill='x')
        Button(self.alignments_listbox_frame, text='Add New', command=self.add_new_alignment).pack(padx=5, fill='y')
        self.dropdown_var.trace("w", self.on_dropdown_change)

        Label(self.alignments_listbox_frame, text="Select a mask:").pack(anchor='w')
        self.alignments_listbox = Listbox(self.alignments_listbox_frame, height=10, selectmode=tk.SINGLE)
        self.alignments_listbox.pack(fill='x', padx=5)
        self.alignments_listbox.bind("<<ListboxSelect>>", self.on_alignments_listbox_select)
        Button(self.alignments_listbox_frame, text="Delete", command=self.delete_selected_alignments_listbox_item).pack(pady=5)
        Button(self.alignments_listbox_frame, text="Save", command=self.save_alignments).pack(pady=5)

    # ---------- Data I/O ----------
    def on_any_click(self, event):
        if isinstance(event.widget, tk.Button):
            timestamp = time.time()
            print(f"Clicked: {event.widget['text']} at {timestamp:.3f}")
            self.annotation_button_click_log[timestamp] = f"{event.widget['text']}"
            save_yaml(self.annotation_button_click_log, self.annotation_button_click_log_path, "w")

    def read_data(self, csv):
        self.csv_path = csv
        base, ext = os.path.splitext(self.csv_path)
        self.alignment_path = base + '_alignment.yaml'
        self.annotation_button_click_log_path = base + '_annotation_button_click_log_' + datetime.now().strftime("%y%m%d_%H%M%S") + '.yaml'
        print(f"alignment path is {self.alignment_path}")
        self.export_path = base + "_labeled.csv"

        df = pd.read_csv(self.csv_path)
        all_data = df.values
        all_data[:, 0] = all_data[:, 0] - all_data[0][0]
        self.data = all_data
        self.timestamps = all_data[:, 0]
        self.headers = df.columns.tolist()
        if "gyro_x" in df.columns: self.column_index = df.columns.get_loc("gyro_x")
        elif "lacc_x" in df.columns: self.column_index = df.columns.get_loc("lacc_x")
        elif "quat_x" in df.columns: self.column_index = df.columns.get_loc("quat_x")
        else: self.column_index = 5

    def reset_radio_buttons(self):
        for widget in self.radio_frame.winfo_children():
            if widget is not self.alignments_listbox_frame: widget.destroy()
        self.radio_buttons = []
        grouped = defaultdict(list)
        for header in self.headers:
            if "timestamp" in header: continue
            if '_' in header:
                prefix, axis = header.split('_', 1)
                grouped[prefix].append(axis)
        for prefix in grouped.keys():
            gf = Frame(self.radio_frame); gf.pack(anchor='w', pady=5)
            Label(gf, text=prefix).pack()
            row = Frame(gf); row.pack()
            for axis in sorted(grouped[prefix]):
                full_name = f"{prefix}_{axis}"
                rb = Radiobutton(row, text=axis, variable=self.radio_var, value=full_name, command=self.update_plot)
                rb.pack(side='left', padx=5)
                self.radio_buttons.append(rb)

    def set_new_file(self, csv):
        if not csv.endswith("_unit_converted.csv"):
            from load_imu_data import load_tag_imu_data_from_csv
            load_tag_imu_data_from_csv(csv)
            name, ext = os.path.splitext(csv)
            csv = f"{name}_unit_converted{ext}"

        self.interval_patches = []
        self.zoom_interval_spans = []
        self.label_to_color = {}
        if hasattr(self, 'listbox_selected_interval_name'):
            delattr(self, 'listbox_selected_interval_name')

        self.read_data(csv)
        self.master.title(self.csv_path)

        self.ax.clear(); self.zoom_ax.clear()
        self.full_plot, = self.ax.plot(self.data[:, 0], self.data[:, self.column_index], label='Time Series A')
        self.zoom_line, = self.zoom_ax.plot([], [], 'r-')

        self.radio_var.set(self.headers[self.column_index])
        self.reset_radio_buttons()

        self.alignments_listbox.delete(0, tk.END)
        self.load_alignments()
        self.annotation_button_click_log = dict()

        # Build / refresh slider
        self.init_or_refresh_slider()

    def init_or_refresh_slider(self):
        if self.zoom_slider is not None:
            try: self.zoom_slider.destroy()
            except Exception: pass
        t0 = float(self.timestamps[0]); t1 = float(self.timestamps[-1])
        span = max(0.0, t1 - t0)
        self.zoom_window_width = min(self.zoom_window_width, max(1.0, span))
        max_start = max(t0, t1 - self.zoom_window_width)
        dt = self._estimate_dt(); resolution = max(dt, 0.001)
        self.zoom_slider = tk.Scale(
            self.slider_frame, from_=t0, to=max_start, orient='horizontal',
            resolution=resolution, showvalue=True, length=500, command=self.on_zoom_slider
        )
        self.zoom_slider.pack(side='left', fill='x', expand=True, padx=6)
        x0_default = self._default_zoom_start()
        self._suspend_slider_callback = True
        try: self.zoom_slider.set(x0_default)
        finally: self._suspend_slider_callback = False
        self.user_set_slider = False

    def _estimate_dt(self):
        if len(self.timestamps) < 3: return 0.01
        diffs = np.diff(self.timestamps[:min(5000, len(self.timestamps))]); diffs = diffs[diffs > 0]
        if len(diffs) == 0: return 0.01
        return float(np.median(diffs))

    def _default_zoom_start(self):
        margin = 10.0; t0 = float(self.timestamps[0])
        if self.interval_patches:
            first_start = float(self.interval_patches[0]['patch'].get_xy()[0][0])
            return max(t0, first_start - margin)
        return t0

    # ---------- Alignments / masks ----------
    def load_alignments(self):
        self.alignment_offsets = load_yaml(self.alignment_path)
        for interval_name in self.alignment_offsets:
            self.alignments_listbox.insert(tk.END, interval_name)
        if len(self.alignment_offsets):
            self.alignments_listbox.selection_set(0)
            mock_event = MockEvent(); mock_event.widget = self.alignments_listbox
            self.on_alignments_listbox_select(mock_event)
        else:
            self.load_color_patches(self.dropdown_var.get())

    def add_new_alignment(self):
        patch = self.interval_patches[0]
        offset = float(patch['patch'].get_xy()[0][0]) - float(patch['limits'][0])
        dropdown_interval_name = self.dropdown_var.get()
        new_key = dropdown_interval_name
        if new_key in self.alignment_offsets:
            counter = 1
            while new_key in self.alignment_offsets:
                new_key = f"{dropdown_interval_name}_{counter}"; counter += 1
        self.alignment_offsets[new_key] = offset
        self.alignments_listbox.insert(tk.END, new_key)
        self.alignments_listbox.selection_clear(0, tk.END)
        self.alignments_listbox.select_set(self.alignments_listbox.get(0, tk.END).index(new_key))

    def save_alignments(self):
        if not self.alignment_offsets:
            messagebox.showwarning("Warning", "No alignments to save. click ADD NEW first if you have something to save.")
            return
        print("Save alignments to:", self.alignment_path)
        save_yaml(self.alignment_offsets, self.alignment_path, 'w')

    def delete_selected_alignments_listbox_item(self):
        selected_items = self.alignments_listbox.curselection()
        if selected_items:
            for index in selected_items[::-1]:
                list_entry_to_be_deleted = self.alignments_listbox.get(index)
                self.alignments_listbox.delete(index)
                del self.alignment_offsets[list_entry_to_be_deleted]

    def on_alignments_listbox_select(self, event):
        selection = event.widget.curselection()
        if selection:
            self.listbox_selected_interval_name = event.widget.get(selection[0])
            self.load_color_patches_offset(self.listbox_selected_interval_name)

    def on_dropdown_change(self, *args):
        name = self.dropdown_var.get()
        self.load_color_patches(name)

    def load_color_patches_offset(self, interval_name):
        offset = self.alignment_offsets.get(interval_name, 0)
        self.load_color_patches(interval_name, offset=offset)

    def load_color_patches(self, interval_name, offset=0):
        base_interval_name = get_base_name(interval_name)
        intervals = [(x, y, z) for (x, y, z) in all_intervals[base_interval_name] if "ready" not in x]
        for p in self.interval_patches:
            p["patch"].remove()
        self.interval_patches = []
        colormap = plt.get_cmap("tab10")
        self.label_to_color = {}
        for label, start, end in intervals:
            if label not in self.label_to_color:
                self.label_to_color[label] = colormap(len(self.label_to_color) % colormap.N)
            color = self.label_to_color[label]
            patch = self.ax.axvspan(start+offset, end+offset, color=color, alpha=0.5, label=label, linewidth=0)
            self.interval_patches.append({'patch': patch, 'label': label, 'color': color, 'limits': (start, end)})

        handles, added = [], set()
        for label, _, _ in intervals:
            if label not in added:
                handles.append(Patch(color=self.label_to_color[label], alpha=0.5, label=label, linewidth=0))
                added.add(label)
        self.patch_legend = self.ax.legend(handles=handles, prop={'size': plt.rcParams['legend.fontsize']}, bbox_to_anchor=(1, 0.5))
        self.patch_legend.get_frame().set_linewidth(0.2)
        self.update_plot()

    # ---------- Interactions ----------
    def connect(self):
        self.cidpress = self.ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cidrelease = self.ax.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion = self.ax.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.zoom_cidpress = self.zoom_ax.figure.canvas.mpl_connect('button_press_event', self.on_zoom_press)
        self.zoom_cidrelease = self.zoom_ax.figure.canvas.mpl_connect('button_release_event', self.on_zoom_release)
        self.zoom_cidmotion = self.zoom_ax.figure.canvas.mpl_connect('motion_notify_event', self.on_zoom_motion)

    # Main ax drag (drag any mask)
    def on_press(self, event):
        if event.inaxes != self.ax: return
        for patch in self.interval_patches:
            contains, _ = patch['patch'].contains(event)
            if contains:
                self.press = event.xdata
                self.background = self.ax.figure.canvas.copy_from_bbox(self.ax.bbox)
                self.update_zoom_according_to_main()  # keep zoom synced
                break

    def on_motion(self, event):
        if self.press is None or event.inaxes != self.ax: return
        dx = event.xdata - self.press
        current_last_patch_end = self.interval_patches[-1]['patch'].get_xy()[2][0]
        current_first_patch_start = self.interval_patches[0]['patch'].get_xy()[0][0]
        xmin, xmax = self.ax.get_xlim()
        if (current_last_patch_end + dx > xmax) or (current_first_patch_start + dx < xmin): return
        for p in self.interval_patches:
            new_x0 = p['patch'].get_xy()[0][0] + dx
            new_x1 = p['patch'].get_xy()[2][0] + dx
            p['patch'].set_xy([(new_x0,0),(new_x0,1),(new_x1,1),(new_x1,0),(new_x0,0)])
        self.press = event.xdata
        self.ax.figure.canvas.restore_region(self.background)
        for p in self.interval_patches: self.ax.draw_artist(p['patch'])
        self.ax.figure.canvas.blit(self.ax.bbox)
        self.update_zoom_according_to_main()  # zoom respects slider if used

    def on_release(self, event):
        if event.inaxes != self.ax: return
        self.press = None
        self.ax.figure.canvas.draw()
        patch = self.interval_patches[0]
        offset = float(patch['patch'].get_xy()[0][0]) - float(patch['limits'][0])
        if hasattr(self, 'listbox_selected_interval_name'):
            self.alignment_offsets[self.listbox_selected_interval_name] = offset
        print("on release", offset, self.alignment_offsets)

    # Zoom ax drag (must start on a visible mask)
    def on_zoom_press(self, event):
        if event.inaxes != self.zoom_ax or event.xdata is None:
            return
        x = float(event.xdata)

        # Start dragging only if pressed on a mask that is visible in zoom window
        self.zoom_dragging = False
        for p in self.interval_patches:
            start = float(p['patch'].get_xy()[0][0])
            end   = float(p['patch'].get_xy()[2][0])
            if start <= x <= end:
                self.zoom_dragging = True
                self.zoom_press = x
                break

        if not self.zoom_dragging:
            self.zoom_press = None

    def on_zoom_motion(self, event):
        if not self.zoom_dragging or self.zoom_press is None or event.inaxes != self.zoom_ax:
            return
        if event.xdata is None:
            return

        dx = float(event.xdata) - float(self.zoom_press)

        """
        # ---- FIX: constrain by the leftmost *visible* span edge within [xmin, xmax]
        #xmin, xmax = self.zoom_ax.get_xlim()
        xmin, xmax = self.ax.get_xlim()
        leftmost_visible_edge = None
        for patch in self.interval_patches:
            start = float(patch['patch'].get_xy()[0][0])
            end   = float(patch['patch'].get_xy()[2][0])
            # overlap with zoom window?
            if end >= xmin and start <= xmax:
                start_vis = max(start, xmin)   # clipped to window
                if leftmost_visible_edge is None or start_vis < leftmost_visible_edge:
                    leftmost_visible_edge = start_vis

        # If nothing is visible (shouldn't happen because we pressed on one), skip
        #if leftmost_visible_edge is None:
        #    return

        # Prevent dragging so that the leftmost visible edge would cross xmin
        if (leftmost_visible_edge + dx < xmin):
            print("leftmost_visible_edge + dx < xmin")
            return
        # ---- END FIX
        """
        current_last_patch_end = self.interval_patches[-1]['patch'].get_xy()[2][0]
        current_first_patch_start = self.interval_patches[0]['patch'].get_xy()[0][0]
        xmin, xmax = self.ax.get_xlim()

        # limit the move range, so that all the spans are contained in the current window
        if (current_last_patch_end + dx > xmax) or (current_first_patch_start + dx < xmin):
            return

        # Shift all patches by dx in the main plot
        for patch in self.interval_patches:
            x0 = float(patch['patch'].get_xy()[0][0]) + dx
            x1 = float(patch['patch'].get_xy()[2][0]) + dx
            patch['patch'].set_xy([(x0,0),(x0,1),(x1,1),(x1,0),(x0,0)])

        # Update anchor
        self.zoom_press = float(event.xdata)

        # Repaint zoom spans to match the moved patches (static zoom window)
        for span in getattr(self, 'zoom_interval_spans', []):
            try: span.remove()
            except Exception: pass
        self.zoom_interval_spans = []
        for patch in self.interval_patches:
            label = patch['label']
            start = float(patch['patch'].get_xy()[0][0])
            end   = float(patch['patch'].get_xy()[2][0])
            color = self.label_to_color[label]
            span = self.zoom_ax.axvspan(start, end, color=color, alpha=0.5, linewidth=0)
            self.zoom_interval_spans.append(span)

        # Redraw both axes so you see the masks move immediately
        self.ax.figure.canvas.draw()

    def on_zoom_release(self, event):
        if event.inaxes != self.zoom_ax:
            return
        self.zoom_press = None
        self.zoom_dragging = False

        # Save current offset against the selected alignment (if any)
        if self.alignments_listbox.curselection():
            selected_item = self.alignments_listbox.get(self.alignments_listbox.curselection()[0])
            patch = self.interval_patches[0]
            offset = float(patch['patch'].get_xy()[0][0]) - float(patch['limits'][0])
            self.alignment_offsets[selected_item] = offset
            print("Updated offset:", offset)

    # Slider → pan zoom window (does not move masks)
    def on_zoom_slider(self, value):
        if self._suspend_slider_callback: return
        try: x0 = float(value)
        except Exception: return
        self.user_set_slider = True
        self.update_zoom_according_to_main(force_x0=x0)

    # ---------- Buttons ----------
    def on_export_button_press(self):
        extract_labeled_data(self.csv_path)

    def load_file(self):
        if not hasattr(self, '_file_dialog_shown_once'):
            initialdir = "./example_data"; self._file_dialog_shown_once = True
        else:
            initialdir = None
        file_path = filedialog.askopenfilename(filetypes=[('csv Files', '*.csv')], initialdir=initialdir)
        if file_path: self.set_new_file(file_path)

    def save_figure(self):
        default_name = os.path.splitext(os.path.basename(self.csv_path))[0] + "_figure" if hasattr(self, 'csv_path') else "figure"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S"); default_name = f"{default_name}_{timestamp}"
        file_path = filedialog.asksaveasfilename(defaultextension=".png",
            filetypes=[("PNG files","*.png"),("PDF files","*.pdf"),("JPEG files","*.jpg")], initialfile=default_name)
        if file_path:
            try:
                self.fig.savefig(file_path, dpi=1000, bbox_inches='tight', pad_inches=0.1)
                messagebox.showinfo("Success", f"Figure saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save figure: {str(e)}")

    # ---------- Plot updates ----------
    def set_font_sizes(self, dpi):
        for ax in [self.ax, self.zoom_ax]:
            ax.title.set_fontsize(plt.rcParams['axes.titlesize'])
            ax.xaxis.label.set_fontsize(plt.rcParams['axes.labelsize'])
            ax.yaxis.label.set_fontsize(plt.rcParams['axes.labelsize'])
            ax.tick_params(axis='both', which='major',
                labelsize=plt.rcParams['xtick.labelsize'],
                width=plt.rcParams['axes.linewidth'], pad=labelpad_scale)
            ax.tick_params(axis='both', which='minor',
                labelsize=plt.rcParams['ytick.labelsize'],
                width=plt.rcParams['axes.linewidth'], pad=labelpad_scale)
            for side in ['top','bottom','left','right']:
                ax.spines[side].set_linewidth(plt.rcParams['axes.linewidth'])

    def update_plot(self):
        label = self.radio_var.get()
        self.column_index = self.headers.index(label)
        self.full_plot.set_ydata(self.data[:, self.column_index])
        self.update_zoom_according_to_main()
        self.ax.relim()
        self.ax.set_xlabel("Time (s)", labelpad=2)
        self.ax.set_ylabel(column_to_unit[label], labelpad=1)
        self.ax.autoscale_view()
        self.ax.set_xlim(0, self.ax.get_xlim()[1])
        self.canvas.draw()

    def update_zoom_according_to_main(self, force_x0=None):
        """Update zoom plot. Slider value (or force_x0) controls the window; masks remain as-is."""
        t0 = float(self.timestamps[0]); t1 = float(self.timestamps[-1])
        max_start = max(t0, t1 - self.zoom_window_width)

        if force_x0 is not None:
            x0 = float(np.clip(force_x0, t0, max_start))
        elif self.zoom_slider is not None and self.user_set_slider:
            x0 = float(np.clip(self.zoom_slider.get(), t0, max_start))
        else:
            margin = 10.0
            if self.interval_patches:
                guess = max(t0, float(self.interval_patches[0]['patch'].get_xy()[0][0]) - margin)
            else:
                guess = t0
            x0 = float(np.clip(guess, t0, max_start))
            if self.zoom_slider is not None:
                self._suspend_slider_callback = True
                try: self.zoom_slider.set(x0)
                finally: self._suspend_slider_callback = False

        x1 = float(min(x0 + self.zoom_window_width, t1))

        mask = (self.timestamps >= x0) & (self.timestamps <= x1)
        zoom_data = self.data[mask]
        zoom_y_data = zoom_data[:, self.column_index]

        self.zoom_ax.clear()
        self.zoom_ax.relim()
        self.zoom_ax.set_xlim(x0, x1)
        self.zoom_ax.set_xlabel("Time (s)", labelpad=2)
        self.zoom_ax.set_ylabel(column_to_unit[self.radio_var.get()], labelpad=1)

        # Recreate zoom spans from current main patches, clipped to [x0, x1]
        for span in getattr(self, 'zoom_interval_spans', []):
            try: span.remove()
            except Exception: pass
        self.zoom_interval_spans = []
        for patch in self.interval_patches:
            label = patch['label']
            start = float(patch['patch'].get_xy()[0][0])
            end   = float(patch['patch'].get_xy()[2][0])
            if end >= x0 and start <= x1:
                color = self.label_to_color[label]
                span = self.zoom_ax.axvspan(max(x0, start), min(x1, end), color=color, alpha=0.5, linewidth=0)
                self.zoom_interval_spans.append(span)

        if zoom_data.size:
            self.zoom_ax.plot(zoom_data[:, 0], zoom_y_data, 'r-')

        self.ax.figure.canvas.draw()

    def disconnect(self):
        self.ax.figure.canvas.mpl_disconnect(self.cidpress)
        self.ax.figure.canvas.mpl_disconnect(self.cidrelease)
        self.ax.figure.canvas.mpl_disconnect(self.cidmotion)

# ---------- Main ----------
if __name__ == "__main__":
    root = Tk()
    root.title("Matplotlib in Tkinter")
    root.tk.call('tk', 'scaling', 2.0)
    print("scaled root DPI", root.winfo_fpixels('1i'))

    app = DraggableIntervals(root, csv="./example_data/doremi_padded_simple_130_ble_imu_data_250429_200238.csv")
    app.connect()
    root.mainloop()
