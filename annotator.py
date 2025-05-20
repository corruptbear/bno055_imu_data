import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
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

class WrappedLabel(tk.Frame):
    def __init__(self, master, text="", width=200, height=30, bg='white', fg='black', relief='flat', borderwidth=1, command=None, **kwargs):
        super().__init__(master, height=height, **kwargs)

        self.command = command  # click callback
        self.canvas = tk.Canvas(self, height=height, width=width, bg=bg, highlightthickness=0, bd=0)
        self.canvas.pack(side='left', fill='both', expand=True)

        self.label = tk.Label(self.canvas, text=text, anchor='w', bg=bg, fg=fg, relief=relief, borderwidth=borderwidth, padx=5)
        self.label_id = self.canvas.create_window((0, 0), window=self.label, anchor='nw')

        self.label.bind("<Button-1>", self._on_click)

        # Optional horizontal scrollbar
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

#matplotlib.use('TkAgg')

dpi = 300
base_dpi = 200
base_font_size = 6
base_linewidth = 0.2
font_size_scale = dpi / base_dpi
scaled_font_size = base_font_size / font_size_scale
linewidth_scale = dpi / base_dpi
labelpad_scale = font_size_scale
#print(plt.rcParams.keys())

plt.rcParams.update({
    'font.size': scaled_font_size,
    'axes.titlesize': scaled_font_size * 1.1,
    'axes.labelsize': scaled_font_size * 0.8,
    'xtick.labelsize': scaled_font_size * 0.7,
    'ytick.labelsize': scaled_font_size * 0.7,
    'legend.fontsize': scaled_font_size * 0.7,  # Adjust the legend font size
    'lines.linewidth': base_linewidth / linewidth_scale,
    'axes.linewidth': 0.15 * (dpi / base_dpi),
    'figure.dpi':600,
    'savefig.dpi':600
})


# Create a mock event object to simulate the event handler
class MockEvent:
    pass

class DraggableIntervals:
    def __init__(self, master, csv=None):
        self.master = master
        self.master.bind_all("<ButtonRelease>", self.on_any_click)

        self.press = None
        self.zoom_press = None

        self.init_components()

        self.set_new_file(csv)

        # load the patch
        # TODO: if the file is already aligned, should load the existing patch
        #self.load_color_patches_offset(self.interval_name)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

    def init_components(self):
        self.fig, (self.ax, self.zoom_ax) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 2]}, figsize=(8, 6), dpi=300)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        #plt.subplots_adjust(bottom=0.1, hspace=0.4)
        plt.subplots_adjust(top=0.92, bottom=0.1, hspace=0.5, right=0.8) 

        print("figure DPI",self.fig.get_dpi())
        # Set font sizes and DPI
        self.set_font_sizes(dpi=300)  # Default DPI

        # Create Tkinter Frame for buttons
        self.button_frame = Frame(self.master)
        self.button_frame.pack(side='bottom', fill='x')

        # export button
        self.export_button = Button(self.button_frame, text='Export Labeled Data', command=self.on_export_button_press)
        self.export_button.pack(side='left', padx=5)

        # load file button
        self.load_button = Button(self.button_frame, text='Load File', command=self.load_file)
        self.load_button.pack(side='left', padx=5)

        self.save_figure_button = Button(self.button_frame, text='Save Figure', command=self.save_figure)
        self.save_figure_button.pack(side='left', padx=5)

        # Add Tkinter RadioButtons
        self.radio_frame = Frame(self.master)
        self.radio_frame.pack(side='right', fill='y')

        self.radio_var = StringVar()
        self.radio_buttons = []

        # Add Tkinter Listbox below RadioButtons
        self.alignments_listbox_frame = Frame(self.radio_frame)
        self.alignments_listbox_frame.pack(side='bottom', fill='x', pady=5)

        # Create and place the dropdown menu
        self.dropdown_var = StringVar()
        self.dropdown_var.set(list(all_intervals.keys())[0])
        self.dropdown_menu = OptionMenu(self.alignments_listbox_frame, self.dropdown_var, *[key for key in all_intervals])
        self.dropdown_menu.pack(pady=5, fill='x')  # also on top

        # Add Tkinter Button for saving alignment
        self.add_new_alignment_button = Button(self.alignments_listbox_frame, text='Add New', command=self.add_new_alignment)
        self.add_new_alignment_button.pack(padx=5, fill='y')  # on top

        self.dropdown_var.trace("w", self.on_dropdown_change)

        self.alignments_listbox_label = Label(self.alignments_listbox_frame, text="Select a mask:")
        self.alignments_listbox_label.pack(anchor='w')
 

        self.alignments_listbox = Listbox(self.alignments_listbox_frame, height=10, selectmode=tk.SINGLE)  # Adjust height as needed
        self.alignments_listbox.pack(fill='x', padx=5)
        self.alignments_listbox.bind("<<ListboxSelect>>", self.on_alignments_listbox_select)

        # Add a button to delete the selected listbox item
        self.delete_button = Button(self.alignments_listbox_frame, text="Delete", command=self.delete_selected_alignments_listbox_item)
        self.delete_button.pack(pady=5)
 
        """
        # Instead of Listbox, use this container
        self.alignments_list_frame = Frame(self.alignments_listbox_frame)
        self.alignments_list_frame.pack(fill='both', expand=True)
        self.selected_alignment_index = None

        # Store the current list of alignments
        self.alignment_items = ["a","b","c"]  # or load from file
        self.render_alignment_items()
        """

        # Add a button to export
        self.save_alignments_button = Button(self.alignments_listbox_frame, text="Save", command=self.save_alignments)
        self.save_alignments_button.pack(pady=5)

    def render_alignment_items(self):
        # Clear previous widgets
        for widget in self.alignments_list_frame.winfo_children():
            widget.destroy()

        for idx, item in enumerate(self.alignment_items):
            item_frame = Frame(self.alignments_list_frame)
            item_frame.pack(fill='x', padx=5, pady=2)

            # Define background color for selected
            bg = '#cce5ff' if idx == self.selected_alignment_index else 'white'

            label = Label(item_frame, text=item, anchor='w', bg=bg, relief='sunken' if idx == self.selected_alignment_index else 'flat')
            label.pack(side='left', fill='x', expand=True)

            # Bind click to select
            label.bind("<Button-1>", lambda e, i=idx: self.on_alignment_label_click(i))

            delete_btn = Button(
                item_frame, text='X', fg='red',
                command=lambda i=idx: self.delete_alignment_item(i),
                bd=0, padx=4, pady=0
            )
            delete_btn.pack(side='right')
    def on_alignment_label_click(self, index):
        self.selected_alignment_index = index
        self.render_alignment_items()
    def delete_alignment_item(self, index):
        del self.alignment_items[index]
        self.render_alignment_items()


    def on_any_click(self, event):
        if isinstance(event.widget, tk.Button):
            timestamp = time.time()  # Unix timestamp (float)
            print(f"Clicked: {event.widget['text']} at {timestamp:.3f}")  # milliseconds precision
            self.annotation_button_click_log[timestamp] = f"{event.widget['text']}"
            save_yaml(self.annotation_button_click_log, self.annotation_button_click_log_path, "w")

    def read_data(self, csv):
        self.csv_path = csv
        base, ext = os.path.splitext(self.csv_path)
        self.alignment_path = base + '_alignment.yaml'
        self.annotation_button_click_log_path = base + '_annotation_button_click_log_' + datetime.now().strftime("%y%m%d_%H%M%S") + '.yaml'
        print(f"alignment path is {self.alignment_path}")
        self.export_path = base + "_labeled.csv"
        # Read the CSV file into a DataFrame
        #df = pd.read_csv("./pkls/0_mix1.csv")
        df = pd.read_csv(self.csv_path)

        all_data = df.values
        # set the timestamp start to 0
        all_data[:, 0] = all_data[:, 0] - all_data[0][0]
        self.data = all_data
        self.timestamps = all_data[:, 0]
        self.headers = df.columns.tolist()
        if "gyro_x" in df.columns:
            self.column_index = df.columns.get_loc("gyro_x")
        elif "lacc_x" in df.columns:
            self.column_index = df.columns.get_loc("lacc_x")
        elif "quat_x" in df.columns:
            self.column_index = df.columns.get_loc("quat_x")
        else:
            self.column_index = 5


    def reset_radio_buttons(self):
        # Destroy everything inside the radio_frame (frames, labels, buttons, etc.)
        for widget in self.radio_frame.winfo_children():
            if widget is not self.alignments_listbox_frame:
                        widget.destroy()

        # Also reset the list
        self.radio_buttons = []
        grouped = defaultdict(list)

        # Group by prefix (before '_')
        for header in self.headers:  # Skip 'timestamp'
            if "timestamp" in header:
                continue
            if '_' in header:
                prefix, axis = header.split('_', 1)
                grouped[prefix].append(axis)

        # Create UI
        for prefix in grouped.keys():
            # Outer group frame
            group_frame = Frame(self.radio_frame)
            group_frame.pack(anchor='w', pady=5)

            # Label on top
            Label(group_frame, text=prefix).pack()

            # Row of buttons
            row = Frame(group_frame)
            row.pack()

            for axis in sorted(grouped[prefix]):  # Ensure x, y, z order
                full_name = f"{prefix}_{axis}"
                rb = Radiobutton(row, text=axis, variable=self.radio_var, value=full_name, command=self.update_plot)
                rb.pack(side='left', padx=5)
                self.radio_buttons.append(rb)

    def set_new_file(self,csv):
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

        # initialize plot
        self.ax.clear()
        self.zoom_ax.clear()

        self.full_plot, = self.ax.plot(self.data[:, 0], self.data[:, self.column_index], label='Time Series A')
        # Initialize zoom plot
        self.zoom_line, = self.zoom_ax.plot([], [], 'r-')

        self.radio_var.set(self.headers[self.column_index])

        #for rb in self.radio_buttons:
        #    rb.destroy()
        #self.radio_buttons = []
        self.reset_radio_buttons()

        #self.dropdown_var.set(self.interval_name)  # Default value
        self.alignments_listbox.delete(0, tk.END)
        print(self.interval_patches)
        self.load_alignments()
        print("after loading alignments:", self.alignment_offsets)
        self.annotation_button_click_log = dict()

    def load_alignments(self):
        """
        load the saved alignments for the current file
        """
        self.alignment_offsets = load_yaml(self.alignment_path)

        # Populate the alignments_listbox
        # keep the _1, _2... endings
        for interval_name in self.alignment_offsets:
            self.alignments_listbox.insert(tk.END, interval_name)

        # default selection: first item
        if len(self.alignment_offsets):
            self.alignments_listbox.selection_set(0)
            # simulate mouse click
            mock_event = MockEvent()
            mock_event.widget = self.alignments_listbox 
            self.on_alignments_listbox_select(mock_event)
        else:
            #for a brand new file: just load the default patches
            self.load_color_patches(self.dropdown_var.get())

    def add_new_alignment(self):
        """
        add new alignment to the listbox
        save alignment offset associated with the new alignment
        """
        patch = self.interval_patches[0]
        # current position minus the original position (not necessary 0 because we remove ready patches)
        offset = float(patch['patch'].get_xy()[0][0])  - float(patch['limits'][0])

        dropdown_interval_name = self.dropdown_var.get()

        #save the new alignment name (as in the dropdown menu) to the offsets list and the listbox
        new_key = dropdown_interval_name
        print("add new alignment:", new_key, self.alignment_offsets)
        if new_key in self.alignment_offsets:
            counter = 1
            while new_key in self.alignment_offsets:
                new_key = f"{dropdown_interval_name}_{counter}"
                counter += 1
            self.alignment_offsets[new_key] = offset
        else:
            # Key does not exist, simply add it
            self.alignment_offsets[new_key] = offset
        #add the new item
        self.alignments_listbox.insert(tk.END, new_key)
        #clear existing selection and select the new item
        self.alignments_listbox.selection_clear(0, tk.END)
        new_index = self.alignments_listbox.get(0, tk.END).index(new_key)
        self.alignments_listbox.select_set(new_index)

        print(self.alignment_offsets)

    def save_alignments(self):
        """
        save alignments (as shown in the listbox)
        """
        if not self.alignment_offsets:
            messagebox.showwarning("Warning", "No alignments to save. click ADD NEW first if you have something to save.")
            return

        print("Save alignments to:", self.alignment_path)
        save_yaml(self.alignment_offsets, self.alignment_path, 'w')

    def delete_selected_alignments_listbox_item(self):
        # Get the index of the selected item
        selected_items = self.alignments_listbox.curselection()
        if selected_items:
            for index in selected_items[::-1]:  # Reverse to avoid index shifting
                list_entry_to_be_deleted = self.alignments_listbox.get(index)
                self.alignments_listbox.delete(index)
                # also delete the entry from the offsets table
                # TODO: rename the existing offsets if there are multiple instances of the same track and you are deleting one of them
                del self.alignment_offsets[list_entry_to_be_deleted]

    def on_alignments_listbox_select(self, event):
        selection = event.widget.curselection()
        if selection:
            self.listbox_selected_interval_name = event.widget.get(selection[0])
            print(f"Selected item: {self.listbox_selected_interval_name}")
            self.load_color_patches_offset(self.listbox_selected_interval_name)

    def set_font_sizes(self, dpi):
        """
        Set font sizes based on DPI.
        """
        # Apply font sizes to both axes
        for ax in [self.ax, self.zoom_ax]:
            ax.title.set_fontsize(plt.rcParams['axes.titlesize'])
            ax.xaxis.label.set_fontsize(plt.rcParams['axes.labelsize'])
            ax.yaxis.label.set_fontsize(plt.rcParams['axes.labelsize'])
            ax.tick_params(axis='both', which='major', labelsize=plt.rcParams['xtick.labelsize'],width=plt.rcParams['axes.linewidth'], pad=labelpad_scale)
            ax.tick_params(axis='both', which='minor', labelsize=plt.rcParams['ytick.labelsize'],width=plt.rcParams['axes.linewidth'], pad=labelpad_scale)
            for axis in ['top','bottom','left','right']:
                ax.spines[axis].set_linewidth(plt.rcParams['axes.linewidth'])

    def on_dropdown_change(self, *args):
        """
        IMPORTANT: merely changing the dropdown menu will not add the item to the listbox
        """
        #TODO: change it to only load new interval
        #self.listbox_selected_interval_name = self.dropdown_var.get()
        listbox_selected_interval_name = self.dropdown_var.get()
        #print(f"Selected value from dropdown: {self.listbox_selected_interval_name}")
        print(f"Selected value from dropdown: {listbox_selected_interval_name}", self.alignment_offsets)

        #load new masks
        #self.load_color_patches(self.listbox_selected_interval_name)
        self.load_color_patches(listbox_selected_interval_name)

    def load_color_patches_offset(self, interval_name):
        """
        load the color patches, not starting from time 0
        """
        #TODO: when there are multiple instances of the same patch in the alignments, currently only the first patch is loaded; should enable selection in such cases
        offset = 0
        if interval_name in self.alignment_offsets:
            offset = self.alignment_offsets[interval_name]

        self.load_color_patches(interval_name, offset = offset)

    def load_color_patches(self, interval_name, offset = 0):
        #print("load_color_patches called")
        base_interval_name = get_base_name(interval_name)
        intervals = [(x, y, z) for (x, y, z) in all_intervals[base_interval_name] if "ready" not in x]
        # the interval name could end with _1, _2...
        #self.interval_name = interval_name
        # Clear the existing patches
        for p in self.interval_patches:
            p["patch"].remove()
        self.interval_patches = []
        # Plot interval patches
        colormap = plt.get_cmap("tab10")
        self.label_to_color = {}
        for label, start, end in intervals:
            if label not in self.label_to_color:
                self.label_to_color[label] = colormap(len(self.label_to_color) % colormap.N)
            color = self.label_to_color[label]
            # apply offset to patches
            patch = self.ax.axvspan(start+offset, end+offset, color=color, alpha=0.5, label=label, linewidth=0)
            self.interval_patches.append({'patch': patch, 'label': label, 'color': color, 'limits': (start, end)})

        # Add custom patches to the legend
        handles = []
        added_labels = set()
        for label, start, end in intervals:
            if label not in added_labels:
                handles.append(Patch(color=self.label_to_color[label], alpha=0.5, label=label, linewidth=0))
                added_labels.add(label)
        self.patch_legend = self.ax.legend(handles=handles, prop={'size': plt.rcParams['legend.fontsize']},bbox_to_anchor=(1, 0.5))
        self.patch_legend.get_frame().set_linewidth(0.2)
        self.update_plot()

    def connect(self):
        self.cidpress = self.ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cidrelease = self.ax.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion = self.ax.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)

        # Connect events for the zoom plot (zoom_ax). 
        # self.ax.figure and self.zoom_ax.figure are the same
        self.zoom_cidpress = self.zoom_ax.figure.canvas.mpl_connect('button_press_event', self.on_zoom_press)
        self.zoom_cidrelease = self.zoom_ax.figure.canvas.mpl_connect('button_release_event', self.on_zoom_release)
        self.zoom_cidmotion = self.zoom_ax.figure.canvas.mpl_connect('motion_notify_event', self.on_zoom_motion)

    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        for patch in self.interval_patches:
            contains, _ = patch['patch'].contains(event)
            if contains:
                self.press = event.xdata
                self.background = self.ax.figure.canvas.copy_from_bbox(self.ax.bbox)
                self.update_zoom_according_to_main()
                break

    def on_motion(self, event):
        if self.press is None:
            return
        if event.inaxes != self.ax:
            return
        dx = event.xdata - self.press

        current_last_patch_end = self.interval_patches[-1]['patch'].get_xy()[2][0]
        current_first_patch_start = self.interval_patches[0]['patch'].get_xy()[0][0]

        xmin, xmax = self.ax.get_xlim()

        # limit the move range, so that all the spans are contained in the current window
        if (current_last_patch_end + dx > xmax) or (current_first_patch_start + dx < xmin):
            return

        for i, p in enumerate(self.interval_patches):
            new_x0 = p['patch'].get_xy()[0][0] + dx
            new_x1 = p['patch'].get_xy()[2][0] + dx
            p['patch'].set_xy([
                (new_x0, 0),
                (new_x0, 1),
                (new_x1, 1),
                (new_x1, 0),
                (new_x0, 0)
            ])
        self.press = event.xdata
        self.ax.figure.canvas.restore_region(self.background)
        self.ax.draw_artist(self.ax.patches[0])
        for p in self.interval_patches:
            self.ax.draw_artist(p['patch'])
        self.ax.figure.canvas.blit(self.ax.bbox)
        self.update_zoom_according_to_main()

    def on_release(self, event):
        if event.inaxes != self.ax:
            return
        self.press = None
        self.ax.figure.canvas.draw()
        patch = self.interval_patches[0]
        # current position minus the original position (not necessary 0 because we remove ready patches)
        offset = float(patch['patch'].get_xy()[0][0])  - float(patch['limits'][0])
        # save the current offset

        # only modify alignment offsets if the listbox is selected
        if hasattr(self, 'listbox_selected_interval_name'):
            self.alignment_offsets[self.listbox_selected_interval_name] = offset
        print("on release",offset, self.alignment_offsets)

    def on_zoom_press(self, event):
        """
        Handle mouse press in the zoom plot.
        """
        if event.inaxes != self.zoom_ax:
            return
        self.zoom_press = event.xdata  # Store the initial x position
        xmin, xmax = self.zoom_ax.get_xlim()
        #print(f"on_zoom_press: xmin={xmin},xmax={xmax}")

    def on_zoom_motion(self, event):
        """
        Handle mouse motion in the zoom plot.
        Effect: move the span but keep the window static
        """
        if self.zoom_press is None:
            return
        if event.inaxes != self.zoom_ax:
            return

        current_last_patch_end = self.interval_patches[-1]['patch'].get_xy()[2][0]
        current_first_patch_start = self.interval_patches[0]['patch'].get_xy()[0][0]

        xmin, xmax = self.zoom_ax.get_xlim()
        # Calculate the drag offset
        dx = event.xdata - self.zoom_press

        # limit the move range, so that all the spans are contained in the current window
        #if (current_last_patch_end + dx > xmax) or (current_first_patch_start + dx < xmin):
        if (current_first_patch_start + dx < xmin):
            return

        # Update the patches in the main plot (ax)
        for patch in self.interval_patches:
            new_x0 = patch['patch'].get_xy()[0][0] + dx
            new_x1 = patch['patch'].get_xy()[2][0] + dx
            patch['patch'].set_xy([
                (new_x0, 0),
                (new_x0, 1),
                (new_x1, 1),
                (new_x1, 0),
                (new_x0, 0)
            ])

        # Update the zoom plot
        self.zoom_press = event.xdata

        # clear the spans in zoom_ax
        for span in self.zoom_interval_spans:
            span.remove()
        self.zoom_interval_spans = []
        # redraw the spans in zoom_ax
        for patch in self.interval_patches:
            label = patch['label']
            start = patch['patch'].get_xy()[0][0]
            end = patch['patch'].get_xy()[2][0]
            color = self.label_to_color[label]
            span = self.zoom_ax.axvspan(start, end, color=color, alpha=0.5, linewidth=0)
            self.zoom_interval_spans.append(span)

        self.ax.figure.canvas.draw()

    def on_zoom_release(self, event):
        """
        Handle mouse release in the zoom plot.
        """
        if event.inaxes != self.zoom_ax:
            return
        self.zoom_press = None
        # Save the new offset if an alignment is selected
        if self.alignments_listbox.curselection():
            selected_index = self.alignments_listbox.curselection()[0]
            selected_item = self.alignments_listbox.get(selected_index)
            patch = self.interval_patches[0]
            offset = float(patch['patch'].get_xy()[0][0]) - float(patch['limits'][0])
            self.alignment_offsets[selected_item] = offset
            print("Updated offset:", offset)

    def on_export_button_press(self):
        """
        export button
        """
        extract_labeled_data(self.csv_path)
 
    def load_file(self):
        """
        load another file
        """
        file_path = filedialog.askopenfilename(filetypes=[('csv Files', '*.csv')], initialdir="./example_data")
        if file_path:
            self.set_new_file(file_path)
    def save_figure(self):
        """
        Save the current figure as an image file
        """
        # Generate default filename based on the current data file
        if hasattr(self, 'csv_path'):
            default_name = os.path.splitext(os.path.basename(self.csv_path))[0] + "_figure"
        else:
            default_name = "figure"
    
        # Add timestamp to filename to avoid overwrites
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"{default_name}_{timestamp}"
    
        # Open file dialog to let user choose save location and format
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("PDF files", "*.pdf"),
                ("JPEG files", "*.jpg")
            ],
            initialfile=default_name
        )
    
        if file_path:
            try:
                # Save the figure with high quality
                self.fig.savefig(
                    file_path, 
                    dpi=1000,
                    bbox_inches='tight',  # This includes the legend in the saved figure
                    pad_inches=0.1
                )
                messagebox.showinfo("Success", f"Figure saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save figure: {str(e)}")

    def update_plot(self):
        """
        Draw both the main plot and the zoom plot according to current column and patches
        """
        label = self.radio_var.get()
        self.column_index = self.headers.index(label)
        time_series = self.data[:, self.column_index]
        self.full_plot.set_ydata(time_series)
        self.update_zoom_according_to_main()

        #x0 = max(self.interval_patches[0]['patch'].get_xy()[0][0] - 10, self.timestamps[0])
        #x1 = min(self.interval_patches[-1]['patch'].get_xy()[2][0] + 10, self.timestamps[-1])

        self.ax.relim()
        self.ax.set_xlabel("Time (s)", labelpad=2)
        self.ax.set_ylabel(column_to_unit[label], labelpad=1)
        self.ax.autoscale_view()
        self.ax.set_xlim(0,self.ax.get_xlim()[1])
        self.canvas.draw()

    def update_zoom_according_to_main(self):
        """
        Update zoom plot, set the scope to be around the start and end of the currently applied intervals
        """
        #print("update_zoom_according_to_main")
        window_size = 100
        x0 = max(self.interval_patches[0]['patch'].get_xy()[0][0] - 10, self.timestamps[0])
        x1 = min(self.interval_patches[-1]['patch'].get_xy()[2][0] + 10, self.timestamps[-1])
        zoom_data = self.data[(self.timestamps >= x0) & (self.timestamps <= x1)]
        zoom_y_data = zoom_data[:, self.column_index]
        x1 = x0+60 #hot apply

        self.zoom_ax.clear()  # Clear the entire axis

        self.zoom_ax.relim()
        self.zoom_ax.set_xlim(x0, x1)
        self.zoom_ax.set_xlabel("Time (s)", labelpad=2)
        self.zoom_ax.set_ylabel(column_to_unit[self.radio_var.get()], labelpad=1)

        for span in self.zoom_interval_spans:
            span.remove()
        self.zoom_interval_spans = []

        # draw the color patches inside the zoom window
        for patch in self.interval_patches:
            label = patch['label']
            start = patch['patch'].get_xy()[0][0]
            end = patch['patch'].get_xy()[2][0]
            color = self.label_to_color[label]
            span = self.zoom_ax.axvspan(max(x0,start), min(x1,end), color=color, alpha=0.5, linewidth=0)
            self.zoom_interval_spans.append(span)

        self.zoom_ax.plot(zoom_data[:, 0], zoom_y_data, 'r-')
        self.ax.figure.canvas.draw()

    def disconnect(self):
        self.ax.figure.canvas.mpl_disconnect(self.cidpress)
        self.ax.figure.canvas.mpl_disconnect(self.cidrelease)
        self.ax.figure.canvas.mpl_disconnect(self.cidmotion)

# Main Tkinter application
if __name__ == "__main__":
    root = Tk()
    root.title("Matplotlib in Tkinter")
    root.tk.call('tk', 'scaling', 2.0)
    print("scaled root DPI", root.winfo_fpixels('1i'))

    # Initialize Tkinter frame and draggable intervals
    app = DraggableIntervals(root, csv="./example_data/doremi_padded_simple_130_ble_imu_data_250429_200238.csv")
    #app = DraggableIntervals(root, csv="./pkls/0_doremi_acc_partial.csv")
    #app = DraggableIntervals(root, csv="./pkls/0_yankee.csv")
    #app = DraggableIntervals(root, csv="./pkls/0_Yankee_doodle_Saloon_style_padded_100.csv")
    #app = DraggableIntervals(root, csv="./pkls/0_doremi_acc_yankee.csv") # weird quat_x???
    #app = DraggableIntervals(root, csv="./pkls/0_k265_device36.csv")

    app.connect()

    root.mainloop()