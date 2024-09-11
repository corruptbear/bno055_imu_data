import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import Tk, Frame, filedialog, Radiobutton, StringVar, Button, OptionMenu
from tkinter import Label, Entry
import tkinter as tk
from tkinter import messagebox
from load_imu_data import formats
import os
import yaml

def save_yaml(dictionary,filepath,write_mode):
    with open(filepath,write_mode) as f:
        yaml.dump(dictionary,f)

def load_yaml(filepath):
    try:
        with open(filepath,'r') as stream:
            dictionary = yaml.safe_load(stream)
            return dictionary
    except:
        return dict()


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
    'axes.titlesize': scaled_font_size * 1.2,
    'axes.labelsize': scaled_font_size,
    'xtick.labelsize': scaled_font_size * 0.8,
    'ytick.labelsize': scaled_font_size * 0.8,
    'legend.fontsize': scaled_font_size * 0.8,  # Adjust the legend font size
    'lines.linewidth': base_linewidth / linewidth_scale,
    'axes.linewidth': 0.15 * (dpi / base_dpi),
    'figure.dpi':300
})

class DraggableIntervals:
    def __init__(self, master, interval_name, csv=None):
        self.master = master
        self.fig, (self.ax, self.zoom_ax) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 2]}, figsize=(8, 6), dpi=300)
        plt.subplots_adjust(bottom=0.1, hspace=0.4)
        print("figure DPI",self.fig.get_dpi())
        self.csv_path = csv
        base, ext = os.path.splitext(self.csv_path)
        self.alignment_path = base + '_alignment.yaml'

        self.load_alignments()
        print(type(self.alignment_offsets))
        print(self.alignment_offsets)

        self.interval_name = interval_name
        
        # Read the CSV file into a DataFrame
        #df = pd.read_csv("./pkls/0_mix1.csv")
        df = pd.read_csv(csv)
    
        all_data = df.values
        headers = df.columns.tolist()

        # set the start to 0
        all_data[:, 0] = all_data[:, 0] - all_data[0][0]
        self.timestamps = all_data[:, 0]
        column_index = 5  # Default column index

        # Set font sizes and DPI
        self.set_font_sizes(dpi=300)  # Default DPI

        self.headers = headers
        self.data = all_data
        self.column_index = column_index
        self.press = None
        self.offsets = []
        self.interval_patches = []
        self.label_to_color = {}

        # Plot initial data
        self.full_plot, = self.ax.plot(self.data[:, 0], self.data[:, column_index], label='Time Series A')

        # Initialize zoom plot
        self.zoom_line, = self.zoom_ax.plot([], [], 'r-')
        self.zoom_ax.set_title("Zoomed In View")
        self.zoom_ax.set_xlabel("Time")
        self.zoom_ax.set_ylabel("Value")

        # Create Tkinter Frame for buttons
        self.button_frame = Frame(self.master)
        self.button_frame.pack(side='bottom', fill='x')

        # export button
        self.export_button = Button(self.button_frame, text='Print A', command=self.on_export_button_press)
        self.export_button.pack(side='left', padx=5)

        # load file button
        self.load_button = Button(self.button_frame, text='Load File', command=self.load_file)
        self.load_button.pack(side='left', padx=5)
        
        # Add Tkinter Button for saving alignment
        self.save_alignment_button = Button(self.button_frame, text='Save Alignment', command=self.save_alignment)
        self.save_alignment_button.pack(side='left', padx=5)

        # Add Tkinter RadioButtons
        self.radio_var = StringVar(value=headers[column_index])
        self.radio_frame = Frame(self.master)
        self.radio_frame.pack(side='right', fill='y')

        self.radio_buttons = []
        for header in headers[5:]:
            rb = Radiobutton(self.radio_frame, text=header, variable=self.radio_var, value=header, command=self.update_plot)
            rb.pack(anchor='w')
            self.radio_buttons.append(rb)

        # Create and place the dropdown menu
        self.dropdown_var = StringVar()
        self.dropdown_var.set(interval_name)  # Default value
        self.dropdown_menu = OptionMenu(self.button_frame, self.dropdown_var, *[key for key in all_intervals])
        self.dropdown_menu.pack(pady=10, side=tk.LEFT)
        self.dropdown_var.trace("w", self.on_dropdown_change)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)

        # load the patch
        self.load_color_patches_offset(self.interval_name)

        self.canvas.get_tk_widget().pack(fill='both', expand=True)

    def load_alignments(self):
        """
        load the saved alignments for the current file
        """
        self.alignment_offsets = load_yaml(self.alignment_path)

    def save_alignment(self):
        """
        Save alignment with handling for duplicated keys (allowing multiple instances of the same song)

        Save one at a time
        """
        print("Save alignments to:", self.alignment_path)
        patch = self.interval_patches[0]
        # current position minus the original position (not necessary 0 because we remove ready patches)
        offset = float(patch['patch'].get_xy()[0][0])  - float(patch['limits'][0])

        if self.interval_name in self.alignment_offsets:
            # Ask user if they want to override or save as a new value
            result = messagebox.askquestion(
                "Overwrite or Save As New",
                f"The key '{self.interval_name}' already exists. Do you want to overwrite it or save as a new key?"
                "\n\nChoose 'Yes' to overwrite or 'No' to save as a new key.",
                icon='warning'
            )
            if result == 'yes':
                # Overwrite existing key
                self.alignment_offsets[self.interval_name] = offset
            else:
                # Save as a new unique key
                new_key = self.interval_name
                counter = 1
                while new_key in self.alignment_offsets:
                    new_key = f"{self.interval_name}_{counter}"
                    counter += 1
                self.alignment_offsets[new_key] = offset
        else:
            # Key does not exist, simply add it
            self.alignment_offsets[self.interval_name] = offset

        print(self.alignment_offsets)

        save_yaml(self.alignment_offsets, self.alignment_path, 'w')

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
        selected_interval_name = self.dropdown_var.get()
        print(f"Selected value from dropdown: {selected_interval_name}")

        self.load_color_patches_offset(selected_interval_name)

    def load_color_patches_offset(self, interval_name):
        """
        load the color patches, not starting from time 0
        """
        offset = 0
        if interval_name in self.alignment_offsets:
            offset = self.alignment_offsets[interval_name]
        self.load_color_patches(interval_name, offset = offset)

    def load_color_patches(self, interval_name, offset = 0):
        intervals = [(x, y, z) for (x, y, z) in all_intervals[interval_name] if "ready" not in x]
        self.interval_name = interval_name
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
        self.patch_legend = self.ax.legend(handles=handles, prop={'size': plt.rcParams['legend.fontsize']})
        self.patch_legend.get_frame().set_linewidth(0.2)
        self.update_plot()

    def connect(self):
        self.cidpress = self.ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cidrelease = self.ax.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion = self.ax.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        for patch in self.interval_patches:
            contains, _ = patch['patch'].contains(event)
            if contains:
                self.press = event.xdata
                self.offsets = [event.xdata - p['patch'].get_xy()[0][0] for p in self.interval_patches]
                self.background = self.ax.figure.canvas.copy_from_bbox(self.ax.bbox)
                self.update_zoom(event.xdata)
                break

    def on_motion(self, event):
        if self.press is None:
            return
        if event.inaxes != self.ax:
            return
        dx = event.xdata - self.press
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
        self.update_zoom(event.xdata)

    def on_release(self, event):
        self.press = None
        self.ax.figure.canvas.draw()

    def on_export_button_press(self):
        """
        export button
        """
        # TODO: save label
        # create empty array of the same shape
        save  = np.empty((0, self.data.shape[1]),dtype=object)
        # the column for labels (add as the last column)
        labels = []
        for patch in self.interval_patches:
            start_timestamp, end_timestamp = patch['patch'].get_xy()[0][0], patch['patch'].get_xy()[2][0]
            print(patch['label'], start_timestamp, end_timestamp)
            selected = [self.data[k,:] for k in range(len(self.data)) if self.data[k][0]>=start_timestamp and  self.data[k][0]<=end_timestamp]
            labels = labels + [patch['label']]*len(selected)
            save = np.vstack([save, selected])
    
        new_column = np.array(labels).reshape(-1,1)

        # Add the new column
        save = np.hstack([save, new_column])
        new_headers = ','.join(headers+["label"])
        new_fmt = formats+["%s"]
        # TODO:
        np.savetxt(f"./test.csv", save, delimiter=',', header=new_headers, comments='', fmt=new_fmt)
 
    def load_file(self):
        """
        TODO: this is for test only; load .txt file for test
        """
        file_path = filedialog.askopenfilename(filetypes=[('Text Files', '*.txt')])
        if file_path:
            with open(file_path, 'r') as file:
                content = file.read()
                print(content)

    def update_plot(self):
        """
        Draw both plots according to current column and patches
        """
        label = self.radio_var.get()
        self.column_index = self.headers.index(label)
        time_series = self.data[:, self.column_index]
        self.full_plot.set_ydata(time_series)
        self.update_zoom(self.ax.get_xlim()[0])
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw()

    def update_zoom(self, x_center):
        window_size = 100
        x0 = max(self.interval_patches[0]['patch'].get_xy()[0][0] - 10, self.timestamps[0])
        x1 = min(self.interval_patches[-1]['patch'].get_xy()[2][0] + 10, self.timestamps[-1])
        self.zoom_ax.set_xlim(x0, x1)
        zoom_data = self.data[(self.timestamps >= x0) & (self.timestamps <= x1)]
        self.zoom_line.set_data(zoom_data[:, 0], zoom_data[:, self.column_index])
        self.zoom_ax.relim()
        self.zoom_ax.autoscale_view()

        self.zoom_ax.clear()  # Clear the entire axis

        for patch in self.interval_patches:
            label = patch['label']
            start = patch['patch'].get_xy()[0][0]
            end = patch['patch'].get_xy()[2][0]
            color = self.label_to_color[label]
            if start >= x0 and end <= x1:
                patch = self.zoom_ax.axvspan(start, end, color=color, alpha=0.5, linewidth=0)

        self.zoom_ax.plot(zoom_data[:, 0], zoom_data[:, self.column_index], 'r-')
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

    all_intervals = {"doremi_acc_padded8":[('get_ready', 0.0, 4.8), ('moving', 4.8, 7.2), ('stop', 7.2, 9.6), ('moving', 9.6, 12.0), ('stop', 12.0, 14.4), ('moving', 14.4, 16.8), ('stop', 16.8, 19.2), ('walk', 19.2, 22.8), ('turn_back', 22.8, 24.0), ('walk', 24.0, 27.6), ('turn_back', 27.6, 28.8), ('walk', 28.8, 32.4), ('turn_back', 32.4, 33.6), ('walk', 33.6, 37.2), ('turn_back', 37.2, 38.4), ('walk', 38.4, 42.0), ('turn_back', 42.0, 43.2), ('walk', 43.2, 46.8), ('turn_back', 46.8, 48.0), ('walk', 48.0, 51.6), ('turn_back', 51.6, 52.8), ('walk', 52.8, 56.4), ('turn_back', 56.4, 57.6), ('walk', 57.6, 60.872730000000004), ('turn_back', 60.872730000000004, 61.96364), ('walk', 61.96364, 65.23637000000001), ('turn_back', 65.23637000000001, 66.32728), ('walk', 66.32728, 69.60001), ('turn_back', 69.60001, 70.69092), ('walk', 70.69092, 73.96365), ('turn_back', 73.96365, 75.05456000000001), ('walk', 75.05456000000001, 78.32729), ('turn_back', 78.32729, 79.4182), ('walk', 79.4182, 82.69093000000001), ('turn_back', 82.69093000000001, 83.78184), ('walk', 83.78184, 87.05457), ('turn_back', 87.05457, 88.14548), ('walk', 88.14548, 91.41821), ('turn_back', 91.41821, 92.50912), ('walk', 92.50912, 95.50912), ('turn_back', 95.50912, 96.50912), ('walk', 96.50912, 99.50912), ('turn_back', 99.50912, 100.50912), ('walk', 100.50912, 103.50912), ('turn_back', 103.50912, 104.50912), ('walk', 104.50912, 107.50912), ('turn_back', 107.50912, 108.50912), ('walk', 108.50912, 111.50912), ('turn_back', 111.50912, 112.50912), ('walk', 112.50912, 115.50912), ('turn_back', 115.50912, 116.50912), ('walk', 116.50912, 119.50912), ('turn_back', 119.50912, 120.50912), ('stop', 120.50912, 126.537143)],
                     "Yankee_doodle_Saloon_style_padded8_120":[('get_ready', 0.0, 4.0), ('moving', 4.0, 6.0), ('stop', 6.0, 8.0), ('moving', 8.0, 10.0), ('stop', 10.0, 12.0), ('moving', 12.0, 14.0), ('stop', 14.0, 16.0), ('get_ready_to_walk_forward', 16.0, 20.0), ('walking_forward', 20.0, 36.0), ('get_ready_to_run_forward', 36.0, 40.0), ('running_forward', 40.0, 56.0), ('get_ready_to_walk_forward', 56.0, 60.0), ('walking_forward', 60.0, 76.0), ('get_ready_to_stand_still', 76.0, 80.0), ('standing_still', 80.0, 98.142041)],
                     "Yankee_doodle_Saloon_style_padded6_100":[('moving', 0.0, 2.4), ('stop', 2.4, 4.8), ('moving', 4.8, 7.2), ('stop', 7.2, 9.6), ('moving', 9.6, 12.0), ('stop', 12.0, 14.4), ('get_ready_to_walk_forward', 14.4, 19.2), ('walking_forward', 19.2, 38.4), ('get_ready_to_run_forward', 38.4, 43.2), ('running_forward', 43.2, 62.4), ('get_ready_to_walk_forward', 62.4, 67.2), ('walking_forward', 67.2, 86.4), ('get_ready_to_stand_still', 86.4, 91.2), ('standing_still', 91.2, 112.53551)],
                     "K265_cutmore_padded_120":[('get_ready', 0.0, 4.0), ('moving', 4.0, 6.0), ('stop', 6.0, 8.0), ('moving', 8.0, 10.0), ('stop', 10.0, 12.0), ('moving', 12.0, 14.0), ('stop', 14.0, 16.0), ('climb_up', 16.0, 24.0), ('have_a_rest', 24.0, 32.0), ('climb_down', 32.0, 40.0), ('have_a_rest', 40.0, 48.0), ('climb_up', 48.0, 56.0), ('have_a_rest', 56.0, 64.0), ('climb_down', 64.0, 72.0), ('have_a_rest', 72.0, 80.0), ('climb_up', 80.0, 88.0), ('have_a_rest', 88.0, 96.0), ('climb_down', 96.0, 104.0), ('have_a_rest', 104.0, 112.0), ('climb_up', 112.0, 120.0), ('have_a_rest', 120.0, 128.0), ('climb_down', 128.0, 136.0), ('have_a_rest', 136.0, 144.0), ('climb_up', 144.0, 152.0), ('have_a_rest', 152.0, 160.0), ('climb_down', 160.0, 168.0), ('have_a_rest', 168.0, 176.0), ('climb_up', 176.0, 184.0), ('have_a_rest', 184.0, 192.0), ('climb_down', 192.0, 200.0), ('have_a_rest', 200.0, 210.050612)],
                     "K265_cutmore_padded_130":[('get_ready', 0.0, 3.692304), ('moving', 3.692304, 5.538456), ('stop', 5.538456, 7.384608), ('moving', 7.384608, 9.23076), ('stop', 9.23076, 11.076912), ('moving', 11.076912, 12.923064), ('stop', 12.923064, 14.769216), ('climb_up', 14.769216, 22.153824), ('have_a_rest', 22.153824, 29.538432), ('climb_down', 29.538432, 36.92304), ('have_a_rest', 36.92304, 44.307648), ('climb_up', 44.307648, 51.692256), ('have_a_rest', 51.692256, 59.076864), ('climb_down', 59.076864, 66.461472), ('have_a_rest', 66.461472, 73.84608), ('climb_up', 73.84608, 81.230688), ('have_a_rest', 81.230688, 88.615296), ('climb_down', 88.615296, 95.999904), ('have_a_rest', 95.999904, 103.384512), ('climb_up', 103.384512, 110.76912), ('have_a_rest', 110.76912, 118.153728), ('climb_down', 118.153728, 125.538336), ('have_a_rest', 125.538336, 132.922944), ('climb_up', 132.922944, 140.307552), ('have_a_rest', 140.307552, 147.69216), ('climb_down', 147.69216, 155.07676800000002), ('have_a_rest', 155.07676800000002, 162.461376), ('climb_up', 162.461376, 169.845984), ('have_a_rest', 169.845984, 177.230592), ('climb_down', 177.230592, 184.61520000000002), ('have_a_rest', 184.61520000000002, 194.037551)]}

    # Initialize Tkinter frame and draggable intervals
    #app = DraggableIntervals(root, "doremi_acc_padded8", csv="./pkls/0_mix1.csv")
    #app = DraggableIntervals(root, "doremi_acc_padded8", csv="./pkls/0_doremi_acc_partial.csv")
    #app = DraggableIntervals(root, "doremi_acc_padded8", csv="./pkls/0_yankee.csv")
    app = DraggableIntervals(root, "doremi_acc_padded8", csv="./pkls/0_Yankee_doodle_Saloon_style_padded_100.csv")
    
    
    
    app.connect()

    root.mainloop()