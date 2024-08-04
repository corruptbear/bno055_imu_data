import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import Tk, Frame, filedialog, Radiobutton, StringVar, Button, OptionMenu
import tkinter as tk
from load_imu_data import formats

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
    def __init__(self, master, interval_name, headers, data, column_index):
        self.master = master
        self.fig, (self.ax, self.zoom_ax) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 2]}, figsize=(8, 6), dpi=300)
        plt.subplots_adjust(bottom=0.1, hspace=0.4)
        print("figure DPI",self.fig.get_dpi())

        # Set font sizes and DPI
        self.set_font_sizes(dpi=300)  # Default DPI

        self.headers = headers
        self.data = data
        self.column_index = column_index
        self.press = None
        self.offsets = []
        self.interval_patches = []
        self.label_to_color = {}

        # Plot initial data
        self.line_A, = self.ax.plot(data[:, 0], data[:, column_index], label='Time Series A')

        # Initialize zoom plot
        self.zoom_line, = self.zoom_ax.plot([], [], 'r-')
        self.zoom_ax.set_title("Zoomed In View")
        self.zoom_ax.set_xlabel("Time")
        self.zoom_ax.set_ylabel("Value")

        # Create Tkinter Frame for buttons
        self.button_frame = Frame(self.master)
        self.button_frame.pack(side='bottom', fill='x')

        # Add Tkinter Buttons
        self.print_button = Button(self.button_frame, text='Print A', command=self.on_button_press)
        self.print_button.pack(side='left', padx=5)

        self.load_button = Button(self.button_frame, text='Load File', command=self.load_file)
        self.load_button.pack(side='left', padx=5)

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
        self.load_color_patches(interval_name)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

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
        selected_value = self.dropdown_var.get()
        print(f"Selected value from dropdown: {selected_value}")
        self.load_color_patches(selected_value)

    def load_color_patches(self, interval_name):
        intervals = [(x, y, z) for (x, y, z) in all_intervals[interval_name] if "ready" not in x]
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
            patch = self.ax.axvspan(start, end, color=color, alpha=0.5, label=label, linewidth=0)
            self.interval_patches.append({'patch': patch, 'label': label, 'color': color})

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

    def on_button_press(self):
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

        np.savetxt(f"./test.csv", save, delimiter=',', header=new_headers, comments='', fmt=new_fmt)
 
    def load_file(self):
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
        self.line_A.set_ydata(time_series)
        self.update_zoom(self.ax.get_xlim()[0])
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw()

    def update_zoom(self, x_center):
        window_size = 100
        x0 = max(self.interval_patches[0]['patch'].get_xy()[0][0] - 10, timestamps[0])
        x1 = min(self.interval_patches[-1]['patch'].get_xy()[2][0] + 10, timestamps[-1])
        self.zoom_ax.set_xlim(x0, x1)
        zoom_data = self.data[(timestamps >= x0) & (timestamps <= x1)]
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
                     "Yankee_doodle_Saloon_style_padded8_120":[('get_ready', 0.0, 4.0), ('moving', 4.0, 6.0), ('stop', 6.0, 8.0), ('moving', 8.0, 10.0), ('stop', 10.0, 12.0), ('moving', 12.0, 14.0), ('stop', 14.0, 16.0), ('get_ready_to_walk_forward', 16.0, 20.0), ('walking_forward', 20.0, 36.0), ('get_ready_to_run_forward', 36.0, 40.0), ('running_forward', 40.0, 56.0), ('get_ready_to_walk_forward', 56.0, 60.0), ('walking_forward', 60.0, 76.0), ('get_ready_to_stand_still', 76.0, 80.0), ('standing_still', 80.0, 98.142041)]}

    # Read the CSV file into a DataFrame
    df = pd.read_csv("./pkls/0_mix1.csv")
    all_data = df.values
    headers = df.columns.tolist()

    # set the start to 0
    all_data[:, 0] = all_data[:, 0] - all_data[0][0]
    timestamps = all_data[:, 0]
    column_index = 5  # Default column index

    # Initialize Tkinter frame and draggable intervals
    app = DraggableIntervals(root, "doremi_acc_padded8", headers, all_data, column_index)
    app.connect()

    root.mainloop()