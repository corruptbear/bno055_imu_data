import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.widgets import Button, RadioButtons
from matplotlib import colormaps

class DraggableIntervals:
    def __init__(self, intervals, headers, data, column_index):
       # Initialize figure and subplots with gridspec_kw
       self.fig, (self.ax, self.zoom_ax) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 2]})
       plt.subplots_adjust(bottom=0.3, hspace=0.4)
       print("DPI",self.fig.get_dpi())

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
       self.zoom_line, = self.zoom_ax.plot([], [], 'r-', lw=2)
       self.zoom_ax.set_title("Zoomed In View")
       self.zoom_ax.set_xlabel("Time")
       self.zoom_ax.set_ylabel("Value")
       
       # Add a button
       self.ax_button = plt.axes([0.7, 0.05, 0.1, 0.075])
       self.button = Button(self.ax_button, 'Print A')
       self.button.on_clicked(self.on_button_press)

       # Initialize radio buttons
       self.radio_ax = plt.axes([0.15, 0.0, 0.15, 0.2], facecolor='lightgoldenrodyellow')
       self.radio = RadioButtons(self.radio_ax, [headers[i] for i in range(5, len(headers))])
       self.radio.on_clicked(self.update_plot)

       self.load_color_patches(intervals)

    def load_color_patches(self, intervals):
        self.interval_patches = []
        # Plot interval patches
        colormap = colormaps["tab10"]
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
        self.ax.legend(handles=handles)

    def connect(self):
        self.cidpress = self.ax.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.ax.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = self.ax.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)

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

    def on_button_press(self, event):
        for patch in self.interval_patches:
            print(patch['label'], patch['patch'].get_xy()[0][0], patch['patch'].get_xy()[2][0])
            #patch['patch'].remove() #clear the drawn patch

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
                self.zoom_ax.axvspan(start, end, color=color, alpha=0.5, linewidth=0)

        self.zoom_ax.plot(zoom_data[:, 0], zoom_data[:, self.column_index], 'r-', lw=2)
        self.ax.figure.canvas.draw()

    def update_plot(self, label):
        self.column_index = self.headers.index(label)
        time_series = self.data[:, self.column_index]
        self.line_A.set_ydata(time_series)
        self.update_zoom(self.ax.get_xlim()[0])
        self.ax.relim()
        self.ax.autoscale_view()
        plt.draw()

    def disconnect(self):
        self.ax.figure.canvas.mpl_disconnect(self.cidpress)
        self.ax.figure.canvas.mpl_disconnect(self.cidrelease)
        self.ax.figure.canvas.mpl_disconnect(self.cidmotion)

intervals = [('get_ready', 0.0, 4.8), ('moving', 4.8, 7.2), ('stop', 7.2, 9.6), ('moving', 9.6, 12.0), ('stop', 12.0, 14.4), ('moving', 14.4, 16.8), ('stop', 16.8, 19.2), ('walk', 19.2, 22.8), ('turn_back', 22.8, 24.0), ('walk', 24.0, 27.6), ('turn_back', 27.6, 28.8), ('walk', 28.8, 32.4), ('turn_back', 32.4, 33.6), ('walk', 33.6, 37.2), ('turn_back', 37.2, 38.4), ('walk', 38.4, 42.0), ('turn_back', 42.0, 43.2), ('walk', 43.2, 46.8), ('turn_back', 46.8, 48.0), ('walk', 48.0, 51.6), ('turn_back', 51.6, 52.8), ('walk', 52.8, 56.4), ('turn_back', 56.4, 57.6), ('walk', 57.6, 60.872730000000004), ('turn_back', 60.872730000000004, 61.96364), ('walk', 61.96364, 65.23637000000001), ('turn_back', 65.23637000000001, 66.32728), ('walk', 66.32728, 69.60001), ('turn_back', 69.60001, 70.69092), ('walk', 70.69092, 73.96365), ('turn_back', 73.96365, 75.05456000000001), ('walk', 75.05456000000001, 78.32729), ('turn_back', 78.32729, 79.4182), ('walk', 79.4182, 82.69093000000001), ('turn_back', 82.69093000000001, 83.78184), ('walk', 83.78184, 87.05457), ('turn_back', 87.05457, 88.14548), ('walk', 88.14548, 91.41821), ('turn_back', 91.41821, 92.50912), ('walk', 92.50912, 95.50912), ('turn_back', 95.50912, 96.50912), ('walk', 96.50912, 99.50912), ('turn_back', 99.50912, 100.50912), ('walk', 100.50912, 103.50912), ('turn_back', 103.50912, 104.50912), ('walk', 104.50912, 107.50912), ('turn_back', 107.50912, 108.50912), ('walk', 108.50912, 111.50912), ('turn_back', 111.50912, 112.50912), ('walk', 112.50912, 115.50912), ('turn_back', 115.50912, 116.50912), ('walk', 116.50912, 119.50912), ('turn_back', 119.50912, 120.50912), ('stop', 120.50912, 126.537143)]
intervals = [(x, y, z) for (x, y, z) in intervals if "ready" not in x]

# Read the CSV file into a DataFrame
df = pd.read_csv("./pkls/0_k265_device59.csv")
# Convert the DataFrame to a 2D NumPy array
all_data = df.values
# Get the headers (column names)
headers = df.columns.tolist()

# timestamps start from 0
all_data[:, 0] = all_data[:, 0] - all_data[0][0]
timestamps = all_data[:, 0]

column_index = 5  # Default column index

#self, ax, interval_patches, headers, data, column_index
# Initialize draggable intervals
dr = DraggableIntervals(intervals, headers, all_data, column_index)
dr.connect()

plt.show()