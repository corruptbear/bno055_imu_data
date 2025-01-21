import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector, Button, RadioButtons
import matplotlib.patches as patches
from tkinter import filedialog
import pandas as pd
import os

matplotlib.use('TkAgg')

class TimeSeriesSpanManager:
    def __init__(self, csv):

        self.column_index=5
        self.fig, self.ax = plt.subplots()
        self.full_plot = None

        self.deleted_ranges = []  # Store deleted ranges as a list of tuples (start, end)
        self.rects = []           # Store matplotlib rectangle patches

        # Set up the plot
        self.fig.subplots_adjust(bottom=0.2)
        self.fig.subplots_adjust(right=0.8)

        # Add a span selector
        self.span_selector = SpanSelector(self.ax, self.on_select, 'horizontal', useblit=False)

        # Add a button to delete selected ranges
        # = self.fig.add_axes([0.8, 0.02, 0.1, 0.05])  # Position for the button
        self.reset_button_ax = self.fig.add_axes([0.39, 0.05, 0.1, 0.075])
        self.delete_button_ax = self.fig.add_axes([0.5, 0.05, 0.1, 0.075])
        self.load_button_ax = self.fig.add_axes([0.61, 0.05, 0.1, 0.075])
        self.export_button_ax = self.fig.add_axes([0.72, 0.05, 0.1, 0.075])

        self.delete_button = Button(self.delete_button_ax, 'Delete')
        self.delete_button.on_clicked(self.delete_selected_ranges)
        self.load_button = Button(self.load_button_ax, 'Load')
        self.load_button.on_clicked(self.load_new_data)
        self.reset_button = Button(self.reset_button_ax, 'Reset')
        self.reset_button.on_clicked(self.reset)
        self.export_button = Button(self.export_button_ax, 'Export')
        self.export_button.on_clicked(self.export)

        # Initialize radio buttons
        # the button labels are initialized in the set_new_file
        self.radio_ax = plt.axes([0.82, 0.3, 0.17, 0.6], facecolor="lightgoldenrodyellow")

        self.set_new_file(csv)

        # Connect events
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

    def load_new_data(self, event=None):
        """
        Load new time series data from a file when the button is clicked.
        """
        # Open file dialog to select a CSV file
        file_path = filedialog.askopenfilename(filetypes=[('csv Files', '*.csv')], initialdir="./pkls")
        if file_path:
            print("load from:",file_path)
            self.set_new_file(file_path)

    def read_data(self, csv):
        self.csv_path = csv

        df = pd.read_csv(self.csv_path)

        all_data = df.values
        # set the timestamp start to 0
        all_data[:, 0] = all_data[:, 0] - all_data[0][0]
        self.original_data = all_data
        self.timestamps = all_data[:, 0]
        self.headers = df.columns.tolist()

    def set_new_file(self,csv):

        self.read_data(csv)
        self.radio_ax.clear()
        self.radio = RadioButtons(self.radio_ax, self.headers[1:])
        self.radio.on_clicked(self.update_radio_selection)

        self.deleted_ranges = []  # Store deleted ranges as a list of tuples (start, end)
        self.rects = []  # Store matplotlib rectangle patches
        self.reset(None)


    def on_select(self, min_val, max_val):
        """
        Callback for span selection. Adds a new span if it does not overlap with existing ones.
        """
        new_start_time = min(min_val, max_val)
        new_end_time = max(min_val, max_val)

        # Check for overlap with existing ranges
        for start_time, end_time in self.deleted_ranges:
            if (new_start_time > start_time and new_start_time < end_time) or \
               (new_end_time > start_time and new_end_time < end_time):
                return

        # Add the new range
        self.deleted_ranges.append((new_start_time, new_end_time))
        self.update_plot()

    def update_plot(self):
        """
        Updates the plot by redrawing the spans.
        """
        # Remove existing rectangles
        for rect in self.rects:
            rect.remove()
        self.rects = []

        # Draw new rectangles for deleted ranges
        for start_time, end_time in self.deleted_ranges:
            rect = self.ax.add_patch(patches.Rectangle(
                (start_time, self.ax.get_ylim()[0]),
                end_time - start_time,
                self.ax.get_ylim()[1] - self.ax.get_ylim()[0],
                linewidth=0.1, edgecolor='red', facecolor='red', alpha=0.5
            ))
            #self.ax.add_patch(rect)
            self.rects.append(rect)
        self.ax.figure.canvas.draw()

    def on_click(self, event):
        """
        Callback for mouse click. Deletes a span if double-clicked.
        """
        if event.dblclick:
            for i, (start_time, end_time) in enumerate(self.deleted_ranges):
                if start_time <= event.xdata <= end_time:
                    self.deleted_ranges.pop(i)
                    self.update_plot()
                    break

    def delete_selected_ranges(self, event):
        """
        Deletes the selected ranges from the data and updates the plot.
        """
        # Create a mask to exclude points within the deleted ranges
        mask = np.ones_like(self.data[:,0], dtype=bool)  # Start with all data points included

        deleted_duration = 0
        new_time = self.data[:,0].copy()
        # Apply exclusions for each deleted range
        for start_time, end_time in self.deleted_ranges:
            mask &= ~((self.data[:,0] >= start_time) & (self.data[:,0] <= end_time))
            #t[t >= 5] -= 2
            adj_end_time = end_time-deleted_duration
            new_time[new_time>=adj_end_time]-=end_time-start_time
            deleted_duration+=end_time-start_time
        self.duration = self.duration-deleted_duration
        print("deleted duration",deleted_duration)

        # Update the time and data arrays
        #self.time = self.time[mask]
        print("before deletion len",len(self.data))
        self.data = self.data[mask,:]
        self.data[:,0] = new_time[mask]
        print("after deletion len",len(self.data))

        # Clear the deleted ranges and rectangles
        self.deleted_ranges = []
        for rect in self.rects:
            rect.remove()
        self.rects = []
        # Redraw the plot with updated data

        #do not use clear! otherwise the rectangle selection will no longer be visible!?
        self.full_plot.set_xdata(self.data[:, 0])
        self.full_plot.set_ydata(self.data[:, self.column_index])
        self.ax.set_title("Select Time Ranges to Delete")
        self.ax.set_xlabel("Time (seconds)")
        self.ax.set_ylabel("Value")

        # Update x-axis limits to fit the new data
        self.ax.set_xlim(0, self.duration)

        # Redraw the canvas
        self.ax.figure.canvas.draw()

    def reset(self, event=None):
        """
        Resets the data and clears all deleted ranges.
        """
        # Restore original data
        self.data = self.original_data.copy()
        self.duration = max(self.original_data[:,0]) - min(self.original_data[:,0])

        # Clear deleted ranges and rectangles
        self.deleted_ranges.clear()
        for rect in self.rects:
            rect.remove()
        self.rects = []

        # Redraw the plot with the original data
        if self.full_plot:
            self.full_plot.set_xdata(self.original_data[:, 0])
            self.full_plot.set_ydata(self.original_data[:, self.column_index])
        else:
            self.full_plot, = self.ax.plot(self.original_data[:,0], self.original_data[:,self.column_index])
        self.ax.set_title("Select Time Ranges to Delete")
        self.ax.set_xlabel("Time (seconds)")
        self.ax.set_ylabel("Value")
        # Update the x-axis to match the original data
        self.ax.set_xlim(min(self.original_data[:,0]), max(self.original_data[:,0]))

        # Redraw the canvas
        self.ax.figure.canvas.draw()

    def show(self):
        """
        Displays the plot.
        """
        plt.show()

    def update_radio_selection(self,label):
        self.column_index = self.headers.index(label)
        time_series = self.data[:, self.column_index]
        self.full_plot.set_ydata(time_series)
        #self.update_zoom(self.ax.get_xlim()[0])
        self.ax.relim()
        self.ax.autoscale_view()
        self.ax.figure.canvas.draw()

    def export(self, event=None):
        base, ext = os.path.splitext(self.csv_path)
        self.export_path = base + '_trimmed.csv'
        np.savetxt(self.export_path, self.data, delimiter=',', header=','.join(self.headers), comments='')

# Usage example
if __name__ == "__main__":
    # Create and display the TimeSeriesSpanManager
    manager = TimeSeriesSpanManager(csv="./pkls/0_mix1.csv")
    manager.show()