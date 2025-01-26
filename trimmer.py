import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import Tk, Button, Frame, filedialog, messagebox, StringVar, Radiobutton
import pandas as pd  # Required for loading CSV files
import matplotlib.patches as patches
from matplotlib.widgets import SpanSelector
import os
from utils import *

matplotlib.use('TkAgg')


class TimeSeriesSpanManager:
    def __init__(self, root, csv):
        self.root = root
        self.column_index = 5
        # Variables to track the selection
        self.start_select = None
        self.current_rect = None
        self.is_selecting = False

        # Create a figure and axis
        self.fig, self.ax = plt.subplots()
        #self.ax.plot(self.time, self.data)
        #self.ax.set_title("Select ranges to delete")
        #self.ax.set_xlabel("Time (seconds)")
        #self.ax.set_ylabel("Value")

        # Embed the figure in a Tkinter canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas_widget = self.canvas.get_tk_widget()

        # Add span selector
        #self.span_selector = matplotlib.widgets.SpanSelector(self.ax, self.on_select, 'horizontal', useblit=False)
        self.span_selector = SpanSelector(self.ax, self.on_select, 'horizontal', useblit=False)

        # Bind mouse events for double-click functionality
        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        #self.fig.canvas.mpl_connect("button_press_event", self.on_click)

        # Add buttons
        self.add_buttons()
        self.set_new_file(csv)

        self.canvas_widget.pack(fill='both', expand=True)


    def on_press(self, event):
        """
        Event handler for mouse button press. Initializes the selection.
        """
        #if event.dblclick:
        #    return
        if event.inaxes != self.ax:
            return
        if event.dblclick:
            self.is_selecting = False
            for i, (start_time, end_time) in enumerate(self.deleted_ranges):
                if start_time <= event.xdata <= end_time:
                    self.deleted_ranges.pop(i)
                    self.update_rects()
                    break
        else:
            self.start_select = event.xdata  # Record the x-coordinate where selection starts
            self.is_selecting = True  # Start selection process

    def on_motion(self, event):
        """
        Event handler for mouse motion. Updates the rectangle during selection.
        """
        if event.inaxes != self.ax or not self.is_selecting:
            return

        # Remove the existing rectangle if it exists
        if self.current_rect:
            self.current_rect.remove()

        # Draw a new rectangle from the start_select to the current x-coordinate
        start_x = min(self.start_select, event.xdata)
        end_x = max(self.start_select, event.xdata)
        self.current_rect = patches.Rectangle(
            (start_x, self.ax.get_ylim()[0]),
            end_x - start_x,
            self.ax.get_ylim()[1] - self.ax.get_ylim()[0],
            linewidth=1, edgecolor="red", facecolor="red", alpha=0.5,
        )
        self.ax.add_patch(self.current_rect)
        self.canvas.draw_idle()  # Redraw the canvas

    def on_release(self, event):
        """
        Event handler for mouse button release. Finalizes the selection.
        """
        if event.inaxes != self.ax:
           return
        self.is_selecting = False

        # Finalize the selection
        start_x = min(self.start_select, event.xdata)
        end_x = max(self.start_select, event.xdata)
        if abs(end_x - start_x) <=0.001:
            return

        # Add the new selection range if it doesn't overlap with existing ranges
        for start_time, end_time in self.deleted_ranges:
            if (start_x > start_time and start_x < end_time) or \
               (end_x > start_time and end_x < end_time):
                #self.is_selecting = False
                self.start_select = None
                if self.current_rect:
                    self.current_rect.remove()
                    self.current_rect = None
                self.canvas.draw_idle()
                return

        # Add the selection to the list of deleted ranges
        self.deleted_ranges.append((start_x, end_x))

        # Reset the current rectangle
        self.is_selecting = False
        self.start_select = None
        if self.current_rect:
            self.current_rect.remove()
            self.current_rect = None

        # Update the plot with the finalized rectangles
        self.update_rects()

    def on_select(self, min_val, max_val):
        """
        Callback for span selection. Adds a new span if it does not overlap with existing ones.
        """
        return
        new_start_time = min(min_val, max_val)
        new_end_time = max(min_val, max_val)

        # Check for overlap with existing ranges
        for start_time, end_time in self.deleted_ranges:
            if (new_start_time > start_time and new_start_time < end_time) or \
               (new_end_time > start_time and new_end_time < end_time):
                return

        # Add the new range
        self.deleted_ranges.append((new_start_time, new_end_time))
        self.update_rects()

    def update_rects(self):
        """
        Updates the plot by redrawing the spans.
        """
        # Remove existing rectangles
        for rect in self.rects:
            rect.remove()
        self.rects = []

        # Draw new rectangles for deleted ranges
        for start_time, end_time in self.deleted_ranges:
            rect = patches.Rectangle(
                (start_time, self.ax.get_ylim()[0]),
                end_time - start_time,
                self.ax.get_ylim()[1] - self.ax.get_ylim()[0],
                linewidth=1, edgecolor='red', facecolor='red', alpha=0.5
            )
            self.ax.add_patch(rect)
            self.rects.append(rect)

        self.canvas.draw()


    def read_data(self, csv):
        self.csv_path = csv
        base, ext = os.path.splitext(self.csv_path)
        #self.alignment_path = base + '_alignment.yaml'
        #print(f"alignment path is {self.alignment_path}")
       # self.export_path = base + "_labeled.csv"
        # Read the CSV file into a DataFrame
        #df = pd.read_csv("./pkls/0_mix1.csv")
        df = pd.read_csv(self.csv_path)

        all_data = df.values
        # set the timestamp start to 0
        all_data[:, 0] = all_data[:, 0] - all_data[0][0]
        self.original_data = all_data
        self.timestamps = all_data[:, 0]
        self.headers = df.columns.tolist()
        self.csv_formats = infer_formats_csv(csv)

    def set_new_file(self,csv):
        self.deleted_ranges = []  # Store deleted ranges as a list of tuples (start, end)
        self.rects = []  # Store matplotlib rectangle patches

        self.read_data(csv)

        # initialize plot
        self.ax.clear()
        self.full_plot, = self.ax.plot(self.original_data[:,0], self.original_data[:,self.column_index], label='Time Series A')
        self.duration = max(self.original_data[:,0]) - min(self.original_data[:,0])
        self.data = self.original_data.copy()

        self.radio_var.set(self.headers[self.column_index])

        for rb in self.radio_buttons:
            rb.destroy()
        self.radio_buttons = []
        for header in self.headers[1:]:
            rb = Radiobutton(self.radio_frame, text=header, variable=self.radio_var, value=header, command=self.update_plot)
            rb.pack(anchor='w')
            self.radio_buttons.append(rb)
        self.reset(None)

        #self.dropdown_var.set(self.interval_name)  # Default value
        #self.alignments_listbox.delete(0, tk.END)
        #self.load_alignments()
        #print(self.alignment_offsets)

    def add_buttons(self):
        """
        Add Tkinter buttons for delete and load data actions.
        """
        self.button_frame = Frame(self.root)
        self.button_frame.pack(side="bottom", fill="x")

        self.export_button = Button(self.button_frame, text="Export", command=self.export_trimmed)
        self.export_button.pack(side="right", padx=5, pady=5)

        self.delete_button = Button(self.button_frame, text="Delete Selected Ranges", command=self.delete_selected_ranges)
        self.delete_button.pack(side="right", padx=5, pady=5)

        self.load_button = Button(self.button_frame, text="Load Data", command=self.load_new_data)
        self.load_button.pack(side="right", padx=5, pady=5)

        self.reset_button = Button(self.button_frame, text='Reset', command=self.reset)
        self.reset_button.pack(side="right", padx=5, pady=5)

        # Add Tkinter RadioButtons
        self.radio_frame = Frame(self.root)
        self.radio_frame.pack(side='right', fill='y')

        self.radio_var = StringVar()
        self.radio_buttons = []

    def delete_selected_ranges(self):
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
        print(deleted_duration)

        # Update the time and data arrays
        #self.time = self.time[mask]
        print(len(self.data))
        self.data = self.data[mask,:]
        self.data[:,0] = new_time[mask]
        print(len(self.data))

        # Clear the deleted ranges and rectangles
        self.deleted_ranges.clear()
        for rect in self.rects:
            rect.remove()
        self.rects = []

        # Redraw the plot with updated data
        self.ax.clear()
        self.full_plot, = self.ax.plot(self.data[:,0], self.data[:,self.column_index], label='Time Series A')
        self.ax.set_title("Select Time Ranges to Delete")
        self.ax.set_xlabel("Time (seconds)")
        self.ax.set_ylabel("Value")

        # Update x-axis limits to fit the new data
        self.ax.set_xlim(0, self.duration)

        # Redraw the canvas
        self.canvas.draw()

    def update_plot(self):
        """
        Draw both the main plot and the zoom plot according to current column and patches
        """
        label = self.radio_var.get()
        self.column_index = self.headers.index(label)
        time_series = self.data[:, self.column_index]
        self.full_plot.set_ydata(time_series)
        #self.update_zoom()
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw()

    def load_new_data(self):
        """
        Load new time series data from a file when the button is clicked.
        """
        # Open file dialog to select a CSV file
        file_path = filedialog.askopenfilename(title="Select Data File", filetypes=[("CSV files", "*.csv")])

        if file_path:
            print("load from:",file_path)
            self.set_new_file(file_path)

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
        self.ax.clear()
        self.full_plot, = self.ax.plot(self.original_data[:,0], self.original_data[:,self.column_index], label='Time Series A')
        self.ax.set_title("Select Time Ranges to Delete")
        self.ax.set_xlabel("Time (seconds)")
        self.ax.set_ylabel("Value")

        # Update the x-axis to match the original data
        self.ax.set_xlim(min(self.original_data[:,0]), max(self.original_data[:,0]))

        self.is_selecting = False
        # Redraw the canvas
        self.fig.canvas.draw()

    def export_trimmed(self, event=None):
        """
        export the trimmed file
        """
        base, ext = os.path.splitext(self.csv_path)
        self.export_path = base + '_trimmed.csv'
        np.savetxt(self.export_path, self.data, delimiter=',', header=','.join(self.headers), comments='',fmt=self.csv_formats)
        print(f"exported to {self.export_path}")


# Main application
if __name__ == "__main__":
    # Generate synthetic time series data (10000 points, sampling every 0.01 seconds)
    time = np.arange(0, 100, 0.01)
    data = np.sin(time)

    # Create Tkinter root window
    root = Tk()
    root.title("Time Series Manager")

    # Create and display the TimeSeriesSpanManager
    manager = TimeSeriesSpanManager(root, csv="./pkls/0_mix1.csv")

    # Start the Tkinter main loop
    root.mainloop()