import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons
from matplotlib.patches import Patch
from collections import defaultdict

def update_plot(col):
    main_plot.set_ydata(df_unlabeled[col].values)
    ax.relim()
    ax.autoscale_view()
    plt.draw()

# Load CSVs
csv_path_labeled = "./button_imu_logs_250507_230833/ble_imu_data_250507_230637_unit_converted_button_labeled.csv"
csv_path_unlabeled = "./button_imu_logs_250507_230833/ble_imu_data_250507_230637_unit_converted.csv"

df = pd.read_csv(csv_path_labeled)
df_unlabeled = pd.read_csv(csv_path_unlabeled)

if "button" not in csv_path_labeled:
    df_unlabeled.iloc[:, 0] -= df_unlabeled.iloc[0, 0]
    timestamps_unlabeled = df_unlabeled.iloc[:, 0].values
    timestamps = df.iloc[:, 0].values
else:
    df_unlabeled.iloc[:, 1] -= df_unlabeled.iloc[0, 1]
    timestamps_unlabeled = df_unlabeled.iloc[:, 1].values
    timestamps = df.iloc[:, 1].values



headers = df_unlabeled.columns.tolist()

# Label intervals

labels = df.iloc[:, -1]
intervals = []
start_time = timestamps[0]
current_label = labels[0]
for i in range(1, len(labels)):
    if labels[i] != current_label:
        intervals.append((current_label, start_time, timestamps[i - 1]))
        start_time = timestamps[i]
        current_label = labels[i]
intervals.append((current_label, start_time, timestamps[-1]))

print(intervals)

# Colormap for labels
col = "acc_x"
label_to_color = {}
colormap = plt.get_cmap("tab10")

# Create a single figure for both plot and radio buttons
fig = plt.figure(figsize=(10, 6))
ax = fig.add_axes([0.3, 0.2, 0.65, 0.7])  # main plot area
radio_ax = fig.add_axes([0.05, 0.2, 0.15, 0.6], facecolor='lightgoldenrodyellow')  # radio button area

# Plot data
main_plot, = ax.plot(timestamps_unlabeled, df_unlabeled[col].values)

for label, start, end in intervals:
    if label not in label_to_color:
        label_to_color[label] = colormap(len(label_to_color) % colormap.N)
    color = label_to_color[label]
    ax.axvspan(start, end, color=color, alpha=0.5, linewidth=0)

# Add legend
handles = []
added_labels = set()
for label, _, _ in intervals:
    if label not in added_labels:
        handles.append(Patch(color=label_to_color[label], alpha=0.5, label=label))
        added_labels.add(label)
patch_legend = ax.legend(handles=handles, prop={'size': plt.rcParams['legend.fontsize']})
patch_legend.get_frame().set_linewidth(0.2)

# Add radio buttons
radio = RadioButtons(radio_ax, [headers[i] for i in range(5, len(headers))])
radio.on_clicked(update_plot)

plt.show()