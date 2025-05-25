import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import json
import glob

# the formats is loaded, needs change for different data collected
#from load_imu_data import formats
import os

from intervals import all_intervals
from utils import *
from load_imu_data import load_tag_imu_data_from_csv

def test_offset_extract_labeled_data(raw_csv_path, interval_name, dst_dir):
    #metronome_padded_bpm60
    base, ext = os.path.splitext(os.path.basename(raw_csv_path))
    dst_dir = os.path.expanduser(dst_dir)

    # Read raw data
    with open(raw_csv_path, 'r') as f:
        reader = csv.reader(f)
        raw_header = next(reader)
        raw_data_rows = list(reader)

    if len(raw_data_rows) == 0:
        raise ValueError("CSV has no data rows")

    first_data_len = len([x for x in raw_data_rows[0] if x.strip() != ""])
    if len(raw_header) > first_data_len:
        headers = [h for h in raw_header if h != "device_id"]
        data_rows = [row[:-1] for row in raw_data_rows]
    else:
        headers = raw_header
        data_rows = raw_data_rows

    df = pd.DataFrame(data_rows, columns=headers).astype(float)
    data = df.values
    data[:, 0] = data[:, 0] - data[0][0]  # Normalize timestamps to start at 0

    formats = generate_formats(headers) + ["%s"]
    headers.append("label")
    header_line = ','.join(headers)

    for i in range(-100, 100, 10): 
        fractional_offset = i / 100.0
        save = np.empty((0, data.shape[1]), dtype=object)
        labels = []

        offset = fractional_offset
        current_interval = [
            (x, y, z) for (x, y, z) in all_intervals[interval_name] if "ready" not in x
        ]
        for label, start, end in current_interval:
            start_ts, end_ts = start + offset, end + offset
            selected = [row for row in data if start_ts <= row[0] <= end_ts]
            labels.extend([label] * len(selected))
            if selected:
                save = np.vstack([save, selected])


        new_column = np.array(labels).reshape(-1, 1)
        save = np.hstack([save, new_column])

        sign = 'm' if i < 0 else 'p'
        suffix = f"{sign}{abs(i):02d}"
        # Create subfolder with subsubfolder called raw_data
        subfolder_path = os.path.join(dst_dir, suffix, "raw_data")
        os.makedirs(subfolder_path, exist_ok=True)

        output_filename = f"{base}_labeled_offset_{suffix}.csv"
        output_path = os.path.join(subfolder_path, output_filename)

        np.savetxt(output_path, save, delimiter=',', header=header_line, comments='', fmt=formats)
        print(f"Saved: {output_path}")
    


def extract_labeled_data(raw_csv_path):
    base, ext = os.path.splitext(raw_csv_path)
    alignment_path = base + '_alignment.yaml'
    export_path = base + "_labeled.csv"

    # load the alignments
    alignment_offsets = load_yaml(alignment_path)

    # Step 1: Read raw data using csv.reader
    with open(raw_csv_path, 'r') as f:
        reader = csv.reader(f)
        raw_header = next(reader)
        raw_data_rows = list(reader)
    print(raw_data_rows[0])

    # Step 2: Detect and remove bad field from header
    if len(raw_data_rows) == 0:
        raise ValueError("CSV has no data rows")

    first_data_len = len([x for x in raw_data_rows[0] if x.strip() != ""])
    if len(raw_header) > first_data_len:
        print(f"Detected extra column in header: {raw_header}")

        headers = [h for h in raw_header if h!="device_id"]
        data_rows = [row[:-1] for row in raw_data_rows]
        print(data_rows[0])
    else:
        headers = raw_header
        data_rows = raw_data_rows

    # Step 3: Convert to float DataFrame
    df = pd.DataFrame(data_rows, columns=headers).astype(float)

    data = df.values
    # set the start of timestamps to 0
    data[:, 0] = data[:, 0] - data[0][0]
    timestamps = data[:, 0]

    # create data to be saved
    save  = np.empty((0, data.shape[1]),dtype=object)
    labels = []

    # the column for labels (add as the last column)
    # TODO: rewrite with the alignments
    for interval_name in alignment_offsets:
        offset = alignment_offsets[interval_name]
        # load the offset for the current interval name (base name; ignoring numbering)
        current_interval = [(x, y, z) for (x, y, z) in all_intervals[get_base_name(interval_name)] if "ready" not in x]
        for label, start, end in current_interval:
            start_timestamp, end_timestamp = start + offset, end + offset
            selected = [data[k,:] for k in range(len(data)) if data[k][0]>=start_timestamp and  data[k][0]<=end_timestamp]
            labels = labels + [label]*len(selected)
            # only concatenate when the selected is not []; otherwise dimension will not match
            if len(selected)>0:
                save = np.vstack([save, selected])

    new_column = np.array(labels).reshape(-1,1)

    # Add the new column
    save = np.hstack([save, new_column])
    headers = headers+["label"]
    new_headers = ','.join(headers)
    #new_fmt = formats+["%s"]
    new_fmt = generate_formats(headers)
    new_fmt = new_fmt + ["%s"]
    print(headers, len(headers),len(new_fmt))
    # TODO:
    np.savetxt(export_path, save, delimiter=',', header=new_headers, comments='', fmt=new_fmt)

def convert_video_annotation(annotation_file_path):
    with open(annotation_file_path, "r") as f:
        json_str = f.read()

    # Load JSON into Python dict
    data = json.loads(json_str)

    # Extract time-label pairs
    video_key = next(iter(data['data']))
    events = sorted(data['data'][video_key], key=lambda x: x['time'])  # Sort by time ascending

    # Build (label, start, end) tuples
    segments = []
    for i in range(len(events) - 1):
        label = events[i]['label']
        start = events[i]['time']
        end = events[i + 1]['time']
        segments.append((label, start, end))

    # Handle last segment: set end to None or max time if known
    segments.append((events[-1]['label'], events[-1]['time'], events[-1]['time']+0.1))
    return segments

#['standing_still', 'walking_forward', 'running_forward', 'climb_up', 'climb_down']
"""
Climbing Down
Climbing Up
Nothing
Running
Standing
Walking
"""
def convert_button_annotation(log_file_path):
    label_mapping = {"Climbing Up":"climb_up","Climbing Down":"climb_down","Running":"running_forward","Standing":"standing_still","Walking":"walking_forward","Nothing":"nothing","walk":"walking_forward"}

    df = pd.read_csv(log_file_path)
    labels = df['label'].values
    timestamps = df['timestamp'].values

    segments = []
    for i in range(len(df) - 1):
        start = timestamps[i]
        end = timestamps[i + 1]
        label = label_mapping[labels[i]]
        segments.append((label, start, end))
    segments.append((labels[-1], timestamps[-1], None))
    return segments


def extract_labeled_data_from_video(sensor_data_path=None, annotation_path=None, sync_value=None):
    #sync_value???
    base, ext = os.path.splitext(sensor_data_path)
    alignment_path = base + '_alignment.yaml'
    export_path = base + "_video_labeled.csv"

    #for_marking_the_start
    video_annotation = convert_video_annotation(annotation_path)

    if sync_value is None:
        alignment = load_yaml(alignment_path)
        if "for_marking_the_start" in alignment:
            ts_first_event_imu = alignment["for_marking_the_start"]
        else:
            ts_first_event_imu = min(list(alignment.values()))
        ts_first_event_video = video_annotation[0][1]
        sync_value = ts_first_event_imu - ts_first_event_video

    #video annotation ts + sync_value     is the ts on imu timeline
    #video_annotation_in_imu_timeline = [(label,start_ts+sync_value,end_ts+sync_value) for (label, start_ts, end_ts) in video_annotation]

    # read the data
    # Step 1: Read raw data using csv.reader
    with open(sensor_data_path, 'r') as f:
        reader = csv.reader(f)
        raw_header = next(reader)
        raw_data_rows = list(reader)
    print(raw_data_rows[0])

    # Step 2: Detect and remove bad field from header
    if len(raw_data_rows) == 0:
        raise ValueError("CSV has no data rows")

    first_data_len = len([x for x in raw_data_rows[0] if x.strip() != ""])
    if len(raw_header) > first_data_len:
        print(f"Detected extra column in header: {raw_header}")

        headers = [h for h in raw_header if h!="device_id"]
        data_rows = [row[:-1] for row in raw_data_rows]
        print(data_rows[0])
    else:
        headers = raw_header
        data_rows = raw_data_rows

    # Step 3: Convert to float DataFrame
    df = pd.DataFrame(data_rows, columns=headers).astype(float)

    data = df.values
    # set the start of timestamps to 0
    data[:, 0] = data[:, 0] - data[0][0]
    timestamps = data[:, 0]

    # create data to be saved
    save  = np.empty((0, data.shape[1]),dtype=object)
    labels = []


    # load the offset for the current interval name (base name; ignoring numbering)
    current_interval = [(x, y, z) for (x, y, z) in video_annotation if "ready" not in x]
    for label, start, end in current_interval:
        start_timestamp, end_timestamp = start + sync_value, end + sync_value
        selected = [data[k,:] for k in range(len(data)) if data[k][0]>=start_timestamp and  data[k][0]<=end_timestamp]
        labels = labels + [label]*len(selected)
        # only concatenate when the selected is not []; otherwise dimension will not match
        if len(selected)>0:
            save = np.vstack([save, selected])

    new_column = np.array(labels).reshape(-1,1)

    # Add the new column
    save = np.hstack([save, new_column])
    new_headers = ','.join(headers+["label"])
    #print(new_headers)
    #new_fmt = formats+["%s"]
    new_fmt = generate_formats(headers)
    new_fmt = new_fmt + ["%s"]
    # TODO:
    np.savetxt(export_path, save, delimiter=',', header=new_headers, comments='', fmt=new_fmt)


def extract_labeled_data_from_button_interface(dir_path = None):
    """
    dir_path: the path of the folder containing both the app button press log and the imu data; can be just a .zip file and it will unzip automatically
    """
    if dir_path.endswith('.zip'):
        dir_path = unzip_to_dir(dir_path)

    # Search for files
    button_log_files = glob.glob(os.path.join(dir_path, "button_press_log*.csv"))
    imu_data_files = glob.glob(os.path.join(dir_path, "ble_imu_data*.csv"))
    # prevent errors on repeated run
    imu_data_files = [f for f in imu_data_files if "unit_converted" not in f]

    button_log_file = button_log_files[0] if button_log_files else None
    imu_data_file = imu_data_files[0] if imu_data_files else None

    print("Button log file:", button_log_file)

    # convert the unit here
    load_tag_imu_data_from_csv(imu_data_file)
    name, ext = os.path.splitext(imu_data_file)
    imu_data_file = f"{name}_unit_converted{ext}"


    print("IMU data file:", imu_data_file)

    button_annotation = convert_button_annotation(button_log_file)
    #print(button_annotation)


    base, ext = os.path.splitext(imu_data_file)
    export_path = base + "_button_labeled.csv"

    #video annotation ts + sync_value     is the ts on imu timeline
    #video_annotation_in_imu_timeline = [(label,start_ts+sync_value,end_ts+sync_value) for (label, start_ts, end_ts) in video_annotation]

    # read the data
    # read the data
    # Step 1: Read raw data using csv.reader
    with open(imu_data_file, 'r') as f:
        reader = csv.reader(f)
        raw_header = next(reader)
        raw_data_rows = list(reader)
    print(raw_data_rows[0])

    # Step 2: Detect and remove bad field from header
    if len(raw_data_rows) == 0:
        raise ValueError("CSV has no data rows")

    first_data_len = len([x for x in raw_data_rows[0] if x.strip() != ""])
    if len(raw_header) > first_data_len:
        print(f"Detected extra column in header: {raw_header}")

        headers = [h for h in raw_header if h!="device_id"]
        data_rows = [row[:-1] for row in raw_data_rows]
        print(data_rows[0])
    else:
        headers = raw_header
        data_rows = raw_data_rows

    # Step 3: Convert to float DataFrame
    df = pd.DataFrame(data_rows, columns=headers).astype(float)
    data = df.values


    # create data to be saved
    save  = np.empty((0, data.shape[1]),dtype=object)
    labels = []


    # load the offset for the current interval name (base name; ignoring numbering)
    current_interval = [(x, y, z) for (x, y, z) in button_annotation if "ready" not in x]
    for label, start, end in current_interval:
        if end is not None:
            start_timestamp, end_timestamp = start/1e9, end/1e9
        else:
            #1e10 is just a large value
            start_timestamp, end_timestamp = start/1e9, 1e10
        #print(start_timestamp, end_timestamp)
        selected = [data[k,:] for k in range(len(data)) if data[k][1]>=start_timestamp and  data[k][1]<=end_timestamp]
        labels = labels + [label]*len(selected)
        # only concatenate when the selected is not []; otherwise dimension will not match
        if len(selected)>0:
            save = np.vstack([save, selected])

    new_column = np.array(labels).reshape(-1,1)

    # Add the new column
    save = np.hstack([save, new_column])
    # set the start of timestamps to 0
    save[:, 1] = save[:, 1] - data[0][1]
    save[:, 0] = save[:, 0] - data[0][0]

    new_headers = ','.join(headers+["label"])
    #print(new_headers)
    #new_fmt = formats+["%s"]
    new_fmt = generate_formats(headers)
    new_fmt = new_fmt + ["%s"]

    np.savetxt(export_path, save, delimiter=',', header=new_headers, comments='', fmt=new_fmt)


def generate_formats(headers):
    formats = []
    for header in headers:
        #print(header)
        if header == "timestamp":
            formats+=["%.2f"]
        if header == "android_nano_timestamp":
            formats+=["%.9f"]
        if header == "temperature":
            formats+=["%d"]
        if header in ["calib_mag","calib_accel","calib_gyro","calib_sys"]:
            formats+=["%d"]
        if header in ["acc_x","acc_y","acc_z","lacc_x","lacc_y","lacc_z","gyro_x","gyro_y","gyro_z","mag_x","mag_y","mag_z","gravity_x","gravity_y","gravity_z"]:
            formats+=["%.4f"]
        if header in ["quat_w","quat_x","quat_y","quat_z"]:
            formats+=["%.14f"]
    return formats

if __name__ == "__main__":
    """
    parser = argparse.ArgumentParser(description="Extract labeled data from IMU CSV using alignments.")
    parser.add_argument("csv_path", help="Path to the input CSV file")
    args = parser.parse_args()
    extract_labeled_data(args.csv_path)
    """

    # example
    #extract_labeled_data_from_video(sensor_data_path="./ble_imu_data_250429_200238_unit_converted.csv", annotation_path="./20250430_030238000_iOS.aucvl")
    #test_offset_extract_labeled_data("ble_imu_data_250516_115403_unit_converted.csv", "metronome_padded_bpm60", "~/Dev/lab_projects/tottag_ranging/ml-model-sandbox/python/datasets_sliding")
    #extract_labeled_data_from_button_interface("./button_imu_logs_250507_230833.zip")

    test_offset_extract_labeled_data("ble_imu_data_250520_161721_unit_converted.csv", "metronome_padded_bpm60", "~/Dev/lab_projects/tottag_ranging/ml-model-sandbox/python/datasets_sliding_gabe")


    #extract_labeled_data("./pkls/0_mix1.csv")
    #extract_labeled_data("./pkls/0_doremi_acc_partial.csv")
    #extract_labeled_data("./pkls/0_k265_device36.csv")
    #extract_labeled_data("./pkls/0_k265_device36_2.csv")
    #extract_labeled_data("./pkls/0_k265_device36_3.csv")
    #extract_labeled_data("./pkls/0_k265_device59.csv")
    #extract_labeled_data("./pkls/0_k265_device59_2.csv")
    #extract_labeled_data("./pkls/0_k265_device59_3.csv")
    #extract_labeled_data("./pkls/0_yankee_device36.csv")
    #extract_labeled_data("./pkls/0_doremi_acc_yankee_device59.csv")
    #extract_labeled_data("./pkls/0_Yankee_doodle_Saloon_style_padded_100.csv") # no quat data; modify load_imu_data
    #extract_labeled_data("/Users/lws/Downloads/exp_data/trial_data/doremi_padded_simple_130_audio_imu_logs_250429_200423/ble_imu_data_250429_200238_unit_converted.csv")

