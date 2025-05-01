import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse

# the formats is loaded, needs change for different data collected
#from load_imu_data import formats
import os

from intervals import all_intervals
from utils import *


def extract_labeled_data(raw_csv_path):
    base, ext = os.path.splitext(raw_csv_path)
    alignment_path = base + '_alignment.yaml'
    export_path = base + "_labeled.csv"

    # load the alignments
    alignment_offsets = load_yaml(alignment_path)

    # read the data
    df = pd.read_csv(raw_csv_path)
    data = df.values
    headers = df.columns.tolist()
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
    new_headers = ','.join(headers+["label"])
    #print(new_headers)
    #new_fmt = formats+["%s"]
    new_fmt = generate_formats(headers)
    new_fmt = new_fmt + ["%s"]
    # TODO:
    np.savetxt(export_path, save, delimiter=',', header=new_headers, comments='', fmt=new_fmt)

def generate_formats(headers):
    """
    if data_type == GYRO_DATA:
        headers = headers + ["gyro_x","gyro_y","gyro_z"]
        formats+=["%.4f"]*3
    if data_type == STAT_DATA:
        headers = headers + ["calib_mag","calib_accel","calib_gyro","calib_sys"]
        formats+=["%d"]*4
    if data_type == LACC_DATA:
        headers = headers + ["lacc_x","lacc_y","lacc_z"]
        formats+=["%.4f"]*3
    if data_type == ACC_DATA:
        headers = headers + ["acc_x","acc_y","acc_z"]
        formats+=["%.4f"]*3
    if data_type == QUAT_DATA:
        headers = headers + ["quat_w","quat_x","quat_y","quat_z"]
        formats+=["%.14f"]*4
    """
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
    parser = argparse.ArgumentParser(description="Extract labeled data from IMU CSV using alignments.")
    parser.add_argument("csv_path", help="Path to the input CSV file")
    args = parser.parse_args()
    extract_labeled_data(args.csv_path)
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

