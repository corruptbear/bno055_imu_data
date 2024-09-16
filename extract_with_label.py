import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# the formats is loaded, needs change for different data collected
from load_imu_data import formats
import os
import yaml

from intervals import all_intervals

def load_yaml(filepath):
    try:
        with open(filepath,'r') as stream:
            dictionary = yaml.safe_load(stream)
            return dictionary
    except:
        return dict()

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
        # handle repeat TODO: needs better numbering system and make it different from the bpm 
        interval_name_splits = interval_name.split("_")
        if interval_name_splits[-1] in ["1","2","3","4","5","6","7","8"]:
            interval_name = "_".join(interval_name_splits[:-1])
        # load the offset for the current interval
        offset = alignment_offsets[interval_name]
        current_interval = [(x, y, z) for (x, y, z) in all_intervals[interval_name] if "ready" not in x]
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
    new_fmt = formats+["%s"]
    # TODO:
    np.savetxt(export_path, save, delimiter=',', header=new_headers, comments='', fmt=new_fmt)


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
#extract_labeled_data("./pkls/0_Yankee_doodle_Saloon_style_padded_100.csv") # no quat data