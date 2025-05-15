import glob
import os
from extract_with_label import extract_labeled_data_from_button_interface
import re
from utils import *

def is_already_unzipped(zip_path):
    # Assume the unzipped directory name is derived from the zip file name
    unzip_dir = os.path.splitext(zip_path)[0]  # remove .zip
    return os.path.isdir(unzip_dir)

def process_all_button_logs_recursive(root_dir):
    root_dir = os.path.expanduser(root_dir)
    # Recursively find all button_*.zip files
    zip_files = glob.glob(os.path.join(root_dir, "**", "button_*.zip"), recursive=True)
    print(zip_files)

    for zip_path in zip_files:
        if is_already_unzipped(zip_path):
            print(f"Skipping already unzipped: {zip_path}")
            continue
        print(f"Processing {zip_path}...")
        extract_labeled_data_from_button_interface(zip_path)

def find_labeled_csvs_with_top_level_p_index(root_dir):
    root_dir = os.path.abspath(os.path.expanduser(root_dir))
    labeled_files = glob.glob(os.path.join(root_dir, '**', '*labeled.csv'), recursive=True)

    result = []
    for path in labeled_files:
        abs_path = os.path.abspath(path)
        rel_path = os.path.relpath(abs_path, root_dir)  # get path relative to root_dir
        parts = rel_path.split(os.sep)

        if len(parts) >= 2:
            top_level_dir = parts[0]  # e.g., "p6"
            match = re.match(r'p(\d+)', top_level_dir)
            if match:
                p_number = int(match.group(1))
                result.append((abs_path, p_number))
            else:
                print(f"Skipping: {abs_path} (no match for p<number> in {top_level_dir})")
        else:
            print(f"Skipping: {abs_path} (not deep enough)")

    return result


process_all_button_logs_recursive("~/Downloads/exp_data/")

files_with_p = find_labeled_csvs_with_top_level_p_index("~/Downloads/exp_data/")
for path, p_num in files_with_p:
    print(f"labeled_csv_path: {path}, p number: {p_num}")
    append_person_information(path, p_num)