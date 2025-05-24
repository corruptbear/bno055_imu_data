import glob
import os
from extract_with_label import *
import re
from utils import *
import shutil

def file_hash(path, chunk_size=8192):
    """Compute SHA256 hash of a file."""
    hasher = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()

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
    """
    the participants data is organized as rootdir/p1, rootdir/p2, ...
    """
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


def extract_audio_labels(root_dir):
    """
    the participants data is organized as rootdir/p1, rootdir/p2, ...
    """
    root_dir = os.path.abspath(os.path.expanduser(root_dir))
    labeled_files = glob.glob(os.path.join(root_dir, '**','*audio_imu_logs*', '*unit_converted.csv'), recursive=True)

    result = []
    for path in labeled_files:
        print("extract_audio_labels\n:",path)
        extract_labeled_data(path)

    return result


#/Users/lws/Dev/lab_projects/tottag_ranging/ml-model-sandbox/python/datasets_audio
def copy_audio_labeled_data(root_dir):
    root_dir = os.path.abspath(os.path.expanduser(root_dir))
    labeled_files = glob.glob(os.path.join(root_dir, '**', '*converted_labeled.csv'), recursive=True)

    dest_dir = "/Users/lws/Dev/lab_projects/tottag_ranging/ml-model-sandbox/python/datasets_audio/raw_data"
    os.makedirs(dest_dir, exist_ok=True)

    for src_path in labeled_files:
        filename = os.path.basename(src_path)
        dest_path = os.path.join(dest_dir, filename)

        if os.path.exists(dest_path):
            print(f"Skipping existing file: {dest_path}")
            continue

        print(f"copy_audio_labeled_data: Copying {src_path} → {dest_path}")
        shutil.copy2(src_path, dest_path)

def copy_button_labeled_data(root_dir):
    root_dir = os.path.abspath(os.path.expanduser(root_dir))
    labeled_files = glob.glob(os.path.join(root_dir, '**', '*button_labeled.csv'), recursive=True)

    dest_dir = "/Users/lws/Dev/lab_projects/tottag_ranging/ml-model-sandbox/python/datasets_button/raw_data"
    os.makedirs(dest_dir, exist_ok=True)

    for src_path in labeled_files:
        filename = os.path.basename(src_path)
        dest_path = os.path.join(dest_dir, filename)

        if os.path.exists(dest_path):
            if file_hash(src_path) == file_hash(dest_path):
                print(f"Skipping identical file: {dest_path}")
                continue
            else:
                print(f"Overwriting different file: {dest_path}")
    
        print(f"Copying {src_path} → {dest_path}")
        shutil.copy2(src_path, dest_path)

process_all_button_logs_recursive("~/Downloads/exp_data/")
extract_audio_labels("~/Downloads/exp_data/")

files_with_p = find_labeled_csvs_with_top_level_p_index("~/Downloads/exp_data/")
for path, p_num in files_with_p:
    print(f"labeled_csv_path: {path}, p number: {p_num}")
    append_person_information(path, p_num)

copy_audio_labeled_data("~/Downloads/exp_data/")
copy_button_labeled_data("~/Downloads/exp_data/")