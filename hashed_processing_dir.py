# -*- coding: utf-8 -*-
import glob
import os
import re
import shutil
import hashlib
import json
import time

# your project imports
from extract_with_label import (
    extract_labeled_data_from_button_interface,
    extract_labeled_data,
    append_person_information,
)
from utils import *  # if you need other helpers

# =========================
# Content-hash based cache
# =========================
def _cache_path_for(root_dir):
    root_dir = os.path.abspath(os.path.expanduser(root_dir))
    return os.path.join(root_dir, ".extraction_cache.json")

def _load_cache(cache_file):
    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except Exception as e:
        print(f"[cache] Could not read cache: {e}. Starting fresh.")
        return {}

def _save_cache(cache_file, cache_obj):
    tmp = cache_file + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache_obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, cache_file)

def file_hash(path, chunk_size=8192):
    """Compute SHA256 hash of a file."""
    hasher = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()

def _should_extract(path, cache, kind):
    """
    Decide if 'path' should be processed:
      - not seen before -> True
      - hash changed since last time -> True
      - otherwise -> False
    `kind` lets you keep separate namespaces if the same file could be treated
    by different pipelines.
    """
    abspath = os.path.abspath(os.path.expanduser(path))
    h = file_hash(abspath)
    key = f"{kind}::{abspath}"
    cached = cache.get(key)
    if cached and cached.get("hash") == h:
        return False, h, key
    return True, h, key

def _mark_extracted(cache, key, h):
    cache[key] = {
        "hash": h,
        "last_extracted_ts": time.time()
    }

# ----- person-information caching -----
def _should_process_person_info(path, p_num, cache):
    """
    Skip if we've already appended person info for this file content and p_num.
    """
    abspath = os.path.abspath(os.path.expanduser(path))
    h = file_hash(abspath)
    key = f"person_info::{abspath}::p{p_num}"
    cached = cache.get(key)
    if cached and cached.get("hash") == h and cached.get("p_num") == p_num:
        return False, h, key
    return True, h, key

def _mark_person_info_done(cache, key, h, p_num):
    cache[key] = {
        "hash": h,
        "p_num": p_num,
        "last_appended_ts": time.time()
    }

# =========================
# Original helpers (with caching)
# =========================
def is_already_unzipped(zip_path):
    # Assume the unzipped directory name is derived from the zip file name
    unzip_dir = os.path.splitext(zip_path)[0]  # remove .zip
    return os.path.isdir(unzip_dir)

def process_all_button_logs_recursive(root_dir):
    """
    Extract only when a 'button_*.zip' is NEW or CHANGED since last successful extraction.
    """
    root_dir = os.path.expanduser(root_dir)
    cache_file = _cache_path_for(root_dir)
    cache = _load_cache(cache_file)

    zip_files = glob.glob(os.path.join(root_dir, "**", "button_*.zip"), recursive=True)
    print("[button] candidates:", len(zip_files))

    extracted = skipped = 0
    for zip_path in zip_files:
        to_extract, h, key = _should_extract(zip_path, cache, kind="button_zip")

        if not to_extract:
            print(f"[skip] unchanged zip: {zip_path}")
            skipped += 1
            continue

        if is_already_unzipped(zip_path):
            print(f"[info] Unzipped dir exists for {zip_path}; zip is new/changed -> re-extracting.")
        print(f"[do] Extracting {zip_path} ...")

        try:
            extract_labeled_data_from_button_interface(zip_path)
            _mark_extracted(cache, key, h)
            extracted += 1
        except Exception as e:
            print(f"[error] Failed extracting {zip_path}: {e}")

    _save_cache(cache_file, cache)
    print(f"[button] extracted={extracted}, skipped={skipped}")

def extract_audio_labels(root_dir):
    """
    For '*audio_imu_logs*/**/*unit_converted.csv' run extraction only if NEW or CHANGED.
    """
    root_dir = os.path.abspath(os.path.expanduser(root_dir))
    cache_file = _cache_path_for(root_dir)
    cache = _load_cache(cache_file)

    labeled_files = glob.glob(
        os.path.join(root_dir, '**', '*audio_imu_logs*', '*unit_converted.csv'),
        recursive=True
    )

    print("[audio] candidates:", len(labeled_files))
    extracted = skipped = 0
    for path in labeled_files:
        to_extract, h, key = _should_extract(path, cache, kind="audio_unit")
        if not to_extract:
            print(f"[skip] unchanged audio unit: {path}")
            skipped += 1
            continue

        print(f"[do] extracting AUDIO labels: {path}")
        try:
            extract_labeled_data(path)
            _mark_extracted(cache, key, h)
            extracted += 1
        except Exception as e:
            print(f"[error] Failed extracting {path}: {e}")

    _save_cache(cache_file, cache)
    print(f"[audio] extracted={extracted}, skipped={skipped}")

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
                print(f"[find_p] Skipping: {abs_path} (no match for p<number> in {top_level_dir})")
        else:
            print(f"[find_p] Skipping: {abs_path} (not deep enough)")

    return result

def process_person_information_for_all(root_dir):
    """
    Append participant info only when the labeled CSV content changed OR the inferred p_num differs.
    """
    root_dir = os.path.abspath(os.path.expanduser(root_dir))
    cache_file = _cache_path_for(root_dir)
    cache = _load_cache(cache_file)

    files_with_p = find_labeled_csvs_with_top_level_p_index(root_dir)

    done = skipped = 0
    for path, p_num in files_with_p:
        to_do, h, key = _should_process_person_info(path, p_num, cache)
        if not to_do:
            print(f"[skip] person-info unchanged: {path} (p{p_num})")
            skipped += 1
            continue

        print(f"[do] append_person_information: {path} (p{p_num})")
        try:
            append_person_information(path, p_num)
            _mark_person_info_done(cache, key, h, p_num)
            done += 1
        except Exception as e:
            print(f"[error] append_person_information failed for {path} (p{p_num}): {e}")

    _save_cache(cache_file, cache)
    print(f"[person-info] done={done}, skipped={skipped}")

# (unchanged) copy helpers already avoid duplicates by comparing hash at copy time
def copy_audio_labeled_data(root_dir):
    root_dir = os.path.abspath(os.path.expanduser(root_dir))
    labeled_files = glob.glob(os.path.join(root_dir, '**', '*converted_labeled.csv'), recursive=True)

    dest_dir = "/Users/lws/Dev/lab_projects/tottag_ranging/ml-model-sandbox/python/datasets_audio/raw_data"
    os.makedirs(dest_dir, exist_ok=True)

    for src_path in labeled_files:
        filename = os.path.basename(src_path)
        dest_path = os.path.join(dest_dir, filename)

        if os.path.exists(dest_path):
            if file_hash(src_path) == file_hash(dest_path):
                print(f"[copy-audio] Skipping identical file: {dest_path}")
                continue
            else:
                print(f"[copy-audio] Overwriting different file: {dest_path}")

        print(f"[copy-audio] Copying {src_path} → {dest_path}")
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
                print(f"[copy-btn] Skipping identical file: {dest_path}")
                continue
            else:
                print(f"[copy-btn] Overwriting different file: {dest_path}")

        print(f"[copy-btn] Copying {src_path} → {dest_path}")
        shutil.copy2(src_path, dest_path)

# =========================
# Prune files that exist only in dest_dir
# =========================
def prune_orphan_files(dest_dir, valid_basenames, dry_run=True):
    """
    Remove files in dest_dir whose base names are NOT present in valid_basenames.
    - valid_basenames: a set of file base names (e.g., {'a.csv', 'b.csv'})
    - dry_run: if True, only print what would be deleted
    """
    dest_dir = os.path.abspath(os.path.expanduser(dest_dir))
    if not os.path.isdir(dest_dir):
        print(f"[prune] dest_dir does not exist: {dest_dir}")
        return

    deletions = 0
    kept = 0
    for entry in os.listdir(dest_dir):
        path = os.path.join(dest_dir, entry)
        if not os.path.isfile(path):
            continue  # ignore subdirs
        if entry not in valid_basenames:
            if dry_run:
                print(f"[prune][DRY] Would delete: {path}")
            else:
                try:
                    os.remove(path)
                    print(f"[prune] Deleted: {path}")
                    deletions += 1
                except Exception as e:
                    print(f"[prune][error] Could not delete {path}: {e}")
        else:
            kept += 1

    print(f"[prune] kept={kept}, {'would_delete' if dry_run else 'deleted'}={deletions}")

def prune_audio_dest(root_dir, dest_dir="/Users/lws/Dev/lab_projects/tottag_ranging/ml-model-sandbox/python/datasets_audio/raw_data", dry_run=True):
    """
    Build the set of expected audio labeled basenames from root_dir, then
    remove extra files in dest_dir that are not in that set.
    """
    root_dir = os.path.abspath(os.path.expanduser(root_dir))
    pattern = os.path.join(root_dir, '**', '*converted_labeled.csv')  # must match copy_audio_labeled_data
    source_files = glob.glob(pattern, recursive=True)
    valid_basenames = {os.path.basename(p) for p in source_files}
    print(f"[prune-audio] source candidates={len(source_files)}, unique basenames={len(valid_basenames)}")
    prune_orphan_files(dest_dir, valid_basenames, dry_run=dry_run)
# =========================
# Run
# =========================
if __name__ == "__main__":
    DATA_ROOT = "~/Downloads/exp_data/"

    process_all_button_logs_recursive(DATA_ROOT)
    extract_audio_labels(DATA_ROOT)

    process_person_information_for_all(DATA_ROOT)

    copy_audio_labeled_data(DATA_ROOT)

    #remove files that are deleted from source folder
    prune_audio_dest(DATA_ROOT, dry_run=False)   # set to False to actually delete

    # copy_button_labeled_data(DATA_ROOT)