#!/usr/bin/env python3
import argparse
import os
import re
import sys
from pathlib import Path

try:
    import yaml  # PyYAML
except ImportError:
    print("This script requires PyYAML. Install with: pip install pyyaml", file=sys.stderr)
    sys.exit(2)

FNAME_SUFFIX = "_unit_converted_alignment.yaml"
BPM_RE = re.compile(r"^(.*?bpm\d+)", re.IGNORECASE)

def extract_prefix_up_to_bpm(s: str) -> str | None:
    """
    Return the substring up to and including 'bpm<digits>' (case-insensitive),
    lower-cased. Returns None if no such pattern is found.
    """
    if s is None:
        return None
    m = BPM_RE.search(s.strip())
    return m.group(1).lower() if m else None

def first_top_level_key(yaml_path: Path) -> str | None:
    """
    Load YAML and return the first top-level key (as a string), or None.
    """
    try:
        with yaml_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception as e:
        print(f"[ERROR] Failed to read YAML: {yaml_path} ({e})", file=sys.stderr)
        return None

    if isinstance(data, dict) and data:
        # Get the first key in insertion order (PyYAML preserves it)
        return next(iter(data.keys()))
    else:
        print(f"[WARN] YAML is not a non-empty mapping: {yaml_path}", file=sys.stderr)
        return None

def check_alignment_file(yaml_path: Path, show_ok: bool = False) -> bool:
    """
    Check one alignment file. Returns True if it matches, False otherwise.
    """
    dir_name = yaml_path.parent.name
    key = first_top_level_key(yaml_path)

    dir_prefix = extract_prefix_up_to_bpm(dir_name)
    key_prefix = extract_prefix_up_to_bpm(key) if key is not None else None

    if dir_prefix is None:
        print(f"[WARN] No 'bpm<digits>' pattern in directory name: {dir_name}  ({yaml_path})")
        return True  # Treat as non-actionable rather than failing

    if key_prefix is None:
        print(f"[ERROR] No top-level key with 'bpm<digits>' in YAML: {yaml_path}")
        return False

    if dir_prefix == key_prefix:
        if show_ok:
            print(f"[OK] {yaml_path}  dir='{dir_prefix}' == key='{key_prefix}'")
        return True
    else:
        print(f"[MISMATCH] {yaml_path}")
        print(f"  dir up to bpm : {dir_prefix}")
        print(f"  key up to bpm : {key_prefix}")
        print(f"  (dir name: '{dir_name}', yaml key: '{key}')")
        return False

def main():
    #usage: python3.10 qa_dirname_alignment_match.py /Users/lws/Downloads/exp_data
    ap = argparse.ArgumentParser(
        description="Recursively verify that alignment YAML keys match their containing directory name up to 'bpm<digits>' (case-insensitive)."
    )
    ap.add_argument("root", help="Root directory to search")
    ap.add_argument("--show-ok", action="store_true", help="Also print matching files")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        print(f"Root does not exist or is not a directory: {root}", file=sys.stderr)
        sys.exit(2)

    total = 0
    mismatches = 0

    for path, dirs, files in os.walk(root):
        for fn in files:
            if fn.endswith(FNAME_SUFFIX):
                total += 1
                yaml_path = Path(path) / fn
                ok = check_alignment_file(yaml_path, show_ok=args.show_ok)
                if not ok:
                    mismatches += 1

    if total == 0:
        print(f"No files ending with '{FNAME_SUFFIX}' found under {root}")

    print(f"\nChecked: {total} file(s). Mismatches: {mismatches}.")
    sys.exit(1 if mismatches > 0 else 0)

if __name__ == "__main__":
    main()