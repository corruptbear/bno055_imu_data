#!/usr/bin/env python3
import os
from pathlib import Path
from extract_with_label import extract_labeled_data  # if needed
import re

BPM_RE = re.compile(r"^(.*?bpm\d+)", re.IGNORECASE)


"""
roman_vi_
roman_ix_
roman_iii_
roman_v_
"""

def extract_prefix_up_to_bpm(s: str) -> str | None:
    """Return substring up to and including 'bpm<digits>' (lowercased)."""
    if s is None:
        return None
    m = BPM_RE.search(s.strip())
    return m.group(1).lower() if m else None

if __name__ == "__main__":
    """
    # label everything as strokes
    root = Path("/Users/lws/Downloads/exp_data").expanduser().resolve()
    for path, dirs, files in os.walk(root):
        folder = Path(path).name
        if "roman" in folder.lower():
            prefix = extract_prefix_up_to_bpm(folder)
            if prefix is None:
                continue
            for f in files:
                if f.endswith("_unit_converted.csv"):
                    csvpath = Path(path) / f
                    prefix= prefix.replace("_padded_", "_strokes_padded_")
                    #prefix = strokes_alignment_name
                    print(prefix, "->", csvpath)
                    extract_labeled_data(csvpath, {prefix:0.05})
    #
    """


    # label the slected numeral as numerals, not strokes,
    # then run python3.10 hashed...
    # these numerals can be recognized using the reaming strokes.
    root = Path("/Users/lws/Downloads/exp_data").expanduser().resolve()
    for path, dirs, files in os.walk(root):
        folder = Path(path).name
        if "roman" in folder.lower():
            prefix = extract_prefix_up_to_bpm(folder)
            if prefix is None:
                continue
            for f in files:
                if f.endswith("_unit_converted.csv"):
                    csvpath = Path(path) / f
                    for name in ["roman_vi_padded_bpm120","roman_ix_padded_bpm120","roman_iii_padded_bpm120","roman_v_padded_bpm120"]:
                        if name in prefix:
                            print(prefix, "->", csvpath)
                            extract_labeled_data(csvpath, {prefix:0.05})
                        #extract_labeled_data(csvpath, {prefix:0.05})
    #
    