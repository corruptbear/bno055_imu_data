import re
import yaml
import csv

def get_base_name(name):
    # Use a regex to remove the numeric suffix preceded by an underscore
    return re.sub(r'_\d+$', '', name)

def save_yaml(dictionary,filepath,write_mode):
    with open(filepath,write_mode) as f:
        yaml.dump(dictionary,f)

def load_yaml(filepath):
    try:
        with open(filepath,'r') as stream:
            dictionary = yaml.safe_load(stream)
            return dictionary
    except:
        return dict()

def infer_formats(line):
    """
    input: "1,2.12,3.5656,10,8.232323"
    output: ['%d', '%.2f', '%.4f', '%d', '%.6f']
    """
    values = line.strip().split(',')

    formats = []

    for value in values:
        try:
            # Try to convert to integer
            int_value = int(value)
            formats.append("%d")  # 'd' for integer
        except ValueError:
            try:
                # Try to convert to float
                float_value = float(value)
                # Determine the number of decimal places
                decimal_places = len(value.split('.')[1]) if '.' in value else 0
                formats.append(f"%.{decimal_places}f")  # '.xf' for float with x decimal places
            except ValueError:
                # If it's neither int nor float, treat it as a string
                formats.append("%s")  # 's' for string
    return formats

def infer_formats_csv(csv_path):
    with open(csv_path, mode='r') as f:
        first_non_header_line = f.readlines()[1].strip()
        return infer_formats(first_non_header_line)