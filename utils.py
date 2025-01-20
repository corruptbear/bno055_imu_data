import re
import yaml

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