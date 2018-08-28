import os
import json

def sample_dirs(base_dir):
    matches = []
    print('Scanning sound store...')
    for root, dirnames, filenames in os.walk(base_dir):
        if 'sample.json' in filenames:
            matches.append(root)
    print('Found %d samples in sound store.' % len(matches))
    return matches

def sample_path(sample_dir):
    return os.path.join(sample_dir, 'sample.dat')

def meta_file_path(sample_dir):
    return os.path.join(sample_dir, 'sample.json')

def load_meta(sample_dir):
    with open(meta_file_path(sample_dir), 'r') as file:
        return json.load(file)

def save_meta(sample_dir, obj):
    with open(meta_file_path(sample_dir), 'w') as file:
        json.dump(obj, file)