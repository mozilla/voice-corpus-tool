#!/usr/bin/env python2
import os
import sys
import csv
import glob
import tqdm
import subprocess
import utils.tree as tree
from multiprocessing import Pool

def fail():
    print('Usage: tocsv.py sample-dir prefix suffix')
    exit(1)

if not len(sys.argv) == 4:
    fail()

base_dir, prefix, suffix = sys.argv[1:]
base_dir = os.path.abspath(base_dir)
sample_dirs = tree.sample_dirs(base_dir)

def convert_sample(sample_dir):
    try:
        meta = tree.load_meta(sample_dir)
        pattern = os.path.join(sample_dir, prefix + '*' + suffix)
        return [(os.path.relpath(filename, base_dir), os.stat(filename).st_size, meta['tags'], meta['duration']) for filename in glob.glob(pattern)]
    except Exception as ex:
        print(ex)
        return []

print('Collecting samples...')

with open(os.path.join(base_dir, prefix + '.csv'), 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['wav_filename', 'wav_filesize', 'transcript', 'tags', 'duration'])
    pool = Pool(processes=8)
    for files in tqdm.tqdm(pool.imap_unordered(convert_sample, sample_dirs), ascii=True, ncols=100, mininterval=0.5, total=len(sample_dirs)):
        for filename, filesize, tags, duration in files:
            writer.writerow([filename, filesize, '', ' '.join(tags), duration])

print('')
print('Done.')