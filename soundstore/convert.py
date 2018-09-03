#!/usr/bin/env python2
import os
import sys
import tqdm
import subprocess
import utils.tree as tree
from multiprocessing import Pool

def fail():
    print('Usage: convert.py sample-dir mode rate prefix codec')
    exit(1)

if not len(sys.argv) == 6:
    fail()

mode, rate, prefix, codec = sys.argv[2:]

if not mode in ['mono', 'split', 'stereo']:
    print('Unknown mode: %s' % mode)
    fail()

try:
    rate = int(rate)
except:
    fail()
if rate <= 0:
    print('Wrong rate: %d' % rate)
    fail()

sample_dirs = tree.sample_dirs(sys.argv[1])
to_percent = 100.0 / float(len(sample_dirs))

def convert_sample(sample_dir):
    try:
        meta = tree.load_meta(sample_dir)

        if not 'codec' in meta:
            return
        codec = meta['codec']

        if not 'channels' in meta:
            return
        channels = meta['channels']
        if channels < 1:
            return

        filename = tree.sample_path(sample_dir)
        args = ['sox', '-t', codec, filename, '-t', codec, '-r', str(rate)]

        if mode == 'mono':
            subprocess.check_output(args + [os.path.join(sample_dir, prefix + '.' + codec), 'remix', '1-' + str(channels)], stderr=subprocess.STDOUT)
        elif mode == 'split':
            for channel in range(1, channels + 1):
                subprocess.check_output(args + [os.path.join(sample_dir, prefix + '.' + str(channel) + '.' + codec), 'remix', str(channel)], stderr=subprocess.STDOUT)
        else:
            if channels == 1:
                subprocess.check_output(args + [os.path.join(sample_dir, prefix + '.' + codec), 'remix', '1', '1'], stderr=subprocess.STDOUT)
            else:
                subprocess.check_output(args + [os.path.join(sample_dir, prefix + '.' + codec), 'remix', '1', '2'], stderr=subprocess.STDOUT)
    except Exception as ex:
        print(ex)

print('Converting samples...')

pool = Pool(processes=8)
for _ in tqdm.tqdm(pool.imap_unordered(convert_sample, sample_dirs), mininterval=0.5, total=len(sample_dirs)):
    pass

print('')
print('Done.')