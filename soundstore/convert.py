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

base_dir, mode, rate, prefix, codec = sys.argv[1:]

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

sample_dirs = tree.sample_dirs(base_dir)

def convert_sample(sample_dir):
    try:
        meta = tree.load_meta(sample_dir)

        if not 'codec' in meta:
            return
        target_codec = meta['codec'].split(',')[0]
        if not 'channels' in meta:
            return
        channels = meta['channels']
        if channels < 1:
            return

        args = ['sox', '-t', target_codec, tree.sample_path(sample_dir), '-t', codec, '-r', str(rate)]

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
for _ in tqdm.tqdm(pool.imap_unordered(convert_sample, sample_dirs), ascii=True, ncols=100, mininterval=0.5, total=len(sample_dirs)):
    pass

print('')
print('Done.')