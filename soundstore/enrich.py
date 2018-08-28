#!/usr/bin/env python2
import os
import re
import sys
import tqdm
import subprocess
import utils.tree as tree
from collections import Counter
from utils.helpers import print_progress
from utils.helpers import format_duration
from multiprocessing import Pool

sample_dirs = tree.sample_dirs(sys.argv[1])
to_percent = 100.0 / float(len(sample_dirs))

total_duration = 0.0
total_duration_channels = 0.0

codec_counter = Counter()
encoding_counter = Counter()
rate_counter = Counter()
channels_counter = Counter()

def enrich_sample(sample_dir):
    meta = tree.load_meta(sample_dir)
    filename = tree.sample_path(sample_dir)
    output = subprocess.check_output(['ffprobe', '-hide_banner', '-i', filename], stderr=subprocess.STDOUT)
    output = output.strip().split('\n')
    codec = 'unknown'
    encoding = 'unknown'
    duration = 0.0
    rate = 0
    channels = 1

    for line in output:
        line = line.strip()
        match = re.match(r'^Input #0, ([0-9,a-z]+), from.*', line)
        if match:
            codec = match.group(1)
        match = re.match(r'^Duration: ([0-9]+):([0-9]+):([0-9]+\.[0-9]+),.*', line)
        if match:
            duration = float(match.group(3)) + 60 * int(match.group(2)) + 60 * 60 * int(match.group(1))
        match = re.match(r'^Stream #0:0: Audio: ([^\s]+) [^,]*, ([0-9]+) Hz, ([0-9]+) channels, ([a-z0-9]+),.*', line)
        if match:
            encoding = match.group(1)
            rate = int(match.group(2))
            channels = int(match.group(3))

    meta['codec'] = codec
    meta['encoding'] = encoding
    meta['duration'] = duration
    meta['rate'] = rate
    meta['channels'] = channels

    tree.save_meta(sample_dir, meta)
    return (codec, encoding, duration, rate, channels)

print('Reading samples and updating JSON files...')

pool = Pool(processes=8)
for codec, encoding, duration, rate, channels in tqdm.tqdm(pool.imap_unordered(enrich_sample, sample_dirs), total=len(sample_dirs)):
    total_duration += duration
    total_duration_channels += duration * channels
    codec_counter[codec] += 1
    encoding_counter[encoding] += 1
    rate_counter[rate] += 1
    channels_counter[channels] += 1

print('')
print('Overall sample duration:                     ' + format_duration(total_duration))
print('Overall sample duration (separate channels): ' + format_duration(total_duration_channels))

def print_counter(counter, caption):
    print('\n%s:' % caption)
    for n, v in counter.most_common():
        print('%10s:% 8.2f%%' % (n, float(v) * to_percent))

print_counter(codec_counter, 'Codecs')
print_counter(encoding_counter, 'Encodings')
print_counter(rate_counter, 'Rates')
print_counter(channels_counter, 'Channels')

print('')
print('Done.')