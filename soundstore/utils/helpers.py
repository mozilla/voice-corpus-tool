
import sys
import datetime

def print_progress(total, current):
    f = float(current) / float(total)
    sys.stdout.write('\r[{0:50s}] {1:.1f}% ({2} of {3})'.format('#' * int(f * 50), f * 100, current, total))
    sys.stdout.flush()

def format_duration(seconds):
    return str(datetime.timedelta(seconds=seconds))