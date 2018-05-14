from argparse import ArgumentParser
import json
import numpy as np
import os
from PIL import Image


def parse_flags():
    a = ArgumentParser()
    a.add_argument('--in_dir', type=str, default='data/umdfaces_batch3/')
    return a.parse_args()


def each_dict(f):
    keys = next(f).strip().split(',')
    for line in f:
        values = next(f).strip().split(',')
        values[0] = int(values[0])
        values[2:] = map(float, values[2:])
        yield dict(zip(keys, values))


def stats(name, x):
    num_buckets = 31
    max_bar_len = 30
    print('-' * 80)
    print(name)
    x = np.array(x, 'float32')
    print('mean', x.mean())
    print('std', x.std())
    orig_x_min = x.min()
    orig_x_max = x.max()
    x = x.astype('float32')
    x -= orig_x_min
    x /= (orig_x_max - orig_x_min)
    x *= (num_buckets - 1)
    x = x.astype('int32')
    buckets = np.zeros(num_buckets, 'int32')
    for a in x:
        buckets[a] += 1
    jump = (orig_x_max - orig_x_min) / (num_buckets - 1)
    for i, b in enumerate(buckets):
        a = int(orig_x_min + jump * i)
        z = int(orig_x_min + jump * (i + 1))
        bar_len = (float(b) - 1e-6) / buckets.max() * max_bar_len
        bar = '=' * int(bar_len)
        print('[%3d] from %5d to %5d :: %7d %s' % (i, a, z, b, bar))
    print('-' * 80)


def main(flags):
    f = os.path.join(flags.in_dir, 'umdfaces_batch3_ultraface.csv')
    f = open(f)
    yaws = []
    pitches = []
    rolls = []
    for d in each_dict(f):
        yaws.append(d['YAW'])
        pitches.append(d['PITCH'])
        rolls.append(d['ROLL'])
    stats('yaw', yaws)
    stats('pitch', pitches)
    stats('roll', rolls)


if __name__ == '__main__':
    main(parse_flags())
