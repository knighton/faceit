from argparse import ArgumentParser
import json
import numpy as np
import os
from PIL import Image
from tqdm import tqdm

from util import load_dataset


def parse_flags():
    a = ArgumentParser()
    a.add_argument('--in_dir', type=str, default='data/proc/')
    a.add_argument('--out_dir', type=str, default='data/proc/look/')
    return a.parse_args()


def draw_sample(crop, info, f):
    im = crop.copy()

    print(info[4:8])
    face_min_x, face_min_y, face_max_x, face_max_y = \
        map(lambda x: x.astype('int32').clip(0, 128), info[4:8])
    im[face_min_y:face_max_y, face_min_x:face_max_x, :] //= 2

    im = Image.fromarray(im)
    im.save(f)


def main(flags):
    assert not os.path.exists(flags.out_dir)
    os.makedirs(flags.out_dir)
    crops, infos = load_dataset(flags.in_dir)
    print(crops.shape)
    print(infos.shape)
    for i in range(len(crops)):
        f = '%d.jpg' % i
        f = os.path.join(flags.out_dir, f)
        draw_sample(crops[i], infos[i], f)


if __name__ == '__main__':
    main(parse_flags())
