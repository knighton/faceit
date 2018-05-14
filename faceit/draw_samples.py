from argparse import ArgumentParser
import json
import numpy as np
import os
from PIL import Image


def parse_flags():
    a = ArgumentParser()
    a.add_argument('--in_dir', type=str, default='data/umdfaces_batch3/')
    a.add_argument('--out_examples_dir', type=str, default='data/examples/')
    return a.parse_args()


def each_dict(f):
    keys = next(f).strip().split(',')
    for line in f:
        values = next(f).strip().split(',')
        values[0] = int(values[0])
        values[2:] = map(float, values[2:])
        yield dict(zip(keys, values))


def main(flags):
    assert not os.path.exists(flags.out_examples_dir)
    os.makedirs(flags.out_examples_dir)

    f = os.path.join(flags.in_dir, 'umdfaces_batch3_ultraface.csv')
    f = open(f)
    dd = []
    for d in each_dict(f):
        dd.append(d)

    np.random.shuffle(dd)

    for d in dd[:100]:
        f = d['FILE']
        f = os.path.join(flags.in_dir, f)
        im = Image.open(f)
        im = np.array(im)

        face_min_x = int(d['FACE_X'])
        face_min_y = int(d['FACE_Y'])
        face_w = int(d['FACE_WIDTH'])
        face_h = int(d['FACE_HEIGHT'])
        face_ctr_x = face_min_x + face_w / 2
        face_ctr_y = face_min_y + face_h / 2
        im[face_min_y:face_min_y + face_h,
           face_min_x:face_min_x + face_w, :] //= 2

        for i in range(1, 22):
            x = int(d['P%dX' % i])
            y = int(d['P%dY' % i])
            im[y - 1:y + 1, x - 1:x + 1, 0] = 255

        f = d['FILE']
        f = f[f.rindex('/') + 1:]
        f = os.path.join(flags.out_examples_dir, f)
        im = Image.fromarray(im)
        im.save(f)


if __name__ == '__main__':
    main(parse_flags())
