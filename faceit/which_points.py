from argparse import ArgumentParser
import json
import numpy as np
import os
from PIL import Image


def parse_flags():
    a = ArgumentParser()
    a.add_argument('--in_dir', type=str, default='data/umdfaces_batch3/')
    a.add_argument('--out_all_keypoints', type=str,
                   default='data/ultraface_all_keypoints.jpg')
    return a.parse_args()


def each_dict(f):
    keys = next(f).strip().split(',')
    for line in f:
        values = next(f).strip().split(',')
        values[0] = int(values[0])
        values[2:] = map(float, values[2:])
        yield dict(zip(keys, values))


def main(flags):
    f = os.path.join(flags.in_dir, 'umdfaces_batch3_ultraface.csv')
    f = open(f)
    for d in each_dict(f):
        print(json.dumps(d, indent=4, sort_keys=True))

        for i in range(1, 22):
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

            x = int(d['P%dX' % i])
            y = int(d['P%dY' % i])
            im[y - 1:y + 1, x - 1:x + 1, 0] = 255

            f = flags.out_all_keypoints.replace('.jpg', '_kp%d.jpg' % i)
            im = Image.fromarray(im)
            im.save(f)
        break

    print('-' * 80)

    print('Results -- the rows of face keypoints mean:')
    print('     1  2  3        left eyebrow (LTR)')
    print('     4  5  6        right eyebrow (LTR)')
    print('     7  8  9        left eye (LTR)')
    print('    10 11 12        right eye (LTR)')
    print('    13 14 15 16 17  left ear, point of noise (3 LTR), right ear')
    print('    18 19 20        smile (3 LTR)')
    print('    21              point of chin')


if __name__ == '__main__':
    main(parse_flags())
