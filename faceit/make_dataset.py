from argparse import ArgumentParser
import json
import numpy as np
import os
from PIL import Image
from tqdm import tqdm

from util import max_pool_2d_nchw


def parse_flags():
    a = ArgumentParser()
    a.add_argument('--in_dir', type=str, default='data/umdfaces_batch3/')
    a.add_argument('--out_dir', type=str, default='data/proc/')
    return a.parse_args()


def each_dict(f):
    keys = next(f).strip().split(',')
    for line in f:
        values = next(f).strip().split(',')
        values[0] = int(values[0])
        values[2:] = map(float, values[2:])
        yield dict(zip(keys, values))


def process_dict(in_dir, d, crops, infos):
    # Load image.
    f = d['FILE']
    f = os.path.join(in_dir, f)
    im = Image.open(f)
    im = np.array(im)

    # Shrink image.
    im = im.transpose([2, 0, 1])
    im = np.expand_dims(im, 0)
    im = max_pool_2d_nchw(im, 2)
    im = im.squeeze()
    im = im.transpose([1, 2, 0])

    # Get whether each pixel is a face (convenient way).
    face_min_x = int(d['FACE_X'] / 2)
    face_min_y = int(d['FACE_Y'] / 2)
    face_w = int(d['FACE_WIDTH'] / 2)
    face_h = int(d['FACE_HEIGHT'] / 2)
    face_max_x = face_min_x + face_w
    face_max_y = face_min_y + face_h

    # Restrict image size.
    if im.shape[0] <= 128 or im.shape[1] <= 128:
        return

    # Restrict face size.
    if not (32 <= face_w <= 96) or not (32 <= face_h <= 96):
        return

    # Get a bounding box that fully contains the face.
    ctr_x = face_min_x + np.random.randint(0, face_w)
    orig_crop_min_x = ctr_x - 128 // 2
    orig_crop_max_x = ctr_x + 128 // 2
    if orig_crop_min_x < 0:
        crop_max_x = orig_crop_max_x - orig_crop_min_x
        crop_min_x = 0
    elif im.shape[1] <= orig_crop_max_x:
        crop_min_x = orig_crop_min_x - orig_crop_max_x + im.shape[1]
        crop_max_x = im.shape[1]
    else:
        crop_min_x = orig_crop_min_x
        crop_max_x = orig_crop_max_x

    ctr_y = face_min_y + np.random.randint(0, face_h)
    orig_crop_min_y = ctr_y - 128 // 2
    orig_crop_max_y = ctr_y + 128 // 2
    if orig_crop_min_y < 0:
        crop_max_y = orig_crop_max_y - orig_crop_min_y
        crop_min_y = 0
    elif im.shape[1] <= orig_crop_max_y:
        crop_min_y = orig_crop_min_y - orig_crop_max_y + im.shape[1]
        crop_max_y = im.shape[1]
    else:
        crop_min_y = orig_crop_min_y
        crop_max_y = orig_crop_max_y

    # Do the crop.
    crop = im[crop_min_y:crop_max_y, crop_min_x:crop_max_x, :]
    if crop.shape != (128, 128, 3):
        return
    crops.append(crop)

    # Gather the info (coordinates and such).
    info = []
    male = d['PR_MALE']
    info += [male]
    yaw = d['YAW']
    pitch = d['PITCH']
    roll = d['ROLL']
    info += [yaw, pitch, roll]
    face_min_x = face_min_x - crop_min_x
    face_min_y = face_min_y - crop_min_y
    face_max_x = face_min_x + face_w
    face_max_y = face_min_y + face_h
    info += [face_min_x, face_min_y, face_max_x, face_max_y]
    for i in range(1, 22):
        x = d['P%dX' % i] // 2 - crop_min_x
        y = d['P%dY' % i] // 2 - crop_min_y
        info += [x, y]
    info = np.array(info, 'float32')
    infos.append(info)


def main(flags):
    assert not os.path.exists(flags.out_dir)
    os.makedirs(flags.out_dir)

    f = os.path.join(flags.in_dir, 'umdfaces_batch3_ultraface.csv')
    f = open(f)
    dd = []
    for d in each_dict(f):
        dd.append(d)
    print('%d entries.' % len(dd))

    np.random.shuffle(dd)

    crops = []
    infos = []
    for d in tqdm(dd):
        process_dict(flags.in_dir, d, crops, infos)
    crops = np.stack(crops)
    infos = np.stack(infos)

    f = 'crops.npy'
    f = os.path.join(flags.out_dir, f)
    crops.tofile(f)

    f = 'infos.npy'
    f = os.path.join(flags.out_dir, f)
    infos.tofile(f)

    d = 'examples'
    d = os.path.join(flags.out_dir, d)
    os.makedirs(d)
    for i in range(500):
        x = Image.fromarray(crops[i])
        f = '%s/%d.jpg' % (d, i)
        x.save(f)


if __name__ == '__main__':
    main(parse_flags())
