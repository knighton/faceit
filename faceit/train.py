from argparse import ArgumentParser
import json
import numpy as np
import os
from PIL import Image
from tqdm import tqdm

from dataset import Dataset
from util import load_dataset


def parse_flags():
    a = ArgumentParser()
    a.add_argument('--in_dir', type=str, default='data/proc/')
    a.add_argument('--out_dir', type=str, default='data/model/')
    a.add_argument('--val_frac', type=float, default=0.2)
    return a.parse_args()


def main(flags):
    assert not os.path.exists(flags.out_dir)
    os.makedirs(flags.out_dir)
    crops, infos = load_dataset(flags.in_dir)
    print('crops:', crops.shape)
    print('infos:', infos.shape)

    gender = infos[:, 0]
    gender = np.expand_dims(gender, 1)
    pose = infos[:, 1:4]
    min_x = infos[:, 4]
    min_y = infos[:, 5]
    max_x = infos[:, 6]
    max_y = infos[:, 7]
    bbox = np.stack([min_x, min_y, max_x, max_y], axis=1)
    print('bbox:', bbox.shape)
    for i in range(len(bbox)):
        assert min_x[i] < max_x[i]
        assert min_y[i] < max_y[i]
    area = (max_y - min_y) * (max_x - min_x) 
    is_face = area / (128 * 128)
    is_face = np.expand_dims(is_face, 1)
    print('faceness:', is_face.mean(), is_face.std(), is_face.min(),
          is_face.max())
    keypoints = infos[:, 8:]

    xx = [crops]
    yy = [is_face, gender, pose, bbox, keypoints]
    dataset = Dataset.split(xx, yy, flags.val_frac)
    print('split:', dataset.train.samples_per_epoch,
          dataset.val.samples_per_epoch)
    print('x:')
    print('- names:', ['crops'])
    print('- shapes:', dataset.x_sample_shapes)
    print('- dtypes:', dataset.x_dtypes)
    print('y:')
    print('- names:', ['isface', 'gender', 'pose', 'bbox', 'keypoints'])
    print('- shapes:', dataset.y_sample_shapes)
    print('- dtypes:', dataset.y_dtypes)


if __name__ == '__main__':
    main(parse_flags())
