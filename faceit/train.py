from argparse import ArgumentParser
import json
import numpy as np
import os
from PIL import Image
import shutil
from torch.optim import SGD
from tqdm import tqdm

from dataset import Dataset
from model import Model
from util import load_dataset


def parse_flags():
    a = ArgumentParser()

    # General knobs.
    a.add_argument('--verbose', type=int, default=2)

    # Dataset knobs.
    a.add_argument('--dataset_dir', type=str, default='data/proc/')
    a.add_argument('--val_frac', type=float, default=0.2)

    # Model knobs.
    a.add_argument('--dim', type=int, default=32)

    # Training knobs.
    a.add_argument('--chk_dir', type=str, default='data/checkpoints/')
    a.add_argument('--lr', type=float, default=0.001)
    a.add_argument('--momentum', type=float, default=0.8)
    a.add_argument('--num_epochs', type=int, default=10000)
    a.add_argument('--max_batches_per_epoch', type=int, default=32)
    a.add_argument('--batch_size', type=int, default=16)

    return a.parse_args()


def get_dataset(dataset_dir, val_frac):
    crops, infos = load_dataset(dataset_dir)
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

    keypoints = np.zeros((infos.shape[0], 4), infos.dtype)
    keypoints[:, 0] = infos[:, 8 + 7 * 2]
    keypoints[:, 1] = infos[:, 8 + 7 * 2 + 1]
    keypoints[:, 2] = infos[:, 8 + 10 * 2]
    keypoints[:, 3] = infos[:, 8 + 10 * 2 + 1]
    print('keypoints:', keypoints.shape)

    xx = [crops]
    yy = [is_face, gender, pose, bbox, keypoints]
    dataset = Dataset.split(xx, yy, val_frac)
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

    return dataset


def main(flags):
    if os.path.exists(flags.chk_dir):
        shutil.rmtree(flags.chk_dir)
    os.makedirs(flags.chk_dir)

    dataset = get_dataset(flags.dataset_dir, flags.val_frac)

    model = Model(flags.dim)
    optimizer = SGD(model.parameters(), lr=flags.lr, momentum=flags.momentum)

    initial_epoch = 0
    model.fit(dataset, optimizer, initial_epoch, flags.num_epochs,
              flags.max_batches_per_epoch, flags.batch_size, flags.chk_dir,
              flags.verbose)


if __name__ == '__main__':
    main(parse_flags())
