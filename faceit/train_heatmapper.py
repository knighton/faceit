from argparse import ArgumentParser
import json
import numpy as np
import os
from PIL import Image
import shutil
from torch.optim import SGD
from tqdm import tqdm

from dataset import Dataset
from heatmapper_model import HeatmapperModel
from util import load_dataset


tqdm.monitor_interval = 0


def parse_flags():
    a = ArgumentParser()

    # General knobs.
    a.add_argument('--verbose', type=int, default=2)

    # Dataset knobs.
    a.add_argument('--dataset_dir', type=str, default='data/proc/')
    a.add_argument('--val_frac', type=float, default=0.2)

    # Model knobs.
    a.add_argument('--dim', type=int, default=64)

    # Training knobs.
    a.add_argument('--chk_dir', type=str,
                   default='data/heatmapper_checkpoints/')
    a.add_argument('--lr', type=float, default=0.001)
    a.add_argument('--momentum', type=float, default=0.8)
    a.add_argument('--num_epochs', type=int, default=4)
    a.add_argument('--max_batches_per_epoch', type=int, default=64)
    a.add_argument('--batch_size', type=int, default=32)

    return a.parse_args()


def get_dataset(dataset_dir, val_frac):
    crops, infos = load_dataset(dataset_dir)
    print('crops:', crops.shape)
    print('infos:', infos.shape)

    num_samples = len(crops)

    min_xx = infos[:, 4]
    min_yy = infos[:, 5]
    max_xx = infos[:, 6]
    max_yy = infos[:, 7]

    heatmaps_shape = num_samples, 128, 128, 1
    heatmaps = np.zeros(heatmaps_shape, 'uint8')
    for i in range(num_samples):
        min_x = int(min_xx[i])
        min_y = int(min_yy[i])
        max_x = int(max_xx[i])
        max_y = int(max_yy[i])
        heatmaps[i, min_y:max_y, min_x:max_x, :] = 1

    dataset = Dataset.split(crops, heatmaps, val_frac)

    print('split:', dataset.train.samples_per_epoch,
          dataset.val.samples_per_epoch)
    print('x:')
    print('- names:', ['crops'])
    print('- shapes:', dataset.x_sample_shapes)
    print('- dtypes:', dataset.x_dtypes)
    print('y:')
    print('- names:', ['heatmaps'])
    print('- shapes:', dataset.y_sample_shapes)
    print('- dtypes:', dataset.y_dtypes)

    return dataset


def main(flags):
    if os.path.exists(flags.chk_dir):
        shutil.rmtree(flags.chk_dir)
    os.makedirs(flags.chk_dir)

    dataset = get_dataset(flags.dataset_dir, flags.val_frac)

    model = HeatmapperModel(flags.dim).cuda()
    optimizer = SGD(model.parameters(), lr=flags.lr, momentum=flags.momentum)

    initial_epoch = 0
    model.fit(dataset, optimizer, initial_epoch, flags.num_epochs,
              flags.max_batches_per_epoch, flags.batch_size, flags.chk_dir,
              flags.verbose)


if __name__ == '__main__':
    main(parse_flags())
