from argparse import ArgumentParser
from glob import glob
import numpy as np
import os
from PIL import Image
import sys
from time import time
import torch
from tqdm import tqdm

from model import Model


tqdm.monitor_interval = 0


def parse_flags():
    a = ArgumentParser()

    # Model.
    a.add_argument('--model', type=str,
                   default='data/checkpoints/epoch_0159_0149_0145/model.bin')
    a.add_argument('--dim', type=int, default=128)

    # Input video.
    a.add_argument('--in_video', type=str,
                   default='/home/frak/dev/commaai/monitoring/video1/')

    # Output video.
    a.add_argument('--out_video', type=str, default='data/anno1/')

    return a.parse_args()


def main(flags):
    assert not os.path.exists(flags.out_video)
    os.makedirs(flags.out_video)

    model = torch.load(flags.model).cuda()

    pattern = os.path.join(flags.in_video, 'frame_*.jpg')
    ff = glob(pattern)

    for i, f in tqdm(enumerate(ff), total=len(ff), leave=False):
        im = Image.open(f)
        dim = 128 * 1152 / 864
        im.thumbnail((dim, dim))
        crop = np.array(im)
        crop = crop[:, -128:, :]

        x = np.expand_dims(crop, 0)
        x = x.astype('float32')
        x /= 127.5
        x -= 1
        x = x.transpose([0, 3, 1, 2])
        x = torch.from_numpy(x).cuda()
        xx = [x]

        is_faces, is_males, poses, bboxes, keypoints = model.forward(xx)
        go = lambda x: x.detach().cpu().numpy()
        min_x, min_y, max_x, max_y = map(int, go(bboxes[0]))
        rect = crop[min_y:max_y, min_x:max_x, :]
        confidence = int(256 * go(is_faces))
        crop[min_y:max_y, min_x:max_x, :] = np.maximum(rect, rect + confidence)

        keypoints = keypoints[0]
        x = int(keypoints[0])
        y = int(keypoints[1])
        crop[y - 1:y + 2, x - 1:x + 2, :] = 255
        x = int(keypoints[2])
        y = int(keypoints[3])
        crop[y - 1:y + 2, x - 1:x + 2, :] = 255

        im = Image.fromarray(crop)
        im = im.resize((256, 256))
        f = 'anno_%05d.jpg' % i
        f = os.path.join(flags.out_video, f)
        im.save(f)


if __name__ == '__main__':
    main(parse_flags())
