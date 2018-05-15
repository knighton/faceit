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

    a.add_argument('--joint_model', type=str,
                   default='data/checkpoints/epoch_0669_0087_0109/model.bin')
    a.add_argument('--batch_size', type=int, default=32)

    a.add_argument('--in_video', type=str,
                   default='/home/frak/dev/commaai/monitoring/video1/')

    a.add_argument('--out_video', type=str, default='data/anno1/')

    return a.parse_args()


def each_frame(video_dir, batch_size):
    # get the list of frames.
    pattern = os.path.join(video_dir, 'frame_*.jpg')
    ff = glob(pattern)

    # read frames from file.
    crops = []
    for i, f in tqdm(enumerate(ff), total=len(ff), leave=False):
        # load the image and scale it down to a 128x128 thumbnail, dropping the
        # left extreme to form a square (the driver is on the far right of
        # image).
        im = Image.open(f)
        dim = 128 * 1152 / 864
        im.thumbnail((dim, dim))
        crop = np.array(im)
        crop = crop[:, -128:, :]
        crops.append(crop)
    crops = np.stack(crops)

    for i in range((len(crops) + batch_size - 1) // batch_size):
        subcrops = crops[i * batch_size:(i + 1) * batch_size]
        if 1 < len(subcrops):
            yield subcrops


def annotate_batch(joint_model, crops):
    # transform the crop to tensor.
    x = crops.astype('float32')
    x /= 127.5
    x -= 1
    x = x.transpose([0, 3, 1, 2])
    x = torch.from_numpy(x).cuda()
    xx = [x]

    # Forward pass.
    is_faces, is_males, poses, bboxes, keypoints = joint_model.forward(xx)

    # Draw the bounding box.
    go = lambda x: x.detach().cpu().numpy()
    for i in range(len(bboxes)):
        min_x, min_y, max_x, max_y = map(int, go(bboxes[i]))
        rect = crops[i, min_y:max_y, min_x:max_x, :]
        confidence = int(256 * go(is_faces[i]))
        crops[i, min_y:max_y, min_x:max_x, :] = \
            np.maximum(rect, rect + confidence)

    # Draw the eye points.
    for i in range(len(bboxes)):
        kp = keypoints[i]
        x = int(kp[0])
        y = int(kp[1])
        crops[i, y - 1:y + 2, x - 1:x + 2, :] = 255
        x = int(kp[2])
        y = int(kp[3])
        crops[i, y - 1:y + 2, x - 1:x + 2, :] = 255

    return crops


def main(flags):
    assert not os.path.exists(flags.out_video)
    os.makedirs(flags.out_video)

    joint_model = torch.load(flags.joint_model).cuda()

    index = 0
    for frame in each_frame(flags.in_video, flags.batch_size):
        # Modify the frame to annotate bounding box, eyes, etc.
        ims = annotate_batch(joint_model, frame)

        # Save to file.
        for i in range(len(ims)):
            im = ims[i]
            im = Image.fromarray(im)
            im = im.resize((256, 256))
            f = 'anno_%05d.jpg' % index
            f = os.path.join(flags.out_video, f)
            im.save(f)
            index += 1


if __name__ == '__main__':
    main(parse_flags())
