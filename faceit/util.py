import numpy as np
import os


def max_pool_2d_nchw(x, k):
    if isinstance(k, int):
        kh = k
        kw = k
    else:
        kh, kw = k
    n, c, h, w = x.shape
    ih = h - h % kh
    iw = w - w % kw
    oh = h // kh
    ow = w // kw
    x = x[:, :, :ih, :iw]
    x = x.reshape(n, c, oh, kh, ow, kw)
    return x.max(axis=(3, 5))


def load_dataset(proc_dir):
    crops_fn = 'crops.npy'
    crops_fn = os.path.join(proc_dir, crops_fn)
    crops = np.fromfile(crops_fn, 'uint8')
    crops = crops.reshape((-1, 128, 128, 3))

    infos_fn = 'infos.npy'
    infos_fn = os.path.join(proc_dir, infos_fn)
    infos = np.fromfile(infos_fn, 'float32')
    infos = infos.reshape((-1, 50))

    return crops, infos
