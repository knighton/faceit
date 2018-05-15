import numpy as np
import os
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from blox import conv_bn_pool, conv_t_bn, IsoConvBlock, ReduceBlock


def mean_to_float(x):
    x = x.mean()
    x = x.detach().cpu().numpy()
    x = np.asscalar(x)
    return float(x)


class HeatmapperModel(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.seq = nn.Sequential(
            conv_bn_pool(3, k),  # 128 -> 64.
            ReduceBlock(k),  # 64 -> 32.
            ReduceBlock(k),  # 32 -> 16.
            ReduceBlock(k),  # 16 -> 8.

            IsoConvBlock(k),
            IsoConvBlock(k),
            IsoConvBlock(k),

            conv_t_bn(k, k),  # 8 -> 16.
            conv_t_bn(k, k),  # 16 -> 32.
            conv_t_bn(k, k),  # 32 -> 64.
            conv_t_bn(k, 1),  # 64 -> 128.
        )

    def forward(self, clips):
        return self.seq(clips)

    def get_loss(self, true, pred):
        pred = pred.clamp(1e-5, 1 - 1e-5)
        return -true * pred.log() - (1 - true) * (1 - pred).log()

    def train_on_batch(self, optimizer, x, y_true, train_losses):
        optimizer.zero_grad()
        y_pred = self.forward(x)
        loss = self.get_loss(y_true, y_pred)
        grad = torch.ones(*loss.shape).cuda()
        torch.autograd.backward([loss], [grad])
        optimizer.step()
        loss = mean_to_float(loss)
        train_losses.append(loss)

    def val_on_batch(self, x, y_true, val_losses):
        y_pred = self.forward(x)
        loss = self.get_loss(y_true, y_pred)
        loss = mean_to_float(loss)
        val_losses.append(loss)

    def fit_on_epoch(self, dataset, optimizer, max_batches_per_epoch=None,
                     batch_size=32, save_dir=None, verbose=2, epoch=None):
        each_batch = dataset.each_batch(batch_size, max_batches_per_epoch)
        total = dataset.batches_per_epoch(batch_size, max_batches_per_epoch)
        if 2 <= verbose:
            each_batch = tqdm(each_batch, total=total, leave=False)

        demo_orig_xx = []
        demo_torch_xx = []
        demo_torch_yy = []
        train_losses = []
        val_losses = []
        for is_training, xx, yy in each_batch:
            orig_x, = xx
            x = orig_x.astype('float32')

            # Randomly darken the input images to better handle nighttime.
            x *= np.random.uniform(0.1, 1)

            x /= 127.5
            x -= 1
            x = x.transpose([0, 3, 1, 2])
            x = torch.from_numpy(x).cuda()

            y, = yy
            y = y.astype('float32')
            y = torch.from_numpy(y).cuda()

            if not is_training and np.random.uniform() < 0.2:
                demo_orig_xx.append(orig_x)
                demo_torch_xx.append(x)
                demo_torch_yy.append(y)

            if is_training:
                self.train()
                self.train_on_batch(optimizer, x, y, train_losses)
            else:
                self.eval()
                self.val_on_batch(x, y, val_losses)

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        print('%4d %.3f %.3f' % (epoch, train_loss, val_loss))

        if save_dir:
            t = int(train_loss * 1000)
            v = int(val_loss * 1000)
            d = 'epoch_%04d_%04d_%04d' % (epoch, t, v)
            print('>', d)
            d = os.path.join(save_dir, d)
            os.makedirs(d)

            f = 'model.bin'
            f = os.path.join(d, f)
            torch.save(self, f)

            self.eval()
            n = len(demo_orig_xx)
            print('demo batches:', n)
            for i in range(n):
                crop = demo_orig_xx[i]
                torch_x = demo_torch_xx[i]
                torch_y_true = demo_torch_yy[i]
                torch_y_pred = self.forward(torch_x)
                y_pred = torch_y_pred.detach().cpu().numpy()
                heatmaps = y_pred.transpose([0, 2, 3, 1])
                heatmaps = heatmaps.repeat(3, 3)
                heatmaps *= 255
                for j in range(len(crop)):
                    heatmap = heatmaps[j]
                    if heatmap.max():
                        heatmap /= heatmap.max()
                    heatmap = heatmap.astype('uint8')
                    canvas = crop[j] // 4
                    canvas = (canvas.astype('uint16') +
                              heatmap.astype('uint16')).astype('uint8')
                    im = Image.fromarray(canvas)
                    index = i * batch_size + j
                    f = 'demo_%05d.jpg' % index
                    f = os.path.join(d, f)
                    im.save(f)

    def fit(self, dataset, optimizer, initial_epoch=0, num_epochs=10,
            max_batches_per_epoch=None, batch_size=32, save_dir=None,
            verbose=2):
        for epoch in range(initial_epoch, num_epochs):
            self.fit_on_epoch(dataset, optimizer, max_batches_per_epoch,
                              batch_size, save_dir, verbose, epoch)
