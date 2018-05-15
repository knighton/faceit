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
            conv_bn_pool(3, k),
            ReduceBlock(k),
            ReduceBlock(k),
            IsoConvBlock(k),
            conv_t_bn(k, k),
            conv_t_bn(k, k),
            conv_t_bn(k, 1),
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

        train_losses = []
        val_losses = []
        for is_training, xx, yy in each_batch:
            x, = xx
            x = x.astype('float32')

            # Randomly darken the input images.
            x *= np.random.uniform(0.1, 1)

            x /= 127.5
            x -= 1
            x = x.transpose([0, 3, 1, 2])
            x = torch.from_numpy(x).cuda()

            y, = yy
            y = y.astype('float32')
            y = torch.from_numpy(y).cuda()

            if is_training:
                self.train()
                self.train_on_batch(optimizer, x, y, train_losses)
            else:
                self.eval()
                self.val_on_batch(x, y, val_losses)

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        print('%4d %.3f %.3f' % (epoch, train_loss, val_loss))

    def fit(self, dataset, optimizer, initial_epoch=0, num_epochs=10,
            max_batches_per_epoch=None, batch_size=32, save_dir=None,
            verbose=2):
        for epoch in range(initial_epoch, num_epochs):
            self.fit_on_epoch(dataset, optimizer, max_batches_per_epoch,
                              batch_size, save_dir, verbose, epoch)
