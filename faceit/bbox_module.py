import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from blox import conv_bn_pool, ReduceBlock, Flatten


def mean_to_float(x):
    x = x.mean()
    x = x.detach().cpu().numpy()
    x = np.asscalar(x)
    return float(x)


class BBoxModule(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.seq = nn.Sequential(
            conv_bn_pool(3, k),  # 128 to 64.
            ReduceBlock(k),  # To 32.
            ReduceBlock(k),  # To 16.
            ReduceBlock(k),  # To 8.
            ReduceBlock(k),  # To 4.
            ReduceBlock(k),  # To 2.
            ReduceBlock(k),  # To 1.
            Flatten(),
            nn.Linear(k, 4),
        )

    def forward(self, x):
        return self.seq(x)

    def dist_to_loss(self, x):
        return x.abs() / 100

    def get_loss(self, true_points, pred_points):
        assert not true_points.shape[1] % 2
        num_points = true_points.shape[1] // 2
        ret_loss = torch.zeros(true_points.shape[0], num_points)
        ret_dist = torch.zeros(true_points.shape[0], num_points)
        for i in range(num_points):
            truex = true_points[:, i * 2 + 0]
            predx = pred_points[:, i * 2 + 0]
            distx = truex - predx
            truey = true_points[:, i * 2 + 1]
            predy = pred_points[:, i * 2 + 1]
            disty = truey - predy
            dist = (distx * distx + disty * disty) ** 0.5
            ret_dist[:, i] = dist
            ret_loss[:, i] = self.dist_to_loss(dist)
        return ret_loss, ret_dist

    def train_on_batch(self, optimizer, xx, yy_true, train_losses, train_dists):
        optimizer.zero_grad()
        x, = xx
        y_true, = yy_true
        y_pred = self.forward(x)
        yy_pred = [y_pred]
        loss, dist = self.get_loss(y_true, y_pred)
        grad = torch.ones(*loss.shape)
        torch.autograd.backward([loss], [grad])
        optimizer.step()
        loss = mean_to_float(loss)
        dist = mean_to_float(dist)
        train_losses.append(loss)
        train_dists.append(dist)

    def val_on_batch(self, xx, yy_true, val_losses, val_dists):
        x, = xx
        y_true, = yy_true
        y_pred = self.forward(x)
        yy_pred = [y_pred]
        loss, dist = self.get_loss(y_true, y_pred)
        loss = mean_to_float(loss)
        dist = mean_to_float(dist)
        val_losses.append(loss)
        val_dists.append(dist)

    def fit_on_epoch(self, dataset, optimizer, max_batches_per_epoch=None,
                     batch_size=32, save_dir=None, verbose=2, epoch=None):
        each_batch = dataset.each_batch(batch_size, max_batches_per_epoch)
        total = dataset.batches_per_epoch(batch_size, max_batches_per_epoch)
        if 2 <= verbose:
            each_batch = tqdm(each_batch, total=total, leave=False)

        train_losses = []
        val_losses = []
        train_dists = []
        val_dists = []
        demo_xx = []
        demo_yy = []
        for is_training, xx, yy in each_batch:
            x, = xx
            x = x.astype('float32')
            x /= 127.5
            x -= 1
            x = x.transpose([0, 3, 1, 2])
            x = torch.from_numpy(x).cuda()
            xx = [x]

            y, = yy
            y = torch.from_numpy(y).cuda()
            yy = [y]

            if is_training:
                self.train()
                self.train_on_batch(optimizer, xx, yy, train_losses,
                                    train_dists)
            else:
                self.eval()
                self.val_on_batch(xx, yy, val_losses, val_dists)

        print()
        print('-' * 80)
        if epoch is not None:
            print('epoch: %d' % epoch)
        print(np.mean(train_losses))
        print(np.mean(train_dists))
        print(np.mean(val_losses))
        print(np.mean(val_dists))
        print('-' * 80)
        print()

    def fit(self, dataset, optimizer, initial_epoch=0, num_epochs=10,
            max_batches_per_epoch=None, batch_size=32, save_dir=None,
            verbose=2):
        for epoch in range(initial_epoch, num_epochs):
            self.fit_on_epoch(dataset, optimizer, max_batches_per_epoch,
                              batch_size, save_dir, verbose, epoch)
