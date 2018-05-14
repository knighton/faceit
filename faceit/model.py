import numpy as np
import os
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from blox import \
    conv_bn, conv_bn_pool, dense_bn, IsoConvBlock, IsoDenseBlock, ReduceBlock, \
    Flatten, Scale, Degrees


def mean_squared_error(true, pred):
    x = true - pred
    return x * x


def binary_cross_entropy(true, pred):
    pred = pred.clamp(1e-5, 1 - 1e-5)
    return -true * pred.log() - (1 - true) * (1 - pred).log()


def binary_accuracy(true, pred):
    true = 0.5 < true
    pred = 0.5 < pred
    return (true == pred).type(torch.float32).mean()


def dist_to_loss(dist):
    return dist / 32


def compare_points(true_points, pred_points):
    assert not true_points.shape[1] % 2
    num_points = true_points.shape[1] // 2
    ret_loss = torch.zeros(true_points.shape[0], num_points)
    ret_dist = torch.zeros(true_points.shape[0], num_points)
    for i in range(num_points):
        truex = true_points[:, i * 2 + 0].clamp(-64, 196)
        predx = pred_points[:, i * 2 + 0].clamp(-64, 196)
        distx = truex - predx
        truey = true_points[:, i * 2 + 1].clamp(-64, 196)
        predy = pred_points[:, i * 2 + 1].clamp(-64, 196)
        disty = truey - predy
        dist = (distx * distx + disty * disty) ** 0.5
        ret_dist[:, i] = dist
        ret_loss[:, i] = dist_to_loss(dist)
    return ret_loss, ret_dist


def mean_to_float(x):
    x = x.mean()
    x = x.detach().cpu().numpy()
    x = np.asscalar(x)
    return float(x)


class Model(nn.Module):
    def __init__(self, k):
        super().__init__()

        self.features = nn.Sequential(
            conv_bn_pool(3, k),

            IsoConvBlock(k),
            IsoConvBlock(k),

            ReduceBlock(k),

            IsoConvBlock(k),
            IsoConvBlock(k),

            ReduceBlock(k),

            IsoConvBlock(k),
            IsoConvBlock(k),

            ReduceBlock(k),

            IsoConvBlock(k),
            IsoConvBlock(k),

            ReduceBlock(k),

            IsoConvBlock(k),
            IsoConvBlock(k),

            ReduceBlock(k),

            IsoConvBlock(k),
            IsoConvBlock(k),
            IsoConvBlock(k),

            ReduceBlock(k),

            IsoConvBlock(k),
            IsoConvBlock(k),
            IsoConvBlock(k),

            Flatten(),

            IsoDenseBlock(k),
            IsoDenseBlock(k),
            IsoDenseBlock(k),
            IsoDenseBlock(k),
        )

        self.is_face = nn.Sequential(
            nn.Linear(k, 1),
            nn.Sigmoid(),
        )

        self.is_male = nn.Sequential(
            nn.Linear(k, 1),
            nn.Sigmoid(),
        )

        self.get_pose = nn.Sequential(
            IsoDenseBlock(k),
            nn.Linear(k, 3),
            Degrees(),
        )

        self.get_face_bbox = nn.Sequential(
            IsoDenseBlock(k),
            nn.Linear(k, 4),
        )

        self.get_keypoints = nn.Sequential(
            IsoDenseBlock(k),
            nn.Linear(k, 4),
        )

    def forward(self, xx):
        crops, = xx
        ff = self.features(crops)
        is_faces = self.is_face(ff)
        is_males = self.is_male(ff)
        poses = self.get_pose(ff)
        bboxes = self.get_face_bbox(ff)
        keypoints = self.get_keypoints(ff)

        return is_faces, is_males, poses, bboxes, keypoints

    def get_faceness_loss(self, true, pred):
        return binary_cross_entropy(true, pred)

    def get_gender_loss(self, true, pred):
        loss = binary_cross_entropy(true, pred)
        acc = binary_accuracy(true, pred)
        return loss, acc

    def get_pose_loss(self, true, pred):
        return (true - pred).abs() / 32

    def get_bbox_loss(self, true, pred):
        return compare_points(true, pred)

    def get_keypoint_loss(self, true, pred):
        return compare_points(true, pred)

    def get_loss(self, yy_true, yy_pred):
        true_is_faces, true_is_males, true_poses, true_bboxes, \
            true_keypoints = yy_true
        pred_is_faces, pred_is_males, pred_poses, pred_bboxes, \
            pred_keypoints = yy_pred
        faceness_loss = self.get_faceness_loss(true_is_faces, pred_is_faces)
        gender_loss, gender_acc = \
            self.get_gender_loss(true_is_males, pred_is_males)
        pose_loss = self.get_pose_loss(true_poses, pred_poses)
        bbox_loss, bbox_dist = self.get_bbox_loss(true_bboxes, pred_bboxes)
        keypoint_loss, keypoint_dist = \
            self.get_keypoint_loss(true_keypoints, pred_keypoints)
        losses = faceness_loss, gender_loss, pose_loss, bbox_loss, keypoint_loss
        extras = gender_acc, bbox_dist, keypoint_dist
        return losses, extras

    def train_on_batch(self, optimizer, xx, yy_true, loss_lists, extra_lists):
        optimizer.zero_grad()
        yy_pred = self.forward(xx)
        losses, extras = self.get_loss(yy_true, yy_pred)
        grads = [torch.ones(*x.shape).cuda() for x in losses]
        torch.autograd.backward(losses, grads)
        optimizer.step()
        losses = [mean_to_float(x) for x in losses]
        extras = map(mean_to_float, extras)
        for i, loss in enumerate(losses):
            loss_lists[i].append(loss)
        for i, extra in enumerate(extras):
            extra_lists[i].append(extra)

    def val_on_batch(self, xx, yy_true, loss_lists, extra_lists):
        yy_pred = self.forward(xx)
        losses, extras = self.get_loss(yy_true, yy_pred)
        losses = [mean_to_float(x) for x in losses]
        extras = map(mean_to_float, extras)
        for i, loss in enumerate(losses):
            loss_lists[i].append(loss)
        for i, extra in enumerate(extras):
            extra_lists[i].append(extra)

    def fit_on_epoch(self, dataset, optimizer, max_batches_per_epoch=None,
                     batch_size=32, save_dir=None, verbose=2, epoch=None):
        each_batch = dataset.each_batch(batch_size, max_batches_per_epoch)
        total = dataset.batches_per_epoch(batch_size, max_batches_per_epoch)
        if 2 <= verbose:
            each_batch = tqdm(each_batch, total=total, leave=False)

        train_loss_lists = [[] for i in range(5)]
        train_extra_lists = [[] for i in range(3)]
        val_loss_lists = [[] for i in range(5)]
        val_extra_lists = [[] for i in range(3)]
        demo_xx = []
        demo_yy = []
        for is_training, xx, yy in each_batch:
            if not is_training and np.random.uniform() < 0.2:
                demo_xx.append(xx)
                demo_yy.append(yy)

            x, = xx
            x = x.astype('float32')
            x /= 127.5
            x -= 1
            x = x.transpose([0, 3, 1, 2])
            x = torch.from_numpy(x).cuda()
            xx = [x]

            is_face, is_male, pose, bbox, keypoint = yy
            is_face = torch.from_numpy(is_face).cuda()
            is_male = torch.from_numpy(is_male).cuda()
            pose = torch.from_numpy(pose).cuda()
            bbox = torch.from_numpy(bbox).cuda()
            keypoint = torch.from_numpy(keypoint).cuda()
            yy = is_face, is_male, pose, bbox, keypoint

            if is_training:
                self.train()
                self.train_on_batch(optimizer, xx, yy, train_loss_lists,
                                    train_extra_lists)
            else:
                self.eval()
                self.val_on_batch(xx, yy, val_loss_lists, val_extra_lists)

        print()
        print('-' * 80)
        if epoch is not None:
            print('epoch: %d' % epoch)

        names = 'faceness', 'gender', 'pose', 'bbox', 'keypoints'
        train_losses = tuple(map(np.mean, train_loss_lists))
        train_extras = tuple(map(np.mean, train_extra_lists))
        val_losses = tuple(map(np.mean, val_loss_lists))
        val_extras = tuple(map(np.mean, val_extra_lists))
        for i in range(5):
            print('loss (%s): %.3f %.3f' %
                  (names[i], train_losses[i], val_losses[i]))

        names = 'gender accuracy', 'avg face bbox dist', 'avg eye dist'
        for i in range(3):
            print('%s: %.3f %.3f' % (names[i], train_extras[i], val_extras[i]))

        print('-' * 80)
        print()

        print('^' * 40)
        tt = []
        ww = []
        for layer in self.features:
            if isinstance(layer, ReduceBlock):
                tt.append('reduce-conv')
                w = layer.weights
                ww.append(w)
            elif isinstance(layer, IsoConvBlock):
                tt.append('iso-conv')
                w = layer.weights
                ww.append(w)
            elif isinstance(layer, IsoDenseBlock):
                tt.append('iso-dense')
                w = layer.weights
                ww.append(w)
        for t, w in zip(tt, ww):
            w = w.detach().cpu().numpy()
            w = list(map(float, w))
            print('[%s]' % t, ' '.join(map(lambda x: '%.3f' % x, w)))
        print('^' * 40)

        if save_dir:
            t = np.mean(train_losses)
            v = np.mean(val_losses)
            d = 'epoch_%04d_%04d_%04d' % (epoch, int(t * 1000), int(v * 1000))
            print('>', d)
            d = os.path.join(save_dir, d)
            os.makedirs(d)

            f = 'model.bin'
            f = os.path.join(d, f)
            torch.save(self, f)

            self.eval()
            n = len(demo_xx)
            print('demo batches:', n)
            for i in range(n):
                xx = demo_xx[i]
                yy = demo_yy[i]

                x, = xx
                crop = x.copy()
                x = x.astype('float32')
                x /= 127.5
                x -= 1
                x = x.transpose([0, 3, 1, 2])
                x = torch.from_numpy(x).cuda()
                xx = [x]

                is_face, is_male, pose, bbox, keypoint = yy
                is_face = torch.from_numpy(is_face).cuda()
                is_male = torch.from_numpy(is_male).cuda()
                pose = torch.from_numpy(pose).cuda()
                bbox = torch.from_numpy(bbox).cuda()
                keypoint = torch.from_numpy(keypoint).cuda()
                yy = is_face, is_male, pose, bbox, keypoint

                yy_pred = self.forward(xx)
                pred_bbox = yy_pred[3].detach().cpu().numpy()
                pred_bbox = pred_bbox.clip(0, 127).astype('int32')
                pred_keypoint = yy_pred[4].detach().cpu().numpy()
                pred_keypoint = pred_keypoint.clip(0, 127).astype('int32')

                for j in range(len(pred_bbox)):
                    minx, miny, maxx, maxy = pred_bbox[j]
                    crop[j, miny, minx, 1] = 255
                    crop[j, maxy, maxx, 2] = 255
                    if minx < maxx and miny < maxy:
                        crop[j, miny:maxy, minx:maxx, :] //= 2

                for j in range(2):
                    x = pred_keypoint[:, j * 2]
                    y = pred_keypoint[:, j * 2 + 1]
                    for k in range(len(pred_keypoint)):
                        if 0 <= y[k] < 128 and 0 <= x[k] < 128:
                            crop[k, y[k] - 2:y[k] + 1,
                                 x[k] - 2:x[k] + 1, :] = 255

                for j in range(len(pred_keypoint)):
                    im = Image.fromarray(crop[j])
                    f = 'demo_%04d_%04d.jpg' % (i, j)
                    f = os.path.join(d, f)
                    im.save(f)

    def fit(self, dataset, optimizer, initial_epoch=0, num_epochs=10,
            max_batches_per_epoch=None, batch_size=32, save_dir=None,
            verbose=2):
        for epoch in range(initial_epoch, num_epochs):
            self.fit_on_epoch(dataset, optimizer, max_batches_per_epoch,
                              batch_size, save_dir, verbose, epoch)
