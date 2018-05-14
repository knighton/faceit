import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm


conv_bn_pool = lambda in_dim, out_dim: nn.Sequential(
    nn.Conv2d(in_dim, out_dim, 5, 2, 2),
    nn.BatchNorm2d(out_dim),
    nn.ReLU(),
)


dense_bn = lambda in_dim, out_dim: nn.Sequential(
    nn.Linear(in_dim, out_dim),
    nn.BatchNorm1d(out_dim),
    nn.ReLU(),
    nn.Dropout(),
)


class IsoBlock(nn.Module):
    """
    A basic spatial shape-preserving block.

    Contains three paths, each multiplied by a learned weight:
    - Skip connection
    - 5x5 conv
    - Gated 5x5 conv

    Initialized to using just the skip connection.  At worst, we waste some
    computation and the information just passes through the skip, but normally
    it balances them against each other.
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv = nn.Conv2d(in_dim, 3 * out_dim, 5, 1, 2)
        self.bn1 = nn.BatchNorm2d(out_dim)
        self.drop1 = nn.Dropout2d()
        self.bn2 = nn.BatchNorm2d(out_dim)
        self.drop2 = nn.Dropout2d()

        weights = torch.FloatTensor([1, 0, 0])
        self.weights = nn.Parameter(weights)

    def forward(self, x):
        t = self.conv(x)
        k = self.out_dim
        conv = t[:, :k, :, :].clone()
        conv = self.bn1(conv).clamp(min=0)
        conv = self.drop1(conv)
        gate = t[:, k:2 * k, :, :].clone()
        gate *= t[:, 2 * k:, :, :].sigmoid()
        gate = self.bn2(gate).clamp(min=0)
        gate = self.drop2(gate)
        w = self.weights
        return w[0] * x + w[1] * conv + w[2] * gate


class ReduceBlock(nn.Module):
    """
    A basic 2x2 pooling block.

    Contains four paths, each multiplied by a learned weight:
    - Average pooling
    - Max pooling
    - Strided 5x5 conv
    - Gated strided 5x5 conv

    Initialized to half avg pooling, half max pooling.  At worst, it just does
    pooling.
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.avg_pool = nn.AvgPool2d(2)
        self.max_pool = nn.MaxPool2d(2)
        self.conv = nn.Conv2d(in_dim, 3 * out_dim, 5, 2, 2)
        self.bn1 = nn.BatchNorm2d(out_dim)
        self.drop1 = nn.Dropout2d()
        self.bn2 = nn.BatchNorm2d(out_dim)
        self.drop2 = nn.Dropout2d()

        weights = torch.FloatTensor([0.5, 0.5, 0, 0])
        self.weights = nn.Parameter(weights)

    def forward(self, x):
        avg_pool = self.avg_pool(x)
        max_pool = self.max_pool(x)
        t = self.conv(x)
        k = self.out_dim
        conv = t[:, :k, :, :].clone()
        conv = self.bn1(conv).clamp(min=0)
        conv = self.drop1(conv)
        gate = t[:, k:2 * k, :, :] * t[:, 2 * k:, :, :].sigmoid()
        gate = gate.clone()
        gate = self.bn2(gate).clamp(min=0)
        gate = self.drop2(gate)
        w = self.weights
        return w[0] * avg_pool + w[1] * max_pool + w[2] * conv + w[3] * gate


class Flatten(nn.Module):
    """
    Flattens the input tensor, returning a view.
    """

    def forward(self, x):
        return x.view(x.shape[0], -1)


class Scale(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return F.tanh(x / self.scale) * self.scale


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
    return (dist / 32).tanh()


def compare_points(true_points, pred_points):
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
            conv_bn_pool(3, k),  # 128 -> 64.
            ReduceBlock(k, k),  # 64 -> 32.
            IsoBlock(k, k),
            ReduceBlock(k, k),  # 32 -> 16.
            IsoBlock(k, k),
            ReduceBlock(k, k),  # 16 -> 8.
            IsoBlock(k, k),
            ReduceBlock(k, k),  # 8 -> 4.
            IsoBlock(k, k),
            IsoBlock(k, k),
            ReduceBlock(k, k),  # 4 -> 2.
            IsoBlock(k, k),
            IsoBlock(k, k),
            IsoBlock(k, k),
            Flatten(),
            dense_bn(2 * 2 * k, 2 * 2 * k),
            dense_bn(2 * 2 * k, 2 * 2 * k),
            dense_bn(2 * 2 * k, k),
        )

        self.is_face = nn.Sequential(
            dense_bn(k, k),
            nn.Linear(k, 1),
            nn.Sigmoid(),
        )

        self.is_male = nn.Sequential(
            dense_bn(k, k),
            nn.Linear(k, 1),
            nn.Sigmoid(),
        )

        self.get_pose = nn.Sequential(
            dense_bn(k, k),
            nn.Linear(k, 3),
            Scale(90),
        )

        self.get_face_bbox = nn.Sequential(
            dense_bn(k, k),
            nn.Linear(k, 4),
        )

        self.get_keypoints = nn.Sequential(
            dense_bn(k, k),
            nn.Linear(k, 21 * 2),
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

    def get_loss(self, yy_true, yy_pred):
        true_is_faces, true_is_males, true_poses, true_bboxes, \
            true_keypoints = yy_true
        pred_is_faces, pred_is_males, pred_poses, pred_bboxes, \
            pred_keypoints = yy_pred
        faceness_loss = mean_squared_error(true_is_faces, pred_is_faces)
        gender_loss = binary_cross_entropy(true_is_males, pred_is_males)
        gender_acc = binary_accuracy(true_is_males, pred_is_males)
        pose_loss = mean_squared_error(true_poses, pred_poses)
        bbox_loss, bbox_dist = compare_points(true_bboxes, pred_bboxes)
        keypoint_loss, keypoint_dist = \
            compare_points(true_keypoints, pred_keypoints)
        losses = faceness_loss, gender_loss, pose_loss, bbox_loss, keypoint_loss
        extras = gender_acc, bbox_dist, keypoint_dist
        return losses, extras

    def train_on_batch(self, optimizer, xx, yy_true, loss_lists, extra_lists):
        optimizer.zero_grad()
        yy_pred = self.forward(xx)
        losses, extras = self.get_loss(yy_true, yy_pred)
        grads = [torch.ones(*x.shape) for x in losses]
        torch.autograd.backward(losses, grads)
        optimizer.step()
        losses = [mean_to_float(x) for x in losses]
        extras = map(mean_to_float, extras)
        for i, loss in enumerate(losses):
            loss_lists[i].append(loss)
        for i, extra in enumerate(extras):
            extra_lists[i].append(extra)
        """
        print()
        print(losses)
        print(gender_acc)
        print(bbox_dist)
        print(keypoint_dist)
        print()
        """

    def val_on_batch(self, xx, yy_true, loss_lists, extra_lists):
        yy_pred = self.forward(xx)
        losses, extras = self.get_loss(yy_true, yy_pred)
        losses = [mean_to_float(x) for x in losses]
        extras = map(mean_to_float, extras)
        for i, loss in enumerate(losses):
            loss_lists[i].append(loss)
        for i, extra in enumerate(extras):
            extra_lists[i].append(extra)
        """
        print()
        print(losses)
        print(gender_acc)
        print(bbox_dist)
        print(keypoint_dist)
        print()
        """

    def fit_on_epoch(self, dataset, optimizer, max_batches_per_epoch=None,
                     batch_size=32, verbose=2, epoch=None):
        each_batch = dataset.each_batch(batch_size, max_batches_per_epoch)
        total = dataset.batches_per_epoch(batch_size, max_batches_per_epoch)
        if 2 <= verbose:
            each_batch = tqdm(each_batch, total=total, leave=False)

        train_loss_lists = [[] for i in range(5)]
        train_extra_lists = [[] for i in range(3)]
        val_loss_lists = [[] for i in range(5)]
        val_extra_lists = [[] for i in range(3)]
        for is_training, xx, yy in each_batch:
            x, = xx
            x = x.astype('float32')
            x /= 127.5
            x -= 1
            x = x.transpose([0, 3, 1, 2])
            x = torch.from_numpy(x)
            xx = [x]

            is_face, is_male, pose, bbox, keypoint = yy
            is_face = torch.from_numpy(is_face)
            is_male = torch.from_numpy(is_male)
            pose = torch.from_numpy(pose)
            bbox = torch.from_numpy(bbox)
            keypoint = torch.from_numpy(keypoint)
            yy = is_face, is_male, pose, bbox, keypoint

            if is_training:
                self.train()
                self.train_on_batch(optimizer, xx, yy, train_loss_lists,
                                    train_extra_lists)
            else:
                self.eval()
                self.val_on_batch(xx, yy, val_loss_lists, val_extra_lists)

        names = 'faceness', 'gender', 'pose', 'bbox', 'keypoints'
        train_losses = tuple(map(np.mean, train_loss_lists))
        train_extras = tuple(map(np.mean, train_extra_lists))
        val_losses = tuple(map(np.mean, val_loss_lists))
        val_extras = tuple(map(np.mean, val_extra_lists))
        print('+' * 40)
        for i in range(5):
            print('loss (%s): %.3f %.3f' %
                  (names[i], train_losses[i], val_losses[i]))

        names = 'gender accuracy', 'avg face bbox dist', 'avg eye dist'
        for i in range(3):
            print('%s: %.3f %.3f' % (names[i], train_extras[i], val_extras[i]))
        print('+' * 40)

    def fit(self, dataset, optimizer, initial_epoch=0, num_epochs=10,
            max_batches_per_epoch=None, batch_size=32, verbose=2):
        for epoch in range(initial_epoch, num_epochs):
            self.fit_on_epoch(dataset, optimizer, max_batches_per_epoch,
                              batch_size, verbose, epoch)
