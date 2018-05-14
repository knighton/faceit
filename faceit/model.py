import torch
from torch import nn
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
            nn.Linear(k, 1),
            nn.Sigmoid(),
        )

        self.is_male = nn.Sequential(
            nn.Linear(k, 1),
            nn.Sigmoid(),
        )

        self.get_pose = nn.Sequential(
            nn.Linear(k, 3),
        )

        self.get_face_bbox = nn.Sequential(
            nn.Linear(k, 4),
        )

        self.get_keypoints = nn.Sequential(
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

    def train_on_batch(self, optimizer, xx, yy_true):
        optimizer.zero_grad()
        yy_pred = self.forward(xx)

    def val_on_batch(self, xx, yy):
        yy_pred = self.forward(xx)

    def fit_on_epoch(self, dataset, optimizer, max_batches_per_epoch=None,
                     batch_size=32, verbose=2, epoch=None):
        each_batch = dataset.each_batch(batch_size, max_batches_per_epoch)
        total = dataset.batches_per_epoch(batch_size, max_batches_per_epoch)
        if 2 <= verbose:
            each_batch = tqdm(each_batch, total=total, leave=False)

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
                self.train_on_batch(optimizer, xx, yy)
            else:
                self.eval()
                self.val_on_batch(xx, yy)


    def fit(self, dataset, optimizer, initial_epoch=0, num_epochs=10,
            max_batches_per_epoch=None, batch_size=32, verbose=2):
        for epoch in range(initial_epoch, num_epochs):
            self.fit_on_epoch(dataset, optimizer, max_batches_per_epoch,
                              batch_size, verbose, epoch)
