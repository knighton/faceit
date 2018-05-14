import torch
from torch import nn


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

        weights = torch.cuda.FloatTensor([1, 0, 0])
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

        weights = torch.cuda.FloatTensor([0.5, 0.5, 0, 0])
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
            ReduceBlock(k),  # 64 -> 32.
            IsoBlock(k),
            ReduceBlock(k),  # 32 -> 16.
            IsoBlock(k),
            ReduceBlock(k),  # 16 -> 8.
            IsoBlock(k),
            ReduceBlock(k),  # 8 -> 4.
            IsoBlock(k),
            IsoBlock(k),
            ReduceBlock(k),  # 4 -> 2.
            IsoBlock(k),
            IsoBlock(k),
            IsoBlock(k),
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

    def forward(self, crops):
        ff = self.features(crops)
        is_faces = self.is_face(ff)
        is_males = self.is_male(ff)
        poses = self.get_pose(ff)
        bboxes = self.get_face_bbox(ff)
        keypoints = self.get_keypoints(ff)
        return is_faces, is_males, poses, bboxes, keypoints
