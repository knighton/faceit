import torch
from torch import nn
from torch.nn import functional as F


conv_bn = lambda in_dim, out_dim: nn.Sequential(
    nn.Conv2d(in_dim, out_dim, 5, 1, 2),
    nn.BatchNorm2d(out_dim),
    nn.ReLU(),
)


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


class IsoConvBlock(nn.Module):
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

    def __init__(self, k):
        super().__init__()
        self.k = k

        self.conv = nn.Conv2d(k, 3 * k, 5, 1, 2)
        self.bn1 = nn.BatchNorm2d(k)
        self.bn2 = nn.BatchNorm2d(k)

        weights = torch.FloatTensor([1, 0, 0])
        self.weights = nn.Parameter(weights)

    def forward(self, x):
        t = self.conv(x)
        k = self.k
        conv = t[:, :k, :, :].clone()
        conv = self.bn1(conv).clamp(min=0)
        gate = t[:, k:2 * k, :, :].clone()
        gate *= t[:, 2 * k:, :, :].sigmoid()
        gate = self.bn2(gate).clamp(min=0)
        w = self.weights
        return w[0] * x + w[1] * conv + w[2] * gate


class IsoDenseBlock(nn.Module):
    """
    A basic  dense block.

    Contains four paths, each multiplied by a learned weight:
    - Skip connection
    - Dense
    - Gated dense #1
    - Gated dense #2

    Initialized to using just the skip connection.  At worst, we waste some
    computation and the information just passes through the skip, but normally
    it balances them against each other.
    """

    def __init__(self, k):
        super().__init__()
        self.k = k

        self.dense = nn.Linear(k, 5 * k)
        self.bn1 = nn.BatchNorm1d(k)
        self.drop1 = nn.Dropout(0.1)
        self.bn2 = nn.BatchNorm1d(k)
        self.drop2 = nn.Dropout(0.1)
        self.bn3 = nn.BatchNorm1d(k)
        self.drop3 = nn.Dropout(0.1)

        weights = torch.FloatTensor([1, 0, 0, 0])
        self.weights = nn.Parameter(weights)

    def forward(self, x):
        t = self.dense(x)
        k = self.k

        one = t[:, k * 0:k * 1].clone()
        one = self.bn1(one)
        one = self.drop1(one)
        one = one.clamp(min=0)

        two = t[:, k * 1:k * 2] * t[:, k * 2:k * 3].sigmoid()
        two = self.bn2(two)
        two = self.drop2(two)
        two = two.clamp(min=0)

        three = t[:, k * 3:k * 4] * t[:, k * 4:k * 5].sigmoid()
        three = self.bn3(three)
        three = self.drop3(three)
        three = three.clamp(min=0)

        w = self.weights
        return w[0] * x + w[1] * one + w[2] * two + w[3] * three


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

    def __init__(self, k):
        super().__init__()
        self.k = k

        self.avg_pool = nn.AvgPool2d(2)
        self.max_pool = nn.MaxPool2d(2)
        self.conv = nn.Conv2d(k, 3 * k, 5, 2, 2)
        self.bn1 = nn.BatchNorm2d(k)
        self.bn2 = nn.BatchNorm2d(k)

        weights = torch.FloatTensor([0.5, 0.5, 0, 0])
        self.weights = nn.Parameter(weights)

    def forward(self, x):
        avg_pool = self.avg_pool(x)
        max_pool = self.max_pool(x)
        t = self.conv(x)
        k = self.k
        conv = t[:, :k, :, :].clone()
        conv = self.bn1(conv).clamp(min=0)
        gate = t[:, k:2 * k, :, :] * t[:, 2 * k:, :, :].sigmoid()
        gate = gate.clone()
        gate = self.bn2(gate).clamp(min=0)
        w = self.weights
        return w[0] * avg_pool + w[1] * max_pool + w[2] * conv + w[3] * gate


class Flatten(nn.Module):
    """
    Flattens the input tensor, returning a view.
    """

    def forward(self, x):
        return x.view(x.shape[0], -1)


class Scale(nn.Module):
    def __init__(self, mul, add):
        super().__init__()
        self.mul = mul
        self.add = add

    def forward(self, x):
        return x * self.mul + self.add


class Degrees(nn.Module):
    def forward(self, x):
        return F.tanh(x / 128) * 180
