from torch import nn
from torch.nn import Conv3d, BatchNorm3d, ReLU, Sigmoid
import numpy as np
import torch

class RatLesNetv2_ResNet(nn.Module):
    def __init__(self, in_filters, conv_num=2):
        super(RatLesNetv2_ResNet, self).__init__()

        self.seq = nn.Sequential(
                ReLU(),
                BatchNorm3d(in_filters),
                Conv3d(in_filters, in_filters, 3, padding=1),
                ReLU(),
                BatchNorm3d(in_filters),
                Conv3d(in_filters, in_filters, 3, padding=1)
            )

    def forward(self, x):
        return x + self.seq(x)

    def __str__(self):
        return "RatLesNetv2_ResNet"

class RatLesNetv2_Bottleneck(nn.Module):
    def __init__(self, in_filters, out_filters):
        super(RatLesNetv2_Bottleneck, self).__init__()

        self.seq = nn.Sequential(
                ReLU(),
                BatchNorm3d(in_filters),
                Conv3d(in_filters, out_filters, 1)
            )

    def forward(self, x):
        return self.seq(x)

    def __str__(self):
        return "RatLesNetv2_Bottleneck"


class RatLesNetv2_DenseNet(nn.Module):
    def __init__(self, in_filters):
        super(RatLesNetv2_DenseNet, self).__init__()

        self.seq = []
        self.seq.append(nn.Sequential(
                ReLU(),
                BatchNorm3d(in_filters),
                Conv3d(in_filters, in_filters, 3, padding=1),
            ))
        self.seq.append(nn.Sequential(
                ReLU(),
                BatchNorm3d(in_filters*2),
                Conv3d(in_filters*2, in_filters, 3, padding=1),
            ))
        # Compression
        self.seq.append(nn.Sequential(
                BatchNorm3d(in_filters*3),
                Conv3d(in_filters*3, in_filters, 1),
            ))

        self.seq = nn.ModuleList(self.seq)

    def forward(self, x):
        x1 = self.seq[0](x)
        x2 = self.seq[1](torch.cat([x, x1], dim=1))
        x3 = self.seq[2](torch.cat([x, x1, x2], dim=1))

        return x3

    def __str__(self):
        return "RatLesNetv2_DenseNet"


class RatLesNetv2_ResNet_v2(nn.Module):
    def __init__(self, in_filters, conv_num):
        super(RatLesNetv2_ResNet_v2, self).__init__()
        self.conv_num = conv_num

        self.seq = []
        for i in range(conv_num):
            self.seq.append(nn.Sequential(
                    ReLU(),
                    BatchNorm3d(in_filters),
                    Conv3d(in_filters, in_filters, 3, padding=1),
                ))
        self.seq = nn.ModuleList(self.seq)

    def forward(self, x):
        for i in range(self.conv_num):
            x = x + self.seq[i](x)

        return x

    def __str__(self):
        return "RatLesNetv2_ResNet_v2"


class RatLesNetv2_SE1(nn.Module):
    def __init__(self, in_filters, bottleneck_filters):
        super(RatLesNetv2_SE1, self).__init__()

        self.seq = nn.Sequential(
                ReLU(),
                Conv3d(in_filters, bottleneck_filters, 1),
                ReLU(),
                Conv3d(bottleneck_filters, in_filters, 1),
                Sigmoid()
            )

    def forward(self, x):
        return self.seq(x)

    def __str__(self):
        return "RatLesNetv2_SE1"


