import torch
from torch import nn
from torch.nn import Conv3d, BatchNorm3d, ReLU
from torch.nn.functional import interpolate

class UNet3D_ConvBlock(nn.Module):
    def __init__(self, filters):
        super(UNet3D_ConvBlock, self).__init__()
        self.seq = nn.Sequential(
                Conv3d(filters[0], filters[1], 3, padding=1),
                BatchNorm3d(filters[1]),
                ReLU(),
                Conv3d(filters[1], filters[2], 3, padding=1),
                BatchNorm3d(filters[2]),
                ReLU(),
                )

    def forward(self, x):
        return self.seq(x)

class UNet3D_UpConv(nn.Module):
    def __init__(self, filters):
        super(UNet3D_UpConv, self).__init__()
        self.conv = Conv3d(filters, filters, 2, padding=1)

    def forward(self, x, size):
        x = interpolate(x, size, mode="trilinear")
        x = self.conv(x)[:,:,1:,1:,1:]
        return x


