import torch
from torch import nn
from torch.nn import Conv3d, BatchNorm3d, ReLU
import random, time
import numpy as np

"""
class MyUnpooling(nn.Module):
    def __init__(self, kernel_size):
        super(MyUnpooling, self).__init__()
        self.unpool = nn.modules.MaxUnpool3d(kernel_size)

    def forward(self, input, indices, output_size):
        return self.unpool(input, indices, output_size)

class ConcatChannels(nn.Module):
    def __init__(self):
        super(ConcatChannels, self).__init__()

    def forward(self, l):
        return torch.cat(l, dim=1)
"""

class Bottleneck3d(nn.Module):
    def __init__(self, in_channels, out_channels, nonlinearity=None):
        super(Bottleneck3d, self).__init__()
        if nonlinearity != None:
            self.conv = nn.Sequential(
                Conv3d(in_channels, out_channels, 1),
                nonlinearity
            )
        else:
            self.conv = nn.Sequential(
                Conv3d(in_channels, out_channels, 1),
            )

    def forward(self, x):
        return self.conv(x)

    def __str__(self):
        return "Bottleneck3d"

class VoxResNet_ResBlock(nn.Module):
    def __init__(self):
        super(VoxResNet_ResBlock, self).__init__()

        self.seq = nn.Sequential(
                BatchNorm3d(64),
                ReLU(),
                Conv3d(64, 64, (1,3,3), padding=(0,1,1)),
                BatchNorm3d(64),
                ReLU(),
                Conv3d(64, 64, (3,3,3), padding=1)
        )

    def forward(self, x):
        return x + self.seq(x)

    def __str__(self):
        return "VoxResNet_ResBlock"

class RatLesNet_DenseBlock(nn.Module):

    def __init__(self, in_filters, concat, growth_rate, dim_reduc=False,
                 nonlinearity=torch.nn.functional.relu):
        super(RatLesNet_DenseBlock, self).__init__()
        self.concat = concat
        self.dim_reduc = dim_reduc
        self.act = nonlinearity
        self.convs = []

        for i in range(self.concat):
            # Add every 3D convolution to self.convs
            self.convs.append(nn.Sequential(
                Conv3d(growth_rate*i+in_filters, growth_rate, 3,
                    stride=1, padding=1),
                nonlinearity
                ))
        self.convs = nn.ModuleList(self.convs)

        # Output shape of the current DenseBLock
        self.out_shape = growth_rate*i+in_filters

        # Reduce dimensions at the end of the block if needed
        if self.dim_reduc:
            self.reduc_conv = nn.Sequential(
                Conv3d(growth_rate*(i+1)+in_filters, in_channels, 3,
                    stride=1, padding=1),
                nonlinearity
            )
            self.out_shape = in_filters

    def forward(self, x):

        inputs = [x]
        for i in range(self.concat):
            #x = self.act(self.convs[i](x)) #Good one
            x = self.convs[i](x)
            inputs.append(x)
            x = torch.cat(inputs, dim=1)

        if self.dim_reduc:
            x = self.act(self.reduc_conv(x))

        return x

    def __str__(self):
        return "RatLesNet_DenseBlock"

class RatLesNet_ResNetBlock(nn.Module):
    def __init__(self, in_filters, concat, growth_rate, dim_reduc=False,
                 nonlinearity=torch.nn.functional.relu):
        super(RatLesNet_ResNetBlock, self).__init__()

        self.seq = nn.Sequential(
                BatchNorm3d(in_filters),
                ReLU(),
                Conv3d(in_filters, in_filters, 3, padding=1),
                BatchNorm3d(in_filters),
                ReLU(),
                Conv3d(in_filters, in_filters, 3, padding=1)
            )

    def forward(self, x):
        return x + self.seq(x)

    def __str__(self):
        return "RatLesNet_ResNetBlock"

class RatLesNet_DenseBlock_133(nn.Module):

    def __init__(self, in_filters, concat, growth_rate, dim_reduc=False,
                 nonlinearity=torch.nn.functional.relu):
        super(RatLesNet_DenseBlock_133, self).__init__()
        self.concat = concat
        self.dim_reduc = dim_reduc
        self.act = nonlinearity
        self.convs = []

        for i in range(self.concat):
            # Add every 3D convolution to self.convs
            self.convs.append(nn.Sequential(
                Conv3d(growth_rate*i+in_filters, growth_rate, (1,3,3),
                    stride=1, padding=(0,1,1)),
                nonlinearity
                ))
        i += 1
        self.convs.append(nn.Sequential(
            Conv3d(growth_rate*i+in_filters, growth_rate, 3,
                stride=1, padding=1),
            nonlinearity
            ))
        self.convs = nn.ModuleList(self.convs)

        # Output shape of the current DenseBLock
        self.out_shape = growth_rate*i+in_filters

        # Reduce dimensions at the end of the block if needed
        if self.dim_reduc:
            self.reduc_conv = nn.Sequential(
                Conv3d(growth_rate*(i+1)+in_filters, in_channels, 3,
                    stride=1, padding=1),
                nonlinearity
            )
            self.out_shape = in_filters

    def forward(self, x):

        inputs = [x]
        for i in range(self.concat):
            #x = self.act(self.convs[i](x)) #Good one
            x = self.convs[i](x)
            inputs.append(x)
            x = torch.cat(inputs, dim=1)

        if self.dim_reduc:
            x = self.act(self.reduc_conv(x))

        return x

    def __str__(self):
        return "RatLesNet_DenseBlock_133"

