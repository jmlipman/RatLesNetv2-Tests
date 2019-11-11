import torch
from torch import nn
from torch.nn import Conv3d
import random, time
import numpy as np
from lib.blocks.Blocks import *


class RatLesNet(nn.Module):

    def __init__(self, config):
        super(RatLesNet, self).__init__()

        act = config["act"]

        self.conv1 = Bottleneck3d(1, config["first_filters"],
                                  nonlinearity=act)
        in_channels = config["first_filters"]
        self.dense1 = RatLesNet_DenseBlock(in_channels,
                                           config["concat"],
                                           config["growth_rate"],
                                           dim_reduc=config["dim_reduc"],
                                           nonlinearity=act)
        self.mp1 = nn.modules.MaxPool3d(2, return_indices=True, ceil_mode=True)

        in_channels = config["first_filters"] + \
            config["concat"]*config["growth_rate"]
        self.dense2 = RatLesNet_DenseBlock(in_channels,
                                           config["concat"],
                                           config["growth_rate"],
                                           dim_reduc=config["dim_reduc"],
                                           nonlinearity=act)


        self.mp2 = nn.modules.MaxPool3d(2, return_indices=True, ceil_mode=True)

        in_channels += config["concat"]*config["growth_rate"]
        self.bottleneck1 = Bottleneck3d(in_channels, in_channels,
                                        nonlinearity=act)

        self.unpool1 = nn.modules.MaxUnpool3d(2)

        in_channels *= 2 # because of the concat at the same level
        self.dense3 = RatLesNet_DenseBlock(in_channels,
                                           config["concat"],
                                           config["growth_rate"],
                                           dim_reduc=config["dim_reduc"],
                                           nonlinearity=act)

        in_channels += config["concat"]*config["growth_rate"]
        out_channels = config["first_filters"] + \
            config["concat"]*config["growth_rate"]
        self.bottleneck2 = Bottleneck3d(in_channels, out_channels,
                                        nonlinearity=act)

        self.unpool2 = nn.modules.MaxUnpool3d(2)
        in_channels = out_channels + config["first_filters"] + \
            config["concat"]*config["growth_rate"]
        self.dense4 = RatLesNet_DenseBlock(in_channels,
                                           config["concat"],
                                           config["growth_rate"],
                                           dim_reduc=config["dim_reduc"],
                                           nonlinearity=act)

        in_channels += config["concat"]*config["growth_rate"]
        self.bottleneck3 = Bottleneck3d(in_channels, 2)

    def forward(self, x):
        x = self.conv1(x)
        dense1_out = self.dense1(x)
        dense1_size = dense1_out.size()


        x, idx1 = self.mp1(dense1_out)
        dense2_out = self.dense2(x)
        dense2_size = dense2_out.size()

        x, idx2 = self.mp2(dense2_out)
        x = self.bottleneck1(x)

        x = self.unpool1(x, idx2, output_size=dense2_size)
        x = torch.cat([x, dense2_out], dim=1)
        x = self.dense3(x)
        x = self.bottleneck2(x)

        x = self.unpool2(x, idx1, output_size=dense1_size)
        x = torch.cat([x, dense1_out], dim=1)
        x = self.dense4(x)
        x = self.bottleneck3(x)
        x = torch.functional.F.softmax(x, dim=1)

        return x

class RatLesNet_ResNet(nn.Module):

    def __init__(self, config):
        super(RatLesNet_ResNet, self).__init__()

        act = config["act"]

        # For total equal params: nfi=nfi2=26
        # For similar params in enc+dec, nfi=28, nfi2=26
        nfi, nfi2 = 28, 26
        self.conv1 = Bottleneck3d(1, nfi,
                                  nonlinearity=act)
        self.dense1 = RatLesNet_ResNetBlock(nfi,
                                           config["concat"],
                                           config["growth_rate"],
                                           dim_reduc=config["dim_reduc"],
                                           nonlinearity=act)
        self.mp1 = nn.modules.MaxPool3d(2, return_indices=True, ceil_mode=True)

        self.dense2 = RatLesNet_ResNetBlock(nfi,
                                           config["concat"],
                                           config["growth_rate"],
                                           dim_reduc=config["dim_reduc"],
                                           nonlinearity=act)


        self.mp2 = nn.modules.MaxPool3d(2, return_indices=True, ceil_mode=True)

        self.bottleneck1 = Bottleneck3d(nfi, nfi,
                                        nonlinearity=act)

        self.unpool1 = nn.modules.MaxUnpool3d(2)


        self.dense3 = RatLesNet_ResNetBlock(nfi*2,
                                           config["concat"],
                                           config["growth_rate"],
                                           dim_reduc=config["dim_reduc"],
                                           nonlinearity=act)

        self.bottleneck2 = Bottleneck3d(nfi*2, nfi,
                                        nonlinearity=act)

        self.unpool2 = nn.modules.MaxUnpool3d(2)

        self.dense4 = RatLesNet_ResNetBlock(nfi*2,
                                           config["concat"],
                                           config["growth_rate"],
                                           dim_reduc=config["dim_reduc"],
                                           nonlinearity=act)

        self.bottleneck3 = Bottleneck3d(nfi*2, 2)

    def forward(self, x):
        x = self.conv1(x)
        dense1_out = self.dense1(x)
        dense1_size = dense1_out.size()


        x, idx1 = self.mp1(dense1_out)
        dense2_out = self.dense2(x)
        dense2_size = dense2_out.size()

        x, idx2 = self.mp2(dense2_out)
        x = self.bottleneck1(x)

        x = self.unpool1(x, idx2, output_size=dense2_size)
        x = torch.cat([x, dense2_out], dim=1)
        x = self.dense3(x)
        x = self.bottleneck2(x)

        x = self.unpool2(x, idx1, output_size=dense1_size)
        x = torch.cat([x, dense1_out], dim=1)
        x = self.dense4(x)
        x = self.bottleneck3(x)
        x = torch.functional.F.softmax(x, dim=1)

        return x

class RatLesNet_3L(nn.Module):

    def __init__(self, config):
        super(RatLesNet_3L, self).__init__()

        act = config["act"]

        self.conv1 = Bottleneck3d(1, config["first_filters"],
                                  nonlinearity=act)
        in_channels = config["first_filters"]
        self.dense1 = RatLesNet_DenseBlock(in_channels,
                                           config["concat"],
                                           config["growth_rate"],
                                           dim_reduc=config["dim_reduc"],
                                           nonlinearity=act)
        self.mp1 = nn.modules.MaxPool3d(2, return_indices=True, ceil_mode=True)

        in_channels += config["concat"]*config["growth_rate"]
        self.dense2 = RatLesNet_DenseBlock(in_channels,
                                           config["concat"],
                                           config["growth_rate"],
                                           dim_reduc=config["dim_reduc"],
                                           nonlinearity=act)


        self.mp2 = nn.modules.MaxPool3d(2, return_indices=True, ceil_mode=True)

        in_channels += config["concat"]*config["growth_rate"]
        self.dense3 = RatLesNet_DenseBlock(in_channels,
                                           config["concat"],
                                           config["growth_rate"],
                                           dim_reduc=config["dim_reduc"],
                                           nonlinearity=act)

        self.mp3 = nn.modules.MaxPool3d(2, return_indices=True, ceil_mode=True)

        in_channels += config["concat"]*config["growth_rate"]
        self.bottleneck1 = Bottleneck3d(in_channels, in_channels,
                                        nonlinearity=act)

        self.unpool1 = nn.modules.MaxUnpool3d(2)

        in_channels *= 2 # because of the concat at the same level
        self.dense4 = RatLesNet_DenseBlock(in_channels,
                                           config["concat"],
                                           config["growth_rate"],
                                           dim_reduc=config["dim_reduc"],
                                           nonlinearity=act)

        in_channels += config["concat"]*config["growth_rate"]
        out_channels = config["first_filters"] + \
            config["concat"]*config["growth_rate"]*2
        self.bottleneck2 = Bottleneck3d(in_channels, out_channels,
                                        nonlinearity=act)

        #in_channels *= 2 # because of the concat at the same level
        in_channels = out_channels + config["first_filters"] + \
            2*config["concat"]*config["growth_rate"]
        self.dense5 = RatLesNet_DenseBlock(in_channels,
                                           config["concat"],
                                           config["growth_rate"],
                                           dim_reduc=config["dim_reduc"],
                                           nonlinearity=act)

        in_channels += config["concat"]*config["growth_rate"]
        out_channels = config["first_filters"] + \
            config["concat"]*config["growth_rate"]
        self.bottleneck3 = Bottleneck3d(in_channels, out_channels,
                                        nonlinearity=act)

        self.unpool2 = nn.modules.MaxUnpool3d(2)
        in_channels = out_channels + config["first_filters"] + \
            config["concat"]*config["growth_rate"]
        self.dense6 = RatLesNet_DenseBlock(in_channels,
                                           config["concat"],
                                           config["growth_rate"],
                                           dim_reduc=config["dim_reduc"],
                                           nonlinearity=act)

        in_channels += config["concat"]*config["growth_rate"]
        self.bottleneck4 = Bottleneck3d(in_channels, 2)

    def forward(self, x):
        x = self.conv1(x)
        dense1_out = self.dense1(x)
        dense1_size = dense1_out.size()


        x, idx1 = self.mp1(dense1_out)
        dense2_out = self.dense2(x)
        dense2_size = dense2_out.size()

        x, idx2 = self.mp2(dense2_out)
        dense3_out = self.dense3(x)
        dense3_size = dense3_out.size()

        x, idx3 = self.mp3(dense3_out)
        x = self.bottleneck1(x)

        x = self.unpool1(x, idx3, output_size=dense3_size)
        x = torch.cat([x, dense3_out], dim=1)

        x = self.dense4(x)
        x = self.bottleneck2(x)

        x = self.unpool2(x, idx2, output_size=dense2_size)
        x = torch.cat([x, dense2_out], dim=1)

        x = self.dense5(x)
        x = self.bottleneck3(x)

        x = self.unpool2(x, idx1, output_size=dense1_size)
        x = torch.cat([x, dense1_out], dim=1)

        x = self.dense6(x)
        x = self.bottleneck4(x)

        x = torch.functional.F.softmax(x, dim=1)

        return x

