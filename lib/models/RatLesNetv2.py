import torch
from torch import nn
from torch.nn import Conv3d
import random, time
import numpy as np
from lib.blocks.RatLesNetv2Blocks import *
from torch.nn.functional import interpolate

class RatLesNet_v2(nn.Module):

    def __init__(self, config):
        super(RatLesNet_v2, self).__init__()

        act = config["act"]
        nfi = config["first_filters"]
        nfi2 = nfi*2

        self.conv1 = Conv3d(1, config["first_filters"], 1)

        self.block1 = RatLesNetv2_ResNet(nfi)
        self.mp1 = nn.modules.MaxPool3d(2, ceil_mode=True)

        self.block2 = RatLesNetv2_ResNet(nfi)
        self.mp2 = nn.modules.MaxPool3d(2, ceil_mode=True)

        self.block3 = RatLesNetv2_ResNet(nfi)
        self.mp3 = nn.modules.MaxPool3d(2, ceil_mode=True)

        self.bottleneck1 = RatLesNetv2_Bottleneck(nfi, nfi)
        self.block4 = RatLesNetv2_ResNet(nfi2)

        self.bottleneck2 = RatLesNetv2_Bottleneck(nfi2, nfi)
        self.block5 = RatLesNetv2_ResNet(nfi2)

        self.bottleneck3 = RatLesNetv2_Bottleneck(nfi2, nfi)
        self.block6 = RatLesNetv2_ResNet(nfi2)

        self.bottleneck4 = RatLesNetv2_Bottleneck(nfi2, 2)

    def forward(self, x):
        x = self.conv1(x)
        block1_out = self.block1(x)
        block1_size = block1_out.size()


        x = self.mp1(block1_out)
        block2_out = self.block2(x)
        block2_size = block2_out.size()

        x = self.mp2(block2_out)
        block3_out = self.block3(x)
        block3_size = block3_out.size()

        x = self.mp3(block3_out)
        x = self.bottleneck1(x)

        x = interpolate(x, block3_size[2:], mode="trilinear")
        x = torch.cat([x, block3_out], dim=1)

        x = self.block4(x)
        x = self.bottleneck2(x)

        x = interpolate(x, block2_size[2:], mode="trilinear")
        x = torch.cat([x, block2_out], dim=1)

        x = self.block5(x)
        x = self.bottleneck3(x)

        x = interpolate(x, block1_size[2:], mode="trilinear")
        x = torch.cat([x, block1_out], dim=1)

        x = self.block6(x)
        x = self.bottleneck4(x)

        x = torch.functional.F.softmax(x, dim=1)

        return x
