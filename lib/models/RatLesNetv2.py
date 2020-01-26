import torch
from torch import nn
from torch.nn import Conv3d
import random, time
import numpy as np
from lib.blocks.RatLesNetv2Blocks import *
from torch.nn.functional import interpolate

class RatLesNetv2(nn.Module):

    def __init__(self, config):
        super(RatLesNetv2, self).__init__()

        act = config["act"]
        nfi = config["first_filters"]
        nfi2 = nfi*2
        conv_num = config["block_convs"]

        self.conv1 = Conv3d(1, config["first_filters"], 1)

        self.block1 = RatLesNetv2_ResNet(nfi, conv_num)
        self.mp1 = nn.modules.MaxPool3d(2, ceil_mode=True)

        self.block2 = RatLesNetv2_ResNet(nfi, conv_num)
        self.mp2 = nn.modules.MaxPool3d(2, ceil_mode=True)

        self.block3 = RatLesNetv2_ResNet(nfi, conv_num)
        self.mp3 = nn.modules.MaxPool3d(2, ceil_mode=True)

        self.bottleneck1 = RatLesNetv2_Bottleneck(nfi, nfi)
        self.block4 = RatLesNetv2_ResNet(nfi2, conv_num)

        self.bottleneck2 = RatLesNetv2_Bottleneck(nfi2, nfi)
        self.block5 = RatLesNetv2_ResNet(nfi2, conv_num)

        self.bottleneck3 = RatLesNetv2_Bottleneck(nfi2, nfi)
        self.block6 = RatLesNetv2_ResNet(nfi2, conv_num)

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
        b1 = self.bottleneck1(x)

        x = interpolate(b1, block3_size[2:], mode="trilinear")
        x = torch.cat([x, block3_out], dim=1)

        block4_out = self.block4(x)
        b2 = self.bottleneck2(block4_out)

        x = interpolate(b2, block2_size[2:], mode="trilinear")
        x = torch.cat([x, block2_out], dim=1)

        block5_out = self.block5(x)
        b3 = self.bottleneck3(block5_out)

        x = interpolate(b3, block1_size[2:], mode="trilinear")
        x = torch.cat([x, block1_out], dim=1)

        block6_out = self.block6(x)
        b4 = self.bottleneck4(block6_out)

        #softed = torch.functional.F.softmax(b4, dim=1)
        softed = torch.functional.F.sigmoid(b4)
        # Must be a tuple
        return (softed, b4)
        #return softed, b4, b1, b2, b3, block2_out, block3_out, block4_out, block5_out, block6_out

class RatLesNet_v2_LVL2(nn.Module):

    def __init__(self, config):
        super(RatLesNet_v2_LVL2, self).__init__()

        act = config["act"]
        nfi = config["first_filters"]
        nfi2 = nfi*2
        conv_num = config["block_convs"]

        self.conv1 = Conv3d(1, config["first_filters"], 1)

        self.block1 = RatLesNetv2_ResNet(nfi, conv_num)
        self.mp1 = nn.modules.MaxPool3d(2, ceil_mode=True)

        self.block2 = RatLesNetv2_ResNet(nfi, conv_num)
        self.mp2 = nn.modules.MaxPool3d(2, ceil_mode=True)

        self.bottleneck2 = RatLesNetv2_Bottleneck(nfi, nfi)
        self.block5 = RatLesNetv2_ResNet(nfi2, conv_num)

        self.bottleneck3 = RatLesNetv2_Bottleneck(nfi2, nfi)
        self.block6 = RatLesNetv2_ResNet(nfi2, conv_num)

        self.bottleneck4 = RatLesNetv2_Bottleneck(nfi2, 2)

    def forward(self, x):
        x = self.conv1(x)
        block1_out = self.block1(x)
        block1_size = block1_out.size()

        x = self.mp1(block1_out)
        block2_out = self.block2(x)
        block2_size = block2_out.size()

        x = self.mp2(block2_out)
        b2 = self.bottleneck2(x)

        x = interpolate(b2, block2_size[2:], mode="trilinear")
        x = torch.cat([x, block2_out], dim=1)

        block5_out = self.block5(x)
        b3 = self.bottleneck3(block5_out)

        x = interpolate(b3, block1_size[2:], mode="trilinear")
        x = torch.cat([x, block1_out], dim=1)

        block6_out = self.block6(x)
        b4 = self.bottleneck4(block6_out)

        softed = torch.functional.F.softmax(b4, dim=1)

        # Must be a tuple
        return (softed, )

class RatLesNet_v2_v2(nn.Module):

    def __init__(self, config):
        super(RatLesNet_v2_v2, self).__init__()

        act = config["act"]
        nfi = config["first_filters"]
        nfi2 = nfi*2

        self.conv1 = Conv3d(1, config["first_filters"], 1)

        self.block1 = RatLesNetv2_ResNet(nfi)
        self.gate1 = RatLesNetv2_SE1(nfi, int(nfi/8))
        self.mp1 = nn.modules.MaxPool3d(2, ceil_mode=True)

        self.block2 = RatLesNetv2_ResNet(nfi)
        self.gate2 = RatLesNetv2_SE1(nfi, int(nfi/8))
        self.mp2 = nn.modules.MaxPool3d(2, ceil_mode=True)

        self.block3 = RatLesNetv2_ResNet(nfi)
        self.gate3 = RatLesNetv2_SE1(nfi, int(nfi/8))
        self.mp3 = nn.modules.MaxPool3d(2, ceil_mode=True)

        self.bottleneck1 = RatLesNetv2_Bottleneck(nfi, nfi)
        self.block4 = RatLesNetv2_ResNet(nfi)

        self.bottleneck2 = RatLesNetv2_Bottleneck(nfi, nfi)
        self.block5 = RatLesNetv2_ResNet(nfi)

        self.bottleneck3 = RatLesNetv2_Bottleneck(nfi, nfi)
        self.block6 = RatLesNetv2_ResNet(nfi)

        self.bottleneck4 = RatLesNetv2_Bottleneck(nfi, 2)

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
        x = x * self.gate3(block3_out)

        x = self.block4(x)
        x = self.bottleneck2(x)

        x = interpolate(x, block2_size[2:], mode="trilinear")
        x = x * self.gate2(block2_out)

        x = self.block5(x)
        x = self.bottleneck3(x)

        x = interpolate(x, block1_size[2:], mode="trilinear")
        x = x * self.gate1(block1_out)

        x = self.block6(x)
        x = self.bottleneck4(x)

        x = torch.functional.F.softmax(x, dim=1)

        return x

class RatLesNet_v2_DenseNet(nn.Module):

    def __init__(self, config):
        super(RatLesNet_v2_DenseNet, self).__init__()

        act = config["act"]
        nfi = config["first_filters"]
        nfi2 = nfi*2
        conv_num = config["block_convs"]

        self.conv1 = Conv3d(1, config["first_filters"], 1)

        self.block1 = RatLesNetv2_DenseNet(nfi)
        self.mp1 = nn.modules.MaxPool3d(2, ceil_mode=True)

        self.block2 = RatLesNetv2_DenseNet(nfi)
        self.mp2 = nn.modules.MaxPool3d(2, ceil_mode=True)

        self.block3 = RatLesNetv2_DenseNet(nfi)
        self.mp3 = nn.modules.MaxPool3d(2, ceil_mode=True)

        self.bottleneck1 = RatLesNetv2_Bottleneck(nfi, nfi)
        self.block4 = RatLesNetv2_DenseNet(nfi2)

        self.bottleneck2 = RatLesNetv2_Bottleneck(nfi2, nfi)
        self.block5 = RatLesNetv2_DenseNet(nfi2)

        self.bottleneck3 = RatLesNetv2_Bottleneck(nfi2, nfi)
        self.block6 = RatLesNetv2_DenseNet(nfi2)

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
        b1 = self.bottleneck1(x)

        x = interpolate(b1, block3_size[2:], mode="trilinear")
        x = torch.cat([x, block3_out], dim=1)

        block4_out = self.block4(x)
        b2 = self.bottleneck2(block4_out)

        x = interpolate(b2, block2_size[2:], mode="trilinear")
        x = torch.cat([x, block2_out], dim=1)

        block5_out = self.block5(x)
        b3 = self.bottleneck3(block5_out)

        x = interpolate(b3, block1_size[2:], mode="trilinear")
        x = torch.cat([x, block1_out], dim=1)

        block6_out = self.block6(x)
        b4 = self.bottleneck4(block6_out)

        softed = torch.functional.F.softmax(b4, dim=1)
        # Must be a tuple
        return (softed, b4)
        #return softed, b4, b1, b2, b3, block2_out, block3_out, block4_out, block5_out, block6_out
