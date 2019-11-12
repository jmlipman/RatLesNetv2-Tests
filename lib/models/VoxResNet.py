import torch
from torch import nn
from torch.nn import Conv3d, BatchNorm3d, ReLU, ConvTranspose3d
import random, time
import numpy as np
from lib.blocks.Blocks import *


class VoxResNet(nn.Module):

    def __init__(self, config):
        super(VoxResNet, self).__init__()


        self.seq1 = nn.Sequential(
            Conv3d(1, 32, 3, padding=1),
            BatchNorm3d(32),
            ReLU(),
            Conv3d(32, 32, (1,3,3), padding=(0,1,1))
            )

        self.seq2 = nn.Sequential(
            BatchNorm3d(32),
            ReLU(),
            Conv3d(32, 64, 3, padding=1, stride=2),
            VoxResNet_ResBlock(),
            VoxResNet_ResBlock()
            )

        self.seq3 = nn.Sequential(
            BatchNorm3d(64),
            ReLU(),
            Conv3d(64, 64, 3, padding=1, stride=2),
            VoxResNet_ResBlock(),
            VoxResNet_ResBlock()
            )

        """
        self.seq4 = nn.Sequential(
            #BatchNorm3d(64),
            ReLU(),
            Conv3d(64, 64, 3, padding=1, stride=2),
            VoxResNet_ResBlock(),
            VoxResNet_ResBlock()
            )
        """

        self.transposed1 = ConvTranspose3d(32, 2, 3, padding=1)
        self.transposed2 = ConvTranspose3d(64, 2, 3, stride=2, padding=1,
                output_padding=1)
        self.transposed3 = ConvTranspose3d(64, 2, 3, stride=4, padding=1,
                output_padding=(1,3,3))
        #self.transposed4 = ConvTranspose3d(64, 2, 3, stride=8, padding=1,
        #        output_padding=(1,7,7))

        #self.bottleneck = nn.Sequential(
        #        BatchNorm3d(64),
        #        ReLU(),
        #        Conv3d(64, 2, 1)
        #        )

    def forward(self, x):
        out1 = self.seq1(x)
        out2 = self.seq2(out1)
        out3 = self.seq3(out2)
        #out4 = self.seq4(out3)

        x1 = self.transposed1(out1)
        x2 = self.transposed2(out2)
        x3 = self.transposed3(out3)
        #x4 = self.transposed4(out4)
        #x4 = torch.functional.F.upsample(out4, out1.size()[2:], mode="trilinear")
        #x4 = self.bottleneck(x4)

        x = torch.functional.F.softmax(x1+x2+x3, dim=1)

        #return x, torch.functional.F.softmax(x1, dim=1), torch.functional.F.softmax(x2, dim=1), torch.functional.F.softmax(x3, dim=1), torch.functional.F.softmax(x4, dim=1)
        return x, torch.functional.F.softmax(x1, dim=1), torch.functional.F.softmax(x2, dim=1), torch.functional.F.softmax(x3, dim=1)
