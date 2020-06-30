import torch
from torch import nn
from torch.nn import Conv3d
import random, time
import numpy as np
from lib.blocks.UNet3DBlocks import *
from torch.nn.functional import interpolate

class UNet3D(nn.Module):

    def __init__(self, config):
        super(UNet3D, self).__init__()
        modalities = 1
        n_classes = 2
        nfi = config["first_filters"]
        # Original nfi = 32

        # Encoder
        self.block1 = UNet3D_ConvBlock([modalities, nfi, nfi*2])
        self.mp1 = nn.modules.MaxPool3d(2, ceil_mode=True)
        self.block2 = UNet3D_ConvBlock([nfi*2, nfi*2, nfi*4])
        self.mp2 = nn.modules.MaxPool3d(2, ceil_mode=True)
        self.block3 = UNet3D_ConvBlock([nfi*4, nfi*4, nfi*8])
        self.mp3 = nn.modules.MaxPool3d(2, ceil_mode=True)

        self.bottleneck = UNet3D_ConvBlock([nfi*8, nfi*8, nfi*16])

        # Decoder
        self.upconv1 = UNet3D_UpConv(nfi*16)
        self.block4 = UNet3D_ConvBlock([nfi*16+nfi*8, nfi*8, nfi*8])
        self.upconv2 = UNet3D_UpConv(nfi*8)
        self.block5 = UNet3D_ConvBlock([nfi*8+nfi*4, nfi*4, nfi*4])
        self.upconv3 = UNet3D_UpConv(nfi*4)
        self.block6 = UNet3D_ConvBlock([nfi*4+nfi*2, nfi*2, nfi*2])
        self.last = Conv3d(nfi*2, n_classes, 1)

    def forward(self, x):
        block1_out = self.block1(x)
        block1_size = block1_out.size()[2:]
        x = self.mp1(block1_out)

        block2_out = self.block2(x)
        block2_size = block2_out.size()[2:]
        x = self.mp2(block2_out)

        block3_out = self.block3(x)
        block3_size = block3_out.size()[2:]
        x = self.mp3(block3_out)

        x = self.bottleneck(x)

        x = self.upconv1(x, size=block3_size)
        x = torch.cat([block3_out, x], dim=1)
        x = self.block4(x)

        x = self.upconv2(x, size=block2_size)
        x = torch.cat([block2_out, x], dim=1)
        x = self.block5(x)

        x = self.upconv3(x, size=block1_size)
        x = torch.cat([block1_out, x], dim=1)
        x = self.block6(x)

        x = self.last(x)
        
        softed = torch.functional.F.softmax(x, dim=1)
        # Must be a tuple
        return (softed, )

