import tensorflow as tf
import numpy as np
from .MainBlocks import *
from tensorflow.keras.layers import Conv3D, BatchNormalization

class UNetBottleneckBlock(BlockBase):

    def __init__(self, conf, nfi, act=None):
        """UnetEncBlock.
           Structure of the block: Conv -> Act
           
           Args:
            `conf`: configuration. Configuration is needed for the weight/bias
                    initialization procedure.
            `nfi`: number of filters.
            `fs`: filter size, typically 3 (3x3x3).
            `act`: activation function.

           Returns:
            Tensor with the final operation after the activation.
        """
        self._name = "UNetBottleneckBlock"
        self.conf = conf
        self.nfi = nfi
        if act is None:
            self.act = self.conf["act"]
        else:
            self.act = act

    def __call__(self, input):
        with tf.variable_scope(self.getBlockName()) as scope1:
            out1 = Conv3D(filters=self.nfi, kernel_size=(1,1,1), strides=(1,1,1),
                    padding="SAME", kernel_initializer=self.conf["initW"],
                    bias_initializer=self.conf["initB"])(input)
            out2 = self.Activation(out1, self.act)

        return out2

class UNetEncBlock(BlockBase):

    def __init__(self, conf, nfi, fs=3, act=None, pad="SAME"):
        """UnetEncBlock.
           Structure of the block: Conv -> Act
           
           Args:
            `conf`: configuration. Configuration is needed for the weight/bias
                    initialization procedure.
            `nfi`: number of filters.
            `fs`: filter size, typically 3 (3x3x3).
            `act`: activation function.
            `pad`: padding, either VALID or SAME.

           Returns:
            Tensor with the final operation after the activation.
        """
        self._name = "UNetEncBlock"
        self.conf = conf
        self.nfi = nfi
        self.fs = fs
        self.pad = pad
        if act is None:
            self.act = self.conf["act"]
        else:
            self.act = act

    def __call__(self, input):
        with tf.variable_scope(self.getBlockName()) as scope1:
            out1 = Conv3D(filters=int(self.nfi/2), kernel_size=self.fs,
                    strides=(1,1,1), padding=self.pad,
                    kernel_initializer=self.conf["initW"],
                    bias_initializer=self.conf["initB"])(input)
            out2 = BatchNormalization()(out1)
            out3 = self.Activation(out2, self.act)

            out4 = Conv3D(filters=self.nfi, kernel_size=self.fs,
                    strides=(1,1,1), padding=self.pad,
                    kernel_initializer=self.conf["initW"],
                    bias_initializer=self.conf["initB"])(out3)
            out5 = BatchNormalization()(out4)
            out6 = self.Activation(out5, self.act)

        return out6


class UNetDecBlock(BlockBase):
    def __init__(self, conf, nfi, outShape, fs=(3,3,3), act=None, pad="SAME"):
        """UnetDecBlock.
           Structure of the block: Upsample -> 2x2 Conv -> Concat -> ConvConv
           
           Args:
            `conf`: configuration. Configuration is needed for the weight/bias
                    initialization procedure.
            `nfi`: number of filters.
            `outShape`: shape of the output of this layer. This is needed for
                        the upsampling procedure.
            `fs`: filter size, typically 3 (3x3x3).
            `act`: activation function.
            `pad`: padding, either VALID or SAME.

           Returns:
            Tensor with the final operation after the activation.
        """
        self._name = "UNetDecBlock"
        self.conf = conf
        self.nfi = nfi
        self.outShape = outShape
        self.fs = fs
        self.pad = pad
        if act is None:
            self.act = self.conf["act"]
        else:
            self.act = act

    def __call__(self, input1, input2):
        """The first input is the one that comes from below, that we need to
           upsample. The second input is the one that we concatenate.
        """

        with tf.variable_scope(self.getBlockName()) as scope1:

            # 1: Upsample input1. Upconvolution part 1/2
            upsampled = UpSampling3DBlock(self.outShape)(input1)

            # 2: 2x2 Conv input1 Upconvolution part 2/2
            out1 = Conv3D(filters=int(self.nfi*2), kernel_size=[2,2,2],
                    strides=(1,1,1), padding=self.pad,
                    kernel_initializer=self.conf["initW"],
                    bias_initializer=self.conf["initB"])(upsampled)

            # 3: Concat input1+input2
            comb1 = CombineBlock([out1, input2]).concat()

            # 4: Conv, Conv the concatenated data
            out2 = Conv3D(filters=self.nfi, kernel_size=self.fs,
                    strides=(1,1,1), padding=self.pad,
                    kernel_initializer=self.conf["initW"],
                    bias_initializer=self.conf["initB"])(comb1)
            out3 = BatchNormalization()(out2)
            out4 = self.Activation(out3, self.act)

            out5 = Conv3D(filters=self.nfi, kernel_size=self.fs,
                    strides=(1,1,1), padding=self.pad,
                    kernel_initializer=self.conf["initW"],
                    bias_initializer=self.conf["initB"])(out4)
            out6 = BatchNormalization()(out5)
            out7 = self.Activation(out6, self.act)

        return out7


