import tensorflow as tf
import numpy as np
from .MainBlocks import BlockBase, CombineBlock, Depthwise_Conv3D
from tensorflow.keras.layers import Conv3D, BatchNormalization, Conv3DTranspose

class RatLesNet_DenseBlock(BlockBase):

    def __init__(self, conf, concat, growth_rate, dim_reduc=False):
        """RatLesNet_DenseBlock.
           Structure of the block: Conv
        """
        self._name = "RatLesNet_DenseBlock"
        self.conf = conf
        self.concat = concat
        self.growth_rate = growth_rate
        self.dim_reduc = dim_reduc

    def __call__(self, input):
        with tf.variable_scope(self.getBlockName()) as scope1:
            feature_maps = int(input.shape[-1])

            x_input = input
            outputs = [x_input]

            for i in range(self.concat):
                conv = Conv3D(filters=self.growth_rate, kernel_size=(3,3,3),
                        strides=(1,1,1), padding="SAME",
                        kernel_initializer=self.conf["initW"],
                        bias_initializer=self.conf["initB"], activation=self.conf["act"])(x_input)
                #act = self.Activation(conv, self.conf["act"])
                #bn = BatchNormalization()(conv)
                # Add BN here.. and put the activation inside Conv3D.
                #conv = BatchNormalization()(conv)
                outputs.append(conv)
                x_input = CombineBlock(outputs).concat()

            # Not in use. Provides slightly worse results although it reduces
            # the number of parameters quite a lot.
            if self.dim_reduc:
                x_input = Conv3D(filters=feature_maps, kernel_size=(1,1,1),
                        strides=(1,1,1), padding="SAME",
                        kernel_initializer=self.conf["initW"],
                        bias_initializer=self.conf["initB"])(x_input)

        return x_input

class MiNet_DenseBlock_DW(BlockBase):

    def __init__(self, conf, concat, growth_rate, dim_reduc=False):
        """MiNet_DenseBlock1.
           Structure of the block: Conv
        """
        self._name = "MiNet_DenseBlock_DW"
        self.conf = conf
        self.concat = concat
        self.growth_rate = growth_rate
        self.dim_reduc = dim_reduc

    def __call__(self, input):
        with tf.variable_scope(self.getBlockName()) as scope1:
            feature_maps = int(input.shape[-1])

            x_input = input
            outputs = [x_input]

            for i in range(self.concat):
                depth = Depthwise_Conv3D(fs=(3,3,3))(x_input)
                act_1 = self.Activation(depth, self.conf["act"])
                pointw = Conv3D(filters=self.growth_rate, kernel_size=(1,1,1),
                        strides=(1,1,1), padding="SAME",
                        kernel_initializer=self.conf["initW"],
                        bias_initializer=self.conf["initB"])(act_1)
                act = self.Activation(pointw, self.conf["act"])

                outputs.append(act)
                x_input = CombineBlock(outputs).concat()

            if self.dim_reduc:
                x_input = Conv3D(filters=feature_maps, kernel_size=(1,1,1),
                        strides=(1,1,1), padding="SAME",
                        kernel_initializer=self.conf["initW"],
                        bias_initializer=self.conf["initB"])(x_input)

        return x_input

