import tensorflow as tf
import random
import numpy as np
import time
from tensorflow.keras.layers import Conv3D, MaxPooling3D
from tensorflow.keras.regularizers import l2
from experiments.lib.blocks.MiNetBratsBlocks import *
from experiments.lib.blocks.MainBlocks import *
from experiments.lib.util import TB_Log, log, dice_coef
from .ModelBase import ModelBase
from experiments.lib import memory_saving_gradients

class MiNetBrats(ModelBase):
    """ My Own model.
    """

    def __init__(self, config):
        self.config = config
        self.create_model()
        self.create_loss()
        self.create_train_step()
        super().__init__()

    #def create_train_step(self):
    #    grads = memory_saving_gradients.gradients_speed(self.loss, tf.trainable_variables())
    #    grads_and_vars = list(zip(grads, tf.trainable_variables()))
    #    self.train_step = self.config["opt"].apply_gradients(grads_and_vars)
    #    super().create_train_step()


    def create_loss(self):
        """Loss has 3 parts. Look at the formula.
           Part 1: L2 Regularization (last piece of the code)
           Part 2: Weighted cross entropy among all classifiers.
           Part 3: Regular cross entropy.
        """

        # Regular cross entropy between the final output and the labels
        self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.placeholders["out_segmentation"]))

    def create_model(self):
        tf.reset_default_graph()

        self.placeholders = {}
        #self.x_input = tf.placeholder(tf.float32, [1, 138, 173, 172, 4])
        self.placeholders["in_volume"] = tf.placeholder(tf.float32, [1, 138, 173, 172, 4])
        self.placeholders["out_segmentation"] = tf.placeholder(tf.float32, [1, 138, 173, 172, 4])
        #self.x_age = tf.placeholder(tf.float32, [1])
        #self.y_true = tf.placeholder(tf.float32, [1, 138, 173, 172, 4])
        self.outputs = [] # I will store all the outputs that need to be opt.
        c1 = self.config["concat"]
        f1 = self.config["growth_rate"] # Initial filters
        g1 = self.config["growth_rate"] # Growth rate

        # input = Conv 1x1 to expand
        out1 = Conv3D(filters=f1, kernel_size=(1,1,1), strides=(1,1,1),
                padding="SAME", kernel_initializer=self.config["initW"],
                bias_initializer=self.config["initB"])(self.placeholders["in_volume"])

        # input = block1
        #out2 = MiNet_DenseBlock1(self.config, concat=c1, growth_rate=g1)(out1)
        out2 = MiNetBrats_DenseBlock1(self.config, concat=c1, growth_rate=g1)(out1)
        out3 = MaxPooling3D((2, 2, 2), (2, 2, 2), padding="SAME")(out2)
        out4 = MiNetBrats_DenseBlock1(self.config, concat=c1, growth_rate=g1)(out3)
        out5 = MaxPooling3D((2, 2, 2), (2, 2, 2), padding="SAME")(out4)

        #bottleneck = Conv3D(filters=f1, kernel_size=(1,1,1), strides=(1,1,1),
        bottleneck = Conv3D(filters=f1+g1*c1+g1*c1, kernel_size=(1,1,1), strides=(1,1,1),
                padding="SAME", kernel_initializer=self.config["initW"],
                bias_initializer=self.config["initB"])(out5)

        # Decoder
        #with tf.device("/device:GPU:1"):
        unpool1 = Unpooling3DBlock(out5, out4, factor=(2,2,2))(bottleneck)
        dec1 = MiNetBrats_DenseBlock1(self.config, concat=c1, growth_rate=g1)(unpool1)

        bottleneck2 = Conv3D(filters=f1+g1*c1, kernel_size=(1,1,1), strides=(1,1,1),
        #bottleneck2 = Conv3D(filters=f1, kernel_size=(1,1,1), strides=(1,1,1),
                padding="SAME", kernel_initializer=self.config["initW"],
                bias_initializer=self.config["initB"])(dec1)

        unpool2 = Unpooling3DBlock(out3, out2, factor=(2,2,2))(bottleneck2)
        dec2 = MiNetBrats_DenseBlock1(self.config, concat=c1, growth_rate=g1)(unpool2)

        # After bottleneck, regression part
        # Does it make sense to add another branch? I won't have the tumor all the time.
        """
        reg1 = Conv3D(filters=g1*c1, kernel_size=(1,1,1), strides=(1,1,1),
                padding="VALID", kernel_initializer=self.config["initW"],
                bias_initializer=self.config["initB"])(bottleneck)
        reg2 = Conv3D(filters=f1, kernel_size=(1,1,1), strides=(1,1,1),
                padding="VALID", kernel_initializer=self.config["initW"],
                bias_initializer=self.config["initB"])(reg1)
        reg3 = tf.layers.Flatten(reg2)
        reg4 = Dense(32*32*32, activation='prelu')(reg3)
        reg5 = Dense(32, activation='prelu')(reg4)
        reg6 = Dense(1)(reg5)
        reg7 = CombineBlock([reg6, 
        """


        # Classifier
        last = Conv3D(filters=self.config["classes"],
                kernel_size=(1,1,1), strides=(1,1,1),
                padding="SAME", kernel_initializer=self.config["initW"],
                bias_initializer=self.config["initB"])(dec2)

        self.logits = last
        self.prediction = tf.nn.softmax(self.logits)


    def measure(self, y_pred, y_true):
        # For now, it is only a segmentation
        # In the future, it can include the survival days

        res = dice_coef(y_pred, y_true["out_segmentation"])
        # Batch-size is always one
        return res

        













