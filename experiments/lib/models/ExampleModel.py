import tensorflow as tf
import random
import numpy as np
import time
from experiments.lib.blocks.UNet3DBlocks import *
from experiments.lib.blocks.MainBlocks import *
from experiments.lib.util import TB_Log, log, dice_coef
from .ModelBase import ModelBase
from tensorflow.keras.layers import MaxPooling3D
from experiments.lib import memory_saving_gradients

class Name(ModelBase):
    """3D U-Net model.
    """
    def __init__(self, config):
        self.config = config

        self.create_model()
        self.create_loss()
        self.create_train_step()
        super().__init__()

    def create_train_step(self):

        with tf.variable_scope("optimizer") as scope:

            #grads = tf.gradients(self.loss, tf.trainable_variables())
            #grads = memory_saving_gradients.gradients_speed(self.loss, tf.trainable_variables())
            #grads_and_vars = list(zip(grads, tf.trainable_variables()))
            #self.train_step = self.config["opt"].apply_gradients(grads_and_vars)
            #self.config["opt"] = tf.train.MomentumOptimizer(learning_rate=self.config["lr"],
            #        momentum=self.config["momentum"])

            self.lr_tensor = tf.Variable(self.config["lr"], trainable=False, name="learning_rate")
            self.config["opt"] = tf.train.GradientDescentOptimizer(learning_rate=self.lr_tensor)
            # For compatibility with the output of ModelBase.py
            self.config["opt"]._lr = self.config["opt"]._learning_rate
            self.train_step = self.config["opt"].minimize(self.loss)


    def create_loss(self):
        """Weighted loss. In the original paper they use another type of weighted loss.
        """
        # Creating this weighted map is a bit of a bottleneck.
        # It would be faster if I do it outside and pass it to the graph.
        prop = 1-tf.reduce_sum(self.placeholders["out_segmentation"], axis=(1,2,3))/tf.reduce_sum(self.placeholders["out_segmentation"], axis=(1,2,3,4))
        weighted_map = tf.reduce_sum(self.placeholders["out_segmentation"] * prop, axis=-1)

        self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.placeholders["out_segmentation"], logits=self.logits)*weighted_map)

    def create_model(self):
        tf.reset_default_graph()

        self.placeholders = {}
        self.placeholders["in_volume"] = tf.placeholder(tf.float32, [None, 18, 256, 256, 1])
        self.placeholders["out_segmentation"] = tf.placeholder(tf.float32, [None, 18, 256, 256, 2])

        # Act: ReLU
        # Padding needs to be the same, otherwise the 18 slices fade out
        enc_out1 = UNetEncBlock(self.config, nfi=64, fs=(3,3,3))(self.placeholders["in_volume"])
        enc_out2 = MaxPooling3D((2,2,2), (2,2,2), padding="SAME")(enc_out1)
        enc_out3 = UNetEncBlock(self.config, nfi=128, fs=(3,3,3))(enc_out2)
        enc_out4 = MaxPooling3D((2,2,2), (2,2,2), padding="SAME")(enc_out3)
        enc_out5 = UNetEncBlock(self.config, nfi=256, fs=(3,3,3))(enc_out4)
        enc_out6 = MaxPooling3D((2,2,2),(2,2,2), padding="SAME")(enc_out5)
        #enc_out7 = UNetEncBlock(self.config, nfi=512, fs=(3,3,3))(enc_out6)
        #enc_out8 = MaxPooling3D((2,2,2),(2,2,2), padding="SAME")(enc_out7)
        # Bottleneck, sort of.
        enc_out7 = UNetEncBlock(self.config, nfi=512, fs=(3,3,3))(enc_out6)

        # Decoder
        dec_out1 = UNetDecBlock(self.config, nfi=256, outShape=enc_out5.shape[1:-1], fs=(3,3,3))(enc_out7, enc_out5)
        dec_out2 = UNetDecBlock(self.config, nfi=128, outShape=enc_out3.shape[1:-1], fs=(3,3,3))(dec_out1, enc_out3)
        dec_out3 = UNetDecBlock(self.config, nfi=64, outShape=enc_out1.shape[1:-1], fs=(3,3,3))(dec_out2, enc_out1)
        #dec_out4 = UNetDecBlock(self.config, nfi=64, outShape=enc_out1.shape[1:-1], fs=(3,3,3))(dec_out3, enc_out1)

        # Bottleneck output
        dec_out4 = UNetBottleneckBlock(self.config, nfi=2)(dec_out3)

        self.logits = dec_out4
        self.prediction = tf.nn.softmax(self.logits)

    def checkSaveModel(self, epoch, val_loss):
        if epoch > 10:
            return True
        return False

    def measure(self, y_pred, y_true):
        res = dice_coef(y_pred, y_true["out_segmentation"])
        return res
