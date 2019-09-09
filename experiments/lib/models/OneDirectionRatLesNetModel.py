import tensorflow as tf
import random
import numpy as np
import time
from experiments.lib.blocks.MainBlocks import *
from experiments.lib.util import TB_Log, log, dice_coef
from .ModelBase import ModelBase
from tensorflow.keras.layers import MaxPooling3D, Conv3D
from experiments.lib import memory_saving_gradients

class Name(ModelBase):
    """
    """
    def __init__(self, config):
        self.config = config

        self.create_model()
        self.create_loss()
        self.create_train_step()
        super().__init__()

    def create_train_step(self):
        with tf.variable_scope("optimizer") as scope:
            # Create optimizer
            global_step = tf.Variable(0, name="global_step", trainable=False)

            decay = tf.train.piecewise_constant(global_step, self.config["wd_epochs"], self.config["wd_rate"])


            self.config["opt"] = RAdam(
                    weight_decay=self.config["weight_decay"]*decay,
                    learning_rate=self.config["lr"]*decay)

            self.train_step = self.config["opt"].minimize(self.loss, global_step=global_step)

    def create_loss(self):
        # Create loss function
        cross_entropy = tf.losses.softmax_cross_entropy(self.placeholders["out_segmentation"], self.logits)
        self.loss = tf.reduce_sum(cross_entropy)

    def create_model(self):
        tf.reset_default_graph()

        self.placeholders = {}
        self.placeholders["in_volume"] = tf.placeholder(tf.float32, [None, 18, 256, 256 ,1])
        self.placeholders["out_segmentation"] = tf.placeholder(tf.float32, [None, 18, 256, 256, 2])

        T = 10

        # Model
        last = Conv3D(filters=1, kernel_size=(1,1,1), strides=(1,1,1),
                padding="SAME", kernel_initializer=self.config["initW"],
                bias_initializer=self.config["initB"])(self.placeholders["in_volume"])
        for i in range(T):
            last = Conv3D(filters=1, kernel_size=(1,1,1), strides=(1,1,1),
                    padding="SAME", kernel_initializer=self.config["initW"],
                    bias_initializer=self.config["initB"])(last)

        self.logits = last 
        self.prediction = tf.nn.softmax(self.logits)

    def checkSaveModel(self, epoch, val_loss):
        # Returns when it returns True, the model will be saved.
        if epoch % 2 == 0:
            return True
        return False

    def measure(self, y_pred, y_true):
        # This function receives batches of y_pred, y_true
        # and needs to return a 2D array where the first dim is the batch.
        res = dice_coef(y_pred, y_true["out_segmentation"])

        return res
