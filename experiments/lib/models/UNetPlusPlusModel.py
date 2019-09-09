import tensorflow as tf
import random
import numpy as np
import time
from experiments.lib.blocks.MainBlocks import *
from experiments.lib.util import TB_Log, log, dice_coef
from .ModelBase import ModelBase
from tensorflow.keras.layers import MaxPooling3D
from experiments.lib import memory_saving_gradients

class UNetPlusPlus(ModelBase):
    """From https://arxiv.org/pdf/1807.10165.pdf
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
            self.config["opt"] = optimizer
            self.train_step = self.config["opt"].minimize(self.loss)

    def create_loss(self):
        # Create loss function
        pass

    def create_model(self):
        tf.reset_default_graph()

        self.placeholders = {}
        self.placeholders["in_volume"] = tf.placeholder(tf.float32, [None, X ,1])
        self.placeholders["out_segmentation"] = tf.placeholder(tf.float32, [None, X, self.config["classes"]])

        # Model
        

        self.logits = last 
        self.prediction = tf.nn.softmax(self.logits)

    def checkSaveModel(self, epoch, val_loss):
        # Returns when it returns True, the model will be saved.
        pass

    def measure(self, y_pred, y_true):
        # This function receives batches of y_pred, y_true
        # and needs to return a 2D array where the first dim is the batch.
        pass
