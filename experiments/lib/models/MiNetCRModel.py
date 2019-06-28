import tensorflow as tf
import random
import numpy as np
import time
from tensorflow.keras.layers import Conv3D, MaxPooling3D
from tensorflow.keras.regularizers import l2
from experiments.lib.blocks.MiNetCRBlocks import *
from experiments.lib.blocks.MainBlocks import *
from experiments.lib.util import TB_Log, log, dice_coef
from .ModelBase import ModelBase
from experiments.lib import memory_saving_gradients

class MiNetCR(ModelBase):
    """Trunk model.
       Paper: 
       Required memory (nvidia-smi): 0., MB, MiB
       Number of parameters: M ()

       Notes:
        * None.

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
        # I don't need to create a model if I will load it

        # Regular cross entropy between the final output and the labels
        # Cross_entropy -> same size as the images without the channels: 18, 256, 256
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits,
                labels=self.placeholders["out_segmentation"])
        #self.loss = tf.reduce_sum(cross_entropy * self.x_weights)
        self.loss = tf.reduce_sum(cross_entropy)

    def create_model(self):
        tf.reset_default_graph()

        self.placeholders = {}
        self.placeholders["in_volume"] = tf.placeholder(tf.float32, [None, 18, 256, 256, 1])
        self.placeholders["out_segmentation"] = tf.placeholder(tf.float32, [None, 18, 256, 256, 2])
        
        #self.x_weights = tf.placeholder(tf.float32, [None, 18, 256, 256])
        self.outputs = [] # I will store all the outputs that need to be opt.
        c1 = self.config["concat"]
        f1 = 12 # Initial filters
        g1 = self.config["growth_rate"] # Growth rate

        # input = Conv 1x1 to expand
        out1 = Conv3D(filters=f1, kernel_size=(1,1,1), strides=(1,1,1),
                padding="SAME", kernel_initializer=self.config["initW"],
                bias_initializer=self.config["initB"])(self.placeholders["in_volume"])

        # input = block1
        #out2 = MiNet_DenseBlock1(self.config, concat=c1, growth_rate=g1)(out1)
        out2 = MiNet_DenseBlock1(self.config, concat=c1, growth_rate=g1)(out1)
        out3 = MaxPooling3D((2, 2, 2), (2, 2, 2), padding="SAME")(out2)
        out4 = MiNet_DenseBlock1(self.config, concat=c1, growth_rate=g1)(out3)
        out5 = MaxPooling3D((2, 2, 2), (2, 2, 2), padding="SAME")(out4)

        #bottleneck = Conv3D(filters=f1, kernel_size=(1,1,1), strides=(1,1,1),
        bottleneck = Conv3D(filters=f1+g1*c1+g1*c1, kernel_size=(1,1,1), strides=(1,1,1),
                padding="SAME", kernel_initializer=self.config["initW"],
                bias_initializer=self.config["initB"])(out5)

        # Decoder
        unpool1 = Unpooling3DBlock(out5, out4, factor=(2,2,2), skip_connection="concat")(bottleneck)
        dec1 = MiNet_DenseBlock1(self.config, concat=c1, growth_rate=g1)(unpool1)

        bottleneck2 = Conv3D(filters=f1+g1*c1, kernel_size=(1,1,1), strides=(1,1,1),
                padding="SAME", kernel_initializer=self.config["initW"],
                bias_initializer=self.config["initB"])(dec1)

        unpool2 = Unpooling3DBlock(out3, out2, factor=(2,2,2), skip_connection="concat")(bottleneck2)

        dec2 = MiNet_DenseBlock1(self.config, concat=c1, growth_rate=g1)(unpool2)

        # Classifier
        last = Conv3D(filters=self.config["classes"],
                kernel_size=(1,1,1), strides=(1,1,1),
                padding="SAME", kernel_initializer=self.config["initW"],
                bias_initializer=self.config["initB"])(dec2)

        self.logits = last
        self.prediction = tf.nn.softmax(self.logits)

    def checkSaveModel(self, epoch, val_loss):
        """This function will decide whether to save the current model or
           not. It is being executed after the epoch and the validation are
           performed during the training. In order to decide whether to save
           the model we can use the parameters provided.

           For example, it might be interesting to save only certain epochs
           (at the end of the training) or the models with the lowest val_loss.

           Args:
            `epoch`: current epoch.
            `val_loss`: validation loss.

           Returns:
            (bool) Whether to save the model.
        """
        if epoch % 2 == 0:
            return True

        return False

    def measure(self, y_pred, y_true):
        # For now, it is only a segmentation
        # In the future, it can include the survival days

        res = dice_coef(y_pred, y_true["out_segmentation"])
        # Batch-size is always one
        return res

        













