import tensorflow as tf
import random
import numpy as np
import time
from tensorflow.keras.layers import Conv3D, MaxPooling3D, BatchNormalization
from tensorflow.keras.regularizers import l2
from experiments.lib.blocks.RatLesNetBlocks import *
from experiments.lib.blocks.MainBlocks import *
from experiments.lib.util import TB_Log, log, dice_coef
from .ModelBase import ModelBase
from experiments.lib import memory_saving_gradients
from experiments.lib.keras_radam import RAdam

class RatLesNet(ModelBase):
    """RatLesNet.
       Paper: Our paper
       Number of parameters: 0.37M

    """

    def __init__(self, config):
        self.config = config
        self.create_model()
        self.create_loss()
        self.create_train_step()
        super().__init__()


    def create_train_step(self):
        """Optimizer to train the network.
        """
        with tf.variable_scope("opt") as scope:
            #super().create_train_step()
            self.val_loss_reduce_lr_counter = 0
            self.val_loss_reduce_lr_thr = 1e-2
            self.lr_tensor = tf.Variable(self.config["lr"], trainable=False, name="learning_rate")

            #### Regular optimization
            if self.config["weight_decay"] is None:

                self.config["opt"] = tf.train.AdamOptimizer(learning_rate=self.lr_tensor)

                #### Memory saving gradients
                #grads = memory_saving_gradients.gradients_speed(self.loss, tf.trainable_variables())
                #grads_and_vars = list(zip(grads, tf.trainable_variables()))
                #self.train_step = self.config["opt"].apply_gradients(grads_and_vars)

                self.train_step = self.config["opt"].minimize(self.loss)

            else:
            #### Using learning rate decay and decoupled weight decay.
                # Global step will increase in each iteration
                global_step = tf.Variable(0, name="global_step", trainable=False)

                # Decay scheduler
                # Typically this accepts 3 params: global_step, vector1, vector2
                # vector 1 indicates when the learning rate will decrease, and
                # vector 2 indicates how much the learning rate will decrease.
                # For instance: (global_step, [5, 10], [1e-0, 1e-1, 1e-2])
                # There is no decay until the 5th step, then decay is 1e-1 until 10th
                # and finally it's 1e-2 all the way until the end.
                #decay = tf.train.piecewise_constant(global_step, [5, 10], [1e-0, 1e-1, 1e-2])
                decay = tf.train.piecewise_constant(global_step, self.config["wd_epochs"], self.config["wd_rate"])

                # Decay scheduler needs to be applied to wd and lr variables, because of the
                # way it is implemented.
                #self.config["opt"] = RAdam(
                self.config["opt"] = tf.contrib.opt.AdamWOptimizer(
                        weight_decay=self.config["weight_decay"]*decay,
                        learning_rate=self.config["lr"]*decay)

                self.train_step = self.config["opt"].minimize(self.loss, global_step=global_step)

    def create_loss(self):
        """Regular cross entropy loss function.
        """
        with tf.variable_scope("loss") as scope:
            # I don't need to create a model if I will load it

            # Regular cross entropy between the final output and the labels
            # Cross_entropy -> same size as the images without the channels: 18, 256, 256
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits,
                    labels=self.placeholders["out_segmentation"])
            self.loss = tf.reduce_mean(cross_entropy)
            #self.loss = tf.reduce_sum(cross_entropy * self.x_weights)

            # Dice loss
            #num = 2 * tf.reduce_sum(self.logits * self.placeholders["out_segmentation"], axis=[1,2,3,4])
            #denom = tf.reduce_sum(tf.square(self.logits) + tf.square(self.placeholders["out_segmentation"]), axis=[1,2,3,4])
            #self.loss = (1 - tf.reduce_sum(num / (denom + 1e-6)))

            if self.config["L2"] != None:
                self.loss += tf.add_n([ tf.nn.l2_loss(v) for v in tf.trainable_variables() ]) * self.config["L2"]


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
        skip = self.config["skip_connection"]

        # input = Conv 1x1 to expand
        out1 = Conv3D(filters=f1, kernel_size=(1,1,1), strides=(1,1,1),
                padding="SAME", kernel_initializer=self.config["initW"], activation=self.config["act"],
                bias_initializer=self.config["initB"])(self.placeholders["in_volume"])

        # input = block1
        out2 = RatLesNet_DenseBlock(self.config, concat=c1, growth_rate=g1)(out1)
        out3 = MaxPooling3D((2, 2, 2), (2, 2, 2), padding="SAME")(out2)
        out4 = RatLesNet_DenseBlock(self.config, concat=c1, growth_rate=g1)(out3)
        out5 = MaxPooling3D((2, 2, 2), (2, 2, 2), padding="SAME")(out4)

        #bottleneck = Conv3D(filters=f1, kernel_size=(1,1,1), strides=(1,1,1),
        bottleneck = Conv3D(filters=f1+g1*c1+g1*c1, kernel_size=(1,1,1), strides=(1,1,1),
                padding="SAME", kernel_initializer=self.config["initW"], activation=self.config["act"],
                bias_initializer=self.config["initB"])(out5)

        # Decoder
        unpool1 = Unpooling3DBlock(out5, out4, factor=(2,2,2), skip_connection=skip)(bottleneck)
        dec1 = RatLesNet_DenseBlock(self.config, concat=c1, growth_rate=g1)(unpool1)

        bottleneck2 = Conv3D(filters=f1+g1*c1, kernel_size=(1,1,1), strides=(1,1,1),
                padding="SAME", kernel_initializer=self.config["initW"], activation=self.config["act"],
                bias_initializer=self.config["initB"])(dec1)

        unpool2 = Unpooling3DBlock(out3, out2, factor=(2,2,2), skip_connection=skip)(bottleneck2)

        dec2 = RatLesNet_DenseBlock(self.config, concat=c1, growth_rate=g1)(unpool2)

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
        """This function measures the segmentation performance.

           Args:
            `y_pred`: batch containing the predictions. BDWHC.
            `y_true`: batch containing the predictions. BDWHC.

           Returns:
            Dice coefficient.
        """
        # For now, it is only a segmentation
        # In the future, it can include the survival days

        res = dice_coef(y_pred, y_true["out_segmentation"])
        # Batch-size is always one
        return res



