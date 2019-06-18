import tensorflow as tf
import numpy as np

class BlockBase:
    """Base class that all Blocks will inherit.
       It contains some basic methods:

       1) getBlockName: This function will calculate the new name
          of the layer where it is being used. It will check how many
          layers of the same Block exist.

       2) Activation: It works as a layer. This is done like this
          to provide with more flexibility in case the act. fun. requires
          some parameters.

    """
    def __init__(self):
        pass

    def getBlockName(self):
        """This function calculates how many layers with the same block name
           are being used in order to calculate its own name.
        """

        candidates = []
        for t in tf.trainable_variables():
            if self._name in t.name:
                candidates.append(t.name)

        scopes = set()
        l = len(self._name)
        for c in candidates:
            # The 3 is because we want to add the _XY where X is a number
            # and Y can be the / or a number. This won't work if I have more
            # Than 100 layers, but it is unlikely that I will have that.
            tmp = c[:c.index(self._name)+l+3]
            scopes.add(tmp)

        return self._name + "_" + str(len(scopes) + 1)

    def Activation(self, input, act):
        """Activation function.
           Currenlty implemented activation functions:
           ReLU, Linear.
        """

        if act == "relu":
            return tf.nn.relu(input)
        elif act == "linear":
            return input
        else:
            raise Exception("Unknown activation function: "+str(act))


class Unpooling3DBlock(BlockBase):

    def __init__(self, maxpooled, maxpooled_inputs, factor=(2,2,2)):
        """Unpooling3DBlock. This operation performs the opposite operation of
           Maxpooling. It will calculate the indices where the maxpooling
           extracted the max values and it will propagate new values.

           Args:
            `maxpooled`: Tensor resulting from the maxpooling operation
             that we want to restore.
            `maxpooled_inputs`: Input of the maxpooling operation we want
             to restore.
            `factor`: upsampling factor.

           Returns:
            Tensor with the final operation after the unpooling.

        """
        self._name = "Unpooling3DBlock"
        self.maxpooled = maxpooled
        self.maxpooled_inputs = maxpooled_inputs
        self.factor = factor

    def __call__(self, input):

        with tf.variable_scope(self.getBlockName()) as scope:
            # The activation maps can be calculated as the gradients.
            grads = tf.gradients(self.maxpooled, self.maxpooled_inputs)[0]
            # Then, we upsample the values we want to propagate, and multiply
            # them by the activation maps.
            reshaped = UpSampling3DBlock(grads.shape[1:-1])(input)
            output = reshaped * grads

        return output
        # Using keras's upsampling is frustrating because it does not
        #  accept float factors, so from 4 I can't upsample to 9.
        #return tf.keras.layers.UpSampling3D(self.factor)(input)*grads


class UpSampling3DBlock(BlockBase):
    def __init__(self, outputSize, interpolation="NN"):
        """UpSampling3DBlock. This block will resize/upsample the given input.
           Format: BXYZC. We can perform different types of interpolation, but
           for now I am only using Upsampling for the unpooling so I only
           need the nearest neighbor interpolation. Extending the
           interpolations is quite straighforward, as well as making a 2D version.

           Source: https://niftynet.readthedocs.io/en/dev/_modules/niftynet/layer/linear_resize.html
        """
        self._name = "UpSampling3DBlock"
        self.outputSize = outputSize
        if interpolation == "NN":
            self.interpolation = tf.image.resize_nearest_neighbor
        else:
            raise Exception("Unknown interpolation method")

    def __call__(self, input):

        with tf.variable_scope(self.getBlockName()) as scope:
            x_size, y_size, z_size, c_size = input.shape.as_list()[1:]
            b_size = -1
            x_size_new, y_size_new, z_size_new = self.outputSize
            # I have removed the ".aslist()"
            # NOTE: This will cause some error in the QuickNAT in the beginning.

            # Resize Y-Z
            squeeze_b_x = tf.reshape(input, [-1, y_size, z_size, c_size])
            resize_b_x = self.interpolation(squeeze_b_x, [y_size_new, z_size_new])
            resume_b_x = tf.reshape(resize_b_x, [b_size, x_size, y_size_new, z_size_new, c_size])

            # Resize X
            # First, reorient
            reoriented = tf.transpose(resume_b_x, [0,3,2,1,4])
            # Squeeze and 2d resize
            squeeze_b_z = tf.reshape(reoriented, [-1, y_size_new, x_size, c_size])
            resize_b_z = self.interpolation(squeeze_b_z, [y_size_new, x_size_new])
            resume_b_z = tf.reshape(resize_b_z, [b_size, z_size_new, y_size_new, x_size_new, c_size])

            output = tf.transpose(resume_b_z, [0, 3, 2, 1, 4])
        return output


class CombineBlock(BlockBase):

    def __init__(self, layers):
        """Combine. This function will combine layers. For now, only
           concatenation and sum are implemented.
           We can study more types of combination methods such as adding
           or subtracting.
        """
        self._name = "CombineBlock"
        self.layers = layers

    def concat(self):
        # Concatenating layers at the end, channels axis.
        self._name += "_Concat"
        with tf.variable_scope(self.getBlockName()) as scope:
            combine = tf.concat(self.layers, axis=-1)
        return combine

    def sum(self):
        self._name += "_Sum"
        with tf.variable_scope(self.getBlockName()) as scope:
            combine = tf.add_n(self.layers)
        return combine
        

class Depthwise_Conv3D(BlockBase):
    """This layer will perform Depthwise 3D convolutions. This is a very specific
       case of grouped convolutions. In this case, every convolution is done
       in every channel separately, and each channel corresponds to different kernels.

       This may be easily convereted into grouped convolutions if some assumptions
       are removed from this code.

       Assumption 2 implies that the output of this layer has the same size as
       the input.

       First the input is reshaped in a way that the channels are first.
       Then, tf.scan is used and the convolution is performed in each of those
       channels (first dim) individually.

       From: Xception and MobileNets.

    """
    def __init__(self, fs=(3,3,3)):
        self.fs = fs

    def __call__(self, inp):
        def f(old, inp):
            x_sample = inp

            y_sample = tf.keras.layers.Conv3D(filters=1, kernel_size=self.fs,
                    strides=[1,1,1], padding="SAME", kernel_initializer="ones")(x_sample)
            return y_sample

        # From BDWHC to CBDWH (channels go first)
        x_scan = tf.transpose(inp, perm=[4, 0, 1, 2, 3])
        # Add another dim at the end.
        x_scan = tf.expand_dims(x_scan, axis=-1)
        #c = tf.scan(f, x_scan, initializer = tf.zeros(inp.shape[:-1].concatenate(1)))
        # Initializer must be specified, otherwise the first element of x_scan
        # will be used as initializer.
        # ASSUMPTION 1: initializer has the shape of the input with 1 channel.
        #     probably this constraints the code if I want to use grouped conv.
        c = tf.scan(f, x_scan, initializer=tf.zeros_like(x_scan[0,:,:,:,:,:]))
        # output of tf.scan: 8, 3, 10, 10, 10, 1
        c = tf.transpose(c,[1, 2, 3, 4, 0, 5])
        #output: 3, 10, 10, 10, 8, 1
        # ASSUMPTION 2: Because of this, I can't do group convs
        c = tf.reshape(c, tf.shape(inp))
        #output: 3, 10, 10, 10, 8

        return c

##################################
## NOT USED ANYMORE
##################################


class BottleneckBlock(BlockBase):

    def __init__(self, conf, nfi=None, act="linear"):
        """BottleneckBlock. These layers are typically used to reduce the
           number of filters.

           Args:
            `conf`: configuration. Configuration is needed for the weight/bias
                    initialization procedure.
            `nfi`: number of filters.
            `act`: activation function.

           Returns:
            Tensor with the final operation after the bottleneck.
        """
        self._name = "BottleneckBlock"
        self.conf = conf
        self.nfi = nfi
        self.act = act

    def __call__(self, input):
        raise Exception("I am even using this?")
        prev_fi = input.shape[-1]
        self.nfi = prev_fi if self.nfi is None else self.nfi

        with tf.variable_scope(self.getBlockName()) as scope1:
            with tf.variable_scope("conv1") as scope2:
                W = tf.get_variable("W", shape=[1, 1, 1, prev_fi, self.nfi],
                        initializer=self.conf["initW"])
                b = tf.get_variable("b", shape=[self.nfi],
                        initializer=self.conf["initB"])
                conv = tf.nn.conv3d(input, W, strides=[1,1,1,1,1], padding="VALID")
                pre_act = tf.nn.bias_add(conv, b)
                act = self.activation(pre_act, self.act)

        return act

class DenseBlock(BlockBase):

    def __init__(self, conf, nfi, fs=3, act="linear", pad="SAME"):
        """DenseBlock.
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
        self._name = "DenseBlock"
        self.conf = conf
        self.nfi = nfi
        self.fs = fs
        self.act = act
        self.pad = pad

    def __call__(self, input):
        raise Exception("Am I even using this?")
        prev_fi = input.shape[-1]
        with tf.variable_scope(self.getBlockName()) as scope1:
            with tf.variable_scope("conv1") as scope2:
                W = tf.get_variable("W", shape=[self.fs, self.fs, self.fs,
                    prev_fi, self.nfi], initializer=self.conf["initW"])
                b = tf.get_variable("b", shape=[self.nfi],
                        initializer=self.conf["initB"])
                conv = tf.nn.conv3d(input, W, strides=[1,1,1,1,1],
                        padding=self.pad)
                pre_act = tf.nn.bias_add(conv, b)
                act = self.activation(pre_act, self.act)

        return act

class MaxPooling3DBlock(BlockBase):
    def __init__(self, size=(2,2,2), strides=(2,2,2), pad="VALID"):
        """Performs 3D MaxPooling.
           NOTE: Try TF's maxpooling, which seems to take much longer.

           Args:
            `size`: size of the pooling operation.
            `strides`: strides of the pooling operation.
            `pad`: padding, either VALID or SAME.

           Returns:
            Tensor with the final operation after the maxpooling.
        """
        raise Exception("No reason to use MaxPooling3DBlock")
        self._name = "MaxPool3DBlock"
        self.size = size
        self.strides = strides
        self.pad = pad

    def __call__(self, input):
        with tf.variable_scope(self.getBlockName()) as scope1:
            return tf.keras.layers.MaxPooling3D(self.size, self.strides, self.pad)(input)

