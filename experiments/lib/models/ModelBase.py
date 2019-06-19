import tensorflow as tf
from tensorflow.keras.layers import Input
from experiments.lib.blocks.MainBlocks import *
import random, os, time, json
import numpy as np
from experiments.lib.util import TB_Log, log
from tensorflow.keras.layers import Conv3D
from tensorflow.keras.regularizers import l2
import nibabel as nib

from tensorflow.core.protobuf import rewriter_config_pb2

#from experiments.lib import memory_saving_gradients

class ModelBase:
    """This class contains the basic functionality that every model regardless
       of their configuration needs such as a training routine.
    """
    def __init__(self):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        #self.sess = tf.Session()
        self.tb = TB_Log(self.config["base_path"], self.sess)
        numParam = np.sum([np.prod(v.shape) for v in tf.trainable_variables()])
        print("Number of parameters: "+str(numParam))
        #raise Exception("llego")

    def create_loss(self):
        """Creates the loss to be minimized.
        """
        self.loss = self.config["loss"](y_true=self.y_true, y_pred=self.logits)

    def create_train_step(self):
        """Optimizer to train the network.
        """
        self.train_step = self.config["opt"].minimize(self.loss)

    def train(self, data):
        """This method will train the network given some data.
           `data` is a wrapper that will retrieve the data in batches.
           `data` wrapper requires the functions getNextTrainingBatch and
           getNextValidationBatch to be implemented.

           Args:
            `data`: data wrapper. It must follow some rules.
        """

        self.sess.run(tf.global_variables_initializer())
        ep = self.config["epochs"]
        bs = self.config["batch"]
        early_stopping_c = 0
        prev_val_loss = 99999999
        val_loss_text = ""
        e = 0 # Epoch counter
        it = 0 # Iteration counter
        losses = [] # Training and validation error
        keep_training = True # Flag to stop training when overfitting occurs.

        log("Starting training")
        while e < ep and keep_training:

            # Training
            d_tmp = data.getNextTrainingBatch()
            while d_tmp != None and keep_training:
                # Gets the inputs and outputs of the network
                feeding = {}
                for pl in d_tmp[0].keys():
                    feeding[self.placeholders[pl]] = d_tmp[0][pl]
                for pl in d_tmp[1].keys():
                    feeding[self.placeholders[pl]] = d_tmp[1][pl]

                it += 1

                #feeding = {self.x_input: X_train, self.y_true: Y_train}
				# I am not using weights for now, so this is fine
				# If I happen to use weights, then add it to the "feeding" here

                run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)
                _, tr_loss = self.sess.run([self.train_step, self.loss], feed_dict=feeding, options=run_options)
                d_tmp = data.getNextTrainingBatch()

            # Save weights every 2 epochs
            if e % 2 == 0:
                saver = tf.train.Saver()
                saver.save(self.sess, self.config["base_path"] + "weights-"+str(e))

            # Validation (after each epoch)
            d_tmp = data.getNextValidationBatch()
            val_loss = 0
            log("Validation")
            while d_tmp != None and keep_training:
                # Gets the inputs and outputs of the network
                feeding = {}
                for pl in d_tmp[0].keys():
                    feeding[self.placeholders[pl]] = d_tmp[0][pl]
                for pl in d_tmp[1].keys():
                    feeding[self.placeholders[pl]] = d_tmp[1][pl]

                val_loss_tmp, pred_tmp = self.sess.run([self.loss, self.prediction], feed_dict=feeding)
                val_loss += val_loss_tmp * 1/data.validation_samples_c
                d_tmp = data.getNextValidationBatch()

            val_loss_text = " Val Loss: {}".format(val_loss)
            self.tb.add_scalar(val_loss, "val_loss", e)

            # Early stopping
            if prev_val_loss < val_loss:
                early_stopping_c += 1
            else:
                early_stopping_c = 0
            prev_val_loss = val_loss

            if early_stopping_c >= self.config["early_stopping_c"]:
                keep_training = False

            log("Epoch: {}. Loss: {}.".format(e, tr_loss) + val_loss_text)
            self.tb.add_scalar(tr_loss, "train_loss", e)
            e += 1

            # If we are getting NaNs, it's pointless to continue
            if np.isnan(tr_loss):
                keep_training = False


    def train_(self, x_train, y_train, x_val=None, y_val=None, x_train_weights=None, x_val_weights=None):
        """Training routine.

           Args:
            `x_train`: numpy.array containing the training data. Format NDHWC.
            `y_train`: numpy.array containing the labels of the training data.
                       Format NDHWL. L indicates the Labels one-hot encoded.
            `x_val`: same as x_train but for validation.
            `y_val`: same as y_train but for validation.
            `x_train_weights`: optional weights used in the calculation of the
             loss function. They need to have the same shape as `x_train`.
            `x_val_weights`: sames as `x_train_weights` for the validation set.

           Returns:
            None. Simply trains the model.
        """
        # NOTE: REDO
        self.sess.run(tf.global_variables_initializer())
        N = x_train.shape[0]
        ep = self.config["epochs"]
        bs = self.config["batch"]
        early_stopping_c = 0
        prev_val_loss = 99999999
        val_loss_text = ""
        e = 0 # Epoch counter
        it = 0 # Iteration counter
        losses = [] # Training and validation error
        keep_training = True # Flag to stop training when overfitting occurs.

        log("Starting training")
        while e < ep and keep_training:
            idx = [i for i in range(N)]
            #random.shuffle(idx)
            x_train = x_train[idx]
            y_train = y_train[idx]
            if not x_train_weights is None:
                x_train_weights = x_train_weights[idx]

            ba = 0 # Mini-batch index

            while ba < N and keep_training:
                it += 1
                X = x_train[ba:ba+bs]
                Y = y_train[ba:ba+bs]

                feeding = {self.x_input: X, self.y_true: Y}
                if not x_train_weights is None:
                    feeding[self.x_weights] = x_train_weights[ba:ba+bs]

                run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)
                _, tr_loss = self.sess.run([self.train_step, self.loss], feed_dict=feeding, options=run_options)
                #_, tr_loss = self.sess.run([self.train_step, self.loss], feed_dict={self.x_input: X, self.y_true: Y})

                ba += bs # End loop - batch

            # Validation
            if not x_val is None and not y_val is None:
                log("Performing validation")
                ba = 0 # Mini-batch index
                val_loss = 0
                while ba < x_val.shape[0]:
                    X = x_val[ba:ba+bs]
                    Y = y_val[ba:ba+bs]

                    feeding = {self.x_input: X, self.y_true: Y}
                    if not x_train_weights is None:
                        feeding[self.x_weights] = x_val_weights[ba:ba+bs]

                    #val_loss += self.sess.run(self.loss, feed_dict={self.x_input: X, self.y_true: Y}) * 1/x_val.shape[0]
                    val_loss_tmp, pred_tmp = self.sess.run([self.loss, self.prediction], feed_dict=feeding)
                    val_loss += val_loss_tmp * 1/x_val.shape[0]
                    print(self.measure_dice(pred_tmp, Y, False))
                    ba += bs # End loop - Validation
                val_loss_text = " Val Loss: {}".format(val_loss)
                self.tb.add_scalar(val_loss, "val_loss", e)
                if prev_val_loss < val_loss:
                    early_stopping_c += 1
                else:
                    early_stopping_c = 0
                prev_val_loss = val_loss
                if early_stopping_c >= self.config["early_stopping_c"]:
                    keep_training = False

            log("Epoch: {}. Loss: {}.".format(e, tr_loss) + val_loss_text)
            self.tb.add_scalar(tr_loss, "train_loss", e)
            # Save in Tensorboard

            # If we are getting NaNs, it's pointless to continue
            if np.isnan(tr_loss):
                keep_training = False
            e += 1


    def predict(self, data, save=False):
        """Prediction routine. Data does not need to provide the expected
           predictions since it will not calculate the dice coefficient.

           Args:
            `data`: data wrapper. It requires getNextTestBatch function.
             Its first element will be the data, and the last the ids.
            `save`: boolean. Whether to save the predictions.

        """
        log("Prediction")

        if save:
            os.makedirs(self.config["base_path"] + "preds")

        d_tmp = data.getNextTestBatch()
        while d_tmp != None:
            # Dictionary that contains the name of the placeholder and the tensor
            # X = {"in_volume": np.array, ...}
            X = d_tmp[0]
            # List of IDs of each subject
            ids = d_tmp[-1]
            pred = self.predict_batch(X)

            # Iterate over all possible predicted subjects
            for i in range(len(ids)):
                if save: # Save predictions
                    np.save(self.config["base_path"] + "preds/pred_"+ids[i]+".npy", pred[i])

            d_tmp = data.getNextTestBatch()

    def predict_batch(self, X):
        """Predicts a single batch.

           Args:
            `X`: dictionary containing the input of the network.
             The keys are the name of the placeholders where the data goes.
             The values are the data.

           Returns:
            Predictions (np.array)
        """
        feeding = {}
        for pl in X.keys():
            feeding[self.placeholders[pl]] = X[pl]
        pred = self.sess.run(self.prediction, feed_dict=feeding)
        return pred
        


    def test(self, data, save=False):
        """This function will generate predictions and measure the results.

           Args:
            `data`: data wrapper. It requires getNextTestBatch function.
             Its first element is the input data, the second are the true values
             and the third the ids of the data.
            `save`: boolean. Whether to save the predictions.

        """

        log("Test")
        if save:
            os.makedirs(self.config["base_path"] + "preds")

        results = {}
        d_tmp = data.getNextTestBatch()
        while d_tmp != None:
            # Dictionary that contains the name of the placeholder and the tensor
            # X = {"in_volume": np.array, ...}
            X = d_tmp[0]
            # Dictionary containing the expected outputs
            # X = {"out_segmentation": np.array, ...}
            Y = d_tmp[1]
            # List of IDs of each subject
            ids = d_tmp[2]
            pred = self.predict_batch(X)
            res = self.measure(pred, Y)

            # Iterate over all possible predicted subjects
            for i in range(len(ids)):
                results[ids[i]] = list(res[i])
                if save: # Save predictions
                    np.save(self.config["base_path"] + "preds/pred_"+ids[i]+".npy", pred[i])
                    #pred_ = np.argmax(pred[i], axis=-1)
                    #nib.save(nib.Nifti1Image(pred_, np.eye(4)), self.config["base_path"] + "pred_"+ids[i]+".npy")

            with open(self.config["base_path"] + "dice_results.json", "w") as f:
                f.write(json.dumps(results))

            d_tmp = data.getNextTestBatch()


    def measure(self, preds, y_test):
        """This function will return different measurements of the
           performance of the developed model.

           Args:
            `preds`: numpy.array containing the predictions. Format NDHWC.
            `y_test`: Labels.

           Returns:
            Different measurements volume-wise. Right now, only dice coeff.
        """
        raise Exception("Implement your own measure function")
        dice_coeff = self.measure_dice(preds, y_test)

        return dice_coeff

    '''
    def measure_dice(self, y_pred, y_true, save=False):
        """This function calculates the dice coefficient.

           Args:
            `preds`: numpy.array containing the predictions. Format NDHWC.
            `y_test`: Labels.

           Returns:
            Dice coeff. per volume.
        """
        # NOTE: REDO
        if save:
            np.save(self.config["base_path"] + "y_pred", y_pred)
            np.save(self.config["base_path"] + "y_true", y_true)

        num_samples = y_pred.shape[0]
        num_classes = y_pred.shape[-1]
        results = np.zeros((num_classes, num_samples))
        y_pred = np.argmax(y_pred, axis=-1)
        y_true = np.argmax(y_true, axis=-1)

        for i in range(num_classes): # 2 CLASSES, BACKGROUND AND LESION
            for j in range(num_samples):
                a = y_pred[j]==i
                b = y_true[j]==i
                if np.sum(b) == 0: # If there is no lesion
                    if np.sum(a) == 0: # If the model didn't predict lesion
                        result = 1.0
                    else:
                        result = (np.sum(b==0)-np.sum(a))*1.0 / np.sum(b==0)
                else:
                    num = 2 * np.sum(a * b)
                    denom = np.sum(a) + np.sum(b)
                    result = num / denom
                results[i, j] = result
        return results
    '''
