import tensorflow as tf
from tensorflow.keras.layers import Input
from experiments.lib.blocks.MainBlocks import *
import random, os, time, json
import numpy as np
from experiments.lib.util import TB_Log, log
from tensorflow.keras.layers import Conv3D
from tensorflow.keras.regularizers import l2
import nibabel as nib
from skimage import measure

from tensorflow.core.protobuf import rewriter_config_pb2

#from experiments.lib import memory_saving_gradients

class ModelBase:
    """This class contains the basic functionality that every model regardless
       of their configuration needs such as a training routine.
    """
    def __init__(self):

        # Create a folder if it does not exist
        if not os.path.isdir(self.config["base_path"] +  "weights"):
            os.makedirs(self.config["base_path"] + "weights")

        # Load weights if needed
        if self.config["find_weights"] != "":
            self.load_model()
        else:
            if self.config["gpu_mem"] == 1:
                self.sess = tf.Session(config=tf.ConfigProto())
            else:
                self.gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.config["gpu_mem"])
                self.sess = tf.Session(config=tf.ConfigProto(gpu_options=self.gpu_options))
            self.saver = tf.train.Saver()

        self.tb = TB_Log(self.config["base_path"], self.sess)
        numParam = np.sum([np.prod(v.shape) for v in tf.trainable_variables()])
        print("Number of parameters: "+str(numParam))

        if self.config["find_weights"] == "":
            self.sess.run(tf.global_variables_initializer())

    def load_model(self):
        """This function will load an existing model (graph + weights).
           I could simply load the weights without loading a graph but this
           would force me to initialize exactly the same graph, which might
           be difficult.
        """
        # I need to reset the graph because it was created assuming there were
        # no weights to load. Now we will load the graph and its weights.
        tf.reset_default_graph()
        # Load the graph. Unnecessary if I can create it again.
        self.saver = tf.train.import_meta_graph(self.config["find_weights"] + ".meta")

        if self.config["gpu_mem"] == 1:
            self.sess = tf.Session(config=tf.ConfigProto())
        else:
            self.gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.config["gpu_mem"])
            self.sess = tf.Session(config=tf.ConfigProto(gpu_options=self.gpu_options))
        #self.sess = tf.Session(config=tf.ConfigProto(gpu_options=self.gpu_options))
        #self.sess = tf.Session(config=tf.ConfigProto())
        # Load the weights
        pathd = "/".join(self.config["find_weights"].split("/")[:-1])
        self.saver.restore(self.sess,tf.train.latest_checkpoint(pathd))

        # Load the tensors/ops in their respective variables
        with open(pathd + "/tensors", "r") as f:
            placeholders, prediction, train_step, loss = json.loads(f.read())

        self.placeholders = {}
        for pl in placeholders.keys():
            self.placeholders[pl] = self.prediction = tf.get_default_graph().get_tensor_by_name(placeholders[pl])
        self.prediction = tf.get_default_graph().get_tensor_by_name(prediction)
        self.loss = tf.get_default_graph().get_tensor_by_name(loss)
        self.train_step = tf.get_default_graph().get_operation_by_name(train_step)

    def create_loss(self):
        """Creates the loss to be minimized.
        """
        raise Exception("Implement your own loss function")
        #self.loss = self.config["loss"](y_true=self.y_true, y_pred=self.logits)

    def create_train_step(self):
        """Optimizer to train the network.
        """
        raise Exception("Implement your own train_step function")
        #self.train_step = self.config["opt"].minimize(self.loss)

    def train(self, data):
        """This method will train the network given some data.
           `data` is a wrapper that will retrieve the data in batches.
           `data` wrapper requires the functions getNextTrainingBatch and
           getNextValidationBatch to be implemented.

           Args:
            `data`: data wrapper. It must follow some rules.
        """
        ep = self.config["epochs"]
        bs = self.config["batch"]
        #early_stopping_c = 0
        prev_val_loss = [99999999]
        val_loss_text = ""
        e = 0 # Epoch counter
        it = 0 # Iteration counter
        losses = [] # Training and validation error
        keep_training = True # Flag to stop training when overfitting occurs.

        self.lr_updated_counter = 0 # Times the that lr was updated

        # Calculate when and how much to decrease the learning rate
        # This is used in the create_train_step, for piecewise_constant function
        #self.config["wd_epochs"] = [(len(data.getFiles("training"))*self.config["wd_epochs"]*i/bs) for i in range(1, int(ep/self.config["wd_epochs"]))]
        # This will always decrease by 0.1
        #self.config["wd_rate"] = [1/(10**i) for i in range(len(self.config["wd_epochs"]+1))]

        #print(tf.global_variables())

        # Track gradients here. After each sample is shown to the network.
        """
        target_grad = ["conv3d/kernel:0", "RatLesNet_DenseBlock_2/conv3d_4/kernel:0", "conv3d_11/kernel:0"]
        if len(target_grad) > 0:
            for v in tf.global_variables():
                if v.name in target_grad:
                    grads_and_vars = self.config["opt"].compute_gradients(self.prediction, [v])[0]
                    tf.summary.histogram(v.name+"_grad", grads_and_vars[0])
            summaries = tf.summary.merge_all()
        """
        # End of tracking gradients.
        #warming_up = np.linspace(0, 1e-4, 36*5)

        # Save evolution of training
        os.makedirs(self.config["base_path"] + "val_evol")

        log("Starting training")
        local_alpha = 1.01
        while e < ep and keep_training:

            # Training
            d_tmp = data.getNextTrainingBatch()
            if d_tmp == None: # This typically happens when the disc where the data is located is unmounted
                raise Exception("No data! Check the script can access to the data.")

            #if local_alpha > 0.01:
            #    local_alpha -= 0.01
            #    self.sess.run(self.alpha_tensor.assign(local_alpha))
            #if e > 400:
            #    self.sess.run(self.alpha_tensor.assign(0.0))

            tr_loss = 0
            while d_tmp != None and keep_training:
                # Gets the inputs and outputs of the network
                feeding = {}
                for pl in d_tmp[0].keys():
                    feeding[self.placeholders[pl]] = d_tmp[0][pl]
                for pl in d_tmp[1].keys():
                    feeding[self.placeholders[pl]] = d_tmp[1][pl]

                #if it < len(warming_up):
                #    self.sess.run(self.lr_tensor.assign(warming_up[it]))

                run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)
                _, tr_loss_tmp = self.sess.run([self.train_step, self.loss], feed_dict=feeding, options=run_options)
                #a1, a2 = self.sess.run([self.prediction, self.placeholders["out_segmentation"]], feed_dict=feeding, options=run_options)

                #summ = self.sess.run(summaries, feed_dict=feeding, options=run_options)
                #self.tb.writer.add_summary(summ, it)

                tr_loss += tr_loss_tmp * 1/len(data.getFiles("training"))
                # This makes sense with _lr is a tensor (AdamWOptimizer)
                #print(it, self.sess.run(self.config["opt"]._lr))
                d_tmp = data.getNextTrainingBatch()

                it += 1

            # Validation (after each epoch)
            log("Validation")
            d_tmp = data.getNextValidationBatch()
            val_loss = 0
            islands = [] # To study indepdent components
            while d_tmp != None and keep_training:
                # Gets the inputs and outputs of the network
                feeding = {}
                for pl in d_tmp[0].keys():
                    feeding[self.placeholders[pl]] = d_tmp[0][pl]
                for pl in d_tmp[1].keys():
                    feeding[self.placeholders[pl]] = d_tmp[1][pl]

                val_loss_tmp, pred_tmp, w_tmp = self.sess.run([self.loss, self.prediction, self.placeholders["in_weights"]], feed_dict=feeding)
                #val_loss_tmp = self.sess.run(self.loss, feed_dict=feeding)

                # TODO: Check if I can do this "outside" in the experiment level.
                # Saving progress.
                if d_tmp[2][0] == "02NOV2016_2h_17" or d_tmp[2][0] == "02NOV2016_24h_5": # For NMR CS3
                #if d_tmp[2][0] == "02NOV2016_2h_40" or d_tmp[2][0] == "02NOV2016_24h_43": # For FUJ PC
                    name = d_tmp[2][0] + "_" + str(e)
                    #np.save(self.config["base_path"] + "val_evol/" + name, w_tmp)
                    s = np.moveaxis(np.reshape(pred_tmp, (18, 256, 256, 2)), 0, 2)
                    #np.save(self.config["base_path"] + "val_evol/" + name, s)
                    s = np.argmax(s, axis=-1)
                    nib.save(nib.Nifti1Image(s, np.eye(4)), self.config["base_path"] + "val_evol/" + name + ".nii.gz")

                val_loss += val_loss_tmp * 1/len(data.getFiles("validation"))
                islands.append(np.max(measure.label(np.argmax(pred_tmp[0], axis=-1))))
                d_tmp = data.getNextValidationBatch()


            prev_val_loss.append(val_loss)

            val_loss_text = " Val Loss: {}".format(prev_val_loss[-1])
            self.tb.add_scalar(prev_val_loss[-1], "val_loss", e)
            self.tb.add_scalar(np.mean(islands), "islands", e)

            # Routine to save the model
            if self.checkSaveModel(e, val_loss):
                self.saver.save(self.sess, self.config["base_path"] + "weights/w-"+str(e))
                # I also save the name of the tensors of the placeholders and pred
                with open(self.config["base_path"] + "weights/tensors", "w") as f:
                    save_pl = {}
                    for k in self.placeholders.keys():
                        save_pl[k] = self.placeholders[k].name
                    f.write(json.dumps([save_pl, self.prediction.name, self.train_step.name, self.loss.name]))

            # Early stopping
            if self.earlyStoppingShouldStop(prev_val_loss, prev_losses_number=self.config["early_stopping_thr"]):
                log("Stop training because of early stopping")
                keep_training = False

            # Decreasing Learning Rate when a plateau is found
            self.decreaseLearningRateOnPlateau(prev_val_loss)

            self.decreaseLearningRateWhenValLossTooBig(prev_val_loss, self.config["lr_valloss_ratio"])
            if self.config["lr_updated_thr"] != -1 and self.lr_updated_counter > self.config["lr_updated_thr"]:
                log("Stop training. I've updated the lr enough times.")
                keep_training = False

            log("Epoch: {}. Loss: {}.".format(e, tr_loss) + val_loss_text)
            self.tb.add_scalar(tr_loss, "train_loss", e)
            e += 1

            # If we are getting NaNs, it's pointless to continue
            if np.isnan(tr_loss):
                keep_training = False


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
            pred = self.predictBatch(X)

            # Iterate over all possible predicted subjects
            for i in range(len(ids)):
                if save: # Save predictions
                    np.save(self.config["base_path"] + "preds/pred_"+ids[i]+".npy", pred[i])

            d_tmp = data.getNextTestBatch()

    def predictBatch(self, X):
        """Predicts an individual batch.

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


    def outputFromOperation(self, X, op_name):
        """Provides with the output of a specific operation.
        
           Args:
            `X`: dictionary containing the input of the network.
             The keys are the name of the placeholders where the data goes.
             The values are the data.
             `op_name`: name of the operation. You can find it by simply
              tf.get_default_graph().get_operations()

           Returns:
            Output from the operation.

        """
        #for op in tf.get_default_graph().get_tensor_by_name(op_name+":0"):
        #    if op_name == op.name:
        #        target_op = op
        target_op = tf.get_default_graph().get_tensor_by_name(op_name+":0")

        feeding = {}
        for pl in X.keys():
            feeding[self.placeholders[pl]] = X[pl]
        pred = self.sess.run(target_op, feed_dict=feeding)
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
            pred = self.predictBatch(X)
            res = self.measure(pred, Y)

            # Iterate over all possible predicted subjects
            for i in range(len(ids)):
                results[ids[i]] = list(res[i])
                if save: # Save predictions
                    np.save(self.config["base_path"] + "preds/pred_"+ids[i]+".npy", pred[i])
                    #np.save(self.config["base_path"] + "preds/true_"+ids[i]+".npy", Y["out_segmentation"])
                    #pred_ = np.argmax(pred[i], axis=-1)
                    #nib.save(nib.Nifti1Image(pred_, np.eye(4)), self.config["base_path"] + "pred_"+ids[i]+".npy")

            d_tmp = data.getNextTestBatch()

        count = str(len([x for x in os.listdir(self.config["base_path"]) if "dice_results" in x]))
        with open(self.config["base_path"] + "dice_results-"+count+".json", "w") as f:
            f.write(json.dumps(results))


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
        #dice_coeff = self.measure_dice(preds, y_test)
        #return dice_coeff


    def earlyStoppingShouldStop(self, val_losses, prev_losses_number=5):
        """This function returns whether the training should stop because the
           validation loss has increased for certain consecutive times.

           Args:
            `val_losses`: list of validation losses.
            `prev_losses_number`: The number of consecutive validation losses
             that need to increase to decide to stop the training.
        """
        if len(val_losses) >= prev_losses_number+1:
            decreases = [(val_losses[-i-1]-val_losses[-i]) < self.val_loss_reduce_lr_thr for i in range(prev_losses_number, 0, -1)]
            return sum(decreases) == prev_losses_number

    def decreaseLearningRateWhenValLossTooBig(self, prev_val_loss, thr, decrease_lr=1e-1):
        # If this is -1, do not decrease learning rate on plateau.
        if self.config["lr_updated_thr"] == -1:
            return False

        if len(prev_val_loss) > 30: # To make sure it won't decrease the lr too soon.
            if prev_val_loss[-1]/np.min(prev_val_loss) > self.config["lr_valloss_ratio"]:
                self.sess.run(self.lr_tensor.assign(self.lr_tensor * decrease_lr))
                self.lr_updated_counter += 1 # In ModelBase.py
                # Check I can do this after vacations.
                #log("Decreasing Learning rate to: "+str(self.sess.run(self.lr_tensor)))
                log("Decreasing Learning rate")
                return True
        return False

    def decreaseLearningRateOnPlateau(self, val_losses, prev_losses_number=5, decrease_lr=1e-1, decrease_thr=1e-1):
        """This function decreases the learning rate when it finds a plateau.
           The method to find a plateau is to check the ratio
           prev_val_loss/val_loss over a number of past iterations (threshold)
           and if that ratio is lower than a threshold (prev_losses_number)
           the learning rate is decreased.

           Args:
            `val_losses`: list of validation losses, including the current (last).
            `prev_losses_number`: after this number of consecutive times finding
             small rates, we consider we found a plateau and consequently
             the learning rate is reduced.
            `decrease_lr`: How much the learning rate will be decreased each
             time it finds a plateau. old_lr = lr*decrease_lr
            `decrease_thr`: How much the ratio threshold needs to be decreased.
             This is important because when the learning rate is decreased,
             the threshold needs to be decrease as well because the steps are
             smaller.

            Returns:
             Whether learning rate was decreased.
        """

        # If this is -1, do not decrease learning rate on plateau.
        if self.config["lr_updated_thr"] == -1:
            return False

        # Note: This is probably a problem when I load a model and
        # I train it again until it decreases the lr because it won't
        # find self.lr_tensor and it will throw an Exception.
        if len(val_losses) >= prev_losses_number+1:
            decreases = [abs(1-val_losses[-i-1]/val_losses[-i]) < self.val_loss_reduce_lr_thr for i in range(prev_losses_number, 0, -1)]
            print(decreases)
            print(self.val_loss_reduce_lr_thr)
            if sum(decreases) == prev_losses_number:
                self.sess.run(self.lr_tensor.assign(self.lr_tensor * decrease_lr))
                self.val_loss_reduce_lr_thr *= decrease_thr
                self.val_loss_reduce_lr_counter = 0 # what was this.
                self.lr_updated_counter += 1 # In ModelBase.py
                # Check I can do this after vacations.
                #log("Decreasing Learning rate to: "+str(self.sess.run(self.lr_tensor)))
                log("Decreasing Learning rate")
                return True

        return False
