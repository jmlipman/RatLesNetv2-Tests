import os
import tensorflow as tf
import numpy as np
from datetime import datetime
import twitter
import random

# Useful resource: https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514#file-tensorboard_logging-py-L41

def dice_coef(y_pred, y_true):
    num_samples = y_pred.shape[0]
    num_classes = y_pred.shape[-1]
    results = np.zeros((num_samples, num_classes))
    y_pred = np.argmax(y_pred, axis=-1)
    y_true = np.argmax(y_true, axis=-1)

    for i in range(num_samples):
        for c in range(num_classes):
            a = y_pred[i] == c
            b = y_true[i] == c
            if np.sum(b) == 0: # If no lesion in the y_true
                if np.sum(a) == 0: # No lesion predicted
                    result = 1.0
                else:
                    result = (np.sum(b==0)-np.sum(a))*1.0 / np.sum(b==0)
            else: # Actual Dice
                num = 2 * np.sum(a * b)
                denom = np.sum(a) + np.sum(b)
                result = num / denom
            results[i, c] = result
    return results

def crossValidation(rows, folds, randomize=True):
    """This function will randomly calculate indices of rows to perform X-val.
            data is a list of "folds" elements
            data[0] list with training indices
            data[1] list with testing indices
    """    

    indices = [i for i in range(rows)]
    if randomize:
        random.shuffle(indices)
    sizeFold = int(rows/folds)

    res = []
    for i in range(folds):
        testIndices = indices[i:i+sizeFold]
        trainIndices = list(set(indices)-set(testIndices))
        res.append([trainIndices,testIndices])

    return res

def log(text):
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # I dont need to save anything because it's being saved by sacred in cout.txt
    # Flush is set to True for SLURM (CSC)
    print(now + ": " + text, flush=True)

def binary_contour(volumes):
    W = np.ones((3,3,3,volumes.shape[-1],1))
    for i in range(volumes.shape[-1]):
        W[1,1,1,i,0] = -26
    conv = tf.nn.conv3d(volumes, W, strides=[1,1,1,1,1], padding="SAME")
    print(conv)
    return tf.cast(tf.less(conv, 0), tf.float32)

def reduce_median(volumes, axis=1):
    median = tf.contrib.distributions.percentile(volumes, 50.0, interpolation='lower', axis=axis)
    median += tf.contrib.distributions.percentile(volumes, 50.0, interpolation='higher', axis=axis)
    median /= 2
    return median

class Twitter:

    def __init__(self):
        # @jmlipman_bot2
        ck = "PAgpS5aX1yduUkvPwwsA"
        cs = "E7YvwTyIBnhQsWctyNN6pXeGqGj19fKtqRPcgpk"
        atk = "562341307-ZBrRQ5PhjCethDPVfpUJQNTk9F6wv366I9sluKy6"
        ats = "eis3zoHx5POw34d3N3m5n16JKjA5eGQ6piHXQ4vAdgce4"
        self.api = twitter.Api(consumer_key=ck, consumer_secret=cs,
                access_token_key=atk, access_token_secret=ats)

    def tweet(self, text):
        text = "@jmlipman " + text
        if len(text) > 280:
            text = text[:277] + "..."
            print("Warning: too long tweet")

        self.api.PostUpdate(text)


class TB_Log:

    def __init__(self, path, sess):
        """TB_Log will help logging info into Tensorboard.
        """
        self.sess = sess
        self.path = path

        if not os.path.isdir(self.path):
            os.makedirs(self.path)
        self.id = len(os.listdir(self.path)) + 1
        #self.writer = tf.summary.FileWriter(self.path + "run_"+str(self.id), self.sess.graph)
        #results/asdasd/asdasd/1/ --> results/asdasd_asdasd_1
        self.path = self.path[:-1].split("/")
        self.path = self.path[0] + "/tensorboard/" + "_".join(self.path[1:])
        self.writer = tf.summary.FileWriter(self.path, self.sess.graph)

    def save_scalar(self, vector, name=None):
        
        if len(vector.shape) > 1:
            raise Exception("'vector' must have a single dimension: "+str(mat.shape))

        if name is None:
            name = "_"
        
        for i in range(vector.shape[0]):
            summary = tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=vector[i])])
            self.writer.add_summary(summary, i)

    def add_scalar(self, val, name, it):
        summary = tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=val)])
        self.writer.add_summary(summary, it)


    def save_histogram(self, mat, name=None, bins=1000):
        """Logs the histogram of a list/vector of values."""

        if name is None:
            name = "_"
        
        for i in range(mat.shape[0]):

            values = np.reshape(mat[i], (-1))
            # Create histogram using numpy        
            counts, bin_edges = np.histogram(values, bins=bins)

            # Fill fields of histogram proto
            hist = tf.HistogramProto()
            hist.min = float(np.min(values))
            hist.max = float(np.max(values))
            hist.num = int(np.prod(values.shape))
            hist.sum = float(np.sum(values))
            hist.sum_squares = float(np.sum(values**2))

            # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
            # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
            # Thus, we drop the start of the first bin
            bin_edges = bin_edges[1:]

            # Add bin edges and counts
            for edge in bin_edges:
                hist.bucket_limit.append(edge)
            for c in counts:
                hist.bucket.append(c)

            # Create and write Summary
            summary = tf.Summary(value=[tf.Summary.Value(tag=name, histo=hist)])
            self.writer.add_summary(summary, i)
            self.writer.flush()
