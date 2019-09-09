import numpy as np
import nibabel as nib
import pandas as pd
import os
import random
from scipy import ndimage
from experiments.lib.data.BaseData import BaseData

# What is this?
"""This is used for testing a "post-processing NN".
"""

class Data(BaseData):
    def __init__(self, batch=1, randomize=False):
        """This function basically gathers in groups the name of the files that will be read.
        """
        # Checks the name of the PC in which the script is being executed.
        pc_name = os.uname()[1]
        if pc_name == "taito-gpu.csc.fi":
            # CSC TAITO-GPU
            raise Exception("It should not be here")
        elif pc_name == "FUJ":
            # LOCAL
            self.PATH = "/home/miguelv/data/in/Postprocessing_Test/"
        elif pc_name == "nmrcs3":
            # NMR-CS3
            self.PATH = "/home/miguelv/data/in/CR_DATA/"
            raise Exception("It should not be here")
        else:
            raise Exception("Unknown PC")

        self.randomize = randomize
        self.batch = batch

        # These dictionaries will contain the X and Y data
        # in case I want it to be loaded in the memory
        self.X_container = {}
        self.Y_container = {}
        self.loading_in_memory = False

        # Studies for "Test"
        # Counters for training, testing and validation
        self.counters = [0, 0, 0]

        # 4 different groups that need to be balanced
        lesion24h = ["24h_12", "24h_13", "24h_2", "24h_21", "24h_22", "24h_25", "24h_3", "24h_32", "24h_35", "24h_43", "24h_45", "24h_5"]
        nolesion24h = ["24h_11", "24h_14", "24h_15", "24h_23", "24h_24", "24h_31", "24h_33", "24h_34", "24h_4", "24h_41", "24h_42", "24h_44"]
        lesion2h = ["2h_16", "2h_17", "2h_18", "2h_19", "2h_26", "2h_28", "2h_29", "2h_36", "2h_38", "2h_40", "2h_50", "2h_6"]
        nolesion2h = ["2h_20", "2h_27", "2h_30", "2h_37", "2h_39", "2h_47", "2h_48", "2h_51", "2h_52", "2h_7", "2h_8", "2h_9"]

        self.groups = [lesion24h, nolesion24h, lesion2h, nolesion2h]

        # At the end, all test/trainging/validation files will be stored
        # in the following lists of lists.
        # all_test_files[K] is a list containing files for the fold K.
        # If not doing K-Folds, then len(all_test_files) = 1.
        self.all_test_files = [[]]
        self.all_training_files = [[]]
        self.all_validation_files = [[]]

    def loadNext(self, files, c):
        """This function will load the next sample from the dataset.

           Args:
            `files`: list of file locations that contain the data to be loaded.
            `c`: indicates which split of the dataset to use (train, test, val)

           Returns:
            Data needed for the neural network, including X, Y, and ids.
        """

        # If I have already retrieved all data, return None
        if self.counters[c] == len(files):
            self.counters[c] += 1
            return None
        # Next time, it will start over
        elif self.counters[c] > len(files):
            self.counters[c] = 0

        target = files[self.counters[c]]
        id_ = target

        # This if controls that the behavior is different when the container
        # used when loadInMemory is not empty.
        try:
            X_train = self.X_container[id_]["in_volume"]
            Y_train = self.Y_container[id_]["out_segmentation"]
        except:
            # Read the actual data
            # Make sure it has the right dimensions
            mri = nib.load(self.PATH + "mri/" + target + ".nii.gz").get_data()
            incomplete = nib.load(self.PATH + "incomplete/" + target + ".nii.gz").get_data()
            incomplete = np.expand_dims(incomplete, axis=-1)
            X_train = np.concatenate([mri, incomplete], axis=-1)
            X_train = np.moveaxis(X_train, 2, 0)
            X_train = np.expand_dims(X_train, axis=0)

            if os.path.isfile(self.PATH + "labels/"+target+".nii.gz"):
                Y_train = nib.load(self.PATH + "labels/" + target + ".nii.gz").get_data()
                Y_train = np.stack([1.0*(Y_train==j) for j in range(2)], axis=-1)
                Y_train = np.moveaxis(Y_train, 2, 0)
            else:
                Y_train = np.ones(list(X_train.shape[1:-1]) + [2])
                Y_train[:,:,:,1] = 0
            Y_train = np.expand_dims(Y_train, axis=0)

        self.counters[c] += 1

        # The ID must be a list, so that I can later iterate over it
        return X_train, Y_train, [id_]


    def getNextTrainingBatch(self):
        # Returns (1,240,240,155,4), Age, Survival
        # Take care of the batch thing in here

        d_tmp = self.loadNext(self.getFiles("training"), 0)
        if d_tmp is None:
            return None
        X_train, Y_train, target = d_tmp
        X = {"in_volume": X_train}
        Y = {"out_segmentation": Y_train}
        return X, Y, target


    def getNextTestBatch(self):
        # Returns (1,240,240,155,4), Age, Survival
        d_tmp = self.loadNext(self.getFiles("test"), 1)
        if d_tmp is None:
            return None
        X_test, Y_test, target = d_tmp
        X = {"in_volume": X_test}
        Y = {"out_segmentation": Y_test}
        return X, Y, target

    def getNextValidationBatch(self):
        # Returns (1,240,240,155,4), Age, Survival
        d_tmp = self.loadNext(self.getFiles("validation"), 2)
        if d_tmp is None:
            return None
        X_val, Y_val, target = d_tmp
        X = {"in_volume": X_val}
        Y = {"out_segmentation": Y_val}
        return X, Y, target

