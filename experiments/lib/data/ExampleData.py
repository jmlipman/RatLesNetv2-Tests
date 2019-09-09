import numpy as np
import nibabel as nib
import pandas as pd
import os
import random
from scipy import ndimage
from experiments.lib.data.BaseData import BaseData


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
            self.PATH = "/media/miguelv/HD1/CR_DATA/"
        elif pc_name == "nmrcs3":
            # NMR-CS3
            self.PATH = "/home/miguelv/data/in/CR_DATA/"
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
        group1 = []
        group2 = []
        group3 = []
        group4 = []

        self.groups = [group1, ... group4]

        # At the end, all test/trainging/validation files will be stored
        # in the following lists of lists.
        # all_test_files[K] is a list containing files for the fold K.
        # If not doing K-Folds, then len(all_test_files) = 1.
        self.all_test_files = [[]]
        self.all_training_files = [[]]
        self.all_validation_files = [[]]

    def split(self, folds=1, prop=[0.8]):
        """This function will split the data from self.groups.
        """

        for g in self.groups:
            split_train = int(len(g)*prop[0])
            self.all_training_files[0].extend(g[:split_train])
            self.all_validation_files[0].extend(g[split_train:])

        # Randomize the data.
        if self.randomize:
            random.shuffle(DATA)

        self.current_fold = 0


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
        id_ = actual ID of the sample.

        # This if controls that the behavior is different when the container
        # used when loadInMemory is not empty.
        try:
            X_train = self.X_container[id_]["in_volume"]
            Y_train = self.Y_container[id_]["out_segmentation"]
        except:
            # Read the actual data
            # Make sure it has the right dimensions
            X_train = LOAD DATA.
            Y_train = LOAD DATA.

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

