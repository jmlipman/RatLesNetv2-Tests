import numpy as np
import nibabel as nib
import pandas as pd
import os
import random
from scipy import ndimage
from lib.data.BaseData import BaseData
import torch

class Data(BaseData):
    def __init__(self, batch=1, randomize=False, seg="miguel"):
        """In the initializer we gather the location of the files that will
           be used for training, testing and validation.
           Counters will be set to 0, so that during the data retrieval
           (one by one) we can know what's the next subject to retrieve.
           It will also read the survival CSV.

           Args:
           `batch`: batch size.
           `randomize`: If True, every time this is executed the data within
            the splits for training, test and validation will be different.
            `seg`: which segmentation to use, mine or CRs.
        """
        # Get the lists
        pc_name = os.uname()[1]
        if pc_name == "taito-gpu.csc.fi":
            # CSC TAITO-GPU
            raise Exception("It should not be here")
        elif pc_name == "FUJ":
            # LOCAL
            self.PATH = "/media/miguelv/HD1/CR_DATA/02NOV2016/"
        elif pc_name == "nmrcs3":
            # NMR-CS3
            self.PATH = "/home/miguelv/data/in/CR_DATA/02NOV2016/"
        else:
            raise Exception("Unknown PC")

        self.randomize = randomize
        self.ext = "_miguel" if seg == "miguel" else "_lesion"
        self.batch = batch
        # These dictionaries will contain the X and Y data
        # in case I want it to be loaded in the memory
        self.X_container = {}
        self.Y_container = {}
        self.loading_in_memory = False

        # Counters for training, testing and validation
        self.counters = [0, 0, 0]

        # 4 different groups that need to be balanced
        lesion2h = []
        no_lesion2h = []
        lesion24h = []
        no_lesion24h = []
        for root, subdirs, files in os.walk(self.PATH + "2h/"):
            if "scan.nii.gz" in files:
                if "scan"+self.ext+".nii.gz" in files:
                    lesion2h.append(root + "/")
                else:
                    no_lesion2h.append(root + "/")

        for root, subdirs, files in os.walk(self.PATH + "24h/"):
            if "scan.nii.gz" in files:
                if "scan"+self.ext+".nii.gz" in files:
                    lesion24h.append(root + "/")
                else:
                    no_lesion24h.append(root + "/")

        self.groups = [lesion2h, no_lesion2h, lesion24h, no_lesion24h]

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
        timepoint, subject = target.split("/")[-3:-1]
        id_ = timepoint+"_"+subject

        # This if controls that the behavior is different when the container
        # used when loadInMemory is not empty.
        try:
            X_train = self.X_container[id_]["in_volume"]
            Y_train = self.Y_container[id_]["out_segmentation"]
        except:
            # Read the actual data
            X_train = nib.load(target+"scan.nii.gz").get_data()
            X_train = np.moveaxis(X_train, -1, 0)
            X_train = np.expand_dims(X_train, axis=0)

            if os.path.isfile(target+"scan"+self.ext+".nii.gz"):
                Y_train = nib.load(target+"scan"+self.ext+".nii.gz").get_data()
                Y_train = np.stack([1.0*(Y_train==j) for j in range(2)], axis=-1)
            else:
                Y_train = np.ones(list(X_train.shape[1:-1])+[2])
                Y_train[:,:,:,1] = 0

            Y_train = np.moveaxis(Y_train, -1, 0)
            Y_train = np.expand_dims(Y_train, 0)

        self.counters[c] += 1

        # The ID must be a list, so that I can later iterate over it
        return torch.from_numpy(X_train.astype(np.float32)).cuda(), torch.from_numpy(Y_train.astype(np.float32)).cuda(), [id_]


    def getNextTrainingBatch(self):
        # Returns (1,240,240,155,4), Age, Survival
        # Take care of the batch thing in here

        d_tmp = self.loadNext(self.getFiles("training"), 0)
        if d_tmp is None:
            return None
        X_train, Y_train, target = d_tmp
        X = [X_train]
        #X = {"in_volume": X_train}
        #Y = {"out_segmentation": Y_train}
        return X, Y_train, target


    def getNextTestBatch(self):
        # Returns (1,240,240,155,4), Age, Survival
        d_tmp = self.loadNext(self.getFiles("test"), 1)
        if d_tmp is None:
            return None
        X_test, Y_test, target = d_tmp
        X = [X_test]
        #X = {"in_volume": X_test}
        #Y = {"out_segmentation": Y_test}
        return X, Y_test, target

    def getNextValidationBatch(self):
        # Returns (1,240,240,155,4), Age, Survival
        d_tmp = self.loadNext(self.getFiles("validation"), 2)
        if d_tmp is None:
            return None
        X_val, Y_val, target = d_tmp
        X = [X_val]
        #X = {"in_volume": X_val}
        #Y = {"out_segmentation": Y_val}
        return X, Y_val, target

