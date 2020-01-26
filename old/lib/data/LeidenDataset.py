import torch, os, random
import nibabel as nib
import numpy as np
from lib.utils import np2cuda, surfacedist
from lib.losses import *


class LeidenDataset(torch.utils.data.Dataset):

    def __init__(self, split, loss=None, dev=None, seg="IAM"):
        """This class will retrieve the whole data set that we have.
           For training/validation it is using 02NOV2016 study.
           For testing it is using the remaining studies.
           This script ensures that the training and validation sets
           are balanced.

           Args:
            `split`: "train", "validation", "test".
            `loss`: loss function used in the code. This is used to compute
             the weights because different loss functions use diff. weights.
            `dev`: device where the data will be brought (cuda/cpu)
            `seg`: Which segmentation is used. Either "IAM" or "SdJ"
        """
        self.split = split
        self.loss = loss
        self.dev = dev
        self.seg = seg

        # At this stage:
        # Train: 2 scans for each (10*2 = 20)
        # Validation: 1 scan for each (10*1 = 10)
        # Test: Rest.

        # Depending on the computer the data is located in a different folder
        pc_name = os.uname()[1]
        if pc_name == "taito-gpu.csc.fi":
            # CSC TAITO-GPU
            raise Exception("It should not be here")
        elif pc_name == "FUJ":
            # LOCAL
            self.PATH = "/media/miguelv/HD1/dataset_mulder/Leiden-Set/"
        elif pc_name == "nmrcs3":
            # NMR-CS3
            raise Exception("It should not be here")
            self.PATH = "/home/miguelv/data/in/CR_DATA/"
        else:
            raise Exception("Unknown PC")

        studies = ["12-14month_old_mice/24h", "12-14month_old_mice/48h", "12-14month_old_mice/4h", "20-24month_old_mice/24h", "20-24month_old_mice/48h", "20-24month_old_mice/4h", "3-5month_old_mice/24h", "3-5month_old_mice/4h", "3-5month_old_mice/48h", "3-5month_old_mice/D8"]

        # Collecting the files in a list to read them when need it
        self.list = []
        for study in studies:
            self.lesion = []
            for root, subdirs, files in os.walk(self.PATH + study + "/"):
                if "scan_lesion"+self.seg+".nii.gz" in files:
                    self.lesion.append(root + "/")
            if split == "train":
                self.list += self.lesion[:2]
            elif split == "validation":
                self.list += self.lesion[2:3]
            else:
                self.list += self.lesion[3:]

        # Randomize
        random.shuffle(self.list)

        # Loading into memory
        # For training and testing only
        if split == "train" or split == "validation":
            self.dataX = []
            self.dataY = []
            self.dataId = []
            self.dataW = []
            for i in range(len(self.list)):
                X, Y, id_, W = self._loadSubject(i)
                self.dataX.append(X)
                self.dataY.append(Y)
                self.dataId.append(id_)
                self.dataW.append(W)

    def __len__(self):
        return len(self.list)

    def __getitem__(self, idx):
        if self.split == "train" or self.split == "validation":
            return self.dataX[idx], self.dataY[idx], self.dataId[idx], self.dataW[idx]
        else:
            X, Y, id_, W = self._loadSubject(idx)
            return X, Y, id_, W

    def _loadSubject(self, idx):
        """This function will load a single subject.
           
           Args:
            `idx`: Index of the subject that will be read.

           Returns:
            `X`: raw brain scan.
            `Y`: labels.
            `id_`: id/name of the scan.
            `W`: weights (or None)
        """

        target = self.list[idx]
        study, timepoint, subject = target.split("/")[-4:-1]
        id_ = study + "_" + timepoint + "_" + subject


        X = nib.load(target+"scan.nii.gz").get_data()
        X = np.moveaxis(X, -1, 0) # Move depth to the beginning
        X = np.expand_dims(X, axis=0) # Add channel dim
        X = np.expand_dims(X, axis=0) # Add batch dim

        Y = nib.load(target+"scan_lesion"+self.seg+".nii.gz").get_data()
        Y = np.moveaxis(Y, -1, 0) # Move depth to the beginning
        Y = np.stack([1.0*(Y==j) for j in range(2)], axis=0)
        Y = np.expand_dims(Y, 0) #BCWHD



        W = self._computeWeight(Y)

        return np2cuda(X, self.dev), np2cuda(Y, self.dev), id_, W

    def _computeWeight(self, Y):
        """This function computes the weights of a manual segmentation.
           The weights are computed based on the loss function used.

           Args:
            `Y`: labels.

           Returns:
            `W`: weights
        """

        if self.loss == WeightedCrossEntropy_DistanceMap:
            return np2cuda(surfacedist(Y), self.dev)

        elif self.loss == WeightedCrossEntropy_ClassBalance:
            # Number of voxels per image
            numvox = np.prod(Y.shape[2:])
            # Number of 1s in each channel in each sample
            ones = np.sum(Y, axis=(2,3,4))
            # The weights are inversely proportional to the number of ones
            # so that if there are very few voxels from one category
            weights_ = 1 - ones/numvox
            Y = np.moveaxis(np.moveaxis(Y, 0, -1), 0, -1)
            weights = np.moveaxis(np.moveaxis(Y*weights_, -1, 0), -1, 0)
            return np2cuda(np.sum(weights, axis=1), self.dev)

        else:
            return None


