import torch, os, random
import nibabel as nib
import numpy as np
from lib.utils import np2cuda, surfacedist
from lib.losses import *


class CRAllDataset(torch.utils.data.Dataset):

    def __init__(self, split, loss=None, seg="miguel"):
        """This class will retrieve the whole data set that we have.
           For training/validation it is using 02NOV2016 study.
           For testing it is using the remaining studies.
           This script ensures that the training and validation sets
           are balanced.

           Args:
            `split`: "train", "validation", "test".
            `loss`: loss function used in the code. This is used to compute
             the weights because different loss functions use diff. weights.
            `seg`: Which segmentation is retrieving.
        """
        self.split = split
        self.loss = loss
        # Split between training and validation from 02NOV2016
        prop = 0.8 

        # Depending on the computer the data is located in a different folder
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

        if split == "train" or split == "validation":
            self.ext = "_miguel"
            studies = ["02NOV2016/2h", "02NOV2016/24h"]
        else:
            self.ext = "_lesion"
            studies = ["08JAN2015", "07MAY2015", "16JUN2015", "21JUL2015", "03AUG2015", "17NOV2015", "22DEC2015", "03MAY2016", "27JUN2017", "02OCT2017", "16NOV2017"]

        # Collecting the files in a list to read them when need it
        self.list = []
        for study in studies:
            self.lesion = []
            self.no_lesion = []
            for root, subdirs, files in os.walk(self.PATH + study + "/"):
                if "scan.nii.gz" in files:
                    if "scan"+self.ext+".nii.gz" in files:
                        self.lesion.append(root + "/")
                    else:
                        self.no_lesion.append(root + "/")
            if split == "train":
                self.list += self.lesion[:int(len(self.lesion)*prop)]
                self.list += self.no_lesion[:int(len(self.no_lesion)*prop)]
            elif split == "validation":
                self.list += self.lesion[int(len(self.lesion)*prop):]
                self.list += self.no_lesion[int(len(self.no_lesion)*prop):]
            else:
                self.list += self.lesion
                self.list += self.no_lesion


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
        X = np.moveaxis(X, -1, 0) # Move channels to the beginning
        X = np.moveaxis(X, -1, 1) # Move depth after channels
        X = np.expand_dims(X, axis=0)

        if os.path.isfile(target+"scan"+self.ext+".nii.gz"):
            Y = nib.load(target+"scan"+self.ext+".nii.gz").get_data()
            Y = np.moveaxis(Y, -1, 0) # Move depth to the beginning
            Y = np.stack([1.0*(Y==j) for j in range(2)], axis=0)
        else:
            #Y = np.ones(list(X.shape[2:]))
            Y = np.ones([2] + list(X.shape[2:]))
            Y[1,:,:,:] = 0
        Y = np.expand_dims(Y, 0) #BCWHD

        W = self._computeWeight(Y)

        return np2cuda(X), np2cuda(Y), id_, W

    def _computeWeight(self, Y):
        """This function computes the weights of a manual segmentation.
           The weights are computed based on the loss function used.

           Args:
            `Y`: labels.

           Returns:
            `W`: weights
        """

        if self.loss == WeightedCrossEntropy_DistanceMap:
            return np2cuda(surfacedist(Y))

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
            return np2cuda(np.sum(weights, axis=1))

        else:
            return None


