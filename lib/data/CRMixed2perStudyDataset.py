import torch, os, random
import nibabel as nib
import numpy as np
from lib.utils import np2cuda, surfacedist
from lib.losses import *


class CRMixed2perStudyDataset(torch.utils.data.Dataset):

    def __init__(self, split, loss=None, brainmask=None, overlap=None, dev=None):
        """
           In this dataset, I will train with 2 samples per study per time-point:
           One with the largest lesion, and one with the smallest lesion.
           Validation is the median lesion.
           Test is the rest.

           Args:
            `split`: "train", "validation", "test".
            `loss`: loss function used in the code. This is used to compute
             the weights because different loss functions use diff. weights.
            `dev`: device where the data will be brought (cuda/cpu)
            `brainmask, overlap`: Nothing, for compatibility reasons.
        """
        self.split = split
        self.loss = loss
        self.dev = dev
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
        elif pc_name == "sampo-tipagpu1":
            # Sampo
            self.PATH = "/home/users/miguelv/data/in/CR_DATA/"
        else:
            raise Exception("Unknown PC")

        studies = ["02NOV2016", "08JAN2015", "07MAY2015", "16JUN2015", "21JUL2015", "03AUG2015", "17NOV2015", "22DEC2015", "03MAY2016", "27JUN2017", "02OCT2017", "16NOV2017"]

        # Collecting the files in a list to read them when need it
        self.list = []
        brains, lesion_size = {}, {}
        nolesions = []
        c = 0
        for study in studies:
            self.lesion = []
            for root, subdirs, files in os.walk(self.PATH + study + "/"):
                if "scan.nii.gz" in files:
                    study_timepoint = "_".join(root.split("/")[-3:-1])
                    if not study_timepoint in brains.keys():
                        brains[study_timepoint] = []
                        lesion_size[study_timepoint] = []
                    
                    if "scan_miguel.nii.gz" in files or "scan_lesion.nii.gz" in files:
                        brains[study_timepoint].append(root + "/")
                        c += 1
                        if "scan_miguel.nii.gz" in files:
                            lesion_file = root + "/scan_miguel.nii.gz"
                        else:
                            lesion_file = root + "/scan_lesion.nii.gz"
                        lesion_size[study_timepoint].append(nib.load(lesion_file).get_data().sum())
                    else:
                        nolesions.append(root + "/")

        if split == "train":
            for i, study_timepoint in enumerate(brains.keys()):
                idx = np.argsort(lesion_size[study_timepoint])
                self.list.append(brains[study_timepoint][idx[0]])
                self.list.append(brains[study_timepoint][idx[-1]])
        elif split == "validation":
            for i, study_timepoint in enumerate(brains.keys()):
                idx = np.argsort(lesion_size[study_timepoint])
                self.list.append(brains[study_timepoint][idx[int(len(idx)/2)]])
        else:
            for i, study_timepoint in enumerate(brains.keys()):
                idx_ = np.argsort(lesion_size[study_timepoint])
                idx = list(idx_[1:int(len(idx_)/2)]) + list(idx_[int(len(idx_)/2+1):-1])
                self.list.extend([brains[study_timepoint][i_] for i_ in idx])

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

        if os.path.isfile(target+"scan_miguel.nii.gz"):
            Y = nib.load(target+"scan_miguel.nii.gz").get_data()
            Y = np.moveaxis(Y, -1, 0) # Move depth to the beginning
            Y = np.stack([1.0*(Y==j) for j in range(2)], axis=0)
        elif os.path.isfile(target+"scan_lesion.nii.gz"):
            Y = nib.load(target+"scan_lesion.nii.gz").get_data()
            Y = np.moveaxis(Y, -1, 0) # Move depth to the beginning
            Y = np.stack([1.0*(Y==j) for j in range(2)], axis=0)
        else:
            #Y = np.ones(list(X.shape[2:]))
            Y = np.ones([2] + list(X.shape[2:]))
            Y[1,:,:,:] = 0
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


