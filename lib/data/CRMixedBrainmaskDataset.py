import torch, os, random
import nibabel as nib
import numpy as np
from lib.utils import np2cuda, surfacedist
from lib.losses import *


class CRMixedBrainmaskDataset(torch.utils.data.Dataset):

    def __init__(self, split, loss=None, brainmask=False, overlap=False, dev=None):
        # 0-1 normal lesions on the brains with brainmask
        # I can make 0-1-2 overlapping
        # I can make 0-1-2 non-overlapping
        """This class creates a train/test/val division with the scans that
           have a brainmask label.

           Args:
            `split`: "train", "validation", "test".
            `loss`: loss function used in the code. This is used to compute
              the weights because different loss functions use diff. weights.
            `brainmask`: if True, Y will include the brain mask, i.e.,
              0 -> background, 1 -> lesion, 2-> brainmask.
              For Multi-task, this should be True.
            `overlap`: if True, lesion will be overlapping brain mask. If False
              lesion voxels won't belong to brainmask class.
              For Multi-Label, this should be True.
            `dev`: device where the data will be brought (cuda/cpu)
        """
        self.split = split
        self.loss = loss
        self.brainmask = brainmask
        self.overlap = overlap
        self.dev = dev
        self.tr_size = 5 # Number of samples per time-point

        if self.tr_size > 10:
            raise Exception("Are you sure? because you only have 12 samples for 2h lesions...")

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
        brains = {}
        nolesions = []
        for study in studies:
            self.lesion = []
            for root, subdirs, files in os.walk(self.PATH + study + "/"):
                if "scan.nii.gz" in files and "scan_brainmask.nii.gz" in files:
                    timepoint = root.split("/")[-2]
                    if not timepoint in brains.keys():
                        brains[timepoint] = []
                    
                    if "scan_miguel.nii.gz" in files or "scan_lesion.nii.gz" in files:
                        brains[timepoint].append(root + "/")
                    else:
                        nolesions.append(root + "/")

        nolesions = sorted(nolesions)
        for timepoint in brains.keys():
            brains[timepoint] = sorted(brains[timepoint])

        if split == "train":
            for data in brains.values():
                self.list.extend(data[:self.tr_size])
        elif split == "validation":
            for data in brains.values():
                self.list.append(data[self.tr_size])
        else:
            for data in brains.values():
                self.list.extend(data[self.tr_size+1:])
            self.list += nolesions

        for l in sorted(self.list):
            print(l)
        raise Exception("llego")

        self.list = nolesions
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

        X_orig = nib.load(target+"scan.nii.gz").get_data()
        X = np.moveaxis(X_orig, -1, 0) # Move channels to the beginning
        X = np.moveaxis(X, -1, 1) # Move depth after channels
        X = np.expand_dims(X, axis=0)

        Y = np.zeros_like(X)
        rep = 2
        if self.brainmask:
            rep += 1
        Y = np.concatenate([Y for _ in range(rep)], axis=1)

        if os.path.isfile(target+"scan_miguel.nii.gz"):
            Y_ = nib.load(target+"scan_miguel.nii.gz").get_data()
        elif os.path.isfile(target+"scan_lesion.nii.gz"):
            Y_ = nib.load(target+"scan_lesion.nii.gz").get_data()
        else:
            Y_ = np.zeros(X_orig.shape[:-1])
        Y_ = np.moveaxis(Y_, -1, 0) # Move depth to the beginning
        #Y_ = np.stack([1.0*(Y_==j) for j in range(2)], axis=0)
        Y[0,1] = Y_
        #Y[0,:2] = Y_ # Put the lesion and non-lesion in Y

        if self.brainmask:
            brainmask = nib.load(target+"scan_brainmask.nii.gz").get_data()
            brainmask = np.moveaxis(brainmask, -1, 0) # Move depth to the beginning

            Y[0,2] = brainmask
            if not self.overlap:
                Y[0,2] -= Y[0,2]*Y[0,1]

        # Important question: should channel 0 represent non-lesion or non-brain?
        if not self.overlap:
            Y[0,0] = 1.0-((np.sum(Y[0,1:], axis=0))>0)
        else:
            Y[0,0] = 1.0-Y[0,1]


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


