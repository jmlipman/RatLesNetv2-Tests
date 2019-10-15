import numpy as np
import nibabel as nib
import pandas as pd
import os
import random
from scipy import ndimage
from experiments.lib.data.BaseData import BaseData
from scipy.ndimage import distance_transform_edt as dist


class Data(BaseData):
    def __init__(self, batch=1, randomize=False, depth_first=True, seg="miguel"):
        """In the initializer we gather the location of the files that will
           be used for training, testing and validation.
           Counters will be set to 0, so that during the data retrieval
           (one by one) we can know what's the next subject to retrieve.
           It will also read the survival CSV.

           Args:
           `batch`: batch size.
           `randomize`: If True, every time this is executed the data within
            the splits for training, test and validation will be different.
            `depth_first`: As TF requires BDWHC, if this is True then the
            slices are moved to the front: DWH.
            `seg`: which segmentation to use, mine or CRs.
        """
        # Get the lists
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
        self.depth_first = depth_first
        self.ext = "_miguel" if seg == "miguel" else "_lesion"
        self.batch = batch
        # These dictionaries will contain the X and Y data
        # in case I want it to be loaded in the memory
        self.X_container = {}
        self.Y_container = {}
        self.loading_in_memory = False

        # Studies for "Test"
        # Note: 10SEP2015 and 26MAY2016 have 17 slices.
        studies = ["08JAN2015", "07MAY2015", "16JUN2015", "21JUL2015", "03AUG2015", "17NOV2015", "22DEC2015", "03MAY2016", "27JUN2017", "02OCT2017", "16NOV2017"]
        #studies = ["03AUG2015"]

        # Counters for training, testing and validation
        self.counters = [0, 0, 0]

        # 4 different groups that need to be balanced
        lesion2h = []
        no_lesion2h = []
        lesion24h = []
        no_lesion24h = []
        for root, subdirs, files in os.walk(self.PATH + "02NOV2016/2h/"):
            if "scan.nii.gz" in files:
                if "scan"+self.ext+".nii.gz" in files:
                    lesion2h.append(root + "/")
                else:
                    no_lesion2h.append(root + "/")

        for root, subdirs, files in os.walk(self.PATH + "02NOV2016/24h/"):
            if "scan.nii.gz" in files:
                if "scan"+self.ext+".nii.gz" in files:
                    lesion24h.append(root + "/")
                else:
                    no_lesion24h.append(root + "/")

        self.groups = [lesion2h, no_lesion2h, lesion24h, no_lesion24h]
        # Brains that will be used for test
        self.all_test_files = [[]]
        for study in studies:
            for root, subdirs, files in os.walk(self.PATH + study + "/"):
                if "scan.nii.gz" in files:
                    self.all_test_files[0].append(root + "/")

    def split(self, folds=1, prop=[0.8]):
        """This function will split the data accordingly.
           The only value seen is prop[0]
           Folds arg is ignored.
        """
        self.all_training_files = [[]]
        self.all_validation_files = [[]]
        for g in self.groups:
            split_train = int(len(g)*prop[0])
            self.all_training_files[0].extend(g[:split_train])
            self.all_validation_files[0].extend(g[split_train:])

        if self.randomize:
            random.shuffle(self.all_training_files[0])
            random.shuffle(self.all_validation_files[0])

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
        study, timepoint, subject = target.split("/")[-4:-1]
        id_ = study + "_" + timepoint + "_" + subject

        # This if controls that the behavior is different when the container
        # used when loadInMemory is not empty.
        # TODO:
        # - Put in a single container all data loaded.
        # - Read X, process Y, pass to the child class.
        # - Create child class to get X,Y,ids and return everything processed.
        try:
            X_train = self.X_container[id_]["in_volume"]
            Y_train = self.Y_container[id_]["out_segmentation"]
        except:
            # Read the actual data
            X_train = nib.load(target+"scan.nii.gz").get_data()

            if self.depth_first:
                X_train = np.moveaxis(X_train, 2, 0)
            X_train = np.expand_dims(X_train, axis=0)

            if c == 1:
                ext = "_lesion"
            else:
                ext = self.ext
            if os.path.isfile(target+"scan"+ext+".nii.gz"):
                Y_train = nib.load(target+"scan"+ext+".nii.gz").get_data()
                #Y_train = np.expand_dims(Y_train, -1)
                Y_train = np.stack([1.0*(Y_train==j) for j in range(2)], axis=-1)
                if self.depth_first:
                    Y_train = np.moveaxis(Y_train, 2, 0)
            else:
                Y_train = np.ones(list(X_train.shape[1:-1])+[2])
                Y_train[:,:,:,1] = 0

            Y_train = np.expand_dims(Y_train, 0)

        self.counters[c] += 1

        # The ID must be a list, so that I can later iterate over it
        return X_train, Y_train, [id_]

    def onehot2prob(self, data):
        distances = np.zeros_like(data)

        for b in range(data.shape[0]):
            for i in range(data.shape[-1]):
                posmask = data[b,:,:,:,i].astype(bool)
                negmask = ~posmask
                distances[b,:,:,:,i] = dist(negmask) * negmask - (dist(posmask) - 1) * posmask
                #distances[b,:,:,:,i] = dist(~data[b,:,:,:,i].astype(bool))

        #return (distances-distances.min())/(distances.max()-distances.min())
        return distances

    def surfacedist(self, data):
        # I am assuming there is only one label
        data = np.argmax(data, axis=-1)
        surface = np.zeros(data.shape)
        distances = np.zeros(data.shape)
        for b in range(data.shape[0]):
            for x in range(data.shape[1]):
                for y in range(data.shape[2]):
                    for z in range(data.shape[3]):
                        if data[b,x,y,z]==1:
                            piece = data[b,x-1:x+2,y-1:y+2,z-1:z+2]
                            if np.sum(piece) != 27:
                                surface[b,x,y,z] = 1
            distances[b,:,:,:] = np.log(dist(1-surface)+np.e)

        return distances

    def getNextTrainingBatch(self):
        # Returns (1,240,240,155,4), Age, Survival
        # Take care of the batch thing in here

        d_tmp = self.loadNext(self.getFiles("training"), 0)
        if d_tmp is None:
            return None
        X_train, Y_train, target = d_tmp
        X = {"in_volume": X_train, "in_weights": self.onehot2prob(Y_train)}
        Y = {"out_segmentation": Y_train}
        #Y = {"out_segmentation": self.onehot2prob(Y_train)}
        #Y = {"out_segmentation": Y_train}
        return X, Y, target


    def getNextTestBatch(self):
        # Returns (1,240,240,155,4), Age, Survival
        d_tmp = self.loadNext(self.getFiles("test"), 1)
        if d_tmp is None:
            return None
        X_test, Y_test, target = d_tmp
        X = {"in_volume": X_test}
        Y = {"out_segmentation": Y_test} # When using boundary loss, I don't need onehot2prob here.
        return X, Y, target

    def getNextValidationBatch(self):
        # Returns (1,240,240,155,4), Age, Survival
        d_tmp = self.loadNext(self.getFiles("validation"), 2)
        if d_tmp is None:
            return None
        X_val, Y_val, target = d_tmp
        X = {"in_volume": X_val, "in_weights": self.onehot2prob(Y_val)}
        Y = {"out_segmentation": Y_val}
        #Y = {"out_segmentation": self.onehot2prob(Y_val)}
        #Y = {"out_segmentation": Y_val}
        return X, Y, target

