import numpy as np
import nibabel as nib
import pandas as pd
import os
import random
from scipy import ndimage
from experiments.lib.data.BaseData import BaseData

#########data.split(folds=1, prop=[0.7, 0.2, 0.1])
class Data(BaseData):
    def __init__(self, batch=1, randomize=False, depth_first=True, only_GTR=False, sliced=False, bounding_box=True):
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
            `only_GTR`: As BraTS only wants us to calculate the survival day
            on those who were GTR, if this is True then it will return None
            instead of the Age and Survival days in those cases it is not GTR.
            `sliced`: whether the data is sliced.
            `bounding_box`: whether to use the data contained in the bounding box
        """
        # Get the lists
        pc_name = os.uname()[1]
        if pc_name == "taito-gpu.csc.fi":
            # CSC TAITO-GPU
            self.PATH = "/wrk/valverde/BraTS/"
        elif pc_name == "FUJ":
            # LOCAL
            self.PATH = "/home/miguelv/Downloads/MICCAI_BraTS_2019_Data_Training/"
        elif pc_name == "nmrcs3":
            # NMR-CS3
            self.PATH = "/home/miguelv/data/in/Brats/"
        else:
            raise Exception("Unknown PC")

        pathHGG = self.PATH + "HGG/"
        pathLGG = self.PATH + "LGG/"
        pathSurvival = self.PATH + "survival_data.csv"

        self.randomize = randomize
        self.depth_first = depth_first
        self.only_GTR = only_GTR
        self.sliced = sliced
        self.bounding_box = bounding_box

        self.pool = []

        # Counters for training, testing and validation
        self.counters = [0, 0, 0]

        filesHGG = ["HGG/" + f for f in os.listdir(pathHGG)]
        totalHGG = len(filesHGG)
        filesLGG = ["LGG/" + f for f in os.listdir(pathLGG)]
        totalLGG = len(filesLGG)

        self.groups = [filesHGG, filesLGG]

        # Reading the CSV
        self.csv_data = pd.read_csv(pathSurvival)

    def loadNext(self, files, c):
        """This function will load the next sample from the dataset.

           Args:
            `files`: list of file locations that contain the data to be loaded.
            `c`: indicates which split of the dataset to use (train, test, val)

           Returns:
            Data needed for the neural network, including X, Y, and ids.
        """

        # In case data is sliced. If there is some data in this pool
        # we simply retrieve it until it is empty.
        if len(self.pool) > 0:
            return self.pool.pop()

        # If I have already retrieved all data, return None
        if self.counters[c] == len(files):
            self.counters[c] += 1
            return None
        # Next time, it will start over
        elif self.counters[c] > len(files):
            self.counters[c] = 0

        target = files[self.counters[c]]
        gtype, target = target.split("/")

        # Read the actual data
        mods = ["flair", "t1", "t1ce", "t2"]
        X_train = np.stack([self.standardize(nib.load(self.PATH+gtype+"/"+target+"/"+target+"_"+mod+".nii.gz").get_data()) for mod in mods], axis=-1)

        if self.depth_first:
            X_train = np.moveaxis(X_train, 2, 0)
        X_train = np.expand_dims(X_train, axis=0)

        tmp_path = self.PATH + gtype + "/" + target + "/" + target + "_seg.nii.gz"
        Y_train = nib.load(tmp_path).get_data()

        if self.depth_first:
            Y_train = np.moveaxis(Y_train, 2, 0)
        Y_train[Y_train==4] = 3
        Y_train = self.one_hot(Y_train, 4)

        Y_train = np.expand_dims(Y_train, 0)

        # Filtering some data based on the initial parameters.
        age, survival = None, None
        if target in list(self.csv_data["BraTS19ID"]):
            row = self.csv_data[self.csv_data["BraTS19ID"].str.contains(target)]
            if not row.Age.isna().bool() and not row.Survival.isna().bool():
                if self.only_GTR:
                    if (row.ResectionStatus == "GTR").bool():
                        try:
                            age = float(row.Age)
                        except:
                            age = None
                        try:
                            survival = float(row.Survival)
                        except:
                            survival = None
                else:
                    try:
                        age = float(row.Age)
                    except:
                        age = None
                    try:
                        survival = float(row.Survival)
                    except:
                        survival = None

        self.counters[c] += 1

        if self.bounding_box:
            # Fixed way. Resulting images are: (138, 173, 172)
            # This is a bounding box that contains all samples.
            loc = (slice(0, 1, None), slice(8, 146, None), slice(40, 213, None), slice(49, 221, None))
            X_train = X_train[loc]
            Y_train = Y_train[loc]

        elif self.sliced:
            # Slicing the data, adding it to the pool.
            self.pool.append((X_train[:,:128,:128,:128,:], Y_train[:,:128,:128,:128,:], age, survival, [gtype+"_"+target]))
            self.pool.append((X_train[:,:128,:128,-128:,:], Y_train[:,:128,:128,-128:,:], age, survival, [gtype+"_"+target]))
            self.pool.append((X_train[:,:128,-128:,:128,:], Y_train[:,:128,-128:,:128,:], age, survival, [gtype+"_"+target]))
            self.pool.append((X_train[:,:128,-128:,-128:,:], Y_train[:,:128,-128:,-128:,:], age, survival, [gtype+"_"+target]))
            self.pool.append((X_train[:,-128:,:128,:128,:], Y_train[:,-128:,:128,:128,:], age, survival, [gtype+"_"+target]))
            self.pool.append((X_train[:,-128:,:128,-128:,:], Y_train[:,-128:,:128,-128:,:], age, survival, [gtype+"_"+target]))
            self.pool.append((X_train[:,-128:,-128:,:128,:], Y_train[:,-128:,-128:,:128,:], age, survival, [gtype+"_"+target]))
            self.pool.append((X_train[:,-128:,-128:,-128:,:], Y_train[:,-128:,-128:,-128:,:], age, survival, [gtype+"_"+target]))

            random.shuffle(self.pool)
            tmp = self.pool.pop()
            return tmp

        # The ID must be a list, so that I can later iterate over it
        return X_train, Y_train, age, survival, [gtype+"_"+target]


    def getNextTrainingBatch(self):
        # Returns (1,240,240,155,4), Age, Survival

        d_tmp = self.loadNext(self.getFiles("training"), 0)
        if d_tmp is None:
            return None
        X_train, Y_train, age, survival, target = d_tmp
        X = {"in_volume": X_train}
        Y = {"out_segmentation": Y_train}
        return X, Y, target

    def getNextTestBatch(self):
        # Returns (1,240,240,155,4), Age, Survival
        d_tmp = self.loadNext(self.getFiles("test"), 1)
        if d_tmp is None:
            return None
        X_test, Y_test, age, survival, target = d_tmp
        X = {"in_volume": X_test}
        Y = {"out_segmentation": Y_test}
        return X, Y, target

    def getNextValidationBatch(self):
        # Returns (1,240,240,155,4), Age, Survival
        d_tmp = self.loadNext(self.getFiles("validation"), 2)
        if d_tmp is None:
            return None
        X_val, Y_val, age, survival, target = d_tmp
        X = {"in_volume": X_val}
        Y = {"out_segmentation": Y_val}
        return X, Y, target

