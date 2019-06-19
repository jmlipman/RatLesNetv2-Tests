import numpy as np
import nibabel as nib
import pandas as pd
import os
import random
from scipy import ndimage

def hotvector(arr, classes):
    if not type(arr) is np.ndarray:
        raise Exception("array must be numpy array!")
    return np.stack([(arr==i)*1.0 for i in range(classes)], axis=-1)

class Data:
    def __init__(self, randomize=False, depth_first=True, only_GTR=False, sliced=False, bounding_box=True):
        """In the initializer we gather the location of the files that will
           be used for training, testing and validation.
           Counters will be set to 0, so that during the data retrieval
           (one by one) we can know what's the next subject to retrieve.
           It will also read the survival CSV.

           Args:
           `randomize`: If True, every time this is executed the data within
            the splits for training, test and validation will be different.
            `depth_first`: As TF requires BDWHC, if this is True then the
            slices are moved to the front: DWH.
            `only_GTR`: As BraTS only wants us to calculate the survival day
            on those who were GTR, if this is True then it will return None
            instead of the Age and Survival days in those cases it is not GTR.
        """
        # Get the lists
        self.PATH = "/wrk/valverde/BraTS/"
        #self.PATH = "/home/miguelv/Downloads/MICCAI_BraTS_2019_Data_Training/"
        pathHGG = self.PATH + "HGG/"
        pathLGG = self.PATH + "LGG/"
        pathSurvival = self.PATH + "survival_data.csv"

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

        # Randomize the files we firstly pick
        if randomize:
            random.shuffle(filesHGG)
            random.shuffle(filesLGG)

        # 70% Training, 20% Testing, 10% Validation
        train_idx_HGG = int(totalHGG*0.7)
        test_idx_HGG = int(totalHGG*0.2) + train_idx_HGG
        train_idx_LGG = int(totalLGG*0.7)
        test_idx_LGG = int(totalLGG*0.2) + train_idx_LGG


        self.train_files = filesHGG[:train_idx_HGG] + filesLGG[:train_idx_LGG]
        self.test_files = filesHGG[train_idx_HGG:test_idx_HGG] + filesLGG[train_idx_LGG:test_idx_LGG]
        self.validation_files = filesHGG[test_idx_HGG:] + filesLGG[test_idx_LGG:]

        # Randomize the order of the files we have picked
        if randomize:
            random.shuffle(self.train_files)
            random.shuffle(self.test_files)
            random.shuffle(self.validation_files)

        # Reading the CSV
        self.csv_data = pd.read_csv(pathSurvival)

        self.train_samples_c = len(self.train_files)
        self.test_samples_c = len(self.test_files)
        self.validation_samples_c = len(self.validation_files)

    def loadNext(self, files, c):
        """Blabla

           In this function I am assuming the following:
            - Size of the nifti files is the same (# of voxels)
            - We have all the modalities and segmentation per subject
           I can assume this because I checked it (assumptions.py).

           Return stuff and id
        """
        def standardize(data):
            return (data-data.mean())/data.std()
        if len(self.pool) > 0:
            return self.pool.pop()

        if self.counters[c] == len(files):
            self.counters[c] += 1
            return None
        elif self.counters[c] > len(files):
            self.counters[c] = 0

        target = files[self.counters[c]]
        gtype, target = target.split("/")

        # Read the actual data
        mods = ["flair", "t1", "t1ce", "t2"]
        X_train = np.stack([standardize(nib.load(self.PATH+gtype+"/"+target+"/"+target+"_"+mod+".nii.gz").get_data()) for mod in mods], axis=-1)

        if self.depth_first:
            X_train = np.moveaxis(X_train, 2, 0)
        X_train = np.expand_dims(X_train, axis=0)

        tmp_path = self.PATH + gtype + "/" + target + "/" + target + "_seg.nii.gz"
        Y_train = nib.load(tmp_path).get_data()

        if self.depth_first:
            Y_train = np.moveaxis(Y_train, 2, 0)
        Y_train[Y_train==4] = 3
        Y_train = hotvector(Y_train, 4)

        Y_train = np.expand_dims(Y_train, 0)

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
            #loc = ndimage.find_objects(1*(X_train!=0))[0]

            # Fixed way. Resulting images are: (138, 173, 172)
            loc = (slice(0, 1, None), slice(8, 146, None), slice(40, 213, None), slice(49, 221, None))
            X_train = X_train[loc]
            Y_train = Y_train[loc]

        elif self.sliced:
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

        d_tmp = self.loadNext(self.train_files, 0)
        if d_tmp is None:
            return None
        X_train, Y_train, age, survival, target = d_tmp
        X = {"in_volume": X_train}
        Y = {"out_segmentation": Y_train}
        return X, Y, target


    def getNextTestBatch(self):
        # Returns (1,240,240,155,4), Age, Survival
        d_tmp = self.loadNext(self.test_files, 1)
        if d_tmp is None:
            return None
        X_test, Y_test, age, survival, target = d_tmp
        X = {"in_volume": X_test}
        Y = {"out_segmentation": Y_test}
        return X, Y, target

    def getNextValidationBatch(self):
        # Returns (1,240,240,155,4), Age, Survival
        d_tmp = self.loadNext(self.validation_files, 2)
        if d_tmp is None:
            return None
        X_val, Y_val, age, survival, target = d_tmp
        X = {"in_volume": X_val}
        Y = {"out_segmentation": Y_val}
        return X, Y, target

"""
data = Data()
for e in range(1):
    print("Epoch: "+str(e))
    d_tmp = data.getNextTrainingSubject()
    while d_tmp != None:
        X_train, Y_train, age, survival = d_tmp
        print(X_train.shape, Y_train.shape, age, survival)
        d_tmp = data.getNextTrainingSubject()
"""

"""
# Calculating the bounding box of all
path = "/home/miguelv/Downloads/MICCAI_BraTS_2019_Data_Training/"
all_brains = np.zeros((240,240,155))
c = 0
for root, subdirs, files in os.walk(path):
    c += 1
    print(c)
    for f in files:
        if f.endswith(".nii.gz"):
            brain = 1*(nib.load(root + "/" + f).get_data()!=0)
        all_brains += brain

loc = ndimage.find_objects(all_brains.astype(int))[0]
print(loc)
print(all_brains[loc].shape)
"""
