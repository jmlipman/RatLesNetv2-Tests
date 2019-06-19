import random

class BaseData:
    def __init__(self):
        raise Exception("You must define a __init__ inside your data wrapper")

    def loadNext(self, files, c):
        raise Exception("You must define a loadNext inside your data wrapper")

    def getNextTrainingBatch(self):
        raise Exception("You must define a getNextTrainingBatch inside your data wrapper")

    def getNextTestBatch(self):
        raise Exception("You must define a getNextTestBatch inside your data wrapper")

    def getNextValidationBatch(self):
        raise Exception("You must define a getNextValidationBatch inside your data wrapper")

    def split(self, folds, prop=[0.7, 0.2, 0.1]):
        """This function will split the data. Split is expected to balance
           the content of self.groups so that the proportion is the same.
           Depending on the number of folds, the splitting is different:
           
           folds = -1: TODO. Leave-one-out. Maybe it makes no sense, no val.
           folds = 0: TODO. Return the whole data in the training.
           folds = 1: Regular splitting into training, test and validation sets.
           folds = N: N-Fold cross-validation.

           Args:
            `folds`: number of folds. Used for cross-validation. If folds is 1
             it will not divide the dataset.
             `prop`: proportion of the data divided into training, test and
              validation respectively.
        """
        assert len(prop) == 3, "prop only accepts 3 numbers: train, test, val"
        assert sum(prop) > 0.9999, "prop must sum up to 1"

        if self.randomize:
            for g in self.groups:
                random.shuffle(g)

        if folds == 1:
            self.all_training_files = [[] for _ in range(folds)]
            self.all_test_files = [[] for _ in range(folds)]
            self.all_validation_files = [[] for _ in range(folds)]
            for g in self.groups:
                # Calculate indices where to split
                split_train = int(len(g)*prop[0])
                split_test = split_train + int(len(g)*prop[1])

                self.all_training_files[0].extend(g[:split_train])
                self.all_test_files[0].extend(g[split_train:split_test])
                self.all_validation_files[0].extend(g[split_test:])

            if self.randomize:
                random.shuffle(self.all_training_files[0])
                random.shuffle(self.all_test_files[0])
                random.shuffle(self.all_validation_files[0])

        elif folds > 1:
            self.all_training_files = [[] for _ in range(folds)]
            self.all_test_files = [[] for _ in range(folds)]
            self.all_validation_files = [[] for _ in range(folds)]
            # For Cross-validation
            for g in self.groups:
                # Length of 20% of the data (fold=5)
                tmp_test_files = [[] for _ in range(folds)]
                tmp_training_files = []
                i = 0
                for elem in g:
                    tmp_test_files[i].append(elem)
                    i += 1
                    if i == folds:
                        i = 0
                for i in range(folds):
                    tmp_training_files_ = list(set(g) - set(tmp_test_files[i]))
                    tmp_validation_files = tmp_training_files_[:int(len(tmp_training_files)*0.1)]
                    tmp_training_files = tmp_training_files_[int(len(tmp_training_files)*0.1):]

                    self.all_training_files[i].extend(tmp_training_files)
                    self.all_test_files[i].extend(tmp_test_files[i])
                    self.all_validation_files[i].extend(tmp_validation_files)
        else:
            raise Exception("Not implemented yet!")

        self.current_fold = 0
        """
                print(len(self.all_training_files[i]))
                print(len(self.all_test_files[i]))
                print(len(self.all_validation_files[i]))
                print("===")
        """

    def getFiles(self, which):
        if which == "training":
            return self.all_training_files[self.current_fold]
        elif which == "test":
            return self.all_test_files[self.current_fold]
        elif which == "validation":
            return self.all_validation_files[self.current_fold]
        else:
            raise Exception("Wrong place to be!")
        

    def nextFold(self):
        self.current_fold += 1

    # Processing operations
    def standardize(self, data):
        """Zero-mean one-variance
        """
        return (data-data.mean())/data.std()

