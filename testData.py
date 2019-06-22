# This script will test that data is what we are expecting
import numpy as np


def checkBatches(getBatch):
    c= 0
    batch_sizes = set()
    size_x = dict()
    size_y = dict()
    #d_tmp = data.getNextTrainingBatch()
    d_tmp = getBatch()
    for k in d_tmp[0].keys():
        size_x[k] = set()
    for k in d_tmp[1].keys():
        size_y[k] = set()

    while d_tmp != None:
        assert len(d_tmp) == 3, "Training batch does not retrieve 3 elements (X, Y, ids)"
        X, Y, ids = d_tmp
        assert type(ids) == list, "IDs must be wrapped in a list"
        c += 1
        #assert len(d_tmp[0]) == len(d_tmp[1]) and len(d_tmp[1]) == len(d_tmp[2]), "Training batches have different sizes"
        bs = []
        for k in X.keys():
            bs.append(X[k].shape[0])
            size_x[k].add(X[k].shape)
        for k in Y.keys():
            bs.append(Y[k].shape[0])
            size_y[k].add(Y[k].shape)
        bs.append(len(ids))
        batch_sizes.update(np.unique(bs))
        #print(size_training_x)
        #print(size_training_y)
        #d_tmp = data.getNextTrainingBatch()
        d_tmp = getBatch()

    print("> Number of batches: " + str(c))
    print("> Batch sizes (one or two diff. numbers are expected): " + str(batch_sizes))
    print("> Size X samples: "+str(size_x))
    print("> Size Y samples: "+str(size_y))

    return str(c), str(batch_sizes), str(size_x), str(size_y)



run_tests = ["CR02NOV16_1fold", "CR02NOV16_5folds", "BraTS_1fold"]
run_tests = ["CR02NOV16_1fold", "CR02NOV16_5folds"]


if "CR02NOV16_1fold" in run_tests:
    print("Test: CR02NOV16_1fold")
    from experiments.lib.data.CR02NOV16 import Data
    data = Data()
    data.split(folds=1)
    ##############################
    # Batch consistency check    #
    # For RegularTrainingTest.py #
    ##############################
    print("Train")
    print("Expected number of samples: "+str(len(data.all_training_files[0])))
    c, bs, sx, sy = checkBatches(data.getNextTrainingBatch)
    assert c == "32"
    assert bs == "{1}"
    assert sx == "{'in_volume': {(1, 18, 256, 256, 1)}}"
    assert sy == "{'out_segmentation': {(1, 18, 256, 256, 2)}}"

    print("Test")
    print("Expected number of samples: "+str(len(data.all_test_files[0])))
    c, bs, sx, sy = checkBatches(data.getNextTestBatch)
    assert c == "8"
    assert bs == "{1}"
    assert sx == "{'in_volume': {(1, 18, 256, 256, 1)}}"
    assert sy == "{'out_segmentation': {(1, 18, 256, 256, 2)}}"

    print("Validation")
    print("Expected number of samples: "+str(len(data.all_validation_files[0])))
    c, bs, sx, sy = checkBatches(data.getNextValidationBatch)
    assert c == "8"
    assert bs == "{1}"
    assert sx == "{'in_volume': {(1, 18, 256, 256, 1)}}"
    assert sy == "{'out_segmentation': {(1, 18, 256, 256, 2)}}"

if "CR02NOV16_5folds" in run_tests:
    print("Test: CR02NOV16_5folds")
    from experiments.lib.data.CR02NOV16 import Data
    data = Data()
    data.split(folds=5)
    data.loadInMemory(["test"])
    ##############################
    # Batch consistency check    #
    # For RegularTrainingTest.py #
    ##############################
    ctr = ["32", "32", "36", "36", "36"]
    cte = ["12", "12", "8", "8", "8"]
    cva = ["4", "4", "4", "4", "4"]
    for f in range(5):
        print("FOLD "+str(f))
        print("Train")
        print("Expected number of samples: "+str(len(data.all_training_files[0])))
        c, bs, sx, sy = checkBatches(data.getNextTrainingBatch)
        assert c == ctr[f]
        assert bs == "{1}"
        assert sx == "{'in_volume': {(1, 18, 256, 256, 1)}}"
        assert sy == "{'out_segmentation': {(1, 18, 256, 256, 2)}}"

        print("Test")
        print("Expected number of samples: "+str(len(data.all_test_files[0])))
        c, bs, sx, sy = checkBatches(data.getNextTestBatch)
        assert c == cte[f]
        assert bs == "{1}"
        assert sx == "{'in_volume': {(1, 18, 256, 256, 1)}}"
        assert sy == "{'out_segmentation': {(1, 18, 256, 256, 2)}}"

        print("Validation")
        print("Expected number of samples: "+str(len(data.all_validation_files[0])))
        c, bs, sx, sy = checkBatches(data.getNextValidationBatch)
        assert c == cva[f]
        assert bs == "{1}"
        assert sx == "{'in_volume': {(1, 18, 256, 256, 1)}}"
        assert sy == "{'out_segmentation': {(1, 18, 256, 256, 2)}}"
        print("Total samples: "+str(len(data.all_validation_files[0]) + len(data.all_training_files[0]) + len(data.all_test_files[0])))
        data.nextFold()

if "BraTS_1fold" in run_tests:
    print("Test: BraTS_1fold")
    from experiments.lib.data.BraTS19 import Data
    data = Data()
    data.split(folds=1)
    ##############################
    # Batch consistency check    #
    # For RegularTrainingTest.py #
    ##############################
    print("Train")
    print("Expected number of samples: "+str(len(data.all_training_files[0])))
    c, bs, sx, sy = checkBatches(data.getNextTrainingBatch)
    assert c == "234"
    assert bs == "{1}"
    assert sx == "{'in_volume': {(1, 138, 173, 172, 4)}}"
    assert sy == "{'out_segmentation': {(1, 138, 173, 172, 4)}}"

    print("Test")
    print("Expected number of samples: "+str(len(data.all_test_files[0])))
    c, bs, sx, sy = checkBatches(data.getNextTestBatch)
    assert c == "66"
    assert bs == "{1}"
    assert sx == "{'in_volume': {(1, 138, 173, 172, 4)}}"
    assert sy == "{'out_segmentation': {(1, 138, 173, 172, 4)}}"

    print("Validation")
    print("Expected number of samples: "+str(len(data.all_validation_files[0])))
    c, bs, sx, sy = checkBatches(data.getNextValidationBatch)
    assert c == "35"
    assert bs == "{1}"
    assert sx == "{'in_volume': {(1, 138, 173, 172, 4)}}"
    assert sy == "{'out_segmentation': {(1, 138, 173, 172, 4)}}"

    print("Total samples: "+str(len(data.all_validation_files[0]) + len(data.all_training_files[0]) + len(data.all_test_files[0])))

