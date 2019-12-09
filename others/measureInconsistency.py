# This script will measure the accuracy and uncertainty of a trained network
# For each network there are hopefully 5 runs.
#
# There are 4 different measures: (values per scan/time-point/study)
#  1) Average performance of the network compare to the ground truth.
#     Dice, Islands and Hausdorff distance. Mean and std.
#
#  2) Unconsistency of the network: How different predictions across the runs are?
#     Comparison among the predictions. If there are 5 runs, there will
#     be 10 comparison per scan. Get the lowest dice, largest HD, largest diff. of islands.
#
#  3) Distribution of the softmax prob. values. How many % of lesion voxels have >0.9
#     softmax prob. values? And >0.8, >0.7, >0.6, >0.5. Retrieve the mean and std across runs.
#
#  4) Histogram of the logits of those values of >0.9, >0.8, ...
#
import os
import numpy as np
import sys 
sys.path.append('..') # So I can import Metric
from lib.metric import Metric


PATH = ""
PATH_DATA = "/media/miguelv/HD1/CR_DATA/"
runs = sorted(os.listdir(PATH))
res1 = {} # res1["id"] = np.array.shape(runs,3) # 3 -> Dice, Islands, HD
predictions = sorted(os.listdir(PATH + "/" + runs[0] + "/preds"))
for pred_id in predictions:
    # Logits
    preds_logits = np.stack([np.load(PATH+"/"+run+"/preds/"+pred_id) for run in runs])
    print(preds_logits.shape) #BWHD2
    # Masks
    #preds = np.stack([np.argmax(preds[i], axis=-1) for i in range(preds.shape[0])])
    #print(preds.shape) #BWDH

    # Now we have to find the ground truth
    # 02NOV2016_24h_43.npy
    path_pred = PATH_DATA + pred_id.split(".")[0].replace("_", "/")
    files_path = os.listdir(path_pred)
    if "scan_miguel.nii.gz" in files_path:
        ground_truth = nib.load(path_pred + "/scan_miguel.nii.gz")
    elif "scan_lesion.nii.gz" in files_path:
        ground_truth = nib.load(path_pred + "/scan_lesion.nii.gz")
    else:
        ground_truth = np.zeros(preds[0].shape)

    ground_truth = np.stack([1.0*(ground_truth==j) for j in range(2)])
    print(ground_truth.shape) #BWHD2

    # This gathers all the results
    tmp_res = np.zeros((len(runs), 3))
    for i in range(len(runs)):
        m = Metric(preds_logits[i], ground_truth)
        tmp_res[i] = [m.dice(), m.islands(), m.hausdorff()]
        # "{},{},{},{}".format(pred_id, tmp_res[i,0], tmp_res[i,1], tmp_res[i,2])


