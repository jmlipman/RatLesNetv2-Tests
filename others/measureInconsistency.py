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
import nibabel as nib
from lib.metric import Metric

def softmax(logits):
    e_x = np.exp(logits+1e-10)
    return e_x / np.expand_dims(e_x.sum(axis=0)+1e-10, 0)


PATH = "/media/miguelv/HD1/Inconsistency/baseline/"
PATH_DATA = "/media/miguelv/HD1/CR_DATA/"

# PART 1. PERFORMANCE WRT. GROUND TRUTH
runs = [i for i in sorted(os.listdir(PATH)) if i.isdigit()] # Remove potential _sources folder
res1 = {} # res1["id"] = np.array.shape(runs,3) # 3 -> Dice, Islands, HD
predictions = sorted(os.listdir(PATH + "/" + runs[0] + "/preds"))
for i, pred_id in enumerate(predictions):
    print("{}/{}".format(i+1, len(predictions)))
    # Logits
    preds_logits = []
    for run in runs:
        tmp = nib.load(PATH+str(run)+"/preds/"+pred_id).get_data()
        tmp = np.moveaxis(np.moveaxis(tmp, -1, 0), -1, 1)
        preds_logits.append(softmax(tmp)) # BCDHW

    # Now we have to find the ground truth
    # 02NOV2016_24h_43_logits.nii.gz
    path_pred = PATH_DATA + "/".join(pred_id.split("_")[:3])
    files_path = os.listdir(path_pred)
    if "scan_miguel.nii.gz" in files_path:
        ground_truth = np.moveaxis(nib.load(path_pred + "/scan_miguel.nii.gz").get_data(), -1, 0)
    elif "scan_lesion.nii.gz" in files_path:
        ground_truth = np.moveaxis(nib.load(path_pred + "/scan_lesion.nii.gz").get_data(), -1, 0)
    else:
        ground_truth = np.zeros(preds_logits[0][0,:,:,:].shape)

    ground_truth = np.stack([1.0*(ground_truth==j) for j in range(2)], axis=0)

    # This gathers all the results
    tmp_res = np.zeros((len(runs), 9))
    for j in range(len(runs)):
        m = Metric(preds_logits[j][np.newaxis,], ground_truth[np.newaxis,])
        m = m.all()
        tmp_res[j] = [m[0][1]] + m[1:]
    res1[pred_id] = tmp_res

# First part done. Run it to confirm it works.
