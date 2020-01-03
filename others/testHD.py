import numpy as np
import nibabel as nib
from metric import Metric

def softmax(logits):
    e_x = np.exp(logits+1e-20)
    suma = np.sum(e_x, axis=-1)
    return e_x / np.expand_dims(suma, -1)

brain1_path = "/media/miguelv/HD1/Inconsistency/baseline/1/preds/02NOV2016_24h_12_logits.nii.gz"
GT_path = "/media/miguelv/HD1/CR_DATA/02NOV2016/24h/12/scan_miguel.nii.gz"


brain = nib.load(brain1_path).get_data()
GT = nib.load(GT_path).get_data()

pred = np.argmax(softmax(brain), -1)

print(pred.shape)
print(GT.shape)

M = Metric(pred, GT)
print(M.hausdorff_distance())
