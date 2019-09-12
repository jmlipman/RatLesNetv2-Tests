from sacred import Experiment
import os
import nibabel as nib
import numpy as np
import time
from datetime import datetime

ex = Experiment("VoxelInfluenceTest")

@ex.main
def main(config, Model, data, base_path, _run):

    base_path = base_path + str(_run._id) + "/"
    config["base_path"] = base_path

    # Load the volume we want to study
    #orig_volume = np.random.random((1, 18, 256, 256, 1))
    path_vol = "/media/miguelv/HD1/CR_DATA/02OCT2017/24h/11/scan.nii.gz"
    t1, t2, t3 = [0, 0, 7]
    orig_volume = nib.load(path_vol).get_data()
    mmin = np.min(orig_volume)
    mmax = np.max(orig_volume)
    orig_volume = np.expand_dims(np.moveaxis(orig_volume, 2, 0), 0)
    orig_volume = np.concatenate([orig_volume, orig_volume], axis=0)

    model = Model(config)


    orig_volume[0, t3, t1, t2, 0] = mmin
    orig_volume[1, t3, t1, t2, 0] = mmax
    X = {"in_volume": orig_volume}
    pred = model.predictBatch(X)

    print(np.abs(pred[0, 7, 165, 126, 1] - pred[1, 7, 165, 126, 1]))
    #pred = np.reshape(pred, (18, 256, 256, 2))
    #pred = np.moveaxis(pred, 0, 2)
    #nib.save(nib.Nifti1Image(pred, np.eye(4)), base_path+"brain.nii.gz")
    #np.save(base_path+"differences.npy", pred)
