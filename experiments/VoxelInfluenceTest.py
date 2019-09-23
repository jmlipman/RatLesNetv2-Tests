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
    path_vol = "/home/miguelv/data/in/CR_DATA/02OCT2017/24h/11/scan.nii.gz"
    t1, t2, t3 = [165, 126, 7]
    orig_volume = nib.load(path_vol).get_data()
    orig_volume = np.expand_dims(np.moveaxis(orig_volume, 2, 0), 0)
    orig_volume = np.concatenate([orig_volume, orig_volume], axis=0)

    model = Model(config)

    mmin = np.min(orig_volume)
    mmax = np.max(orig_volume)
    res = np.zeros((256, 256, 18))

    #X = {"in_volume": orig_volume}
    #pred = model.predictBatch(X)
    #ref_value = pred[0, t3, t1, t2, 1]

    for i in range(t1-43, t1+43):
        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S')+" "+str(i)+"/"+str(167+30))
        for j in range(t2-30, t2+30):
            for k in range(18):
                vol = np.copy(orig_volume)
                vol[0, k, i, j, 0] = mmin
                vol[1, k, i, j, 0] = mmax
                X = {"in_volume": vol}

                pred = model.predictBatch(X)
                diff = np.abs(pred[0, t3, t1, t2, 1] - pred[1, t3, t1, t2, 1])
                #pred = np.moveaxis(np.reshape(pred, (18, 256, 256, 2)), 0, 2)
                res[i, j, k] = diff
                #print(i, j, k, diff)

    #nib.save(nib.Nifti1Image(res, np.eye(4)), base_path+"asd.nii.gz")
    np.save(base_path+"differences.npy", res)
