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
    #path_vol = "/media/miguelv/HD1/CR_DATA/02OCT2017/24h/11/scan.nii.gz"
    t1, t2, t3 = [165, 126, 7]
    #t1, t2, t3 = [64, 64, 7]
    #t1, t2, t3 = [32, 32, 3]
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

    firstRange = [5, 5]
    secondRange = [5, 5]

    for i in range(t1-firstRange[0], t1+firstRange[1]):
        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S')+" "+str(i)+"/"+str(t1+firstRange[1]))
        for j in range(t2-secondRange[0], t2+secondRange[1]):
            for k in range(18):
                vol = np.copy(orig_volume)
                vol[0, k, i, j, 0] = mmin
                vol[1, k, i, j, 0] = mmax
                X = {"in_volume": vol}

                
                #pred = model.predictBatch(X)
                pred = model.outputFromOperation(X, "max_pooling3d/MaxPool3D")
                #pred = model.outputFromOperation(X, "RatLesNet_DenseBlock_1/CombineBlock_Concat_1/concat")
                diff = 0
                for c in range(pred.shape[-1]):
                    tmp_diff = np.abs(pred[0, t3, t1, t2, c] - pred[1, t3, t1, t2, c])
                    if tmp_diff > diff:
                        diff = tmp_diff

                #pred = np.moveaxis(np.reshape(pred, (18, 256, 256, 2)), 0, 2)
                #if diff != 0:
                #    print(i, j, k, diff)
                res[i, j, k] = diff
                #print(i, j, k, diff)

    #nib.save(nib.Nifti1Image(res, np.eye(4)), base_path+"asd.nii.gz")
    np.save(base_path+"differences.npy", res)


#### Some notes
# I was calculating the receptive field by max(np.where(d!=0)[1]) - min(np.where(d!=0)[1])
# When I calculated after the first DenseBlock I was getting 2, which makes no sense because
# one convolution has a receptive field of 3. Turns out I was getting 65 - 63, which is 2
# indeed, but that means that 65, 64, and 63 were affected, meaning 3 numbers.
# This problem could only be seen in the encoding part of the network, where I was getting
# 9 instead of 10 (concat=1). At the end I was then getting 17 instead of 18.
