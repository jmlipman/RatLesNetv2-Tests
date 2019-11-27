from sacred import Experiment
import nibabel as nib
import numpy as np
from datetime import datetime
import os, time, torch
from lib.utils import log, np2cuda

ex = Experiment("ReceptiveField")

@ex.main
def main(config, Model, data, base_path, _run):
    log("Calculating the Receptive field")

    base_path = base_path + str(_run._id) + "/"
    config["base_path"] = base_path

    # Load the volume we want to study
    # In theory, any of them should be okay
    test_data = data("test", loss=config["loss_fn"], dev=config["device"])
    X, Y, id_, W = test_data[0]
    X = X.cpu().numpy()

    # Model
    model = Model(config)
    model.to(config["device"])
    model.eval()

    if config["model_state"] == "":
        raise Exception("A model needs to be loaded")
    model.load_state_dict(torch.load(config["model_state"]))

    mmin = np.min(X)
    mmax = np.max(X)

    # Target voxels. I will study how modifying other voxels affects this particular one.
    t1, t2, t3 = 9, 80, 80

    with torch.no_grad():
        log("Checking in Depth dimension. Reference: {},{},{}".format(t1, t2, t3))
        results = np.zeros(X.shape[2])
        for i in range(X.shape[2]): # Studying in Depth (first) dimension
            if i%50==0:
                log("{}/{}".format(i, X.shape[2]-1))
            vol = np.copy(X)
            vol[0, 0, i, t2, t3] = mmin
            pred_min = model(np2cuda(vol, config["device"]))
            pred_min = pred_min.cpu().numpy()

            vol = np.copy(X)
            vol[0, 0, i, t2, t3] = mmax
            pred_max = model(np2cuda(vol, config["device"]))
            pred_max = pred_max.cpu().numpy()

            results[i] = np.max(np.abs(pred_min[0, :, t1, t2, t3] - pred_max[0, :, t1, t2, t3]))
        r = np.where(results!=0)[0][[0, -1]]
        log("Receptive field range in Depth-dim: {}. Size: {}".format(r, r[1]-r[0]+1))

        log("Checking in Height dimension. Reference: {},{},{}".format(t1, t2, t3))
        results = np.zeros(X.shape[3])
        for i in range(X.shape[3]): # Studying in Depth (first) dimension
            if i%50==0:
                log("{}/{}".format(i, X.shape[3]-1))
            vol = np.copy(X)
            vol[0, 0, t1, i, t3] = mmin
            pred_min = model(np2cuda(vol, config["device"]))
            pred_min = pred_min.cpu().numpy()

            vol = np.copy(X)
            vol[0, 0, t1, i, t3] = mmax
            pred_max = model(np2cuda(vol, config["device"]))
            pred_max = pred_max.cpu().numpy()

            results[i] = np.max(np.abs(pred_min[0, :, t1, t2, t3] - pred_max[0, :, t1, t2, t3]))
        r = np.where(results!=0)[0][[0, -1]]
        log("Receptive field range in Height-dim: {}. Size: {}".format(r, r[1]-r[0]+1))

        log("Checking in Width dimension. Reference: {},{},{}".format(t1, t2, t3))
        results = np.zeros(X.shape[4])
        for i in range(X.shape[4]): # Studying in Depth (first) dimension
            if i%50==0:
                log("{}/{}".format(i, X.shape[4]-1))
            vol = np.copy(X)
            vol[0, 0, t1, t2, i] = mmin
            pred_min = model(np2cuda(vol, config["device"]))
            pred_min = pred_min.cpu().numpy()

            vol = np.copy(X)
            vol[0, 0, t1, t2, i] = mmax
            pred_max = model(np2cuda(vol, config["device"]))
            pred_max = pred_max.cpu().numpy()

            results[i] = np.max(np.abs(pred_min[0, :, t1, t2, t3] - pred_max[0, :, t1, t2, t3]))
        r = np.where(results!=0)[0][[0, -1]]
        log("Receptive field range in Width-dim: {}. Size: {}".format(r, r[1]-r[0]+1))

