from sacred import Experiment
import nibabel as nib
import numpy as np
from datetime import datetime
import os, time, torch
from lib.utils import log, np2cuda
from lib.metrics import *

ex = Experiment("GeneratePredictions")

@ex.main
def main(config, Model, data, base_path, _run):
    log("Generating predictions")

    base_path = base_path + str(_run._id) + "/"
    config["base_path"] = base_path

    # Model
    model = Model(config)
    model.to(config["device"])
    model.eval()

    if config["model_state"] == "":
        raise Exception("A model needs to be loaded")
    model.load_state_dict(torch.load(config["model_state"]))

    os.makedirs(config["base_path"] + "preds")

    test_data = data("test", loss=config["loss_fn"], dev=config["device"])

    stats = []
    with torch.no_grad():
        for i in range(len(test_data)): # Studying in Depth (first) dimension
            if i % 10 == 0:
                print("{}/{}".format(i, len(test_data)))
            X, Y, id_, W = test_data[i]
            
            pred = model(X)
            pred = pred[0].cpu().numpy()
            Y = Y.cpu().numpy()

            # Stats
            dice_res = list(dice_coef(pred, Y)[0])
            haus_res = hausdorff_distance(pred, Y)[0]
            islands_res = islands_num(pred)[0]
            stats.append("{}\t{}\t{}\t{}\n".format(id_, dice_res, haus_res, islands_res))

            _out = np.argmax(np.moveaxis(np.reshape(pred, (2,18,256,256)), 1, -1), axis=0)
            nib.save(nib.Nifti1Image(_out, np.eye(4)), config["base_path"] + "preds/" + id_ + "_pred.nii.gz")
            _out = np.argmax(np.moveaxis(np.reshape(Y, (2,18,256,256)), 1, -1), axis=0)
            nib.save(nib.Nifti1Image(_out, np.eye(4)), config["base_path"] + "preds/" + id_ + "_label.nii.gz")
            _out = np.moveaxis(np.reshape(X.cpu().numpy(), (18,256,256)), 0, -1)
            nib.save(nib.Nifti1Image(_out, np.eye(4)), config["base_path"] + "preds/" + id_ + ".nii.gz")
            
            
    with open(config["base_path"] + "results", "w") as f:
        for s in stats:
            f.write(s)

