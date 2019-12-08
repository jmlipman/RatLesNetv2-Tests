from sacred import Experiment
import nibabel as nib
import numpy as np
from datetime import datetime
import os, time, torch
from lib.utils import log, np2cuda
from lib.metrics import *

ex = Experiment("CalculateSensivityPrecision")

@ex.main
def main(config, Model, data, base_path, _run):
    log("Calculating sensivity and precision")

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
            X, Y, id_, W = test_data[i]
            if "_24h_" not in id_:
                continue

            Y = Y.cpu().numpy()
            if np.sum(Y[0,0,:,:,:]) == 256*256*18:
                continue # Reportedly sham-animal

            pred = model(X)
            pred = pred[0].cpu().numpy()

            # Stats
            sens, prec, tp = sensivity_precision_TP(pred, Y)
            dice = dice_coef(pred, Y)[0]
            stats.append("{},{},{},{},{}\n".format(id_, dice, sens, prec, tp))

            
    with open(config["base_path"] + "results", "w") as f:
        for s in stats:
            f.write(s)

