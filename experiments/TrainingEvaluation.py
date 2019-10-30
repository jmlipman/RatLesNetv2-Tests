from sacred import Experiment
import os, time, torch
#from torchsummary import summary
import numpy as np
import nibabel as nib
#from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
from lib.utils import log
from lib.metrics import *
import json

ex = Experiment("TrainingEvaluation")

@ex.main
def main(config, Model, data, base_path, _run):
    log("Start TrainingEvaluation test")

    base_path = base_path + str(_run._id) + "/"
    config["base_path"] = base_path

    # Data
    tr_data = data("train", loss=config["loss_fn"])
    val_data = data("validation", loss=config["loss_fn"])

    # Model
    model = Model(config)
    model.cuda()

    # Weight initialization
    def weight_init(m):
        if isinstance(m, torch.nn.Conv3d):
            config["initW"](m.weight)
            config["initB"](m.bias)
    model.apply(weight_init)

    # Save graph
    X, _, _, _ = tr_data[0]
    tb_path = base_path[:-1].split("/")
    tb_path = tb_path[0] + "/tensorboard/" + "_".join(tb_path[1:])
    writer = SummaryWriter(tb_path)
    writer.add_graph(model, X)
    writer.close()

    # Test how long each operation take
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        model(X)
    with open(base_path + "profile", "w") as f:
        f.write(str(prof))

    # Create folder for saving the model and validation results
    if len(config["save_validation"]) > 0:
        os.makedirs(config["base_path"] + "val_evol")
    os.makedirs(config["base_path"] + "model")

    # Config
    ep = config["epochs"]
    bs = config["batch"]
    loss_fn = config["loss_fn"]
    opt = config["opt"](model.parameters(), lr=config["lr"])
    lr_scheduler = config["lr_scheduler"]
    if not lr_scheduler is None:
        lr_scheduler.setOptimizer(opt)

    # Save model and optimizer's state dict
    param_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log("Number of parameters: " + str(param_num))
    with open(base_path + "state_dict", "w") as f:
        f.write(">> Model's state dict:\n") # Important for debugging
        for param_tensor in model.state_dict():
            f.write(param_tensor + "\t" + str(model.state_dict()[param_tensor].size()) + "\n")

        f.write("\n>> Optimizer's state dict:\n") # Important for reproducibility
        for var_name in opt.state_dict():
            f.write(var_name + "\t" + str(opt.state_dict()[var_name]) + "\n")

    # Load weights if necessary
    if config["model_state"] != "":
        log("Loading previous model")
        model.load_state_dict(torch.load(config["model_state"]))

    # Counters and flags
    e = 0 # Epoch counter
    it = 0 # Iteration counter
    keep_training = True # Flag to stop training when overfitting

    log("Training")
    while e < ep and keep_training:
        model.train()

        tr_loss = 0
        tr_islands = 0
        tr_i = 0
        while tr_i < len(tr_data) and keep_training:
            X, Y, id_, W = tr_data[tr_i]

            out = model(X)
            if W is None:
                tr_loss_tmp = loss_fn(out, Y)
            else:
                tr_loss_tmp = loss_fn(out, Y, W)
            tr_loss += tr_loss_tmp
            tr_islands += islands_num(out.detach().cpu())

            # Optimization
            opt.zero_grad()
            tr_loss_tmp.backward()
            opt.step()

            it += 1
            tr_i += 1

        tr_loss /= len(tr_data)
        tr_islands /= len(tr_data)

        # Get summaries to add to tensorboard
        writer = SummaryWriter(tb_path)
        writer.add_scalar("tr_loss", tr_loss, e)
        writer.add_scalar("tr_islands", tr_islands, e)
        writer.close()

        log("Validation")
        val_loss = 0
        val_islands = 0
        val_dice = 0 # Dice coef in the lesions
        val_i = 0
        model.eval()
        with torch.no_grad():
            while val_i < len(val_data) and keep_training:
                X, Y, id_, W = val_data[val_i]

                out = model(X)
                if W is None:
                    val_loss_tmp = loss_fn(out, Y)
                else:
                    val_loss_tmp = loss_fn(out, Y, W)
                val_loss += val_loss_tmp
                val_islands += islands_num(out.cpu())
                val_dice += dice_coef(out.cpu().numpy(), Y.cpu().numpy())[0][1] # Lesion dice
                # Calculate Dice coef during validation

                if id_ in config["save_validation"]:
                    name = id_ + "_" + str(e)
                    out = np.moveaxis(np.reshape(out.cpu().numpy(), (2,256,256,18)), 0, -1)
                    if config["save_npy"]:
                        np.save(config["base_path"] + "val_evol/" + name, out)
                    out = np.argmax(out, axis=-1)
                    nib.save(nib.Nifti1Image(out, np.eye(4)), config["base_path"] + "val_evol/" + name + ".nii.gz")

                val_i += 1

        val_loss /= len(val_data)
        val_islands /= len(val_data)
        val_dice /= len(val_data)

        # Get summaries to add to tensorboard
        writer = SummaryWriter(tb_path)
        writer.add_scalar("val_loss", val_loss, e)
        writer.add_scalar("val_islands", val_islands, e)
        writer.add_scalar("val_dice", val_dice, e)
        writer.close()

        # Reduce learning rate if needed, and stop if limit is reached.
        if lr_scheduler != None:
            lr_scheduler.step(val_loss)
            #keep_training = lr_scheduler.limit_cnt > -1 # -1 -> stop training
            if lr_scheduler.limit_cnt < 0:
                keep_training = False
                lr_scheduler.limit_cnt = lr_scheduler.limit # Needed if we run ex. more than once!

        log("Epoch: {}. Loss: {}. Val Loss: {}".format(e, tr_loss, val_loss))

        # Save model after every epoch
        torch.save(model.state_dict(), config["base_path"] + "model/model-" + str(e))
        if e > 4 and os.path.exists(config["base_path"] + "model/model-"+str(e-5)):
            os.remove(config["base_path"] + "model/model-"+str(e-5))

        e += 1

    log("Testing")
    test_data = data("test", loss=config["loss_fn"])
    if config["save_prediction"]:
        os.makedirs(config["base_path"] + "preds")

    results = {}
    model.eval()
    with torch.no_grad():
        # Assuming that batch_size is 1
        for test_i in range(len(test_data)):
            X, Y, id_, _ = test_data[test_i]
            out = model(X)
            out = out.cpu().numpy()
            Y = Y.cpu().numpy() # NBWHC

            if config["save_prediction"]:
                _out = np.argmax(np.reshape(out, (2,256,256,18)), axis=0)
                nib.save(nib.Nifti1Image(_out, np.eye(4)), config["base_path"] + "preds/" + id_ + ".nii.gz")

            dice_res = list(dice_coef(out, Y)[0])
            haus_res = hausdorff_distance(out, Y)[0]
            islands_res = islands_num(out)[0]
            results[id_] = [dice_res, islands_res, haus_res]

    with open(config["base_path"] + "results.json", "w") as f:
        f.write(json.dumps(results))

    log("End")