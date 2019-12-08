from sacred import Experiment
import os, time, torch
#from torchsummary import summary
import numpy as np
import nibabel as nib
#from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
from lib.utils import log, removeSmallIslands
from lib.metrics import *
import json
from lib.models.VoxResNet import VoxResNet

from lib.metric import Metric

ex = Experiment("TrainingEvaluation")

@ex.main
def main(config, Model, data, base_path, _run):
    log("Start TrainingEvaluation")

    base_path = base_path + str(_run._id) + "/"
    config["base_path"] = base_path

    # Data
    tr_data = data("train", loss=config["loss_fn"], dev=config["device"])
    val_data = data("validation", loss=config["loss_fn"], dev=config["device"])

    # Model
    model = Model(config)
    #model.cuda()
    model.to(config["device"])

    # Weight initialization
    def weight_init(m):
        if isinstance(m, torch.nn.Conv3d):
            config["initW"](m.weight)
            config["initB"](m.bias)
    model.apply(weight_init)


    # Save graph
    X, _, _, _ = tr_data[0]
    tb_path = base_path[:-1].split("/")
    tb_path = "/".join(tb_path[:-2]) + "/tensorboard/" + "_".join(tb_path[-2:])
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
    #p1 = sum(p.numel() for p in model.conv1.parameters() if p.requires_grad)
    #p2 = sum(p.numel() for p in model.dense1.parameters() if p.requires_grad)
    #p3 = sum(p.numel() for p in model.dense2.parameters() if p.requires_grad)
    #p4 = sum(p.numel() for p in model.bottleneck1.parameters() if p.requires_grad)
    #p5 = sum(p.numel() for p in model.dense3.parameters() if p.requires_grad)
    #p6 = sum(p.numel() for p in model.bottleneck2.parameters() if p.requires_grad)
    #p7 = sum(p.numel() for p in model.dense4.parameters() if p.requires_grad)
    #p8 = sum(p.numel() for p in model.bottleneck3.parameters() if p.requires_grad)
    #print("Encoder: "+str(p1+p2+p3+p4) + " (before: 83052)")
    #print("Decoder: "+str(p5+p6+p7+p8) + " (before: 284282)")

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

            output = model(X)
            pred = output[0]
            #print(activations.mean().cpu(), activations.std().cpu())
            if W is None:
                tr_loss_tmp = loss_fn(pred, Y, config)
            else:
                tr_loss_tmp = loss_fn(pred, Y, config, W)
            tr_loss += tr_loss_tmp
            tr_islands += np.sum(Metric(pred.detach().cpu(), None).islands())

            # Optimization
            opt.zero_grad()
            tr_loss_tmp.backward()
            opt.step()

            it += 1
            tr_i += 1

        tr_loss /= len(tr_data)
        tr_islands /= len(tr_data)

        # Tensorboard summaries
        writer = SummaryWriter(tb_path)
        writer.add_scalar("tr_loss", tr_loss, e)
        writer.add_scalar("tr_islands", tr_islands, e)
        writer.close()

        log("Validation")
        val_loss = 0
        val_islands = 0
        val_dice = 0
        val_i = 0
        model.eval()
        with torch.no_grad():
            while val_i < len(val_data) and keep_training:
                X, Y, id_, W = val_data[val_i]

                output = model(X)
                pred = output[0]
                if W is None:
                    val_loss_tmp = loss_fn(pred, Y, config)
                else:
                    val_loss_tmp = loss_fn(pred, Y, config, W)
                val_loss += val_loss_tmp
                m = Metric(pred.cpu().numpy(), Y.cpu().numpy())
                val_islands += np.sum(m.islands())
                val_dice += m.dice()[:,1] # Lesion Dice

                if id_ in config["save_validation"]:
                    name = id_ + "_" + str(e)
                    pred = np.moveaxis(np.moveaxis(np.reshape(pred.cpu().numpy(), (2,18,256,256)), 1, -1), 0, -1)
                    if config["save_npy"]:
                        np.save(config["base_path"] + "val_evol/" + name, pred)
                    pred = np.argmax(pred, axis=-1)
                    nib.save(nib.Nifti1Image(pred, np.eye(4)), config["base_path"] + "val_evol/" + name + ".nii.gz")

                val_i += 1

        val_loss /= len(val_data)
        val_islands /= len(val_data)
        val_dice /= len(val_data)

        # Tensorboard summaries
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
    test_data = data("test", loss=config["loss_fn"], dev=config["device"])
    if config["save_prediction_mask"] or config["save_prediction_softmaxprob"]:
        os.makedirs(config["base_path"] + "preds")


    results = {}
    results_post = {}
    model.eval()
    with torch.no_grad():
        # Assuming that batch_size is 1
        for test_i in range(len(test_data)):
            X, Y, id_, _ = test_data[test_i]
            output = model(X)
            pred = output[0].cpu().numpy()
            Y = Y.cpu().numpy() # NBWHC

            if config["save_prediction_mask"]:
                _out = np.argmax(np.moveaxis(np.reshape(pred, (2,18,256,256)), 1, -1), axis=0)
                nib.save(nib.Nifti1Image(_out, np.eye(4)), config["base_path"] + "preds/" + id_ + "_mask.nii.gz")

            if config["save_prediction_softmaxprob"]:
                _out = np.moveaxis(np.moveaxis(np.reshape(pred, (2,18,256,256)), 1, -1), 0, -1)
                nib.save(nib.Nifti1Image(_out, np.eye(4)), config["base_path"] + "preds/" + id_ + "_softmaxprob.nii.gz")

            results[id_] = Metric(pred, Y).all()

            # Results after post-processing
            if config["removeSmallIslands_thr"] != -1:
                pred = removeSmallIslands(pred, thr=config["removeSmallIslands_thr"])
                results_post[id_] = Metric(pred, Y).all()

    with open(config["base_path"] + "results.json", "w") as f:
        f.write(json.dumps(results))

    if config["removeSmallIslands_thr"] != -1:
        # Results after post-processing
        with open(config["base_path"] + "results-post.json", "w") as f:
            f.write(json.dumps(results_post))

    log("End")
