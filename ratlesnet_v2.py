from sacred.observers import FileStorageObserver
from experiments.TrainingEvaluation import ex
#from experiments.ReceptiveField import ex
from lib.models.RatLesNetv2 import *
from lib.data.CRAllDataset import CRAllDataset as DataOrig
from lib.data.CRMixedDataset import CRMixedDataset as DataMixed
import itertools, os
import time, torch
import numpy as np
from lib.losses import *
from lib.utils import he_normal
from lib.lr_scheduler import CustomReduceLROnPlateau
import argparse

#os.environ["CUDA_VISIBLE_DEVICES"]="-1"

# Speed up the script.
# This seems to not be a good idea when the input sizes change during the training.
torch.backends.cudnn.benchmark = True

pc_name = os.uname()[1]

parser = argparse.ArgumentParser()
# -gpu arg indicates which gpu to use. For CPU-only, -gpu -1
parser.add_argument("-gpu", dest="gpu", default=0)
args = parser.parse_args()

# Setting GPU-CPU
if args.gpu >= torch.cuda.device_count():
    if torch.cuda.device_count() == 0:
        print("No available GPUs. Run with -gpu -1")
    else:
        print("Available GPUs:")
        for i in range(torch.cuda.device_count()):
            print(" > GPU #"+str(i)+" ("+torch.cuda.get_device_name(i)+")")
    raise Exception("The GPU #"+str(args.gpu)+" does not exist. Check available GPUs.")

if args.gpu > -1:
    device = torch.device("cuda:"+str(args.gpu))
else:
    device = torch.device("cpu")

### TODO
# - Decrease learning rate options should be modelable from here.
# - Check "predict" method from ModelBase class.

BASE_PATH = "results_ablation_RatLesNetv2/"
messageTwitter = "ratlesnet_"

os.environ["CUDA_VISIBLE_DEVICES"]="0"

### Fixed configuration
# TODO let's see how to put the data.
#data = Data
#data.split(folds=1, prop=[0.7, 0.2, 0.1]) # 0.8
config = {}
config["data"] = DataOrig
config["Model"] = RatLesNet_v2_v1
config["config.device"] = device
config["config.lr"] = 1e-4
config["config.epochs"] = 700 # Originally 700
config["config.batch"] = 1
#config["config.initW"] = torch.nn.init.kaiming_normal_
config["config.initW"] = he_normal
config["config.initB"] = torch.nn.init.zeros_
config["config.act"] = torch.nn.ReLU()
#config["config.loss_fn"] = torch.nn.BCELoss()
config["config.loss_fn"] = CrossEntropyDiceLoss
config["config.opt"] = torch.optim.Adam
config["config.classes"] = 2


### Model architecture
config["config.first_filters"] = 39 #32 for RatLesNetv2, 21 for same params as RatLesNet
config["config.block_convs"] = 2 # Number of convolutions within block

### Save validation results
# The following brains will be saved during validation. If not wanted, empty list.
if pc_name == "FUJ":
    config["config.save_validation"] = ["02NOV2016_2h_40", "02NOV2016_24h_43"]
elif pc_name == "nmrcs3":
    config["config.save_validation"] = ["02NOV2016_24h_5", "02NOV2016_2h_6"]
elif pc_name == "sampo-tipagpu1":
    config["config.save_validation"] = ["02NOV2016_24h_5", "02NOV2016_2h_6"]
else:
    raise Exception("Unknown PC: "+pc_name)
config["config.save_npy"] = False
config["config.save_prediction"] = False # Save preds on Testing section.
config["config.removeSmallIslands_thr"] = 20 # Remove independent connected components. Use 20.

### Loading Weights
#config["config.model_state"] = "/home/miguelv/data/out/Lesion/Journal/1-connectivity-loss/vox_CEDice_orig/3/model/model-699"
config["config.model_state"] = ""

### LR Scheduler. Reduce learning rate on plateau
config["config.lr_scheduler_patience"] = 4
config["config.lr_scheduler_factor"] = 0.1
config["config.lr_scheduler_improvement_thr"] = 0.01 # Percentage to improve
# After this number of times that the lr is updated, the training is stopped.
config["config.lr_scheduler_limit"] = 3
config["config.lr_scheduler_verbose"] = True
config["config.lr_scheduler"] = CustomReduceLROnPlateau(
        patience=config["config.lr_scheduler_patience"],
        factor=config["config.lr_scheduler_factor"],
        improvement_thr=config["config.lr_scheduler_improvement_thr"],
        limit=config["config.lr_scheduler_limit"],
        verbose=config["config.lr_scheduler_verbose"]
        )
config["config.lr_scheduler"] = None

### Regularization
config["config.alpha_length"] = 0.000001


####### Not migrated confs:
### L2 regularization
config["config.L2"] = None

### Early stopping
config["config.early_stopping_thr"] = 999

### Weight Decay
# TODO I am not sure I need all of this now, since weight decay can be
# set in a much easier way now.
#config["config.weight_decay"] = None # None will use Adam
# Every X epochs, it will decrease Y rate.
#config["config.wd_epochs"] = 200
#config["config.wd_rate"] = 0.1 # Always 0.1
#config["config.wd_epochs"] = [int(len(data.getFiles("training"))*config["config.wd_epochs"]*i/config["config.batch"]) for i in range(1, int(config["config.epochs"]/config["config.wd_epochs"]+1))]
#config["config.wd_rate"] = [1/(10**i) for i in range(len(config["config.wd_epochs"])+1)]

#lrs = [1e-4, 1e-5]
#concats = [1, 2, 3, 4, 5, 6]
#skips = [False, "sum", "concat"]
#fsizes = [3, 6, 12, 18, 22, 25]

#params = [lrs, concats, skips, fsizes]
#all_configs = list(itertools.product(*params))
ci = 0

#all_configs = [CrossEntropyLoss, DiceLoss, CrossEntropyDiceLoss, WeightedCrossEntropy_ClassBalance, WeightedCrossEntropy_DistanceMap]
all_configs = [DataOrig, DataMixed]

for data in all_configs:

    for __ in range(3):
        ci += 1
        # Name of the experiment and path
        exp_name = "level2_sameparams"
        if not config["config.lr_scheduler"] is None:
            exp_name += "_ES"
        if data == DataOrig:
            exp_name += "_orig"
        else:
            exp_name += "_mixed"

        try:
            print("Trying: "+exp_name)
            experiment_path = BASE_PATH + exp_name + "/"
            ex.observers = [FileStorageObserver.create(experiment_path)]
            config["base_path"] = experiment_path

            # Testing paramenters
            config["data"] = data
            #config["config.growth_rate"] = fsize
            #config["config.concat"] = concat
            #config["config.skip_connection"] = skip
            #config["config.lambda_length"] = l2

            ex.run(config_updates=config)

            show_text = messageTwitter + exp_name + " ({}/{})".format(ci, len(all_configs))
            print(show_text)
        except KeyboardInterrupt:
            raise
        except:
            #Twitter().tweet("Error " + str(time.time()))
            raise

#Twitter().tweet("Done " + str(time.time()))
