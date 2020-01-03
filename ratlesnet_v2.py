from sacred.observers import FileStorageObserver
from experiments.TrainingEvaluation import ex
#from experiments.ReceptiveField import ex
from lib.models.RatLesNetv2 import *
from lib.data.CRAllDataset import CRAllDataset as DataOrig
from lib.data.CRMixedDataset import CRMixedDataset as DataMixed
from lib.data.CR24hDataset import CR24hDataset as Data24h
import itertools, os
import time
import numpy as np
from lib.losses import *
from lib.utils import he_normal
from lib.lr_scheduler import CustomReduceLROnPlateau
import argparse
from sacred import SETTINGS
SETTINGS.CONFIG.READ_ONLY_CONFIG = False

# NOTE verify whether this should be before "import torch"
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Speed up the script.
# This seems to not be a good idea when the input sizes change during the training.
torch.backends.cudnn.benchmark = True

pc_name = os.uname()[1]

parser = argparse.ArgumentParser()
# -gpu arg indicates which gpu to use. For CPU-only, -gpu -1
parser.add_argument("-gpu", dest="gpu", default=0)
args = parser.parse_args()

import torch
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

### Fixed configuration
config = {}
config["data"] = DataMixed
config["Model"] = RatLesNetv2
config["config.device"] = device
config["config.lr"] = 1e-4
config["config.epochs"] = 700 # Originally 700
config["config.batch"] = 1
config["config.initW"] = he_normal
config["config.initB"] = torch.nn.init.zeros_
config["config.act"] = torch.nn.ReLU()
#config["config.loss_fn"] = torch.nn.BCELoss()
config["config.loss_fn"] = CrossEntropyDiceLoss
config["config.opt"] = torch.optim.RAdam
config["config.classes"] = 2


### Model architecture
# 32 for RatLesNetv2
# 21 for same params as RatLesNet
# 39 for LVL2_sameparams
# 26 for DenseNet
config["config.first_filters"] = 32
config["config.block_convs"] = 2 # Number of convolutions within block

### Save validation results
# The following brains will be saved during validation. If not wanted, empty list.
if pc_name == "FUJ":
    config["config.save_validation"] = ["02NOV2016_2h_40", "02NOV2016_24h_43"]
    BASE_PATH = "/home/miguelv/data/out/RAW/"
elif pc_name == "nmrcs3":
    config["config.save_validation"] = ["02NOV2016_24h_5", "02NOV2016_2h_6"]
    BASE_PATH = "/home/miguelv/data/out/RAW/"
elif pc_name == "sampo-tipagpu1":
    config["config.save_validation"] = ["02NOV2016_24h_5", "02NOV2016_2h_6"]
    BASE_PATH = "/home/users/miguelv/data/out/RAW/"
else:
    raise Exception("Unknown PC: "+pc_name)
config["config.save_npy"] = False
config["config.save_prediction_mask"] = False # Save masks on Testing section. (mask = np.argmax(...))
config["config.save_prediction_softmaxprob"] = False # Save softmax predictions on Testing section.
config["config.save_prediction_logits"] = True # Save logits of the predictions on Testing section.
config["config.removeSmallIslands_thr"] = 20 # Remove independent connected components. Use 20. If not, -1

### Loading Weights
#config["config.model_state"] = "/home/miguelv/data/out/Lesion/Journal/2-baseline/1-ratlesnetv2/baseline_mixed/1/model/model-699"
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
#####################

BASE_PATH += "RatLesNetv2_Inconsistency/"

#lrs = [1e-4, 1e-5]
#concats = [1, 2, 3, 4, 5, 6]
#skips = [False, "sum", "concat"]
#fsizes = [3, 6, 12, 18, 22, 25]
#datas = [DataOrig, DataMixed]

#params = [datas]
#all_configs = list(itertools.product(*params))
ci = 0

#all_configs = [CrossEntropyLoss, DiceLoss, CrossEntropyDiceLoss, WeightedCrossEntropy_ClassBalance, WeightedCrossEntropy_DistanceMap]
all_configs = [0]

for ss in all_configs:

    for i in range(5):
        ci += 1
        # Name of the experiment and path
        exp_name = "baseline_improved"
        if not config["config.lr_scheduler"] is None:
            exp_name += "_ES"
        #if data == DataOrig:
        #    exp_name += "_orig"
        #else:
        #    exp_name += ""

        try:
            print("Trying: "+exp_name)
            experiment_path = BASE_PATH + exp_name + "/"
            ex.observers = [FileStorageObserver.create(experiment_path)]
            config["base_path"] = experiment_path

            # Testing paramenters
            #config["data"] = data
            #config["config.growth_rate"] = fsize
            #config["config.concat"] = concat
            #config["config.skip_connection"] = skip

            ex.run(config_updates=config)

        except KeyboardInterrupt:
            raise
        except:
            #Twitter().tweet("Error " + str(time.time()))
            raise

#Twitter().tweet("Done " + str(time.time()))
