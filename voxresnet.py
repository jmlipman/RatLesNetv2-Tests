from sacred.observers import FileStorageObserver
from experiments.TrainingEvaluation import ex
#from experiments.VoxelIndividualTest import ex
#from experiments.VoxelInfluenceTest import ex
#from experiments.lib.util import Twitter
from lib.models.VoxResNet import VoxResNet
from lib.data.CRAllDataset import CRAllDataset as DataOrig
from lib.data.CRMixedDataset import CRMixedDataset as DataMixed
#from lib.data.LeidenDataset import LeidenDataset as Data
import itertools, os
import time, torch
import numpy as np
from lib.losses import *
from lib.utils import he_normal
from lib.lr_scheduler import CustomReduceLROnPlateau
import argparse
from sacred import SETTINGS
SETTINGS.CONFIG.READ_ONLY_CONFIG = False

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

BASE_PATH = "results_VoxResNet/"
messageTwitter = "ratlesnet_"

os.environ["CUDA_VISIBLE_DEVICES"]="0"

### Fixed configuration
# TODO let's see how to put the data.
#data = Data
#data.split(folds=1, prop=[0.7, 0.2, 0.1]) # 0.8
config = {}
#config["data"] = Data
config["Model"] = VoxResNet
config["config.device"] = device
config["config.lr"] = 1e-4
config["config.epochs"] = 700 # Originally 700
config["config.batch"] = 1
#config["config.initW"] = torch.nn.init.kaiming_normal_
config["config.initW"] = he_normal
config["config.initB"] = torch.nn.init.zeros_
config["config.act"] = torch.nn.ReLU()
#config["config.loss_fn"] = torch.nn.BCELoss()
config["config.loss_fn"] = VoxResNet_CE
config["config.opt"] = torch.optim.Adam
config["config.classes"] = 2


### Model architecture
config["config.growth_rate"] = 18
config["config.concat"] = 2
config["config.first_filters"] = 12
config["config.skip_connection"] = "concat" #sum, False
config["config.dim_reduc"] = False

### Save validation results
# The following brains will be saved during validation. If not wanted, empty list.
config["config.pc_name"] = pc_name
if pc_name == "FUJ":
    config["config.save_validation"] = []
    BASE_PATH = "/home/miguelv/data/out/RAW/"
elif pc_name == "nmrcs3":
    config["config.save_validation"] = []
    BASE_PATH = "/home/miguelv/data/out/RAW/"
elif pc_name == "sampo-tipagpu1":
    config["config.save_validation"] = []
    BASE_PATH = "/home/users/miguelv/data/out/RAW/"
else:
    raise Exception("Unknown PC: "+pc_name)
config["config.save_npy"] = False
config["config.save_prediction_mask"] = True # Save masks on Testing section. (mask = np.argmax(...))
config["config.save_prediction_softmaxprob"] = False # Save softmax predictions on Testing section.
config["config.save_prediction_logits"] = False # Save logits of the predictions on Testing section
config["config.removeSmallIslands_thr"] = 20 # Remove independent connected components. Use 20.

### Loading Weights
#config["config.model_state"] = "/home/miguelv/data/out/Lesion/Journal/2-baseline/0-voxrat1/700ep/VoxResNet_mixed/1/model/model-699"
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
config["config.weight_decay"] = 0

### Data-related
config["config.brainmask"] = None
config["config.overlap"] = None

####### Not migrated confs:
### L2 regularization
config["config.L2"] = None

### Early stopping
config["config.early_stopping_thr"] = 999

BASE_PATH += "VoxResNet/"

#lrs = [1e-4, 1e-5]
#concats = [1, 2, 3, 4, 5, 6]
#skips = [False, "sum", "concat"]
#fsizes = [3, 6, 12, 18, 22, 25]

#params = [lrs, concats, skips, fsizes]
#all_configs = list(itertools.product(*params))
ci = 0

all_configs = [DataOrig, DataMixed] # Run 5 times

for dat in all_configs:

    for i in range(3):
        ci += 1
        # Name of the experiment and path
        exp_name = "VoxResNet"
        if dat == DataOrig:
            exp_name += "_orig"
        elif dat == DataMixed:
            exp_name += "_mixed"


        try:
            print("Trying: "+exp_name)
            experiment_path = BASE_PATH + exp_name + "/"
            ex.observers = [FileStorageObserver.create(experiment_path)]
            config["base_path"] = experiment_path

            # Testing paramenters
            config["data"] = dat

            ex.run(config_updates=config)

            show_text = messageTwitter + exp_name + " ({}/{})".format(ci, len(all_configs))
            print(show_text)
        except KeyboardInterrupt:
            raise
        except:
            #Twitter().tweet("Error " + str(time.time()))
            raise

#Twitter().tweet("Done " + str(time.time()))
