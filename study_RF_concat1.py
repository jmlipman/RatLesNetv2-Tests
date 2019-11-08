from sacred.observers import FileStorageObserver
from experiments.FullVoxelInfluenceTest import ex
from experiments.lib.util import Twitter
from experiments.lib.models.RatLesNetModel import RatLesNet
from experiments.lib.data.CRAll import Data
import tensorflow as tf
import itertools, os
import time
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-gpu_mem", dest="gpu_mem", default=1)
parser.add_argument("-start", dest="start", default="-1")
results = parser.parse_args()

if results.start == "-1":
    raise Exception("You must provide a -start")

BASE_PATH = "results_voxelinfluence/"
messageTwitter = "ratlesnet_"

os.environ["CUDA_VISIBLE_DEVICES"]="0"

### Fixed configuration
data = Data()
data.split(folds=1, prop=[0.8])
config = {}
config["data"] = data
config["Model"] = RatLesNet
config["start"] = int(results.start)

config["config.lr"] = 1e-4
config["config.epochs"] = 10 # Originally 700
config["config.batch"] = 1
config["config.initW"] = tf.keras.initializers.he_normal()
config["config.initB"] = tf.constant_initializer(0)
config["config.act"] = "relu"
config["config.classes"] = 2

### Model architecture
config["config.growth_rate"] = 18
config["config.concat"] = 3
config["config.skip_connection"] = "concat" #sum, False

### L2 regularization
config["config.L2"] = None

### Loading Weights
#config["config.find_weights"] = "/home/miguelv/data/out/Lesion/RatLesNet/multipleConfigurations/lr0.0001_concat1_f18_skipFalse/1/weights/w-293"
#config["config.find_weights"] = "/home/miguelv/pythonUEF/MiNet/results_RatLesNet/differences_concat1_skipFalse/1/weights/w-699"
config["config.find_weights"] = "/home/miguelv/data/in/weights/w-293"
#config["config.find_weights"] = ""

### Early stopping
config["config.early_stopping_thr"] = 999

### Decrease Learning Rate On Plateau
# After this number of times that the lr is updated, the training is stopped.
# If this is -1, LR won't decrease.
config["config.lr_updated_thr"] = -1 # Originally 3

### Decrease Learning Rate when Val Loss is too high
# 1.1 -> If val loss is 10% larger than the minimum recorded, decrease lr.
config["config.lr_valloss_ratio"] = 1.1

### Weight Decay
config["config.weight_decay"] = None # None will use Adam
# Every X epochs, it will decrease Y rate.
config["config.wd_epochs"] = 200
config["config.wd_rate"] = 0.1 # Always 0.1
config["config.wd_epochs"] = [int(len(data.getFiles("training"))*config["config.wd_epochs"]*i/config["config.batch"]) for i in range(1, int(config["config.epochs"]/config["config.wd_epochs"]+1))]
config["config.wd_rate"] = [1/(10**i) for i in range(len(config["config.wd_epochs"])+1)]

# Other
config["config.gpu_mem"] = float(results.gpu_mem)

### Legacy
#config["config.opt"] = tf.train.AdamOptimizer(learning_rate=config["config.lr"])
#config["config.opt"] = tf.contrib.opt.AdamWOptimizer(weight_decay=config["weight_decay"], learning_rate=config["lr"])
#config["config.loss"] = "own"
#config["config.alpha_l2"] = 0.01 # Typical value

#lrs = [1e-4, 1e-5]
#concats = [1, 2, 3, 4, 5, 6]
#skips = [False, "sum", "concat"]
#fsizes = [3, 6, 12, 18, 22, 25]

#params = [lrs, concats, skips, fsizes]
#all_configs = list(itertools.product(*params))


# Name of the experiment and path
exp_name = "concat1_skipFalse"

experiment_path = BASE_PATH + exp_name + "/"
ex.observers = [FileStorageObserver.create(experiment_path)]
config["base_path"] = experiment_path

ex.run(config_updates=config)


