from sacred.observers import FileStorageObserver
from experiments.RegularTrainingTest import ex
#from experiments.VoxelIndividualTest import ex
#from experiments.VoxelInfluenceTest import ex
from experiments.lib.util import Twitter
from experiments.lib.models.RatLesNetModel import RatLesNet
from experiments.lib.data.CRAll import Data
import tensorflow as tf
import itertools, os
import time
import argparse

### TODO
# - Decrease learning rate options should be modelable from here.
# - Check "predict" method from ModelBase class.

parser = argparse.ArgumentParser()
parser.add_argument("-gpu_mem", dest="gpu_mem", default=1)
results = parser.parse_args()

BASE_PATH = "delete/"
messageTwitter = "ratlesnet_"

os.environ["CUDA_VISIBLE_DEVICES"]="0"

### Fixed configuration
data = Data()
data.split(folds=1, prop=[0.8])
config = {}
config["data"] = data
config["Model"] = RatLesNet
config["config.lr"] = 1e-4
config["config.epochs"] = 700 # Originally 700
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
#config["config.find_weights"] = "/home/miguelv/data/in/weights/w-293"
config["config.find_weights"] = ""

### Early stopping
config["config.early_stopping_thr"] = 999

### Decrease Learning Rate On Plateau
# After this number of times that the lr is updated, the training is stopped.
# If this is -1, LR won't decrease.
config["config.lr_updated_thr"] = 3 # Originally 3

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

ess = [-1, 3]
#lrs = [1e-4, 1e-5]
#concats = [1, 2, 3, 4, 5, 6]
#skips = [False, "sum", "concat"]
#fsizes = [3, 6, 12, 18, 22, 25]

params = [ess]
all_configs = list(itertools.product(*params))
ci = 0

#all_configs = [0] # Run 5 times

for es, in all_configs:

    ci += 1
    # Name of the experiment and path
    exp_name = "CE_weighted_log"
    if es == 3:
        exp_name += "_ES"

    try:
        print("Trying: "+exp_name)
        experiment_path = BASE_PATH + exp_name + "/"
        ex.observers = [FileStorageObserver.create(experiment_path)]
        config["base_path"] = experiment_path

        # Testing paramenters
        #config["config.lr"] = lr
        #config["config.growth_rate"] = fsize
        #config["config.concat"] = concat
        config["config.lr_updated_thr"] = es
        #config["config.lambda_length"] = l2

        ex.run(config_updates=config)

        show_text = messageTwitter + exp_name + " ({}/{})".format(ci, len(all_configs))
        print(show_text)
    except KeyboardInterrupt:
        raise
    except tf.errors.ResourceExhaustedError:
        with open("no_memory_errors", "a") as f:
            f.write(exp_name + "\n")
    except:
        raise

#Twitter().tweet("Done " + str(time.time()))
