from sacred.observers import FileStorageObserver
from experiments.RegularTrainingTest import ex
from experiments.lib.util import Twitter
from experiments.lib.models.UNet3DModel import UNet3D
from experiments.lib.data.CRAll import Data
import tensorflow as tf
import itertools, os
import time

BASE_PATH = "results_3DUNet_DELETE/"
messageTwitter = "ratlesnet_"

os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Fixed configuration
data = Data()
data.split(folds=1, prop=[0.8])
config = {}
config["data"] = data
config["Model"] = UNet3D 
config["config.lr"] = 1e-9
config["config.epochs"] = 30
config["config.batch"] = 1
config["config.initW"] = tf.keras.initializers.he_normal()
config["config.initB"] = tf.constant_initializer(0)
config["config.act"] = "relu"
config["config.classes"] = 2

# Model architecture
config["config.momentum"] = 0.99

### Loading Weights
config["config.find_weights"] = ""
config["config.find_weights"] = "/home/miguelv/pythonUEF/MiNet/results_3DUNet_DELETE/test/1/weights/w-11"

### Early stopping
config["config.early_stopping_thr"] = 999

### Decrease Learning Rate On Plateau
# After this number of times that the lr is updated, the training is stopped.
# If this is -1, LR won't decrease.
config["config.lr_updated_thr"] = -1

### Weight Decay
config["config.weight_decay"] = None # None will use Adam
# Every X epochs, it will decrease Y rate.
config["config.wd_epochs"] = 200
config["config.wd_rate"] = 0.1 # Always 0.1
config["config.wd_epochs"] = [int(len(data.getFiles("training"))*config["config.wd_epochs"]*i/config["config.batch"]) for i in range(1, int(config["config.epochs"]/config["config.wd_epochs"]+1))]
config["config.wd_rate"] = [1/(10**i) for i in range(len(config["config.wd_epochs"])+1)]

### Legacy
#config["config.opt"] = tf.train.AdamOptimizer(learning_rate=config["config.lr"])
#config["config.opt"] = tf.contrib.opt.AdamWOptimizer(weight_decay=config["weight_decay"], learning_rate=config["lr"])
#config["config.loss"] = "own"
#config["config.alpha_l2"] = 0.01 # Typical value


for _ in [1]:

    # Name of the experiment and path
    #exp_name = "lr" + str(lr) + "_concat" + str(concat) + "_f" + str(fsize) + "_skip" + str(skip)
    exp_name = "test"

    print("Trying: "+exp_name)
    experiment_path = BASE_PATH + exp_name + "/"
    ex.observers = [FileStorageObserver.create(experiment_path)]
    config["base_path"] = experiment_path
    
    # Testing paramenters
    # Empty for now.

    ex.run(config_updates=config)

    #show_text = messageTwitter + exp_name + " ({}/{})".format(ci, len(all_configs))
    #print(show_text)

#Twitter().tweet("Done" + str(time.time()))
