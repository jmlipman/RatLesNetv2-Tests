from sacred.observers import FileStorageObserver
from experiments.RegularTrainingTest import ex
from experiments.lib.util import Twitter
from experiments.lib.models.MiNetCRModel import MiNetCR
from experiments.lib.data.CRAll import Data
import tensorflow as tf
import itertools, os

BASE_PATH = "results_MiNet/"
messageTwitter = "minet_d2_concat2"

os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Fixed configuration
config = {}
config["config.lr"] = 1e-5
config["config.epochs"] = 1000
config["config.batch"] = 1
config["config.initW"] = tf.keras.initializers.he_normal()
config["config.initB"] = tf.constant_initializer(0)
config["config.act"] = "relu"
config["config.classes"] = 2
config["config.early_stopping_c"] = 999
#config["config.find_weights"] = "/home/miguelv/MiNet/results_MiNet/delete/growthrate_16/1/weights/w-15"
config["config.find_weights"] = ""
config["config.growth_rate"] = 18
config["config.concat"] = 2
config["Model"] = MiNetCR
data = Data()
data.split(folds=1, prop=[0.8])
config["data"] = data

# Related to weight decay.
config["config.weight_decay"] = 1e-4 # None will use Adam
# Every X epochs, it will decrease Y rate.
config["config.wd_epochs"] = 200
config["config.wd_rate"] = 0.1 # Always 0.1

config["config.wd_epochs"] = [int(len(data.getFiles("training"))*config["config.wd_epochs"]*i/config["config.batch"]) for i in range(1, int(config["config.epochs"]/config["config.wd_epochs"]+1))]
config["config.wd_rate"] = [1/(10**i) for i in range(len(config["config.wd_epochs"])+1)]
print(config["config.wd_epochs"])
print(config["config.wd_rate"])

# Legacy
#config["config.opt"] = tf.train.AdamOptimizer(learning_rate=config["config.lr"])
#config["config.opt"] = tf.contrib.opt.AdamWOptimizer(weight_decay=config["weight_decay"], learning_rate=config["lr"])
#config["config.loss"] = "own"
#config["config.alpha_l2"] = 0.01 # Typical value
for _ in [1]:

    # Name of the experiment and path
    exp_name = "normal"
    experiment_path = BASE_PATH + exp_name + "/"
    ex.observers = [FileStorageObserver.create(experiment_path)]
    config["base_path"] = experiment_path
    
    # Testing paramenters

    ex.run(config_updates=config)

    #Twitter().tweet(messageTwitter + exp_name)
