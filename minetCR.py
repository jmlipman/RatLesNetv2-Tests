from sacred.observers import FileStorageObserver
from experiments.RegularTrainingTest import ex
from experiments.lib.util import Twitter
from experiments.lib.models.MiNetCRModel import MiNetCR
from experiments.lib.data.CR02NOV16 import Data
import tensorflow as tf
import itertools, os

BASE_PATH = "results_MiNet/delete/"
messageTwitter = "minet_d2_concat2"

os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Fixed configuration
config = {}
config["config.lr"] = 1e-5
#config["config.weight_decay"] = 1e-4
config["config.opt"] = tf.train.AdamOptimizer(learning_rate=config["config.lr"])
#config["config.opt"] = tf.contrib.opt.AdamWOptimizer(weight_decay=config["weight_decay"], learning_rate=config["lr"])
config["config.loss"] = "own"
config["config.epochs"] = 100
config["config.batch"] = 1
config["config.initW"] = tf.keras.initializers.he_normal()
config["config.initB"] = tf.constant_initializer(0)
config["config.act"] = "relu"
config["config.classes"] = 2
config["config.alpha_l2"] = 0.01 # Typical value
config["config.early_stopping_c"] = 99
config["config.find_weights"] = "/home/miguelv/MiNet/results_MiNet/delete/growthrate_16/1/weights/w-15"
#config["config.find_weights"] = ""
config["config.growth_rate"] = 16
config["config.concat"] = 2
config["Model"] = MiNetCR
data = Data()
data.split(folds=1, prop=[0.7, 0.2, 0.1])
config["data"] = data

for _ in [1]:

    # Name of the experiment and path
    exp_name = "growthrate_16"
    experiment_path = BASE_PATH + exp_name + "/"
    ex.observers = [FileStorageObserver.create(experiment_path)]
    config["base_path"] = experiment_path
    
    # Testing paramenters
    config["config.growth_rate"] = 16
    config["config.concat"] = 2 # This must be one...

    ex.run(config_updates=config)

    #Twitter().tweet(messageTwitter + exp_name)
