from sacred.observers import FileStorageObserver
from experiments.RegularTrainingTest import ex
from experiments.lib.util import Twitter
from experiments.lib.models.MiNetBratsModel import MiNetBrats
from experiments.lib.data.BraTS19 import Data
import tensorflow as tf
import itertools, os

os.environ["CUDA_VISIBLE_DEVICES"]="0"

BASE_PATH = "results_MiNet/minet_d2_concat2_BN"
messageTwitter = "minet_d2_concat2"

# 12 -> ok
# 13 -> ok
# 14 -> ok
# 15 -> ok
# 16 -> memory error. 2506050 > 516096
# 17 -> memory error. 2740366 > 516096
# sbatch mem is 4096, priority is 1622
# sbatch mem is 512, priority is 1620

# Fixed configuration
config = {}
config["config.lr"] = 1e-8
#config["config.weight_decay"] = 1e-4
config["config.opt"] = tf.train.AdamOptimizer(learning_rate=config["config.lr"])
#config["config.opt"] = tf.contrib.opt.AdamWOptimizer(weight_decay=config["weight_decay"], learning_rate=config["lr"])
config["config.loss"] = "own"
config["config.epochs"] = 100
config["config.batch"] = 1
config["config.initW"] = tf.keras.initializers.he_normal()
config["config.initB"] = tf.constant_initializer(0)
config["config.act"] = "relu"
config["config.classes"] = 4
config["config.alpha_l2"] = 0.01 # Typical value
config["config.early_stopping_c"] = 99
config["Model"] = MiNetBrats
config["Data"] = Data

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
