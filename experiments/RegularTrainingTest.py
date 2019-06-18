from experiments.lib.data.dataBrats import Data
from experiments.lib.models.MiNetBratsModel import MiNetBrats
from sacred import Experiment
import os

#os.environ["CUDA_VISIBLE_DEVICES"]="-1"

ex = Experiment("RegularTrainingTest")

@ex.config
def my_config():
    """
    # Model's default configuration
    # However, for 100 epochs, lr is better to be 1e-3
    config = {}
    config["lr"] = 1e-5
    #config["weight_decay"] = 1e-4
    config["opt"] = tf.train.AdamOptimizer(learning_rate=config["lr"])
    #config["opt"] = tf.contrib.opt.AdamWOptimizer(weight_decay=config["weight_decay"], learning_rate=config["lr"])
    config["loss"] = "own" 
    config["epochs"] = 100
    config["batch"] = 1
    config["initW"] = tf.keras.initializers.he_normal()
    config["initB"] = tf.constant_initializer(0)
    config["act"] = "relu"
    config["classes"] = 4
    config["alpha_l2"] = 0.01 # Typical value
    config["early_stopping_c"] = 99
    config["growth_rate"] = 1
    #return config
    """
    pass


@ex.main
def main(config, Model, base_path, _run):

    base_path = base_path + str(_run._id) + "/"
    config["base_path"] = base_path

    data = Data()

    model = Model(config)
    model.train(data)
    model.predict(data, save=True)
