from experiments.lib.models.MiNetBratsModel import MiNetBrats
from sacred import Experiment
import os


ex = Experiment("RegularTrainingTest")

@ex.main
def main(config, Model, Data, base_path, _run):

    base_path = base_path + str(_run._id) + "/"
    config["base_path"] = base_path

    data = Data()
    data.split(folds=1, prop=[0.7, 0.2, 0.1])
    model = Model(config)
    model.train(data)
    model.test(data)
