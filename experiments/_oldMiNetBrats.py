from experiments.lib.data.dataBrats import Data
from experiments.lib.models.MiNetBratsModel import MiNetBrats
import os
import numpy as np
from sacred import Experiment
import tensorflow as tf
import nibabel as nib

os.environ["CUDA_VISIBLE_DEVICES"]="-1"

ex = Experiment("MiNet")

@ex.config
def my_config():
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
    XFolds = 5 # Not in use.
    TrainingType = "temporal"
    Segmentation = "miguel"
    #return config


@ex.main
def main(config, XFolds, TrainingType, Segmentation, base_path, _run):
    #data = Data()
    # As for now, I have 12 samples.
    # Training: 6. Testing: 4. Validation: 2.
    if TrainingType == "temporal":

        #X, Y, idx, brain_ids = data.get_02NOV16(seg=Segmentation)

        base_path = base_path + str(_run._id) + "/"

        config["base_path"] = base_path
        # Create model
        # Trying 5-fold xval

        data = Data()
        model = MiNetBrats(config)
        model.train(data)
        model.predict(data, save=True)
        raise Exception("llego1")
        #model.train(x_train, y_train)
        preds, dice = model.predict(data, save=False)
        #xval_dice = model.measure(preds, y_test)

        print(xval_dice)
        #np.save(config["base_path"]+"preds_"+str(f)+".npy", preds)
        #for i,id_ in enumerate(idx[f][1]):
            #tmp_br = np.argmax(preds[i], axis=-1)
            #tmp_br = np.moveaxis(tmp_br, 0, 2)
            #nib.save(nib.Nifti1Image(tmp_br, np.eye(4)), config["base_path"]+str(id_)+".nii.gz")
            #_run.log_scalar("brain_"+str(id_), xval_dice[:,i], 0)

        d_tmp = data.getNextTestSubject()
        c = 0
        while d_tmp != None:
            X_test, Y_test, age, survival = d_tmp
            preds = model.predict(X_test)
            xval_dice = model.measure(preds, y_test)
            _run.log_scalar("brain_"+str(c), xval_dice[:,i], 0)

            d_tmp = data.getNextTestSubject()
            c += 1

    elif TrainingType == "testing6studies":

       
        config["base_path"] = base_path + str(_run._id) + "/"

        for _ in range(1):
            X, Y, idx, _ = data.get_02NOV16(seg=Segmentation, rand=True)
            val_idx = idx[0][2] # since it's random, the fold 0 is irrelevant
            train_idx = idx[0][0] + idx[0][1]

            x_train = X[train_idx]
            y_train = Y[train_idx]
            x_val = X[val_idx]
            y_val = Y[val_idx]

            # Create model
            model = MiNet(config)
            model.train(x_train, y_train, x_val, y_val)

            x_test, y_test, curr_dirs = data.get_6studies_data_for_prediction()
            c = 0
            while not x_test is None:
                preds = model.predict(x_test)
                xval_dice = model.measure(preds, y_test)

                print(preds.shape[0])
                np.save(config["base_path"]+"preds_"+str(c)+".npy", preds)
                for i in range(preds.shape[0]):
                    id_ = str(c*20 + i) # This 20 comes from the batch size
                    tmp_br = np.argmax(preds[i], axis=-1)
                    tmp_br = np.moveaxis(tmp_br, 0, 2)
                    nib.save(nib.Nifti1Image(tmp_br, np.eye(4)), config["base_path"]+str(id_)+".nii.gz")
                    _run.log_scalar("brain_"+str(id_), xval_dice[:,i], 0)

                with open(config["base_path"] + "brain_ids", "a") as f:
                    for cu in curr_dirs:
                        f.write(cu + "\n")

                x_test, y_test, curr_dirs = data.get_6studies_data_for_prediction()
                c += 1


    elif TrainingType == "fixed":
        X, Y = data.get_all()

        config["base_path"] = base_path + str(_run._id) + "/"

        # Do the xval in here
        for _ in range(1):
            x_train = X[0:15]
            y_train = Y[0:15]
            x_test = X[15:21]
            y_test = Y[15:21]
            x_val = X[21:]
            y_val = Y[21:]

            # Create model
            model = MiNet(config)
            model.train(x_train, y_train, x_val, y_val)
            #model.train(x_train, y_train)
            preds = model.predict(x_test)
            xval_dice = model.measure(preds, y_test)

            np.save(config["base_path"]+"preds.npy", preds)
            ids = [34, 35, 4, 41, 42, 43]
            for i,id_ in enumerate(ids):
                tmp_br = np.argmax(preds[i], axis=-1)
                nib.save(nib.Nifti1Image(tmp_br, np.eye(4)), config["base_path"]+str(id_)+".nii.gz")

            print(xval_dice)
            for i in range(xval_dice.shape[1]):
                _run.log_scalar("brain_"+str(i), xval_dice[:,i], 0)

#if __name__ == "__main__":
#    config = my_config()
#    main(config, 0, 0, 0)
