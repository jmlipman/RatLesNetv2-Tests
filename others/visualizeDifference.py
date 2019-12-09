import tensorflow as tf
import pdb, os
import numpy as np
import matplotlib.pyplot as plt
# Visualize difference in learning between CE and CE+Dice

def getTBvalue(path, value):
    """path where all files can be found
       value we are looking for
    """
    files = os.listdir(path)
    results = []
    for f in files:
        for e in tf.train.summary_iterator(path + "/" + f):
            v = e.summary.value
            if len(v) > 0:
                if v[0].tag == value:
                    results.append([e.wall_time, v[0].simple_value])

    results = np.array(results)
    idx = np.argsort(results[:,0]) # Sort by time_wall
    return results[idx]

base_path = "tensorboard/RatLesNetv2_"
dataset = "mixed"

res1 = getTBvalue(base_path + dataset+"_CrossEntropyDiceLoss_1", "val_loss")
res2 = getTBvalue(base_path + dataset+"_CrossEntropyDiceLoss_2", "val_loss")
res3 = getTBvalue(base_path + dataset+"_CrossEntropyDiceLoss_3", "val_loss")

res_CED = np.zeros((res1.shape[0], 3))
res_CED[:,0] = res1[:,1]
res_CED[:,1] = res2[:,1]
res_CED[:,2] = res3[:,1]


res1 = getTBvalue(base_path + dataset+"_CrossEntropyLoss_1", "val_loss")
res2 = getTBvalue(base_path + dataset+"_CrossEntropyLoss_2", "val_loss")
res3 = getTBvalue(base_path + dataset+"_CrossEntropyLoss_3", "val_loss")

res_CE = np.zeros((res1.shape[0], 3))
res_CE[:,0] = res1[:,1]
res_CE[:,1] = res2[:,1]
res_CE[:,2] = res3[:,1]

avg_CED = np.sum(res_CED, axis=1)
avg_CE = np.sum(res_CE, axis=1)

plt.plot(avg_CED, color="red")
plt.plot(avg_CE, color="blue")
plt.show()
