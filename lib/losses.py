# Notes on the Weighted Cross Entropy (WCE) functions
# They look the same, but they need to be separated in diff functions
# Because data._computeWeight will generate diff weights based on the
# names of the WCE functions.

import torch
import numpy as np
from lib.utils import np2cuda

#def CrossEntropyLoss(y_pred, y_true):
#    return -torch.mean(y_true * torch.log(y_pred + 1e-15))
def CrossEntropyLoss(y_pred, y_true, config, weights=1):
    """Regular Cross Entropy loss function.
       It is possible to use weights with the shape of BWHD (no channel).

       Args:
        `y_pred`: predictions after softmax, BCWHD.
        `y_true`: labels one-hot encoded, BCWHD.
        `weights`: weights tensor, BWHD.

    """
    ce = torch.sum(y_true * torch.log(y_pred + 1e-15), axis=1)
    return -torch.mean(ce*weights)


def DiceLoss(y_pred, y_true, config):
    """Binary Dice loss function.

       Args:
        `y_pred`: predictions after softmax, BCWHD.
        `y_true`: labels one-hot encoded, BCWHD.
    """
    num = 2 * torch.sum(y_pred * y_true, axis=(1,2,3,4))
    denom = torch.sum(torch.pow(y_pred, 2) + torch.pow(y_true, 2), axis=(1,2,3,4))
    return (1 - torch.sum(num / (denom + 1e-6)))

def CrossEntropyDiceLoss(y_pred, y_true, config):
    """Cross Entropy combined with Dice Loss.

       Args:
        `y_pred`: predictions after softmax, BCWHD.
        `y_true`: labels one-hot encoded, BCWHD.
    """
    return CrossEntropyLoss(y_pred, y_true, config) + DiceLoss(y_pred, y_true, config)

def WeightedCrossEntropy_ClassBalance(y_pred, y_true, config, weights):
    """Weighted Cross Entropy in which the weights are based on the
       number of voxels that belong to a certain label.
       For instance: [0, 0, 1, 0, 0, 1]
                     [1, 1, 0, 1, 1, 0]
       would generate:
                     [2/6 2/6 4/6 2/6 2/6 4/6]
       In other words, voxels that belong to common classes will have an
       inversely proportional weight/importance.

       Args:
        `y_pred`: predictions after softmax, BCWHD.
        `y_true`: labels one-hot encoded, BCWHD.
        `weights`: weights tensor, BWHD.
    """
    # Number of voxels per image
    #numvox = y_true[0,0].flatten().shape[0]
    # Number of 1s in each channel in each sample
    #ones = torch.sum(y_true, axis=(2,3,4)) # Size: BC
    #weights_ = 1 - ones/numvox
    # weights size: BWHD (no channel)a
    # Broadcast them so that the weights can multiply the cross entropy
    #weights = torch.sum((y_true.permute(2,3,4,0,1)*weights_).permute(3,4,0,1,2), axis=1)


    #ce = torch.sum(y_true * torch.log(y_pred + 1e-15), axis=1)
    #return -torch.mean(ce*weights)
    return CrossEntropyLoss(y_pred, y_true, config, weights)

def WeightedCrossEntropy_DistanceMap(y_pred, y_true, config, weights):
    """Weighted Cross Entropy in which the weights are based on the
       the distance to the boundaries of the labels.
       It gives less importance/weight to the voxels close to the boundaries
       because they can be less accurate.

       Args:
        `y_pred`: predictions after softmax, BCWHD.
        `y_true`: labels one-hot encoded, BCWHD.
        `weights`: weights tensor, BWHD.
    """
    #ce = torch.sum(y_true * torch.log(y_pred + 1e-15), axis=1)
    #return -torch.mean(ce*weights)
    return CrossEntropyLoss(y_pred, y_true, config, weights)

def _LengthRegularization(y_pred):
    x_ = y_pred[:,:,1:,:,:] - y_pred[:,:,:-1,:,:]
    y_ = y_pred[:,:,:,1:,:] - y_pred[:,:,:,:-1,:]
    z_ = y_pred[:,:,:,:,1:] - y_pred[:,:,:,:,:-1]
    dx_ = torch.abs(x_[:,:,:,:-1,:-1])
    dy_ = torch.abs(y_[:,:,:-1,:,:-1])
    dz_ = torch.abs(z_[:,:,:-1,:-1,:])
    length = torch.sum(dx_ + dy_ + dz_)
    return length

def CrossEntropy_LengthRegularization(y_pred, y_true, config):
    return CrossEntropyLoss(y_pred, y_true, config) + config["alpha_length"]*_LengthRegularization(y_pred)

def Dice_LengthRegularization(y_pred, y_true, config):
    return DiceLoss(y_pred, y_true, config) + config["alpha_length"]*_LengthRegularization(y_pred)


# Boundary loss
