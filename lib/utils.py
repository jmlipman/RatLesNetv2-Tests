from datetime import datetime
import torch
import numpy as np
from scipy import ndimage
from scipy.ndimage import distance_transform_edt as dist

def np2cuda(inp):
    return torch.from_numpy(inp.astype(np.float32)).cuda()

def log(text):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(now + ": " + text, flush=True)

def he_normal(w):
    """ He Normal initialization.
    """
    fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(w)
    return torch.nn.init.normal_(w, 0, np.sqrt(2/fan_in))

def border_np(y):
    """Calculates the border of a 3D binary map.
       From NiftyNet.
    """
    west = ndimage.shift(y, [-1, 0, 0], order=0)
    east = ndimage.shift(y, [1, 0, 0], order=0)
    north = ndimage.shift(y, [0, 1, 0], order=0)
    south = ndimage.shift(y, [0, -1, 0], order=0)
    top = ndimage.shift(y, [0, 0, 1], order=0)
    bottom = ndimage.shift(y, [0, 0, -1], order=0)
    cumulative = west + east + north + south + top + bottom
    border = ((cumulative < 6) * y) == 1
    return border

def surfacedist(data):
    """
    data is BCWHD
    """
    # I am assuming there is only one label
    data = np.argmax(data, axis=1)
    surface = np.zeros(data.shape)
    distances = np.zeros(data.shape)
    for b in range(data.shape[0]):
        for x in range(data.shape[1]):
            for y in range(data.shape[2]):
                for z in range(data.shape[3]):
                    if data[b,x,y,z]==1:
                        piece = data[b,x-1:x+2,y-1:y+2,z-1:z+2]
                        if np.sum(piece) != 27:
                            surface[b,x,y,z] = 1
        distances[b,:,:,:] = np.log(dist(1-surface)+np.e)

    return distances

