import numpy as np
from tqdm import tqdm
from numba import jit, njit, prange
from scipy.ndimage.morphology import *

@njit()
def get_window(image1, i, j,k, w1):
    return image1[
        np.max(np.array([i-w1, 0])):np.min(np.array([image1.shape[0], i+w1])),
        np.max(np.array([j-w1, 0])):np.min(np.array([image1.shape[1], j+w1])),
        np.max(np.array([k-w1, 0])):np.min(np.array([image1.shape[2], k+w1])),
        ]

@njit(fastmath = True)
def handle_window(segmentation, ground_truth, index, window):
    seg = get_window(segmentation, index[0], index[1], index[2], window)
    gt = get_window(ground_truth, index[0], index[1], index[2], window)
    seg_and_gt = seg & gt
    intersection = np.sum(seg_and_gt)
    union = np.sum(seg) + np.sum(gt)
    if (np.prod(seg_and_gt) == 0 and union > 0):
        return 2 * intersection / union
    else:
        return 0
@jit(fastmath = True)
def evaluate(segmentation : np.ndarray, 
             ground_truth : np.ndarray)->float:
    radius = 5
    res = 0
    n = 0

    combined = np.logical_or(segmentation, ground_truth)
    obj_indicies = np.argwhere(combined == True)
    for i in range(obj_indicies.shape[0]):
        res_i = handle_window(segmentation, ground_truth, obj_indicies[i], radius)
        if (res_i > 0):
            n += 1
            res += res_i
    return res / n