import numpy as np
from numba import jit
from scipy.ndimage.morphology import *


@jit()
def dice(segmentation : np.ndarray, 
             ground_truth : np.ndarray)->float:
    intersection = segmentation & ground_truth
    union = segmentation | ground_truth
    return 2*np.sum(intersection) / (np.sum(segmentation)+ np.sum(ground_truth)) if np.sum(union) > 0 else 0

@jit()
def get_window(image1, i, j,k, w1):
    return image1[
        np.max(np.array([i-w1, 0])):np.min(np.array([image1.shape[0], i+w1])),
        np.max(np.array([j-w1, 0])):np.min(np.array([image1.shape[1], j+w1])),
        np.max(np.array([k-w1, 0])):np.min(np.array([image1.shape[2], k+w1])),
        ]

@jit()
def boundary(segmentation : np.ndarray, 
             ground_truth : np.ndarray,
             window : int):
    combined = np.logical_or(segmentation, ground_truth)
    obj_indicies = np.argwhere(combined == True)
    for index in obj_indicies:
        if (( np.any(get_window(segmentation, index[0], index[1], index[2], window) == 0) ) or
         ( np.any(get_window(ground_truth, index[0], index[1], index[2], window) == 0))):
            yield index

@jit()
def evaluate(segmentation : np.ndarray, 
             ground_truth : np.ndarray)->float:
    radius = 5
    res = 0
    n = 0
    for index in boundary(segmentation, ground_truth, radius):
        res += dice(
            get_window(segmentation, index[0], index[1], index[2], radius),
            get_window(ground_truth, index[0], index[1], index[2], radius))
        n += 1
    return res / n