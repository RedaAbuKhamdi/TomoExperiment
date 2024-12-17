import numpy as np
from numba import njit

@njit
def jaccard(segmentation : np.ndarray, 
             ground_truth : np.ndarray)->float:
    
    segmentation = segmentation.astype(np.int8)
    ground_truth = ground_truth.astype(np.int8)

    intersection = np.logical_and(segmentation, ground_truth)
    union = np.logical_or(segmentation, ground_truth)

    return np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 1