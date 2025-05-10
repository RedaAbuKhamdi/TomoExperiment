import numpy as np
from numba import njit

@njit
def evaluate(segmentation : np.ndarray, 
             ground_truth : np.ndarray)->float:

    if segmentation.shape != ground_truth.shape:
        raise ValueError(f"Shape mismatch: segmentation {segmentation.shape} vs ground_truth {ground_truth.shape}")
    u1 = np.unique(segmentation)
    u2 = np.unique(ground_truth)
    if not (set(u1) <= {0,1} and set(u2) <= {0,1}):
        raise ValueError(f"Both inputs must be binary masks (0/1). Got unique(seg)={u1}, unique(gt)={u2}")
    if set(u1) != set(u2):
        raise ValueError(f"segmentation and ground_truth must share the same binary labels. Got {u1} vs {u2}")
    seg_f = segmentation.astype(np.float32)
    gt_f  = ground_truth.astype(np.float32)
    se = (seg_f - gt_f) ** 2
    return float(se.mean())