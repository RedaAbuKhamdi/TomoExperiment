import numpy as np
from numba import njit, prange

# Inline get_window to avoid creating temporary arrays.
@njit(inline='always')
def get_window(image, i, j, k, w):
    i0 = i - w if i - w >= 0 else 0
    i1 = i + w if i + w < image.shape[0] else image.shape[0]
    j0 = j - w if j - w >= 0 else 0
    j1 = j + w if j + w < image.shape[1] else image.shape[1]
    k0 = k - w if k - w >= 0 else 0
    k1 = k + w if k + w < image.shape[2] else image.shape[2]
    return image[i0:i1, j0:j1, k0:k1]

@njit(fastmath=True)
def handle_window(segmentation, ground_truth, index, window):
    seg = get_window(segmentation, index[0], index[1], index[2], window)
    gt = get_window(ground_truth, index[0], index[1], index[2], window)
    
    # Instead of using bitwise operations that allocate new arrays,
    # we iterate once over the window.
    intersection = 0
    sum_seg = 0
    sum_gt = 0
    # We'll also check if every element of the element–wise AND is True.
    all_true = True
    s0, s1, s2 = seg.shape[0], seg.shape[1], seg.shape[2]
    for ii in range(s0):
        for jj in range(s1):
            for kk in range(s2):
                s_val = seg[ii, jj, kk]
                gt_val = gt[ii, jj, kk]
                if s_val:
                    sum_seg += 1
                if gt_val:
                    sum_gt += 1
                if s_val and gt_val:
                    intersection += 1
                else:
                    all_true = False
    union = sum_seg + sum_gt
    # If not all elements are simultaneously True and union > 0, return 2*intersection/union.
    if (not all_true) and union > 0:
        return 2.0 * intersection / union
    else:
        return 0.0

@njit(parallel=True, fastmath=True)
def evaluate(segmentation, ground_truth):
    radius = 5
    shape = segmentation.shape
    # Precompute a combined mask of foreground.
    combined = np.empty(shape, dtype=np.bool_)
    for i in prange(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                combined[i, j, k] = segmentation[i, j, k] or ground_truth[i, j, k]
                
    # Count the number of foreground voxels.
    count = 0
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                if combined[i, j, k]:
                    count += 1
                    
    # Allocate an array for the indices.
    obj_indices = np.empty((count, 3), dtype=np.int64)
    idx = 0
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                if combined[i, j, k]:
                    obj_indices[idx, 0] = i
                    obj_indices[idx, 1] = j
                    obj_indices[idx, 2] = k
                    idx += 1
                    
    # Allocate an array to hold the result for each index.
    res_arr = np.empty(obj_indices.shape[0], dtype=np.float64)
    for i in prange(obj_indices.shape[0]):
        res_arr[i] = handle_window(segmentation, ground_truth, obj_indices[i], radius)
        
    total = 0.0
    valid = 0.0
    for i in range(res_arr.shape[0]):
        if res_arr[i] > 0:
            total += res_arr[i]
            valid += 1.0
            
    return total / valid if valid > 0 else 0.0