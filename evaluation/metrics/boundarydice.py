import numpy as np
from numba import njit, prange

@njit()
def get_window(image, i, j, k, w):
    # Compute window bounds without extra array allocations
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
    gt  = get_window(ground_truth, index[0], index[1], index[2], window)
    seg_and_gt = seg & gt
    intersection = np.sum(seg_and_gt)
    union = np.sum(seg) + np.sum(gt)
    # Instead of computing np.prod(seg_and_gt), we check if not all elements are 1
    if (intersection != seg_and_gt.size and union > 0):
        return 2 * intersection / union
    else:
        return 0.0

@njit(parallel=True, fastmath=True)
def evaluate(segmentation, ground_truth):
    radius = 5
    # Use bitwise OR to combine the arrays
    combined = segmentation | ground_truth
    indices = np.argwhere(combined)
    
    # Compute scores for each index in parallel.
    results = np.zeros(indices.shape[0])
    for idx in prange(indices.shape[0]):
        results[idx] = handle_window(segmentation, ground_truth, indices[idx], radius)
    
    # Aggregate results in a serial pass
    total_score = 0.0
    count = 0
    for r in results:
        if r > 0:
            total_score += r
            count += 1
    
    return total_score / count if count > 0 else 0.0
