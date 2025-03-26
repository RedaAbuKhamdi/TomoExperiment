import math
import numpy as np
import tqdm
from numba import jit, prange
from imagedata import ImageData
from metrics import jaccard

# Precompute the integral images (summed-volume tables)
@jit(nopython=True, fastmath=True, parallel=True)
def compute_integrals(image: np.ndarray, integral: np.ndarray, integral_sq: np.ndarray):
    D, H, W = image.shape
    for x in range(1, D + 1):
        for y in range(1, H + 1):
            for z in range(1, W + 1):
                val = image[x - 1, y - 1, z - 1]
                integral[x, y, z] = (
                    val +
                    integral[x - 1, y,   z] +
                    integral[x,   y - 1, z] +
                    integral[x,   y,   z - 1] -
                    integral[x - 1, y - 1, z] -
                    integral[x - 1, y,   z - 1] -
                    integral[x,   y - 1, z - 1] +
                    integral[x - 1, y - 1, z - 1]
                )
                integral_sq[x, y, z] = (
                    val * val +
                    integral_sq[x - 1, y,   z] +
                    integral_sq[x,   y - 1, z] +
                    integral_sq[x,   y,   z - 1] -
                    integral_sq[x - 1, y - 1, z] -
                    integral_sq[x - 1, y,   z - 1] -
                    integral_sq[x,   y - 1, z - 1] +
                    integral_sq[x - 1, y - 1, z - 1]
                )

# Thresholding using precomputed integral images.
@jit(nopython=True, fastmath=True, parallel=True)
def thresholding_precomputed(result: np.ndarray, image: np.ndarray,
                             integral: np.ndarray, integral_sq: np.ndarray,
                             q: float, window: int, beta : float):
    # image is assumed to be 3D: (D, H, W)
    D, H, W = image.shape
    r = window // 2

    for x in prange(D):
        for y in range(H):
            for z in range(W):
                # Clamp window boundaries
                x0 = x - r if x - r >= 0 else 0
                y0 = y - r if y - r >= 0 else 0
                z0 = z - r if z - r >= 0 else 0
                x1 = x + r if x + r < D else D - 1
                y1 = y + r if y + r < H else H - 1
                z1 = z + r if z + r < W else W - 1

                count = (x1 - x0 + 1) * (y1 - y0 + 1) * (z1 - z0 + 1)
                # Use inclusion-exclusion with precomputed integrals.
                sum_val = (integral[x1 + 1, y1 + 1, z1 + 1] -
                           integral[x0,     y1 + 1, z1 + 1] -
                           integral[x1 + 1, y0,     z1 + 1] -
                           integral[x1 + 1, y1 + 1, z0] +
                           integral[x0,     y0,     z1 + 1] +
                           integral[x0,     y1 + 1, z0] +
                           integral[x1 + 1, y0,     z0] -
                           integral[x0,     y0,     z0])
                sum_sq_val = (integral_sq[x1 + 1, y1 + 1, z1 + 1] -
                              integral_sq[x0,     y1 + 1, z1 + 1] -
                              integral_sq[x1 + 1, y0,     z1 + 1] -
                              integral_sq[x1 + 1, y1 + 1, z0] +
                              integral_sq[x0,     y0,     z1 + 1] +
                              integral_sq[x0,     y1 + 1, z0] +
                              integral_sq[x1 + 1, y0,     z0] -
                              integral_sq[x0,     y0,     z0])
                mean = sum_val / count
                var = sum_sq_val / count - mean * mean
                if var < 0:
                    var = 0.0
                std = math.sqrt(var)
                thresh = mean + q * std + beta

                result[x, y, z] = 1 if image[x, y, z] > thresh else 0

def segment(imageData: ImageData, params: dict = None):
    print("Start Niblack")
    img = imageData.image
    D, H, W = img.shape
    best_image = np.zeros(img.shape, dtype=np.uint8)
    best_val = 0.0
    best_beta = params["beta"] if params is not None else np.min(img)
    best_k = params["k"] if params is not None else 0.0
    best_window = int(params["w"]) if params is not None else 25

    # Precompute the integral images once.
    integral = np.zeros((D + 1, H + 1, W + 1), dtype=img.dtype)
    integral_sq = np.zeros((D + 1, H + 1, W + 1), dtype=img.dtype)
    compute_integrals(img, integral, integral_sq)

    if params is not None:
        segmentation = np.zeros(img.shape, dtype=np.uint8)
        thresholding_precomputed(segmentation, img, integral, integral_sq, best_k, best_window, best_beta)
        mean_metric = 0.0
        gt_count = 0
        for index, ground_truth_slice in imageData.get_ground_truth_slices():
            segmented_slice = segmentation[index]
            mean_metric += jaccard(segmented_slice, ground_truth_slice)
            gt_count += 1
        mean_metric /= gt_count
        best_val = mean_metric
        best_image = segmentation.copy()
    else:
        # Define search ranges.
        ks = np.linspace(-2, 2, 40)
        betas = np.array([0, np.min(img), np.std(img)])
        max_window = max(1, min(min(img.shape) // 2, 500))
        # Create a range of window sizes from best_window up to max_window.
        windows = np.arange(best_window, max_window, 
                            (max_window - best_window) // 30 if max_window > best_window else 1, 
                            dtype=np.int16)
        for window in tqdm.tqdm(windows, desc="Window sizes"):
            for i in tqdm.tqdm(range(ks.size), desc="k values", leave=False):
                for j in tqdm.tqdm(range(betas.size), desc="beta values", leave=False):
                    segmentation = np.zeros(img.shape, dtype=np.uint8)
                    thresholding_precomputed(segmentation, img, integral, integral_sq, ks[i], window, betas[j])
                    mean_metric = 0.0
                    gt_count = 0
                    for index, ground_truth_slice in imageData.get_ground_truth_slices():
                        segmented_slice = segmentation[index]
                        mean_metric += jaccard(segmented_slice, ground_truth_slice)
                        gt_count += 1
                    mean_metric /= gt_count

                    if mean_metric > best_val:
                        print(f"Improved metric: {mean_metric}")
                        best_val = mean_metric
                        best_k = ks[i]
                        best_beta = betas[j]
                        best_image = segmentation.copy()
                        best_window = window

    params = {
        "k": float(best_k),
        "w": int(best_window),
        "beta": float(best_beta),
        "jaccard": float(best_val)
    }
    return best_image, params
