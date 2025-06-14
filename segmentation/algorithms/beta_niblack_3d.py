from os import path
import os
import tqdm
import sys
import numpy as np
import pandas as pd

import cupy as cp
import cupyx.scipy.ndimage as ndi_gpu

from matplotlib import pyplot as plt

sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
import config
from imagedata import ImageData
import logging

def thresholding_niblack_gpu(img_f: cp.ndarray,
                             q: float,
                             mean: cp.ndarray,
                             std: cp.ndarray,
                             beta: float) -> cp.ndarray:
    """
    Perform Niblack threshold:
      seg = img >= (mean_local + q * std_local)
    entirely on the GPU.
    """
    return img_f >= (mean + std * q + beta)


def calculate_mean_std_gpu(img: cp.ndarray,
                                    window: int) -> tuple[cp.ndarray,cp.ndarray]:
    """
    Compute local mean & std on each (y,x) slice with a window of half‐width=window//2,
    but automatically shrink at the borders (so you never rely on padding).
    Uses a mask+box‐filter trick to count valid pixels per window.
    """
    # img has shape (D, H, W)
    D, H, W = img.shape
    size = (1, window, window)
    area = window * window

    # 1) sum of values in each local box (zeros outside)
    mean_zero = ndi_gpu.uniform_filter(img, size=size,
                                      mode='constant', cval=0.0)
    sum_vals  = mean_zero * area

    # 2) build a mask of “1” for real pixels, 0 for outside
    mask      = cp.ones_like(img, dtype=cp.float32)
    mean_m    = ndi_gpu.uniform_filter(mask, size=size,
                                      mode='constant', cval=0.0)
    count     = mean_m * area   # how many real pixels in each local window

    # 3) true local mean = sum_vals / count  (avoid division by zero)
    mean_loc  = sum_vals / cp.maximum(count, 1.0)

    # 4) same for sum of squares
    sumsq_zero = ndi_gpu.uniform_filter(img*img, size=size,
                                        mode='constant', cval=0.0)
    sumsq      = sumsq_zero * area

    # 5) variance = E[x²] – (E[x])²
    var_loc    = sumsq / cp.maximum(count, 1.0) - mean_loc*mean_loc
    std_loc    = cp.sqrt(cp.clip(var_loc, 0.0, None))

    return mean_loc, std_loc

# at top‐level once per module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s",
                                      datefmt="%H:%M:%S"))
    logger.addHandler(ch)


def segment(imageData: ImageData, params: dict = None):
    """
    If params provided: single Niblack with (w, k, beta).
    Otherwise: sweep (w, k, beta) to minimize mean-squared error over GT slices.
    """
    name    = imageData.settings["name"]
    img     = imageData.image
    img_gpu = cp.asarray(img, dtype=cp.float32)

    # single-call mode
    if params:
        w = params["w"]; k = params["k"]; b = params["beta"]
        mean, std   = calculate_mean_std_gpu(img_gpu, w)
        seg_gpu     = thresholding_niblack_gpu(img_gpu, k, mean, std, b)
        return cp.asnumpy(seg_gpu), params

    # --- full sweep mode ---
    logger.info("No params provided; running full w,k,beta sweep for MSE")

    # 1) load GT once
    gt_list       = list(imageData.get_ground_truth_slices())
    indices, gts  = zip(*gt_list)
    indices       = np.array(indices, dtype=np.int64)
    gt_stack_gpu  = cp.stack([cp.asarray(g, dtype=cp.bool_) for g in gts], axis=0)
    S             = gt_stack_gpu.shape[0]
    logger.info("Evaluating MSE on %d ground-truth slices", S)

    # 2) define parameter grids
    windows = np.arange(5, min(125, min(img.shape)), 20, dtype=int)      # e.g. [5,25,45,65,85,105]
    ks      = np.linspace(-1, 1, 80, dtype=float)  # 100 values
    betas   = np.linspace(-cp.max(img_gpu).item() / 2,
                          cp.max(img_gpu).item() / 2,
                          40, dtype=float)
    total_runs = windows.size * ks.size * betas.size
    logger.info("Grid size: %d windows × %d ks × %d betas = %d runs",
                windows.size, ks.size, betas.size, total_runs)

    # 3) precompute mean/std for each window
    stats = {}
    logger.info("Precomputing mean/std for each window…")
    for w in tqdm.tqdm(windows, desc="Precompute windows"):
        stats[w] = calculate_mean_std_gpu(img_gpu, w)

    # 4) sweep to minimize MSE
    best_val    = float('inf')
    best_params = {}
    best_image  = None
    pbar = tqdm.tqdm(total=total_runs, desc="Sweeping", unit="run")

    for w in windows:
        mean_gpu, std_gpu = stats[w]
        for k in ks:
            for b in betas:
                seg_gpu    = thresholding_niblack_gpu(img_gpu, k, mean_gpu, std_gpu, b)
                seg_slices = seg_gpu[indices]  # shape (S, H, W)

                # compute MSE per slice, then average
                diff          = seg_slices.astype(cp.float32) - gt_stack_gpu.astype(cp.float32)
                mse_per_slice = cp.mean(diff*diff, axis=(1,2))
                mean_mse      = float(mse_per_slice.mean().item())

                if mean_mse < best_val:
                    best_val    = mean_mse
                    best_params = {
                        "w":     int(w),
                        "k":     float(k),
                        "beta":  float(b),
                        "mse":   mean_mse
                    }
                    best_image  = cp.asnumpy(seg_gpu)
                    logger.info(" New best MSE=%.4f at w=%d, k=%.3f, beta=%.4f",
                                mean_mse, w, k, b)

                pbar.update()
    pbar.close()

    logger.info("Sweep finished; best params = %s", best_params)
    return best_image, best_params
