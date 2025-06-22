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



def tversky_gpu(seg: cp.ndarray,
                gt:  cp.ndarray,
                alpha: float = 0.7,
                beta:  float = 0.3) -> float:
    """
    seg, gt: boolean cupy arrays of the same shape.
    alpha: weight on false positives
    beta:  weight on false negatives
    returns float Tversky index in [0..1].
    """
    tp = cp.logical_and(seg, gt).sum()
    fp = cp.logical_and(seg, cp.logical_not(gt)).sum()
    fn = cp.logical_and(cp.logical_not(seg), gt).sum()
    denom = tp + alpha*fp + beta*fn
    return float((tp / denom).item()) if denom > 0 else 1.0

def calculate_mean_std_gpu(img: cp.ndarray, window: int) -> tuple[cp.ndarray, cp.ndarray]:
    """Compute local mean & std using 3D windows"""
    D, H, W = img.shape
    size = (window, window, window)  # Now truly 3D
    area = window * window * window  # Volume instead of area

    # Rest of the implementation remains similar but operates in 3D
    mean_zero = ndi_gpu.uniform_filter(img, size=size, mode='constant', cval=0.0)
    sum_vals = mean_zero * area
    
    mask = cp.ones_like(img, dtype=cp.float32)
    mean_m = ndi_gpu.uniform_filter(mask, size=size, mode='constant', cval=0.0)
    count = mean_m * area
    
    mean_loc = sum_vals / cp.maximum(count, 1.0)
    
    sumsq_zero = ndi_gpu.uniform_filter(img*img, size=size, mode='constant', cval=0.0)
    sumsq = sumsq_zero * area
    
    var_loc = sumsq / cp.maximum(count, 1.0) - mean_loc*mean_loc
    std_loc = cp.sqrt(cp.clip(var_loc, 0.0, None))
    
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
    logger.info("No params provided; running full w,k,beta sweep for iou")

    # 1) load GT once
    gt_list       = list(imageData.get_ground_truth_slices())
    indices, gts  = zip(*gt_list)
    indices       = np.array(indices, dtype=np.int64)
    gt_stack_gpu  = cp.stack([cp.asarray(g, dtype=cp.bool_) for g in gts], axis=0)
    S             = gt_stack_gpu.shape[0]
    logger.info("Evaluating iou on %d ground-truth slices", S)

    # 2) define parameter grids
    windows = np.arange(20, min(70, min(img.shape)), 13, dtype=int)
    ks      = np.linspace(-0.1, 0.1, 40, dtype=float)  # 100 values
    betas   = np.linspace(cp.min(img_gpu).item() ,
                          cp.max(img_gpu).item() ,
                          50, dtype=float)
    total_runs = windows.size * ks.size * betas.size
    logger.info("Grid size: %d windows × %d ks × %d betas = %d runs",
                windows.size, ks.size, betas.size, total_runs)

    # 3) precompute mean/std for each window
    stats = {}
    logger.info("Precomputing mean/std for each window…")
    for w in tqdm.tqdm(windows, desc="Precompute windows"):
        stats[w] = calculate_mean_std_gpu(img_gpu, w)

    # 4) sweep to minimize iou
    best_val    = float('-inf')
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
                score = tversky_gpu(seg_slices, gt_stack_gpu, alpha=0.7, beta=0.3)

                if score > best_val:
                    best_val    = score
                    best_params = {
                        "w":     int(w),
                        "k":     float(k),
                        "beta":  float(b),
                        "iou":   score
                    }
                    best_image  = cp.asnumpy(seg_gpu)
                    logger.info(" New best iou=%.4f at w=%d, k=%.3f, beta=%.4f",
                                score, w, k, b)

                pbar.update()
    pbar.close()

    logger.info("Sweep finished; best params = %s", best_params)
    return best_image, best_params
