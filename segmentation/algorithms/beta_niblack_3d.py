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

def thresholding_niblack_gpu(img_gpu: cp.ndarray,
                             q: float,
                             mean: cp.ndarray,
                             std: cp.ndarray,
                             beta: float) -> cp.ndarray:
    """
    Perform Niblack threshold:
      seg = img >= (mean_local + q * std_local)
    entirely on the GPU.
    """
    img_f = img_gpu.astype(cp.float32)
    return img_f >= (mean + std * q + beta)


def calculate_mean_std_gpu(img_gpu: cp.ndarray, window: int) -> tuple:
    # cast once
    img_f = img_gpu.astype(cp.float32)

    # compute local mean of I and I^2, but only in-plane:
    # size = (1, window, window) → radius 0 in z, radius window//2 in y,x
    mean   = ndi_gpu.uniform_filter(img_f,
                                    size=(1, window, window),
                                    mode="reflect")
    meansq = ndi_gpu.uniform_filter(img_f * img_f,
                                    size=(1, window, window),
                                    mode="reflect")

    # std = sqrt(E[x^2] - E[x]^2), clamp negative round‐off to zero
    std = cp.sqrt(cp.clip(meansq - mean * mean, 0, None))

    return mean, std
import logging
import tqdm
import numpy as np
import cupy as cp

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
    If params given: do a single Niblack with (w, k, beta).
    Otherwise: sweep (w in windows, k in ks), find best by mean‐Jaccard,
    logging progress and bests as we go.
    """
    name = imageData.settings["name"]
    img   = imageData.image
    img_gpu = cp.asarray(img, dtype=cp.float32)
    beta    = abs(imageData.get_noise_std())
    logger.info("Segment called for '%s'; beta=%.4f", name, beta)

    # quick single‐call case
    if params:
        w = params["w"]
        k = params["k"]
        mean, std = calculate_mean_std_gpu(img_gpu, w)
        seg_gpu   = thresholding_niblack_gpu(img_gpu, k, mean, std, beta)
        best_image = cp.asnumpy(seg_gpu)
        return best_image, params

    # --- otherwise, do full sweep ---
    logger.info("No params provided; running full sweep")

    # get GT slices once
    gt_list = list(imageData.get_ground_truth_slices())
    indices, gt_arrays = zip(*gt_list)
    indices = np.array(indices, dtype=np.int64)
    gt_stack_gpu = cp.stack([cp.asarray(gt, dtype=cp.bool_) for gt in gt_arrays], axis=0)
    S = gt_stack_gpu.shape[0]
    logger.info("Evaluating on %d ground-truth slices", S)

    # define grid
    windows = np.arange(20, 250, 20, dtype=int)      # e.g. 5,15,...,55
    ks      = np.linspace(-0.5, 0.5, 50, dtype=float)
    logger.info("Grid = %d windows × %d ks = %d runs",
                windows.size, ks.size, windows.size*ks.size)

    # precompute mean/std per window
    logger.info("Precomputing slice‐wise mean/std for each window…")
    stats = {}
    for w in tqdm.tqdm(windows, desc="Precompute windows"):
        stats[w] = calculate_mean_std_gpu(img_gpu, w)

    # sweep
    best_val    = -1.0
    best_params = {}
    best_image  = None

    pbar = tqdm.tqdm(total=windows.size*ks.size, desc="Sweeping", unit="run")
    for w in windows:
        mean_gpu, std_gpu = stats[w]
        for k in ks:
            seg_gpu     = thresholding_niblack_gpu(img_gpu, k, mean_gpu, std_gpu, beta)
            seg_slices  = seg_gpu[indices]                   # (S,H,W)
            I =   cp.logical_and(seg_slices, gt_stack_gpu).sum(axis=(1,2))
            U =   cp.logical_or (seg_slices, gt_stack_gpu).sum(axis=(1,2))
            jaccs = cp.where(U>0, I/U, 0.0)
            mean_j = float(jaccs.mean().item())

            if mean_j > best_val:
                best_val    = mean_j
                best_params = {"w":int(w), "k":float(k), "beta":beta, "jaccard":mean_j}
                best_image  = cp.asnumpy(seg_gpu)
                logger.info(" New best IoU=%.4f at w=%d, k=%.3f", mean_j, w, k)

            pbar.update()
    pbar.close()

    logger.info("Sweep finished; best params = %s", best_params)
    return best_image, best_params
