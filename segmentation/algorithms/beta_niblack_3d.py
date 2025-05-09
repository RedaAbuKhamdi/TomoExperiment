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
    img_f = img_gpu.astype(cp.float32)
    mean   = ndi_gpu.uniform_filter(img_f,      size=window, mode="reflect")
    meansq = ndi_gpu.uniform_filter(img_f*img_f, size=window, mode="reflect")
    std    = cp.sqrt(cp.abs(meansq - mean*mean))
    return mean, std

def segment(imageData: ImageData, params: dict = None):
    print("Start Niblack")

    best_val    = 0.0
    best_k      = params["k"] if params else 0.0
    best_window = params["w"] if params else 10
    best_image  = None

    # 1) upload raw image once
    img_gpu = cp.asarray(imageData.image, dtype=cp.float32)
    
    beta = cp.std(img_gpu).item()

    if params:
        # single (w,k) provided → just compute and return
        mean, std = calculate_mean_std_gpu(img_gpu, best_window)
        seg_gpu    = thresholding_niblack_gpu(img_gpu, best_k, mean, std, beta)
        best_image = cp.asnumpy(seg_gpu)

    else:
        shape   = imageData.image.shape
        ks      = np.linspace(-2, 1, 100)
        max_w   = max(1, min(2 * min(shape)//3, 250))
        windows = np.arange(best_window, max_w,
                            max(1, (max_w - best_window)//15),
                            dtype=np.int16)

        # 2) pull GT once and move to GPU
        gt_slices = list(imageData.get_ground_truth_slices())
        gt_gpu    = [(idx, cp.asarray(gt, dtype=cp.bool_))
                     for idx, gt in gt_slices]

        # 3) sweep (window, k)
        for window in tqdm.tqdm(windows, desc="Window"):
            mean, std = calculate_mean_std_gpu(img_gpu, window)
            for q in tqdm.tqdm(ks, desc="k"):
                seg_gpu = thresholding_niblack_gpu(img_gpu, q, mean, std, beta)

                # 4) GPU‐side Jaccard average
                total = 0.0
                for idx, gt_slice_gpu in gt_gpu:
                    sg = seg_gpu[idx]               # boolean 2D cupy array
                    I  = cp.logical_and(sg, gt_slice_gpu).sum().item()
                    U  = cp.logical_or (sg, gt_slice_gpu).sum().item()
                    if (U > 0):
                        total += (I / U)
                mean_metric = total / len(gt_gpu)

                # 5) track best
                if mean_metric > best_val:
                    best_val    = mean_metric
                    best_k      = float(q)
                    best_window = int(window)
                    best_image  = cp.asnumpy(seg_gpu)

    out_params = {"k": best_k, "w": best_window, "beta": beta}
    if best_val > 0:
        out_params["jaccard"] = best_val
    return best_image, out_params


def parameters_experiment(imageData: ImageData):
    print("Start Niblack Experiment (GPU)")

    # 0) Prepare output folder
    folder = (config.EXPERIMENTS_PARAMETERS_PATH /
              "beta_niblack_3d" /
              imageData.settings["name"])
    os.makedirs(folder, exist_ok=True)

    # 1) Upload image once
    img_gpu = cp.asarray(imageData.image, dtype=cp.float32)

    # 2) Pull & stack only your GT slices, keep their indices
    gt_list = list(imageData.get_ground_truth_slices())  # [(idx, gt), ...]
    indices, gt_arrays = zip(*gt_list)
    indices = np.array(indices, dtype=np.int64)
    # stack the GT masks on GPU: shape (S, H, W)
    gt_stack_gpu = cp.stack([cp.asarray(gt, dtype=cp.bool_) 
                              for gt in gt_arrays], axis=0)
    S = gt_stack_gpu.shape[0]

    # 3) Parameter grids
    shape   = imageData.image.shape
    ks      = np.linspace(-2,  2, 60, dtype=np.float32)
    max_w   = max(1, min(2 * min(shape)//3, 250))
    windows = np.arange(10, max_w,
                        max(1, (max_w - 10)//15),
                        dtype=np.int16)

    rows = []  # will collect dicts for Pandas

    # 4) Sweep over (window, k, beta)
    for window in tqdm.tqdm(windows, desc="Window", total=windows.size):
        mean_gpu, std_gpu = calculate_mean_std_gpu(img_gpu, int(window))

        for j, q in enumerate(tqdm.tqdm(ks, desc="k", leave=False)):
            # build beta array
            lin_sz = 40
            betas = np.linspace(-1.5, 1, lin_sz, dtype=np.float32)
            metric_values = np.empty_like(betas, dtype=np.float32)

            for k, beta in enumerate(betas):
                # 4a) segmentation on GPU
                seg_gpu = thresholding_niblack_gpu(
                    img_gpu, q, mean_gpu, std_gpu, beta
                )  # full (D, H, W) boolean mask

                # 4b) pull only the GT‐indexed slices
                seg_slices_gpu = seg_gpu[indices, :, :]  # shape (S,H,W)

                # 4c) vectorized Jaccard over those S slices
                I =   cp.logical_and(seg_slices_gpu, gt_stack_gpu).sum(axis=(1,2))
                U =   cp.logical_or (seg_slices_gpu, gt_stack_gpu).sum(axis=(1,2))
                jaccs = cp.where(U>0, I/U, 0.0)  # (S,)
                mean_metric = float(jaccs.mean().item())
                metric_values[k] = mean_metric

                # 4d) record
                rows.append({
                    "window": int(window),
                    "k":       float(q),
                    "beta":    float(beta),
                    "jaccard": mean_metric
                })

    # 6) dump results
    df = pd.DataFrame(rows)
    df.to_csv(f"{folder}/results.csv", index=False)
    return df
            

    