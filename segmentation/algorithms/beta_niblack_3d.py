import tqdm
import numpy as np
from imagedata import ImageData

import cupy as cp
import cupyx.scipy.ndimage as ndi_gpu


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
    return img_f - beta >= (mean + std * q)


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
        mean, std = calculate_mean_std_gpu(img_gpu, best_window, beta)
        seg_gpu    = thresholding_niblack_gpu(img_gpu, best_k, mean, std)
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
    print("Start Niblack")

    # 1) upload raw image once
    img_gpu = cp.asarray(imageData.image, dtype=cp.float32)
    beta = cp.std(img_gpu)

    shape   = imageData.image.shape
    ks      = np.linspace(-5, 5, 100)
    max_w   = max(1, min(2 * min(shape)//3, 250))
    windows = np.arange(5, max_w,
                        max(1, (max_w - 5)//15),
                        dtype=np.int16)

    # 2) pull GT once and move to GPU
    gt_slices = list(imageData.get_ground_truth_slices())
    gt_gpu    = [(idx, cp.asarray(gt, dtype=cp.bool_))
                    for idx, gt in gt_slices]
    result = {
        "title": [],
        "x": [],
        "y": []
    }
    # 3) sweep (window, k)
    for i, window in tqdm.tqdm(enumerate(windows), desc="Window"):
        mean, std = calculate_mean_std_gpu(img_gpu, window)
        result["title"].append(f"w={window}")
        result["x"].append(ks)
        result["y"].append(np.zeros_like(ks))
        for j, q in tqdm.tqdm(enumerate(ks), desc="k"):
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
            result["y"][i][j] = mean_metric

    return result
            

    