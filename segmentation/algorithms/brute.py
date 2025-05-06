import tqdm
import numpy as np
import cupy as cp

from imagedata import ImageData

def segment(imageData: ImageData, params: dict = None):
    """
    Brute‐force threshold segmentation, fully on GPU until the end.
    """
    print("Start Brute (GPU)")
    best_threshold = params["threshold"] if params else 0.0
    best_val       = 0.0
    best_image     = None

    # 1) upload raw image once
    img_gpu = cp.asarray(imageData.image, dtype=cp.float32)

    # 2) prepare ground‐truth on GPU
    gt_slices = list(imageData.get_ground_truth_slices())
    gt_gpu    = [(idx, cp.asarray(gt, dtype=cp.bool_))
                 for idx, gt in gt_slices]

    if params:
        # single threshold provided → compute once
        seg_gpu    = img_gpu >= best_threshold
        best_image = cp.asnumpy(seg_gpu)
    else:
        # full sweep of thresholds
        tmin, tmax = img_gpu.min().item(), img_gpu.max().item()
        thresholds = np.linspace(tmin, tmax, 500, dtype=np.float32)

        for t in tqdm.tqdm(thresholds, desc="Threshold"):
            # 3) threshold on GPU
            seg_gpu = img_gpu >= t

            # 4) GPU‐side average Jaccard
            total = 0.0
            for idx, gt_gpu_slice in gt_gpu:
                sg = seg_gpu[idx]  # 2D cupy bool array
                I  = cp.logical_and(sg, gt_gpu_slice).sum()
                U  = cp.logical_or (sg, gt_gpu_slice).sum()
                total += (I / U).item()
            mean_metric = total / len(gt_gpu)

            # 5) track best
            if mean_metric > best_val:
                best_val       = mean_metric
                best_threshold = float(t)
                best_image     = cp.asnumpy(seg_gpu)

    out_params = {"threshold": best_threshold}
    if best_val > 0:
        out_params["jaccard"] = best_val
    return best_image, out_params


def parameters_experiment(imageData: ImageData):
    """
    Sweep thresholds and return curve data, fully on GPU for speed.
    """
    print("Start Brute Experiment (GPU)")
    img_gpu = cp.asarray(imageData.image, dtype=cp.float32)

    # prepare GT once
    gt_slices = list(imageData.get_ground_truth_slices())
    gt_gpu    = [(idx, cp.asarray(gt, dtype=cp.bool_))
                 for idx, gt in gt_slices]

    # thresholds to scan
    tmin, tmax = img_gpu.min().item(), img_gpu.max().item()
    thresholds = np.linspace(tmin, tmax, 500, dtype=np.float32)

    # preallocate results
    metrics = np.empty_like(thresholds, dtype=np.float32)

    for i, t in enumerate(tqdm.tqdm(thresholds, desc="Threshold")):
        seg_gpu = img_gpu >= t

        total = 0.0
        for idx, gt_gpu_slice in gt_gpu:
            sg = seg_gpu[idx]
            I  = cp.logical_and(sg, gt_gpu_slice).sum()
            U  = cp.logical_or (sg, gt_gpu_slice).sum()
            total += (I / U).item()

        metrics[i] = total / len(gt_gpu)

    return {
        "title": ["Threshold"],
        "x": [thresholds],
        "y": [metrics]
    }
