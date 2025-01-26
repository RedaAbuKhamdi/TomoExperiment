import tqdm

import numpy as np

from numba import jit
from metrics import jaccard
from imagedata import ImageData

def thresholding(image : np.ndarray, q  : int, mean : np.ndarray, std : np.ndarray, beta : int):
    return image >= (mean + std * q + beta)

def segment(imageData : ImageData):
    print("Start Niblack")
    shape = imageData.image.shape
    image = imageData.image
    ks, w, betas = np.linspace(-2, 2, 24), max([1, min(min(shape)//2, 80)]), np.linspace(np.min(image), np.mean(image), 30)
    best_val = 0
    best_k = 0
    best_image = None
    best_beta = 0
    best_window = 5
    windows = np.arange(5, w, (w-5) // 8, dtype=np.int16)
    for window in tqdm.tqdm(windows):
        means, stds = imageData.get_e_sd(window)
        stds = np.sqrt(np.abs(stds))
        for a in tqdm.tqdm(range(betas.size)):
                for i in range(ks.size):
                    segmentation = thresholding(image, ks[i], means, stds, betas[a])
                    mean_metric = 0
                    ground_truth_slice_amount = 0
                    for index, ground_truth_slice in imageData.get_ground_truth_slices():
                        segmented_slice = segmentation[index]   
                        mean_metric += jaccard(segmented_slice, ground_truth_slice)
                        ground_truth_slice_amount += 1
                    mean_metric /= ground_truth_slice_amount
                    if mean_metric > best_val:
                        best_val = mean_metric
                        best_k = ks[i]
                        best_image = np.copy(segmentation)
                        best_beta = betas[a]
                        best_window = window

    params = {
        "k" : float(best_k),
        "w" : float(best_window),
        'beta' : float(best_beta),
        "jaccard" : float(best_val)
    }
    return best_image, params
