import tqdm
import numpy as np
from matplotlib import pyplot as plt

from numba import jit
from metrics import jaccard
from imagedata import ImageData

def thresholding(image : np.ndarray, q  : int, mean : np.ndarray, std : np.ndarray):
    return image >= mean + std * q

def segment(imageData : ImageData):
    print("Start Niblack")
    shape = imageData.image.shape
    ks, w = np.linspace(-3, 3, 60), max([1, min(min(shape)//2, 250)])
    best_val = 0
    best_k = 0
    best_image = None
    best_window = 10
    windows = np.arange(best_window, w, (w-best_window) // 8, dtype=np.int16)
    for window in tqdm.tqdm(windows):
        means, stds = imageData.get_e_sd(window)
        stds = np.sqrt(np.abs(stds))
        for i in tqdm.tqdm(range(ks.size)):
            segmentation = thresholding(imageData.image, ks[i], means, stds)
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
                best_window = window

    params = {
        "k" : float(best_k),
        "w" : float(best_window),
        "jaccard" : float(best_val)
    }
    return best_image, params
