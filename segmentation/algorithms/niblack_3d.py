import tqdm
import numpy as np
from matplotlib import pyplot as plt

from numba import jit
from metrics import jaccard
from imagedata import ImageData

def thresholding(image : np.ndarray, q  : int, mean : np.ndarray, std : np.ndarray):
    return image >= mean + std * q
# visualize change of parameters and choose best

def segment(imageData : ImageData, params : dict = None):
    print("Start Niblack")
    
    best_val = 0
    best_k = params["k"] if params is not None else 0
    best_image = None
    best_window = params["w"] if params is not None else 10
    if params is not None:
        means, stds = imageData.get_e_sd(best_window)
        stds = np.sqrt(np.abs(stds))
        best_image = thresholding(imageData.image, best_k, means, stds)
    else:
        shape = imageData.image.shape
        ks, w = np.linspace(-3, 3, 60), max([1, min(min(shape)//2, 250)])
        windows = np.arange(best_window, w, (w-best_window) // 10, dtype=np.int16)
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
        "w" : int(best_window)
    }
    if best_val != 0:
        params["jaccard"] =  float(best_val)
    return best_image, params
