import tqdm
import numpy as np
from matplotlib import pyplot as plt

from numba import jit
from metrics import jaccard
from imagedata import ImageData

def thresholding(image : np.ndarray, threshold : float):
    return image >= threshold

def segment(imageData : ImageData):
    print("Start Brute")
    shape = imageData.image.shape
    thresholds = np.linspace(np.min(imageData.image), np.max(imageData.image), 120)
    best_val = 0
    best_threshold = 0
    best_image = None
    for i in tqdm.tqdm(range(thresholds.size)):
        segmentation = thresholding(imageData.image, thresholds[i])
        mean_metric = 0
        ground_truth_slice_amount = 0
        for index, ground_truth_slice in imageData.get_ground_truth_slices():
            segmented_slice = segmentation[index]   
            mean_metric += jaccard(segmented_slice, ground_truth_slice)
            ground_truth_slice_amount += 1
        mean_metric /= ground_truth_slice_amount
        
        if mean_metric > best_val:
            best_val = mean_metric
            best_threshold = thresholds[i]
            best_image = np.copy(segmentation)

    params = {
        "threshold" : float(best_threshold),
        "jaccard" : float(best_val)
    }
    return best_image, params
