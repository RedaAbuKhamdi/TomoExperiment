import tqdm
import numpy as np

from numba import jit
from metrics import jaccard
from imagedata import ImageData
@jit()
def thresholding(image : np.ndarray, threshold : float):
    return image >= threshold

def segment(imageData : ImageData, params : dict = None):
    print("Start Brute")
    best_threshold = params["threshold"] if params is not None else 0
    best_val = 0
    best_image = None
    if params is not None:
        best_image = thresholding(imageData.image, best_threshold)
    else:
        thresholds = np.linspace(np.min(imageData.image), np.max(imageData.image), 500)
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
        "threshold" : float(best_threshold)
    }
    if best_val != 0:
        params["jaccard"] = float(best_val)
    
    return best_image, params
