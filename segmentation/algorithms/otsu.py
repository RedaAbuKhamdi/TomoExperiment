import tqdm
import numpy as np
from skimage.filters import threshold_otsu

from numba import jit
from metrics import jaccard
from imagedata import ImageData

def thresholding(image : np.ndarray, threshold : float):
    return image >= threshold

def segment(imageData : ImageData, params : dict = None):
    print("Otsu")
    threshold = threshold_otsu(imageData.image)
    best_image = thresholding(imageData.image, threshold)

    params = {
        "threshold" : float(threshold)
    }
    return best_image, params