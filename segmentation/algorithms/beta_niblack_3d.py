import tqdm
from matplotlib import pyplot as plt
import numpy as np
from numba import jit, prange
from metrics import jaccard
from imagedata import ImageData


@jit()
def thresholding(segmented, image,q, mean, std, beta):
    for i in range(segmented.shape[0]):
        for j in range(segmented.shape[1]):
            for k in range(segmented.shape[2]):
                segmented[i, j, k] = 1 if image[i, j, k] >= (mean[i, j, k] + q*std[i, j, k] + beta) else 0 

    return segmented


def segment(imageData : ImageData):
    image = imageData.image
    ks, w, betas = np.linspace(-1, 1, 24), np.max(np.array([1, np.min(image.shape)])), np.linspace(np.min(image), np.max(image), 40)
    best_val = 0
    best_k = 0
    best_image = None
    best_beta = 0
    best_window = 5
    windows = np.arange(5, w, (w-5)//5)
    for window in windows:
        std_field = imageData.get_window_e_sd_field(window)
        means, stds = std_field[0, :, :, :], std_field[1, :, :, :]
        for a in tqdm.tqdm(range(betas.size)):
                for i in range(ks.size):
                    segmentation = np.zeros((image.shape[0], image.shape[1], image.shape[2])).astype(bool)
                    segmentation = thresholding(segmentation, image, ks[i], means, stds, betas[a])
                    mean_metric = 0
                    ground_truth_slice_amount = 0
                    for index, ground_truth_slice in imageData.get_ground_truth_slices():
                        segmented_slice = segmentation[:,:,index]
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
