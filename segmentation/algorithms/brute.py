import tqdm
from matplotlib import pyplot as plt
import numpy as np
from numba import jit, njit, prange
from metrics import jaccard
from imagedata import ImageData

@jit()
def binarization(image : np.ndarray, threshold : float,  indicies : np.ndarray, ground_truth_slices : np.ndarray) -> np.ndarray:
    segmentation = (image > threshold)
    mean_metric = 0
    ground_truth_slice_amount = 0
    for i in range(indicies.shape[0]):
        index = indicies[i]
        ground_truth_slice = ground_truth_slices[i]
        segmented_slice = segmentation[index]
        mean_metric += jaccard(segmented_slice, ground_truth_slice)
        ground_truth_slice_amount += 1
    mean_metric /= ground_truth_slice_amount
    return mean_metric, segmentation


def segment(imageData : ImageData):
        image = imageData.image.reshape(imageData.image.shape[2], imageData.image.shape[0], imageData.image.shape[1])
        image = image.astype(np.float32)
        search_space = np.linspace(np.min(image) , np.max(image), 300)
        biggest_val = 0
        thresh = 0
        best_img = None
        ground_truth_indicies_slices = list(imageData.get_ground_truth_slices())
        indicies = np.zeros(len(ground_truth_indicies_slices))
        ground_truth_slices = []
        i = 0
        for index, ground_truth_slice in imageData.get_ground_truth_slices():
             indicies[i] = index
             i += 1
             ground_truth_slices.append(ground_truth_slice)
        ground_truth_slices = np.array(ground_truth_slices)
        indicies = indicies.astype(int)
        for val in tqdm.tqdm(search_space):
            mean_metric, segmented = binarization(image, val, indicies, ground_truth_slices)
            if biggest_val < mean_metric:
                biggest_val = mean_metric
                thresh = val
                best_img = np.copy(segmented)

        params = {
            "Threshold": thresh,
            "Jaccard": biggest_val
        }
        return best_img, params