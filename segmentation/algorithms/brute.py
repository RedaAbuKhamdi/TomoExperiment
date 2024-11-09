import tqdm
from matplotlib import pyplot as plt
import numpy as np
from numba import jit, njit, prange
from metrics import jaccard
from imagedata import ImageData



def segment(imageData : ImageData):
        image = imageData.image
        search_space = np.linspace(np.min(image) , np.max(image), 300)
        biggest_val = 0
        thresh = 0
        best_img = None

        for val in tqdm.tqdm(search_space):
            segmented = (image > val).astype(np.uint8)
            mean_metric = 0
            ground_truth_slice_amount = 0
            for index, ground_truth_slice in imageData.get_ground_truth_slices():
                segmented_slice = segmented[:,:,index]
                mean_metric += jaccard(segmented_slice, ground_truth_slice)
                ground_truth_slice_amount += 1
            mean_metric /= ground_truth_slice_amount
            if biggest_val < mean_metric:
                biggest_val = mean_metric
                thresh = val
                best_img = np.copy(segmented)

        params = {
            "Threshold": thresh,
            "Jaccard": biggest_val
        }
        return best_img, params