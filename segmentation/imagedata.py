import json
import sys
import os

import numpy as np

from skimage import io
from numba import jit
from os import path, makedirs

sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from common.imagedata import ImageDataBase

@jit(nopython = True)
def get_sum(prefix_sum, i, j, k, w):
    
    upper_left = np.array([i - w - 1, j - w - 1])
    upper_right = np.array([i - w - 1, min(j + w, prefix_sum.shape[2] - 1)])
    lower_left = np.array([min(i + w, prefix_sum.shape[1] - 1), j - w - 1])
    lower_right = np.array([min(i + w, prefix_sum.shape[1] - 1), min(j + w, prefix_sum.shape[2] - 1)])
    slices = np.array([(k - w) * (k - w >= 0), min (k + w + 1, prefix_sum.shape[0])])
    result = 0
    for z in range(slices[0], slices[1]):
        lower_right_value = prefix_sum[z, lower_right[0], lower_right[1]]
        lower_left_value = prefix_sum[z, lower_left[0], lower_left[1]] * (lower_left[1] >= 0)
        upper_right_value = prefix_sum[z, upper_right[0], upper_right[1]] * (upper_right[0] >= 0)
        upper_left_value = prefix_sum[z, upper_left[0], upper_left[1]] * (upper_left[0] >= 0 and upper_left[1] >= 0)
        result +=  lower_right_value - lower_left_value - upper_right_value + upper_left_value
    return result
@jit(nopython = True)
def calculate_e_sd_field(prefix_sums : np.ndarray, 
                         prefix_sums_squared : np.ndarray,  window : int):
    means = np.zeros(prefix_sums.shape, dtype=np.float32)
    stds = np.zeros(prefix_sums.shape, dtype=np.float32)
    for k in range(prefix_sums.shape[0]):
        for i in range(prefix_sums.shape[1]):  
            for j in range(prefix_sums.shape[2]):
                means[k, i, j] = get_sum(prefix_sums, i, j, k, window) / window ** 3
                stds[k, i, j] = get_sum(prefix_sums_squared, i, j, k, window) / window ** 3 - means[k, i, j] ** 2
    return means, stds


@jit(nopython=True, fastmath=True)
def compute_prefix_sums(image: np.ndarray):
    """
    Compute the 3D integral image (prefix sum) and the 3D square integral image
    for a given 3D image, with output arrays of shape (D, H, W).

    For each voxel (k, i, j), the prefix sum is defined as:
      prefix[k, i, j] = sum_{u=0}^{k} sum_{v=0}^{i} sum_{w=0}^{j} image[u, v, w]

    Similar definition applies for prefix_sq using image[u,v,w]^2.
    """
    D, H, W = image.shape
    prefix = np.empty((D, H, W), dtype=image.dtype)
    prefix_sq = np.empty((D, H, W), dtype=image.dtype)
    
    for k in range(D):
        for i in range(H):
            for j in range(W):
                val = image[k, i, j]
                sum_val = val
                sum_sq_val = val * val

                if k > 0:
                    sum_val += prefix[k-1, i, j]
                    sum_sq_val += prefix_sq[k-1, i, j]
                if i > 0:
                    sum_val += prefix[k, i-1, j]
                    sum_sq_val += prefix_sq[k, i-1, j]
                if j > 0:
                    sum_val += prefix[k, i, j-1]
                    sum_sq_val += prefix_sq[k, i, j-1]
                if k > 0 and i > 0:
                    sum_val -= prefix[k-1, i-1, j]
                    sum_sq_val -= prefix_sq[k-1, i-1, j]
                if k > 0 and j > 0:
                    sum_val -= prefix[k-1, i, j-1]
                    sum_sq_val -= prefix_sq[k-1, i, j-1]
                if i > 0 and j > 0:
                    sum_val -= prefix[k, i-1, j-1]
                    sum_sq_val -= prefix_sq[k, i-1, j-1]
                if k > 0 and i > 0 and j > 0:
                    sum_val += prefix[k-1, i-1, j-1]
                    sum_sq_val += prefix_sq[k-1, i-1, j-1]
                
                prefix[k, i, j] = sum_val
                prefix_sq[k, i, j] = sum_sq_val
    return prefix, prefix_sq
class ImageData(ImageDataBase):
    def __init__(self, path):
        super().__init__( path)
    
    def get_folder(self, algorithm):
        return self.path.replace("reconstructions", "binarizations") + "/" + algorithm

    def save_result(self, segmentation : np.ndarray, params : dict, algorithm : str):
        folder = self.get_folder(algorithm)
        makedirs(folder, exist_ok=True)
        params["angles"] = len(self.settings["angles"]["values"])
        io.imsave("{0}/segmentation.tiff".format(folder),  segmentation)
        with open("{}/parameters.json".format(folder), "w") as f:
            f.write(json.dumps(params))

    def check_if_done(self, algorithm : str):
        folder = self.get_folder(algorithm)
        if (path.isfile("{}/parameters.json".format(folder))):
            with open("{}/parameters.json".format(folder), "r") as f:
                return json.loads(f.read())
        else:
            return None
    def get_e_sd(self, window : int):
        path = self.path
        print("window {0}".format(window))
        if os.path.isfile("{0}/means_{1}.npy".format(path, window)) and path.isfile("{0}/stds_{1}.npy".format(path, window)):
            return np.load("{0}/means_{1}.npy".format(path, window)), np.load("{0}/stds_{1}.npy".format(path, window)) 
        prefix_sum, square_prefix_sum = compute_prefix_sums(self.image)
        means, stds = calculate_e_sd_field(prefix_sum, square_prefix_sum, window)
        np.save("{0}/means_{1}.npy".format(path, window), means, False)
        np.save("{0}/stds_{1}.npy".format(path, window), stds, False)
        return means, stds
    