import json
import sys
import os

import numpy as np

from skimage import io
from numba import njit, jit, prange
from os import path, makedirs

sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from common.imagedata import ImageDataBase
import cupy as cp
import cupyx.scipy.ndimage as ndi

def sliding_window_mean_std(arr, size, mode='reflect'):
    """
    Compute the sliding window mean and standard deviation on GPU.
    
    Parameters:
      arr (cp.ndarray): Input array (can be 1D, 2D, etc.)
      size (int or sequence of ints): The size of the sliding window. 
          For multi-dimensional arrays, provide one size per axis.
      mode (str): How to handle boundaries. Options like 'reflect', 'constant', etc.
    
    Returns:
      mean (cp.ndarray): The sliding window mean.
      std (cp.ndarray): The sliding window standard deviation.
    """
    # Compute the mean using a uniform filter
    mean = ndi.uniform_filter(arr, size=size, mode=mode)
    
    # Compute the mean of squares using the same filter on the squared array
    mean_sq = ndi.uniform_filter(arr**2, size=size, mode=mode)
    
    # Calculate the variance and then standard deviation.
    # We use cp.maximum to avoid negative values due to numerical issues.
    variance = mean_sq - mean**2
    std = cp.sqrt(cp.abs(variance))
    
    return mean, std
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
    def parse_settings(self, path):
        settings_path = path + "/" + [f for f in os.listdir(path) if f.endswith(".json")][0]
        with open(settings_path, "r") as f:
            self.settings = json.loads(f.read())
            self.settings["name"] = self.path.split("/")[-2]

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
        if os.path.isfile("{0}/means_{1}.npy".format(path, window)) and os.path.isfile("{0}/stds_{1}.npy".format(path, window)):
            return np.load("{0}/means_{1}.npy".format(path, window)), np.load("{0}/stds_{1}.npy".format(path, window)) 
        means, stds = sliding_window_mean_std(cp.asarray(self.image), (window, window, window))
        means = cp.asnumpy(means)
        stds = cp.asnumpy(stds)
        np.save("{0}/means_{1}.npy".format(path, window), means, False)
        np.save("{0}/stds_{1}.npy".format(path, window), stds, False)
        return means, stds
    