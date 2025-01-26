import json
import os 
import re
import sys

import numpy as np

from PIL import Image
from skimage import io
from numba import jit
from os import path

sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
import config

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

class ImageData:
    def __init__(self, prefix, path):
        self.path = path
        self.prefix = prefix
        image_path = prefix + path
        self.image = io.imread("{0}/reconstruction_experiment.tiff".format(image_path)).astype(np.float32)
        self.parse_settings(("{0}/settings.json".format(image_path)))
        print(path)
        self.prefix_sum = np.cumsum(np.cumsum(self.image, axis = 2), axis = 1)
        self.square_prefix_sum = np.cumsum(np.cumsum(self.image ** 2, axis = 2), axis = 1)
        self.load_ground_truth()
            

    def parse_settings(self, path):
        with open(path, "r") as f:
            self.settings = json.loads(f.read())

    def load_ground_truth(self):
        try:
            ground_truth_path = config.GROUND_TRUTH_PATH / self.settings["name"]
            ground_truth_slices = {}
            for file in os.listdir(ground_truth_path):
                index = int(re.findall(r"\d+", file)[0])
                ground_truth_slices[index] = np.asarray(Image.open("{}\\{}".format(ground_truth_path, file)))
            self.ground_truth_slices = ground_truth_slices
        except:
            raise Exception("Error loading ground truth")
    
    def get_ground_truth_slices(self):
        try:
            self.ground_truth_slices
        except AttributeError:
            self.load_ground_truth()
        for key in self.ground_truth_slices:
                yield key, self.ground_truth_slices[key]
    
    def get_folder(self, algorithm):
        return "./results/binarizations/{0}/{1}".format(algorithm, self.path)

    def save_result(self, segmentation : np.ndarray, params : dict, algorithm : str):
        folder = self.get_folder(algorithm)
        os.makedirs(folder, exist_ok=True)
        io.imsave("{0}/segmentation.tiff".format(folder),  segmentation)
        with open("{}/parameters.json".format(folder), "w") as f:
            params["ground_truth_path"] = self.settings["groud_truth_path"]
            f.write(json.dumps(params))
    def get_e_sd(self, window : int):
        path = self.prefix + self.path
        print("window {0}".format(window))
        if os.path.isfile("{0}/means_{1}.npy".format(path, window)) and os.path.isfile("{0}/stds_{1}.npy".format(path, window)):
            return np.load("{0}/means_{1}.npy".format(path, window)), np.load("{0}/stds_{1}.npy".format(path, window)) 
        means, stds = calculate_e_sd_field(self.prefix_sum, self.square_prefix_sum, window)
        np.save("{0}/means_{1}.npy".format(path, window), means, False)
        np.save("{0}/stds_{1}.npy".format(path, window), stds, False)
        return means, stds
    