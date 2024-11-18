import json
import os 
import re

import numpy as np

from numba import jit, prange
from PIL import Image
from skimage import io

@jit()
def get_delta(image : np.ndarray, i, j, k, w):
    x_left = 0 if i - w < 0 else image[i - w, j, k]
    x_right = 0 if i + w >= image.shape[0] else image[i + w, j, k]
    y_left = 0 if j - w < 0 else image[i, j - w, k]
    y_right = 0 if j + w >= image.shape[1] else image[i, j + w, k]
    z_left = 0 if k - w < 0 else image[i, j, k - w]
    z_right = 0 if k + w >= image.shape[2] else image[i, j,  + w]

    return -x_left + x_right - y_left + y_right - z_left + z_right,  -(x_left**2) + x_right**2 - y_left**2 + y_right**2 - z_left**2 + z_right**2

@jit()
def calculate_fixed_window_e_sd_field(image : np.ndarray, 
                                    w : int):
    std_field = np.zeros((2, image.shape[0], image.shape[1], image.shape[2]))
    
    shape = image.shape
    n = shape[0] * shape[1] * shape[2]
    w_sum = np.sum(image[0:w, 0:w, 0:w])
    w_square_sum = np.sum(image[0:w, 0:w, 0:w]**2)
    std_field[0, 0, 0, 0] = w_sum / n
    std_field[1, 0, 0, 0] = (w_square_sum / n - std_field[0, 0, 0, 0]**2)**0.5
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for k in range(image.shape[2]):
                if (i + j + k == 0):
                    continue
                delta, squared_delta = get_delta(image, i, j, k, w)
                w_sum += delta
                w_square_sum += squared_delta
                std_field[0, i, j, k] = w_sum / n
                std_field[1, i, j, k] = (w_square_sum / n - std_field[0,  i, j, k]**2)**0.5
    return std_field

class ImageData:
    def __init__(self, prefix, path):
        self.path = path
        image_path = prefix + path
        self.image = np.load("{0}/image.npy".format(image_path))
        self.image = self.image.transpose(1, 2, 0)
        self.parse_settings(("{0}/settings.json".format(image_path)))
        self.load_ground_truth()

    def parse_settings(self, path):
        with open(path, "r") as f:
            self.settings = json.loads(f.read())

    def load_ground_truth(self):
        try:
            ground_truth_path = self.settings["groud_truth_path"]
            ground_truth_slices = {}
            for file in os.listdir(ground_truth_path):
                index = int(re.findall(r"\d+", file)[0])
                ground_truth_slices[index] = np.asarray(Image.open("{}/{}".format(ground_truth_path, file)))
            self.ground_truth_slices = ground_truth_slices
        except AttributeError:
            raise "No ground_truth_path in settings.json of the reconstruction"
    
    def get_ground_truth_slices(self):
        try:
            self.ground_truth_slices
        except AttributeError:
            self.load_ground_truth()
        for key in self.ground_truth_slices:
                yield key, self.ground_truth_slices[key]

    def get_window_e_sd_field(self, window : int):
        print("Calculation of e sd field for window {}".format(window))
        return calculate_fixed_window_e_sd_field(self.image, window)
    
    def get_folder(self, algorithm):
        return "./results/binarizations/{0}/{1}".format(algorithm, self.path)

    def save_result(self, segmentation : np.ndarray, params : dict, algorithm : str):
        folder = self.get_folder(algorithm)
        os.makedirs(folder, exist_ok=True)
        io.imsave("{0}/segmentation.tiff".format(folder),  segmentation)
        with open("{}/parameters.json".format(folder), "w") as f:
            params["ground_truth_path"] = self.settings["groud_truth_path"]
            f.write(json.dumps(params))
    