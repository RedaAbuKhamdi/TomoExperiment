import json
import os 
import re

import numpy as np

from numba import jit, prange
from PIL import Image
from skimage import io

@jit()
def e_sd(window_sum, window_square_sum, size):

    mean = window_sum/size
    standard_deviation = (window_square_sum/size - mean**2)**0.5

    return mean, standard_deviation


@jit()
def construct_cumulative_sum(result : np.ndarray, result_square : np.ndarray, image : np.ndarray):
    for k in range(image.shape[2]):
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if j == 0:
                    result[i,j,k, 0] = image[i,j,k]
                    result_square[i,j,k, 0] = image[i,j,k]**2
                else:
                    result[i, j,k,0] = result[i, j-1,k, 0] + image[i, j,k]
                    result_square[i, j,k,0] = result_square[i, j-1, k,0] + image[i, j,k]**2
                if i == 0:
                    result[i, j,k,1] = result[i, j,k, 0]
                    result_square[i, j,k,1] = result_square[i, j,k, 0]
                else:
                    result[i, j, k,1] = result[i-1, j, k,1] + result[i, j,k, 0]
                    result_square[i, j,k, 1] = result_square[i-1, j, k,1] + result_square[i, j,k, 0]
    
    return result, result_square

@jit()
def window_sum(shape, slice_cumulative_rows, slice_cumulative_cols, i, j, k, w):
    
    res = 0
    
    for z in range(
        np.max(np.array([k-w, 0])), 
        np.min(np.array([k+w+1, shape[2]]))
        ):
        cumulative_cols = slice_cumulative_cols[:,:,z]
        cumulative_rows = slice_cumulative_rows[:,:,z]
        right_down_corner = (np.min(np.array([i + w, shape[0]-1])), np.min(np.array([j + w, shape[1]-1])))
        left_down_corner = (np.min(np.array([i + w, shape[0] - 1])) ,  np.max(np.array([j - w, 0])))
        left_up_corner = (np.max(np.array([i - w, 0])), np.max(np.array([j - w, 0])))
        right_up_corner = (np.max(np.array([i - w, 0])),  np.min(np.array([j + w, shape[1] - 1])))
        res = cumulative_cols[right_down_corner] - (cumulative_cols[right_up_corner[0]-1, right_up_corner[1]] if right_up_corner[0]-1 >= 0 else 0)
        res += - ( cumulative_cols[left_down_corner[0], left_down_corner[1]-1] if 
                left_down_corner[1] - 1 >= 0 and j - w >= 0 else 0) + (
            cumulative_cols[left_up_corner[0]-1, left_up_corner[1]-1] if 
            left_up_corner[0]-1 >= 0 and left_up_corner[1]-1 >= 0 and j - w >= 0 else 0)
    return res

@jit()
def calculate_fixed_window_e_sd_field(image : np.ndarray, 
                                    w : int, cumulative_sum : np.ndarray, 
                                    cumulative_square_sum : np.ndarray):
    std_field = np.zeros((2, image.shape[0], image.shape[1], image.shape[2]))
    shape = image.shape
    for k in range(image.shape[2]):
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                right_down_corner = (np.min(np.array([i + w, shape[0]-1])), np.min(np.array([j + w, shape[1]-1])), np.min(np.array([k + w, shape[2]-1])))
                left_up_corner = (np.max(np.array([i - w, 0])), np.max(np.array([j - w, 0])), np.max(np.array([k - w, 0])))
                psize = (right_down_corner[0] - left_up_corner[0] + 1)*(right_down_corner[1] - left_up_corner[1] + 1)*(right_down_corner[2] - left_up_corner[2] + 1)
                m, sd = e_sd(window_sum(shape, cumulative_sum[:,:,:,0], cumulative_sum[:,:,:,1], i, j,k, w),
                            window_sum(shape, cumulative_square_sum[:,:,:,0], cumulative_square_sum[:,:,:,1], i, j,k, w), psize)
                std_field[0, i, j, k] = m
                std_field[1, i, j, k] = sd
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
        c_sum = np.zeros((
            self.image.shape[0],
            self.image.shape[1],
            self.image.shape[2],
            2))
        c_square_sum = np.zeros((
            self.image.shape[0],
            self.image.shape[1],
            self.image.shape[2],
            2))
        c_sum, c_square_sum = construct_cumulative_sum(c_sum, c_square_sum, self.image)
        return calculate_fixed_window_e_sd_field(self.image, window, c_sum, c_square_sum)
    
    def save_result(self, segmentation : np.ndarray, params : dict, algorithm : str):
        folder = "./results/binarizations/{0}/{1}".format(algorithm, self.path)
        os.makedirs(folder, exist_ok=True)
        io.imsave("{0}/segmentation.tiff".format(folder),  segmentation)
        with open("{}/parameters.json".format(folder), "w") as f:
            params["ground_truth_path"] = self.settings["groud_truth_path"]
            f.write(json.dumps(params))
    