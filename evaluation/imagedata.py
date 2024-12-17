import json
import os 
import re

import numpy as np

from PIL import Image
from skimage import io

class ImageData:
    def __init__(self, prefix, path):
        self.path = path
        image_path = prefix + path
        self.image = io.imread("{0}/segmentation.tiff".format(image_path))
        self.parse_settings(("{0}/parameters.json".format(image_path)))
        self.load_ground_truth()

    def parse_settings(self, path):
        with open(path, "r") as f:
            self.settings = json.loads(f.read())

    def load_ground_truth(self):
        try:
            ground_truth_path = self.settings["ground_truth_path"]
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
    
    def get_data(self):
        segmentation = []
        ground_truth = []
        for index, gt in self.get_ground_truth_slices():
            segmentation.append(self.image[index])
            ground_truth.append(gt)
        return np.array(segmentation), np.array(ground_truth)
    
    def get_folder(self, algorithm):
        return "./results/binarizations/{0}/{1}".format(algorithm, self.path)

    def save_result(self, segmentation : np.ndarray, params : dict, algorithm : str):
        folder = self.get_folder(algorithm)
        os.makedirs(folder, exist_ok=True)
        io.imsave("{0}/segmentation.tiff".format(folder),  segmentation)
        np.save("{0}/segmentation.npy".format(folder), segmentation, False)
        with open("{}/parameters.json".format(folder), "w") as f:
            params["ground_truth_path"] = self.settings["groud_truth_path"]
            f.write(json.dumps(params))
    