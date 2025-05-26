import json
import os 
import re
import sys

import numpy as np

from PIL import Image
from skimage import io
from os import path

sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
import config

class ImageDataBase:
    def __init__(self, path):
        self.path = path
        self.image = io.imread(path + "/" + [f for f in os.listdir(path) if f.endswith(".tiff")][0])
        self.parse_settings(path)
        self.load_ground_truth()

    def parse_settings(self, path):
        settings_path = path + "/" + [f for f in os.listdir(path) if f.endswith(".json")][0]
        with open(settings_path, "r") as f:
            self.settings = json.loads(f.read())
            self.settings["name"] = self.path.split("/")[-3]

    def load_ground_truth(self):
        try:
            if "name" in self.settings.keys():
                ground_truth_path = config.GROUND_TRUTH_PATH / self.settings["name"]
            else:
                ground_truth_path = config.GROUND_TRUTH_PATH / self.path.split("/")[-3]
            ground_truth_slices = {}
            for file in os.listdir(ground_truth_path):
                index = int(re.findall(r"\d+", file)[0])
                ground_truth_slices[index] = np.asarray(Image.open("{}/{}".format(ground_truth_path, file)))
            self.ground_truth_slices = ground_truth_slices
        except AttributeError:
            raise "Error loading ground truth"
    
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
        print(self.settings["name"])
        print(self.settings["name"])
        for index, gt in self.get_ground_truth_slices():
            segmentation.append(self.image[index])
            ground_truth.append(gt)
        return np.array(segmentation), np.array(ground_truth)