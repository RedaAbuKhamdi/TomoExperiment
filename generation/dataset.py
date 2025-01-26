import astra
import json
import re

import numpy as np

from PIL import Image
from typing import List

SAVE_SETTINGS_KEYWORDS = [
    "name"
]

def natural_sort_key(s, _nsre=re.compile(r'(\d+)')):
    return [int(text) if text.isdigit() else text.lower()
            for text in _nsre.split(s)]

class Data:
    def __init__(self, image_paths : List[str]) -> None:
        settings_matches = [i for i in range(len(image_paths)) if image_paths[i].find("json") != -1]
        if len(settings_matches) != 1:
            raise "More than one setting files found!" if len(settings_matches) > 1 else "No settings file found!"
        self.parse_settings(image_paths.pop(settings_matches[0]))
        self.n = len(image_paths)
        image_paths.sort(key=natural_sort_key)
        self.volumetric = self.n > 1
        self._create_image_with_geometry(image_paths)

    def parse_settings(self, path):
        with open(path, "r") as file:
            self.settings = json.loads(file.read())

    def get_experiments(self):
        for i in range(len(self.settings["experiments"])):
            yield self.settings["experiments"][i]

    def _create_image_with_geometry(self, image_paths):
        image = [None] * self.n
        for i in range(self.n):
            image[i] = np.asarray(Image.open(image_paths[i]))
        
        image = np.array(image) if self.volumetric else image[0]
        data_geometry = astra.create_vol_geom((image.shape[2], image.shape[1], image.shape[0])) if self.volumetric else  astra.create_vol_geom(image.shape[1], image.shape[0])
       
        if self.volumetric:
            self.data_id = astra.data3d.create("-vol", data_geometry, data = image)
        else: 
            self.data_id = astra.data2d.create("-vol", data_geometry, data = image)
    
        self.sinogram_id = None
        self.data_geometry = data_geometry

    def calculate_sinogram(self, projector, projection_geometry): 
        image = astra.data3d.get(self.data_id) if self.volumetric else astra.data2d.get(self.data_id)
        if self.volumetric:
            self.data_id = astra.data3d.create("-vol", self.data_geometry, data = image)
            self.sinogram_id = astra.create_sino3d_gpu(self.data_id, projection_geometry, self.data_geometry, returnData = False)
        else: 
            self.data_id = astra.data2d.create("-vol", self.data_geometry, data = image)
            self.sinogram_id = astra.create_sino(image, projector, returnData = False)
        return self.sinogram_id
    def __del__(self):
        try:
            if self.volumetric:
                astra.data3d.delete(self.data_id)
                astra.data3d.delete(self.sinogram_id)
            else:
                astra.data2d.delete(self.data_id)
                astra.data2d.delete(self.sinogram_id)
        except:
            print("Dataset {} did not need reconstruction".format(self.settings["name"]))
    def get_volume_geometry(self):
        return self.data_geometry
    def save_settings(self, path : str, angles : np.ndarray, strategy : str, step : str):
        new_settings = {
            "angles": {
                "step": step,
                "strategy": strategy,
                "values": angles.tolist()
            }
        }
        for key in SAVE_SETTINGS_KEYWORDS:
            if key in self.settings:
                new_settings[key] = self.settings[key]
        with open("{0}/settings.json".format(path), "w") as f:
            f.write(json.dumps(new_settings))
