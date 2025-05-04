import PIL.Image
import numpy as np
import PIL
import os
import json
from numba import jit, prange
from ellipses import SyntheticEllipsesImage
from polygon import PolygonSceneGenerator
from grid import Grid3DGenerator
from gaussian import SyntheticGaussianImage


def save_dataset(image, dataset_name):
    folder = "../data/" + dataset_name
    os.makedirs(folder, exist_ok=True)

    for k in range(image.shape[0]):
        slice = PIL.Image.fromarray(np.uint8(image[k] + np.random.normal(20, 5, image[k].shape)) , 'L')
        slice.save(folder + "/" + str(k) + ".png")

    with open(folder + "/settings.json", "w") as f:
        f.write(json.dumps({
        "name": dataset_name,
        "detector_spacing": {
            "x": 1,
            "y": 1
        },
        "detector_count": {
            "rows": 800,
            "columns": 800
        },
        "iterations": 120,
        "algorithm": "SIRT3D_CUDA"
    }))
    
def save_ground_truth(image, dataset_name):
    folder = "../ground_truth/" + dataset_name
    os.makedirs(folder, exist_ok=True)
    for k in range(image.shape[0]):
        slice = PIL.Image.fromarray(np.uint8((image[k] > 0) * (255)) , 'L')
        slice.save(folder + "/" + str(k) + ".png")

dataset_name = input("Which dataset to generate (ellipses, polygons, grid)")
if ("polygons" in dataset_name):
    image = PolygonSceneGenerator().generate_dataset(np.zeros((256, 512, 512)), float(input("Cluster radius: "))) * 124
elif dataset_name == "ellipses":
    image = SyntheticEllipsesImage().generate_dataset(np.zeros((256, 512, 512)), 3)
elif dataset_name == "anglegrid":
    image = Grid3DGenerator.generate_dataset(64, 5, True) * 104
elif dataset_name == "grid":
    image = Grid3DGenerator.generate_dataset(64, 5, False) * 104
elif dataset_name == "gaussian":
    image = SyntheticGaussianImage().generate_dataset(np.zeros((256, 512, 512))) * 104
else:
    raise Exception("Unknown dataset name")

save_ground_truth(image, dataset_name)
save_dataset(image, dataset_name)
