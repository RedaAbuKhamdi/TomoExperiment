import PIL.Image
import numpy as np
import PIL
import os
import json
from numba import jit, prange
from ellipses import SyntheticEllipsesImage


def save_dataset(image):
    folder = "../data/ellipses"
    os.makedirs(folder, exist_ok=True)

    for k in range(image.shape[0]):
        slice = PIL.Image.fromarray(np.uint8(image[k] + np.random.normal(20, 5, image[k].shape)) , 'L')
        slice.save(folder + "/" + str(k) + ".png")

    with open(folder + "/settings.json", "w") as f:
        f.write(json.dumps({
        "name": "ellipses",
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
    
def save_ground_truth(image):
    folder = "../ground_truth/ellipses"
    os.makedirs(folder, exist_ok=True)
    for k in range(image.shape[0]):
        slice = PIL.Image.fromarray(np.uint8((image[k] > 0) * (255)) , 'L')
        slice.save(folder + "/" + str(k) + ".png")


image = SyntheticEllipsesImage().generate_dataset(np.zeros((256, 512, 512)), 3)

save_ground_truth(image)
save_dataset(image)
