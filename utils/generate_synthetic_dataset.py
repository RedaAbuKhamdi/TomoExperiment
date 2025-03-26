import PIL.Image
import numpy as np
import PIL
import os
import json
from numba import njit

@njit()
def generate_synthetic_dataset(n):
    image = np.zeros((256, 512, 512)).astype(np.uint8)
    max_size = 200
    min_size = 64
    color = 255 // n
    for _ in range(n):
        pts = np.argwhere(image == 0)
        center = pts[np.random.choice(pts.shape[0], 1)]
        a = np.random.randint(min_size, max_size)
        b = np.random.randint(min_size, max_size)
        c = np.random.randint(min_size, max_size)

        for k in range(image.shape[0]):
            for i in range(image.shape[1]):  
                for j in range(image.shape[2]):
                    image[k, i, j] += color if ((i - center[0][0])/a) ** 2 + ((j - center[0][1])/b) ** 2 + ((k - center[0][2])/c) ** 2 <= 1 else 0
    return image

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
            "x": 2.1,
            "y": 2.1
        },
        "detector_count": {
            "rows": 400,
            "columns": 400
        },
        "iterations": 200,
        "algorithm": "SIRT3D_CUDA"
    }))
    
def save_ground_truth(image):
    folder = "../ground_truth/ellipses"
    os.makedirs(folder, exist_ok=True)
    for k in range(image.shape[0]):
        slice = PIL.Image.fromarray(np.uint8((image[k] > 0) * (255)) , 'L')
        slice.save(folder + "/" + str(k) + ".png")

image = generate_synthetic_dataset(8)

save_ground_truth(image)
save_dataset(image)
