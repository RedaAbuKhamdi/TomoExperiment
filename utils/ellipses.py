import numpy as np
from tqdm import tqdm
from numba import jit

@jit(nopython = True)
def paint_ellipse(image, center, a, b, c, color):
    for k in range(image.shape[0]):
        for i in range(image.shape[1]):  
            for j in range(image.shape[2]):
                if ((k - center[0])/a) ** 2 + ((i - center[1])/b) ** 2 + ((j - center[2])/c) ** 2 <= 1:
                    image[k, i, j] += color
@jit(nopython = True)
def break_down_volume(image : np.ndarray, cube_amount : int):
        cube_size = np.array([
            image.shape[0] // cube_amount,
            image.shape[1] // cube_amount,
            image.shape[2] // cube_amount
        ])
        result = np.zeros((cube_amount ** 3, 3))
        for i in range(cube_amount):
            for j in range(cube_amount):
                for k in range(cube_amount):
                    result[i * cube_amount * cube_amount + j * cube_amount + k] = np.array([
                        (cube_size[0] * (i + 1) + cube_size[0] * i) // 2,
                        (cube_size[1] * (j + 1) + cube_size[1] * j) // 2,
                        (cube_size[1] * (k + 1) + cube_size[2] * k) // 2
                    ])

        return result
class SyntheticEllipsesImage:
    def generate_dataset(self, image : np.ndarray, cube_amount : int):
        max_radius = np.min(image.shape) // 2
        min_radius = max_radius // 4
        for cube_center in tqdm(break_down_volume(image, cube_amount)):
            paint_ellipse(image, cube_center, np.random.randint(min_radius, max_radius), np.random.randint(min_radius, max_radius), np.random.randint(min_radius, max_radius), 210 // cube_amount)
        print(np.unique(image))
        return image, {}

    
