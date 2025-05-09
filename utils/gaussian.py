import numpy as np
from tqdm import tqdm
from numba import jit
import transformations
import random
from scipy.ndimage import affine_transform
from typing import Tuple

class Gaussian:
    def __init__(self,
                 x0: float,
                 y0: float,
                 A: float = 1,
                 theta: float = 0,
                 deltax: float = 1,
                 deltay: float = 1,
                 cut_off: int = 0,
                 orientation: str = 'up',
                 z_limit: int = None,
                 min_height: int = 1):
        self.x0 = x0
        self.y0 = y0

        if orientation not in ('up', 'down'):
            raise ValueError("orientation must be 'up' or 'down'")
        self.orientation = orientation

        # For 'down', we invert A so the bell goes below cut_off
        self.A = float(A) if orientation == 'up' else -abs(A)
        self.cut_off = int(cut_off)

        # Optional z-limit and minimum painted thickness
        self.z_limit    = None if z_limit is None else int(z_limit)
        self.min_height = int(min_height)

        # Validate and store rotation angle
        if not (0 <= theta <= 2*np.pi):
            raise ValueError("theta must be in [0, 2π]")
        self.theta = theta
        # Precompute Gaussian coefficients
        self.direction = 1 if theta < np.pi else -1
        self.a = (np.cos(theta)**2) / (2 * deltax**2) \
               + (np.sin(theta)**2) / (2 * deltay**2)
        self.b = self.direction * (
                - (np.sin(theta)*np.cos(theta)) / (2*deltax**2)
                + (np.sin(theta)*np.cos(theta)) / (2*deltay**2)
            )
        self.c = (np.sin(theta)**2) / (2 * deltax**2) \
               + (np.cos(theta)**2) / (2 * deltay**2)

    def value(self, x: float, y: float) -> float:
        """Return the z‐coordinate of the Gaussian surface at (x, y)."""
        return self.cut_off + self.A * np.exp(
            -self.a * (x - self.x0)**2
            - 2*self.b * (x - self.x0)*(y - self.y0)
            - self.c * (y - self.y0)**2
        )

    def rotate(self, volume: np.ndarray) -> np.ndarray:
        """Rotate the 3D mask by a random principal axis rotation."""
        R, rotation = transformations.choose_random_rotate(self.theta)
        R_inv = np.linalg.inv(R)
        return affine_transform(
            volume,
            R_inv,
            offset=0,
            order=0,
            mode='constant',
            cval=0
        ), rotation

    def paint_volume(self, volume: np.ndarray):
        D, H, W = volume.shape
        img = np.zeros_like(volume, dtype=np.uint8)

        # 1) Raster the bell up or down into img
        for x in tqdm(range(W), desc="Painting Gaussian"):
            for y in range(H):
                z_val = int(round(self.value(x, y)))

                # enforce absolute z_limit if set
                if self.z_limit is not None:
                    if self.A >= 0 and z_val > self.z_limit:
                        continue
                    if self.A <  0 and z_val < self.z_limit:
                        continue

                # clamp into [0, D]
                z_val = max(0, min(D, z_val))

                # determine painting interval
                if self.A >= 0:
                    z_lo, z_hi = self.cut_off, z_val
                else:
                    z_lo, z_hi = z_val, self.cut_off

                thickness = z_hi - z_lo
                # only paint if at least min_height thick
                if thickness >= self.min_height:
                    img[z_lo:z_hi, y, x] = 1

        # 2) Rotate and scatter into the main volume
        rotated, rotation = self.rotate(img)
        volume[rotated > 0] = 1
        return rotation


class SyntheticGaussianImage:
    def generate_dataset(self, image: np.ndarray) -> np.ndarray:
        peak = (image.shape[1] // 2, image.shape[2] // 2)
        gaussian1 = Gaussian(
            x0=peak[0], y0=peak[1],
            A=np.random.randint(140, 240),
            theta=np.pi / 14,
            deltax=np.random.uniform(image.shape[1] // 10,
                                     image.shape[1] // 4),
            deltay=np.random.uniform(image.shape[1] // 10,
                                     image.shape[1] // 4),
            cut_off=0,
            orientation='up',
            z_limit=None,
            min_height=10
        )
        gaussian2 = Gaussian(
            x0=peak[0], y0=image.shape[2] // 3,
            A=np.random.randint(70, 230),
            theta=np.pi / 22,
            deltax=np.random.uniform(image.shape[1] // 10,
                                     image.shape[1] // 4),
            deltay=np.random.uniform(image.shape[1] // 10,
                                     image.shape[1] // 4),
            cut_off=image.shape[0] - 1,
            orientation='down',
            z_limit=None,
            min_height=15
        )
        rotation2 = gaussian2.paint_volume(image)
        rotation1 = gaussian1.paint_volume(image)
        return image, {
            "gaussian1": rotation1,
            "gaussian2": rotation2
        }
