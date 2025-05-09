import numpy as np
import transformations
from tqdm import tqdm

class Grid3DGenerator:
    @staticmethod
    def generate_dataset(spacing: int = 64,
                         thickness: int = 5,
                         random_angle: bool = False):
        D, H, W = 256, 512, 512
        # build coordinate grids once
        z = np.arange(D)[:, None, None]
        y = np.arange(H)[None, :, None]
        x = np.arange(W)[None, None, :]

        parameters = {}

        if random_angle:
            # 1) build a single random rotation R
            a, b, c = np.random.uniform(0, np.pi/2, size=3)

            parameters = {
                "roll": c,
                "pitch": a,
                "yaw": b
            }

            R = (
                transformations.Direction.ROLL(c) @
                transformations.Direction.YAW(b) @
                transformations.Direction.PITCH(a)
            )

            # 2) define the three base‐normals
            base_normals = [
                np.array([0, 0, 1]),   # z‐plane
                np.array([0, 1, 0]),   # y‐plane
                np.array([1, 0, 0]),   # x‐plane
            ]
            normals = [R @ n for n in base_normals]

            # 3) for each normal, slice the volume into parallel planes
            grid = np.zeros((D, H, W), dtype=bool)
            for n in normals:
                # signed distance along that normal
                d = n[0]*x + n[1]*y + n[2]*z
                # tile planes every `spacing`, thickness `thickness`
                grid |= (d % spacing) < thickness

            # finally cast to 0/1
            grid = grid.astype(np.uint8)

        else:
            # axis‐aligned backup
            mask_z = (z % spacing) < thickness
            mask_y = (y % spacing) < thickness
            mask_x = (x % spacing) < thickness
            grid = (mask_z | mask_y | mask_x).astype(np.uint8)

        return grid, parameters
