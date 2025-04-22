import numpy as np
import cupy as cp

class Grid3DGenerator:
    @staticmethod
    def generate_dataset(spacing: int = 64,
                         thickness: int = 5,
                         use_gpu: bool = False):
        """
        Generate a 3D grid volume of fixed shape, with grid lines every `spacing` voxels
        and line thickness `thickness`.

        Parameters:
          spacing: int - distance between grid lines
          thickness: int - thickness of each grid line
          use_gpu: bool - if True, use CuPy for GPU acceleration

        Returns:
          volume: np.ndarray - binary volume with 1 on grid lines, 0 elsewhere
        """
        # fixed volume shape
        shape = (256, 512, 512)
        D, H, W = shape
        xp = cp if use_gpu else np

        # Create axis arrays of indices
        z = xp.arange(D)[:, None, None]
        y = xp.arange(H)[None, :, None]
        x = xp.arange(W)[None, None, :]

        # Compute masks for grid planes along each axis
        mask_z = (z % spacing) < thickness  # planes perpendicular to z
        mask_y = (y % spacing) < thickness  # planes perpendicular to y
        mask_x = (x % spacing) < thickness  # planes perpendicular to x

        # Combine masks: grid lines appear where any mask is True
        grid = mask_z | mask_y | mask_x

        # If on GPU, explicitly transfer back to NumPy
        if use_gpu:
            grid = cp.asnumpy(grid)
        return grid

