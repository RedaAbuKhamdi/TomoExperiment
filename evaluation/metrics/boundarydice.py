import numpy as np
from numba import njit, prange
from tqdm import tqdm

import cupy as cp

# Define a custom CUDA kernel for a 3D sliding window sum with zero-padding.
kernel_code = r'''
extern "C" __global__
void sliding_window_sum_3d(const int* arr, int* out, 
                           int D0, int D1, int D2, 
                           int radius)
{
    // Compute the 3D index handled by this thread.
    // We use z for the first dimension (i), y for the second (j), and x for the third (k)
    int k = blockIdx.x * blockDim.x + threadIdx.x;  // third dimension index
    int j = blockIdx.y * blockDim.y + threadIdx.y;  // second dimension index
    int i = blockIdx.z * blockDim.z + threadIdx.z;  // first dimension index

    if (i >= D0 || j >= D1 || k >= D2)
        return;

    float sum = 0.0f;
    // Loop over the window in the first dimension.
    for (int di = -radius; di <= radius; di++) {
        int ii = i + di;
        if (ii < 0 || ii >= D0)
            continue;
        // Loop over the window in the second dimension.
        for (int dj = -radius; dj <= radius; dj++) {
            int jj = j + dj;
            if (jj < 0 || jj >= D1)
                continue;
            // Loop over the window in the third dimension.
            for (int dk = -radius; dk <= radius; dk++) {
                int kk = k + dk;
                if (kk < 0 || kk >= D2)
                    continue;
                // Compute the flat index for the 3D array.
                int idx = ii * D1 * D2 + jj * D2 + kk;
                sum += arr[idx];
            }
        }
    }
    // Write the result to the output.
    out[i * D1 * D2 + j * D2 + k] = sum;
}
'''

# Compile the custom CUDA kernel.
sliding_window_sum_kernel = cp.RawKernel(kernel_code, 'sliding_window_sum_3d')

def sliding_window_sum_3d_gpu(arr, radius):
    """
    Compute a 3D sliding window sum on a GPU using a custom CUDA kernel with zero-padding.
    
    Each output voxel (i, j, k) is computed as the sum of all values in the window
    from (i - radius, j - radius, k - radius) to (i + radius, j + radius, k + radius).
    Voxels outside the boundaries are treated as zero.
    
    Parameters:
      arr (cp.ndarray): 3D input array on the GPU (dtype float32).
      radius (int): Radius of the window.
    
    Returns:
      cp.ndarray: A 3D output array with the same shape as `arr`, containing the sliding window sum.
    """
    if arr.ndim != 3:
        raise ValueError("This function supports 3D arrays only.")
    if arr.dtype != cp.int32:
        arr = arr.astype(cp.int32)
    
    D0, D1, D2 = arr.shape
    out = cp.empty_like(arr)
    
    # Define block size. A good starting point is (8, 8, 8).
    threads_per_block = (8, 8, 8)
    # Grid dimensions are computed based on the volume dimensions.
    blocks_x = (D2 + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_y = (D1 + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_z = (D0 + threads_per_block[2] - 1) // threads_per_block[2]
    grid = (blocks_x, blocks_y, blocks_z)
    
    # Launch the kernel.
    sliding_window_sum_kernel(grid, threads_per_block,
                              (arr, out, D0, D1, D2, radius))
    return out

# Define a custom CUDA kernel that checks if any element in the 3D neighborhood is zero.
kernel_code = r'''
extern "C" __global__
void sliding_window_has_zero(const int* arr, int* out,
                             int D0, int D1, int D2, int radius)
{
    // Compute the 3D index for the current thread.
    // Here: i = first dimension, j = second, k = third.
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= D0 || j >= D1 || k >= D2)
        return;

    bool found = false;

    // Loop over the neighborhood: from i - radius to i + radius, etc.
    for (int di = -radius; di <= radius && !found; di++) {
        int ii = i + di;
        if (ii < 0 || ii >= D0)
            continue;
        for (int dj = -radius; dj <= radius && !found; dj++) {
            int jj = j + dj;
            if (jj < 0 || jj >= D1)
                continue;
            for (int dk = -radius; dk <= radius; dk++) {
                int kk = k + dk;
                if (kk < 0 || kk >= D2)
                    continue;

                int idx = ii * D1 * D2 + jj * D2 + kk;
                if (arr[idx] == 0.0f) {
                    found = true;
                    break;
                }
            }
        }
    }

    // Write 1 if a zero was found, 0 otherwise.
    out[i * D1 * D2 + j * D2 + k] = found ? 1.0f : 0.0f;
}
'''

# Compile the CUDA kernel.
sliding_window_has_zero_kernel = cp.RawKernel(kernel_code, 'sliding_window_has_zero')

def sliding_window_has_zero_3d_gpu(arr, radius):
    """
    Check with a sliding window whether there is any zero in the neighborhood for a 3D array.
    
    For each voxel (i, j, k) in the input 3D array, the function examines
    the values in the window spanning from (i - radius) to (i + radius),
    (j - radius) to (j + radius), and (k - radius) to (k + radius). If any element in the window is zero,
    the output voxel is set to 1; otherwise, it is set to 0.
    
    Out-of-bound indices (where the window extends past the array boundary)
    are skipped.
    
    Parameters:
      arr (cp.ndarray): A 3D CuPy array with dtype float32.
      radius (int): The radius defining the window size.
    
    Returns:
      cp.ndarray: A 3D CuPy array of the same shape as `arr` containing 1's and 0's.
    """
    if arr.ndim != 3:
        raise ValueError("This function supports 3D arrays only.")
    if arr.dtype != cp.int32:
        arr = arr.astype(cp.int32)
        
    D0, D1, D2 = arr.shape
    out = cp.empty_like(arr)
    
    # Define a 3D block size. Here (8, 8, 8) is chosen as a starting point.
    threads_per_block = (8, 8, 8)
    blocks_x = (D2 + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_y = (D1 + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_z = (D0 + threads_per_block[2] - 1) // threads_per_block[2]
    grid = (blocks_x, blocks_y, blocks_z)
    
    # Launch the kernel.
    sliding_window_has_zero_kernel(grid, threads_per_block,
                                   (arr, out, D0, D1, D2, radius))
    return out

def evaluate(segmentation, ground_truth):
    radius = 1

    # binarize
    ground_truth = (ground_truth > 0).astype(np.int32)
    segmentation = (segmentation > 0).astype(np.int32)

    # upload
    cp_seg = cp.asarray(segmentation, dtype=cp.int32)
    cp_gt  = cp.asarray(ground_truth,  dtype=cp.int32)

    # 1) detect *all* boundary voxels (including interior holes) via has‐zero
    has0_seg = sliding_window_has_zero_3d_gpu(cp_seg, radius)
    has0_gt  = sliding_window_has_zero_3d_gpu(cp_gt,  radius)
    b_seg = (cp_seg & has0_seg).astype(cp.int32)
    b_gt  = (cp_gt  & has0_gt ).astype(cp.int32)

    # 2) compute local sums *only* on those boundary masks
    I_map   = sliding_window_sum_3d_gpu(b_seg & b_gt, radius)
    S_seg   = sliding_window_sum_3d_gpu(b_seg,       radius)
    S_gt    = sliding_window_sum_3d_gpu(b_gt,        radius)

    # 3) sample coordinates of each boundary set
    zs_seg, ys_seg, xs_seg = cp.where(b_seg)
    zs_gt,  ys_gt,  xs_gt  = cp.where(b_gt)

    # 4) compute local Dice at each boundary point
    dice_seg = 2 * I_map[zs_seg,ys_seg,xs_seg] / (
                  S_seg[zs_seg,ys_seg,xs_seg] + S_gt[zs_seg,ys_seg,xs_seg]
               )
    dice_gt  = 2 * I_map[zs_gt, ys_gt, xs_gt] / (
                  S_seg[zs_gt, ys_gt, xs_gt]  + S_gt[zs_gt, ys_gt, xs_gt]
               )

    # 5) average
    total = dice_seg.sum() + dice_gt.sum()
    count = zs_seg.size + zs_gt.size
    return float((total / count).item())
