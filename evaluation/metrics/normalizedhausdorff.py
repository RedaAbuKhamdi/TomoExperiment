import numpy as np
from scipy.spatial import cKDTree

def hausdorff_distance_cpu_parallel(A: np.ndarray, B: np.ndarray, workers: int = -1) -> float:
    """
    Compute the undirected Hausdorff distance between two sets of 3D points using parallel CPU processing.
    
    Parameters:
        A (np.ndarray): Array of shape (n, 3) containing 3D points.
        B (np.ndarray): Array of shape (n, 3) containing 3D points.
        workers (int): Number of parallel workers to use (default: -1 uses all available cores).
        
    Returns:
        float: The Hausdorff distance between sets A and B.
    """
    # Build a kd-tree for set B and query all points in A.
    tree_B = cKDTree(B)
    dists_A, _ = tree_B.query(A, workers=workers)
    directed_AB = dists_A.max()
    
    # Build a kd-tree for set A and query all points in B.
    tree_A = cKDTree(A)
    dists_B, _ = tree_A.query(B, workers=workers)
    directed_BA = dists_B.max()
    
    # The undirected Hausdorff distance is the maximum of the two directed distances.
    return max(directed_AB, directed_BA)

def evaluate(segmentation : np.ndarray, 
             ground_truth : np.ndarray)->float:
    
    segmentation_points = np.argwhere(segmentation!=False)
    ground_truth_points = np.argwhere(ground_truth!=False)

    print(segmentation_points.shape)
    hd = hausdorff_distance_cpu_parallel(segmentation_points, ground_truth_points, 28)

    return (1 - hd/(np.sum(np.array(segmentation.shape)**2)**0.5))