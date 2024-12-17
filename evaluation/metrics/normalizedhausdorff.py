import numpy as np
from scipy.spatial.distance import directed_hausdorff

def evaluate(segmentation : np.ndarray, 
             ground_truth : np.ndarray)->float:
    
    segmentation_points = np.argwhere(segmentation!=False)
    ground_truth_points = np.argwhere(ground_truth!=False)

    print(segmentation_points.shape)

    return (1 - max(
        directed_hausdorff(segmentation_points, ground_truth_points)[0],
        directed_hausdorff(ground_truth_points, segmentation_points)[0])/(np.sum(np.array(segmentation.shape)**2)**0.5))