import numpy as np
from numba import jit

@jit
def evaluate(segmentation : np.ndarray, 
             ground_truth : np.ndarray)->float:
    
    segmentation = segmentation.astype(np.int8)
    ground_truth = ground_truth.astype(np.int8)

    square_error = (segmentation - ground_truth)**2

    return np.mean(square_error)