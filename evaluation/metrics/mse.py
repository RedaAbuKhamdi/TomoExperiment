import numpy as np
def evaluate(segmentation : np.ndarray, 
             ground_truth : np.ndarray)->float:
    segmentation[segmentation > 0] = 1
    ground_truth[ground_truth > 0] = 1
    seg_f = segmentation.astype(np.float32)
    gt_f  = ground_truth.astype(np.float32)
    se = (seg_f - gt_f) ** 2
    return float(se.mean())