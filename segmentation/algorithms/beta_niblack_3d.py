import tqdm
import torch

import numpy as np

from numba import jit
from metrics import jaccard
from imagedata import ImageData

def thresholding(image : torch.tensor, q  : int, mean : torch.tensor, std : torch.tensor, beta : int):
    seg = torch.add(mean, std, alpha=q)
    seg.add(beta)
    return torch.ge(image, seg.view(image.size()[0], image.size()[1], image.size()[2])).cpu().detach().numpy()

def segment(imageData : ImageData):
    print("Start Niblack")
    shape = imageData.image.shape
    image = torch.from_numpy(imageData.image.reshape(shape[2], shape[0], shape[1]))
    image_gpu = image.clone().detach().to(torch.device("cuda"))
    ks, w, betas = np.linspace(-1, 1, 24), max([1, min(image.shape)//2]), np.linspace(torch.min(image).item(), torch.mean(image).item(), 30)
    best_val = 0
    best_k = 0
    best_image = None
    best_beta = 0
    best_window = 5
    windows = np.arange(5, w, (w-5)//5)
    for window in tqdm.tqdm(windows):
        window = window - (window % 2 == 0)
        means, stds = [], []
        for i in range(image.size()[0]):
            folded = torch.nn.functional.unfold(image_gpu[0].view(
                1, 1, image_gpu.size()[1], image_gpu.size()[2]
            ), window, padding=window // 2)
            means.append(folded.mean(dim=1))
            stds.append(folded.std(dim=1))
            del folded
        means_cuda = torch.stack(means, 0)
        means_cuda = means_cuda.view(shape[2], shape[0], shape[1])
        stds_cuda = torch.stack(stds, 0).view(shape[2], shape[0], shape[1])
        for a in tqdm.tqdm(range(betas.size)):
                for i in range(ks.size):
                    segmentation = thresholding(image_gpu, ks[i], means_cuda, stds_cuda, betas[a])
                    mean_metric = 0
                    ground_truth_slice_amount = 0
                    for index, ground_truth_slice in imageData.get_ground_truth_slices():
                        segmented_slice = segmentation[index]
                        mean_metric += jaccard(segmented_slice, ground_truth_slice)
                        ground_truth_slice_amount += 1
                    mean_metric /= ground_truth_slice_amount
                    if mean_metric > best_val:
                        best_val = mean_metric
                        best_k = ks[i]
                        best_image = np.copy(segmentation)
                        best_beta = betas[a]
                        best_window = window
        del means_cuda
        del stds_cuda
        torch.cuda.empty_cache()

    params = {
        "k" : float(best_k),
        "w" : float(best_window),
        'beta' : float(best_beta),
        "jaccard" : float(best_val)
    }
    return best_image, params
