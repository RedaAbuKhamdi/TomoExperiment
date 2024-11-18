import metrics
import importlib
import json
import pandas as pd

from imagedata import ImageData
from os import environ, listdir, makedirs, path
from tqdm import tqdm


data_paths = json.loads(environ["paths"])
prefix = environ["prefix"]
for dataset_path in data_paths:
    print("Evaluating dataset {}".format(dataset_path))
    folder = "./results/evaluation/{0}".format(dataset_path)
    if not path.isdir(folder):
        data = {}
        for experiment in tqdm(listdir(prefix + dataset_path)):
            image = ImageData(prefix, "{}/{}".format(dataset_path, experiment))
            segmentation, ground_truth = image.get_data()
            data[experiment] = {}
            for metric in metrics.__all__:
                data[experiment][metric] = importlib.import_module("metrics.{}".format(metric)).evaluate(
                    segmentation,
                    ground_truth
                )
        data = pd.DataFrame(data)
        
        makedirs(folder, exist_ok=True)
        data.to_csv("{}/metrics.csv".format(folder))
    else:
        print("Dataset {} already evaluated".format(dataset_path))