import metrics
import importlib
import json
import pandas as pd
import concurrent.futures as cf

from imagedata import ImageData
from os import environ, listdir, makedirs, path
from tqdm import tqdm

CORES = 4
def calculate_metrics(experiment : str, prefix : str, dataset_path : str):
    image = ImageData(prefix, "{}/{}".format(dataset_path, experiment))
    segmentation, ground_truth = image.get_data()
    data = {}
    for metric in metrics.__all__:
        print("metric {}, experiment {}".format(metric, experiment), segmentation.shape, ground_truth.shape)
        data[metric] = importlib.import_module("metrics.{}".format(metric)).evaluate(
            segmentation,
            ground_truth
        )
    return (experiment, data)

def parallel_calculate_metrics(prefix : str, dataset_path : str):
    with cf.ProcessPoolExecutor(max_workers=CORES) as executor:
        futures = [executor.submit(calculate_metrics, experiment, prefix, dataset_path) for experiment in listdir(prefix + dataset_path)]
        cf.wait(futures)
        results = [future.result() for future in futures]
    return results
def sequential_calculate_metrics(prefix : str, dataset_path : str):
    results = []
    for experiment in listdir(prefix + dataset_path):
        results.append(calculate_metrics(experiment, prefix, dataset_path))
    return results
if __name__ == "__main__":
    data_paths = json.loads(environ["paths"])
    prefix = environ["prefix"]
    parallel = environ["parallel"] if "parallel" in environ.keys() else False
    for dataset_path in tqdm(data_paths):
        print("Evaluating dataset {}".format(dataset_path))
        folder = "./results/evaluation/{0}".format(dataset_path)
        if not path.isdir(folder):
            data = {}
            results = parallel_calculate_metrics(prefix, dataset_path) if parallel else sequential_calculate_metrics(prefix, dataset_path)
            for experiment, metrics_Data in results:
                data[experiment] = metrics_Data
            data = pd.DataFrame(data)
            
            makedirs(folder, exist_ok=True)
            data.to_csv("{}/metrics.csv".format(folder))
        else:
            print("Dataset {} already evaluated".format(dataset_path))