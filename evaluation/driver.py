import metrics
import importlib
import json
import pandas as pd
import sys

from imagedata import ImageData
from os import environ, listdir, makedirs, path
from tqdm import tqdm

sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
import config

def calculate_metrics_ground_truth(data : dict, image : ImageData):
    segmentation, ground_truth = image.get_data()
    for metric in metrics.__all__:
        metric_value = importlib.import_module("metrics.{}".format(metric)).evaluate(
            segmentation,
            ground_truth
        )
        if metric not in data.keys():
            data[metric] = []
        data[metric].append(metric_value)

def calculate_metrics_neighbor(data : dict, image1 : ImageData, image2 : ImageData):
    for metric in metrics.__all__:
        metric_value = importlib.import_module("metrics.{}".format(metric)).evaluate(
            image1.image,
            image2.image
        )
        if metric not in data.keys():
            data[metric] = []
        data[metric].append(metric_value)

def iterate_datasets(data_paths : list):
    datasets = {}
    for dataset_path in tqdm(data_paths):
        strategy = dataset_path.split("/")[-3]
        name = dataset_path.split("/")[-2]
        print(dataset_path)
        for algorithm in listdir(dataset_path):
            folder = "./results/evaluation/{0}/{1}/{2}".format(strategy, name, algorithm)
            if not path.isdir(folder):
                if name not in datasets.keys():
                    datasets[name] = {}
                if algorithm not in datasets[name].keys():
                    datasets[name][algorithm] = []
                datasets[name][algorithm].append("{}/{}".format(dataset_path, algorithm))
    return datasets 

def save_result(data : pd.DataFrame, save_path, name : str):
    makedirs(save_path, exist_ok=True)
    data.to_csv("{}/{}.csv".format(save_path, name))

def run_evaluations(dataset : list, name : str, algorithm : str):
    data_gt = {}
    angles = []
    data_neighbor = {}
    for i, experiment_image in tqdm(enumerate(dataset), "Evaluating dataset " + name + " with algorithm " + algorithm):
        calculate_metrics_ground_truth(data_gt, experiment_image)
        angles.append(experiment_image.settings["angles"])
        if (i > 0):
            calculate_metrics_neighbor(data_neighbor, experiment_image, dataset[i - 1])
    return data_gt, angles, data_neighbor

if __name__ == "__main__":
    data_paths = json.loads(environ["paths"])
    datasets = iterate_datasets(data_paths)
    for i, name in tqdm(enumerate(datasets.keys()), "Evaluating datasets"):
        for algorithm in datasets[name].keys():
            dataset = [ImageData(datasets[name][algorithm][i]) for i in range(len(datasets[name][algorithm]))]
            save_path = config.EVALUATION_PATH / dataset[0].strategy / name / algorithm
            data_gt, angles, data_neighbor = run_evaluations(dataset, name, algorithm)
            save_result(pd.DataFrame(data_gt, index = angles), save_path, "ground_truth_metrics")
            save_result(pd.DataFrame(data_neighbor), save_path, "neighbor_metrics")