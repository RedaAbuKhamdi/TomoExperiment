import json
import sys
import pathlib
import os
import pandas as pd
import numpy as np

from os import environ, listdir, makedirs, path
from matplotlib import pyplot as plt

from metricsdata import MetricsData
from tqdm import tqdm

from matplotlib import colors as mcolors
import random

def generate_distinct_colors(labels):
    color_names = list(mcolors.TABLEAU_COLORS.values())
    random.shuffle(color_names)
    return dict(zip(labels, color_names[:len(labels)]))

sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
import config
def iterate_datasets():
    for experiment_path in config.EVALUATION_PATH.iterdir():
        experiment = experiment_path.as_posix().split("/")[-1]
        for dataset_path in experiment_path.iterdir():
            dataset = dataset_path.as_posix().split("/")[-1]
            for algorithm_path in dataset_path.iterdir():
                algorithm = algorithm_path.as_posix().split("/")[-1]
                dataset_config = {
                    "experiment": experiment,
                    "name": dataset,
                    "algorithm": algorithm,
                    "evaluations": []
                }
                for data in algorithm_path.glob("*.csv"):
                    data_name = data.as_posix().split("/")[-1]
                    dataset_config["evaluations"].append({
                        "type": data_name.split(".")[0].replace("_", " ").capitalize(),
                        "path": data.as_posix()
                    })
                yield dataset_config
def delta_from_values(values : np.ndarray) -> np.ndarray:
    result = np.zeros_like(values)
    for i in range(result.shape[0] - 1):
        result[i] = values[i+1] - values[i]
    return result
def scatter_plots(algorithm_data: dict, algorithm : str, colors : list):
    datasets = algorithm_data["datasets"]
    angles = algorithm_data["angles"]
    metric_values = algorithm_data["metric_values"]
    metrics = metric_values[datasets[0]].keys()

    for metric_name in metrics:
        plt.clf()
        for i, dataset in enumerate(datasets):
            for j, metric in enumerate(metric_values[dataset].keys()):
                if metric != metric_name:
                    continue
                values = metric_values[dataset][metric_name]
                plt.scatter(angles.astype(str), values, color=colors[dataset], label=f"{dataset} - {metric}", s = 16)

        plt.title("{} metric values by angle for each image".format(metric_name))
        plt.xlabel("Number of Angles")
        plt.ylabel("Metric Value")
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.legend()
        folder = config.VISUALIZATION_PATH / algorithm
        makedirs(folder, exist_ok=True)
        plt.savefig("{0}/{1}_scatter_plot_neighbor.png".format(folder, metric_name))

def scatter_plots_per_dataset(algorithm_data: dict, algorithm : str, colors : list, delta : bool = False):
    datasets = algorithm_data["datasets"]
    angles = algorithm_data["angles"]
    metric_values = algorithm_data["metric_values"]
    metrics = metric_values[datasets[0]].keys()

    x = np.arange(len(angles))  
    for dataset in datasets:
        plt.clf()
        for metric in metric_values[dataset].keys():
            values = delta_from_values(metric_values[dataset][metric]) if delta else metric_values[dataset][metric]
            plt.scatter(angles.astype(str), values, label=f"{dataset} - {metric}")

        plt.title("Metric values by angle for image {}".format(dataset))
        plt.xlabel("Number of Angles")
        plt.ylabel("Metric Value")
        plt.tight_layout()
        plt.legend()
        folder = config.VISUALIZATION_PATH / algorithm / "per_dataset"
        makedirs(folder, exist_ok=True)
        plt.savefig("{0}/{1}_scatter_plot.png".format(folder, dataset))

def scatter_plots_per_dataset_ground_truth(algorithm_data: dict, algorithm : str, colors : list, delta : bool = False):
    datasets = algorithm_data["datasets"]
    angles = algorithm_data["angles"]
    metric_values = algorithm_data["metric_values"]
    metrics = metric_values[datasets[0]].keys()

    x = np.arange(len(angles))  
    for dataset in datasets:
        plt.clf()
        for metric in metric_values[dataset].keys():
            values = delta_from_values(metric_values[dataset][metric]) if delta else metric_values[dataset][metric]
            plt.scatter(angles.astype(str), values, label=f"{dataset} - {metric}")

        plt.title("Metric values by angle for image {}".format(dataset))
        plt.xlabel("Number of Angles")
        plt.ylabel("Metric Value")
        plt.tight_layout()
        plt.legend()
        folder = config.VISUALIZATION_PATH / "threshold" / algorithm / "per_dataset"
        makedirs(folder, exist_ok=True)
        plt.savefig("{0}/{1}_scatter_plot.png".format(folder, dataset))

def scatter_plots_mean_ground_truth(algorithm_data: dict, algorithm : str, colors : list, delta : bool = False):
    datasets = algorithm_data["datasets"]
    angles = algorithm_data["angles"]
    metric_values = algorithm_data["metric_values"]
    metrics = metric_values[datasets[0]].keys()

    x = np.arange(len(angles))  
    for metric_name in metrics:
        plt.clf()
        values = np.zeros(angles.shape)
        for i, dataset in enumerate(datasets):
            values += delta_from_values(metric_values[dataset][metric_name]) if delta else metric_values[dataset][metric_name]
        values /= len(datasets)
        plt.scatter(angles.astype(str), values, color=colors[dataset])

        if delta:
            plt.title("Bubble Chart of mean {} differences values across datasets".format(metric_name))
        else:
            plt.title("Bubble Chart of mean {} values across datasets".format(metric_name))

        plt.xlabel("Number of Angles")
        plt.ylabel("Metric Value")
        plt.tight_layout()
        plt.legend()
        folder = config.VISUALIZATION_PATH / "threshold" / algorithm 
        makedirs(folder, exist_ok=True)
        plt.savefig("{0}/{1}_scatter_plot_mean.png".format(folder, metric_name))

if __name__ == "__main__":
    print("Start visualization")
    metrics_data = MetricsData()
    datasets = metrics_data.get_datasets()
    colors = generate_distinct_colors(datasets)
    for algorithm, data in metrics_data.get_per_algorithm_data("Neighbor metrics"):
        print("Processing algorithm {}".format(algorithm))
        scatter_plots(data, algorithm, colors)
        scatter_plots_per_dataset(data, algorithm, colors)
    
    for algorithm in metrics_data.get_algorithms():
        print("Processing algorithm {} - thresholds".format(algorithm))
        thresholds = np.arange(0, 1, 0.01)
        for metric in ["iou", "boundarydice"]:
            angles = []
            metrics = []
            for threshold in tqdm(thresholds):
                angle, metric_value = metrics_data.get_threshold_data(threshold, algorithm, metric)
                angles.append(angle)
                metrics.append(metric_value)
            angles = np.array(angles)
            metrics = np.array(metrics) 
            plt.clf()
            plt.scatter(angles, metrics, label = "Threshold")
            plt.title("Mean {} values by angle for each threshold".format(metric))
            plt.xlabel("angles")
            plt.ylabel("Metric Value")
            plt.tight_layout()
            plt.legend()
            folder = config.VISUALIZATION_PATH / "threshold" / algorithm
            makedirs(folder, exist_ok=True)
            plt.savefig("{0}/{1}_threshold.png".format(folder, metric))
        for algorithm, data in metrics_data.get_per_algorithm_data("Ground truth metrics"):
            print("Processing algorithm {}".format(algorithm))
            scatter_plots_per_dataset_ground_truth(data, algorithm, colors)
            scatter_plots_mean_ground_truth(data, algorithm, colors)