import json
import sys
import pathlib
import os
import pandas as pd
import numpy as np

from os import environ, listdir, makedirs, path
from matplotlib import pyplot as plt

from metricsdata import MetricsData

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

def bar_charts(algorithm_data: dict, algorithm: str, delta: bool = False):
    datasets       = algorithm_data["datasets"]
    angles         = algorithm_data["angles"]
    metric_values  = algorithm_data["metric_values"]
    metrics        = list(next(iter(metric_values.values())).keys())

    N = len(datasets)
    M = len(metrics)
    x = np.arange(len(angles))
    bar_width = 0.8 / M

    # create one row per dataset
    fig, axes = plt.subplots(
        nrows=N, ncols=1,
        figsize=(12, 4 * N),
        sharex=True
    )

    # if there's only one dataset, axes is not a list
    if N == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        for j, metric_name in enumerate(metrics):
            vals = np.array(metric_values[dataset][metric_name], dtype=float)
            if delta:
                vals = np.gradient(vals, angles)
            # center the group of M bars at x=k
            offset = (j - (M - 1) / 2) * bar_width
            ax.bar(x + offset, vals, bar_width, label=metric_name)

        ax.set_ylabel("Metric Value")
        ax.set_title(f"{dataset}")
        ax.legend(loc="upper right")
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    # common labels
    axes[-1].set_xlabel("Number of Angles")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(angles)
    fig.suptitle(f"{algorithm} — {'Δ ' if delta else ''}Metrics per Dataset", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # save
    folder = config.VISUALIZATION_PATH / algorithm
    makedirs(folder, exist_ok=True)
    filename = f"{'delta_' if delta else ''}bar_charts.png"
    fig.savefig(f"{folder}/{filename}")
    plt.close(fig)

def scatter_plots(algorithm_data: dict, algorithm : str, colors : list, delta : bool = False):
    datasets = algorithm_data["datasets"]
    angles = algorithm_data["angles"]
    metric_values = algorithm_data["metric_values"]
    metrics = metric_values[datasets[0]].keys()

    x = np.arange(len(angles))  
    for metric_name in metrics:
        plt.clf()
        for i, dataset in enumerate(datasets):
            for j, metric in enumerate(metric_values[dataset].keys()):
                if metric != metric_name:
                    continue
                values = metric_values[dataset][metric]
                if delta:
                    values = np.gradient(values, angles)
                plt.scatter(angles.astype(str), values, color=colors[dataset], label=f"{dataset} - {metric}")

        plt.title("Bubble Chart of Metric Values by Angle and Dataset")
        plt.xlabel("Number of Angles")
        plt.ylabel("Metric Value")
        plt.tight_layout()
        plt.legend()
        folder = config.VISUALIZATION_PATH / algorithm
        makedirs(folder, exist_ok=True)
        plt.savefig("{0}/{1}scatter_plot{2}.png".format(folder, metric_name ,"_delta" if delta else ""))


if __name__ == "__main__":
    print("Start visualization")
    metrics_data = MetricsData()
    datasets = metrics_data.get_datasets()
    colors = generate_distinct_colors(datasets)
    for algorithm, data in metrics_data.get_per_algorithm_data("Neighbor metrics"):
        print("Processing algorithm {}".format(algorithm))
        print(data)
        bar_charts(data, algorithm, delta=True)
        scatter_plots(data, algorithm, colors, delta=True)
        bar_charts(data, algorithm, delta=False)
        scatter_plots(data, algorithm, colors, delta=False)
        
            
