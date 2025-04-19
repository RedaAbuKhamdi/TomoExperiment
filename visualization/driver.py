import json
import sys
import pathlib
import os
import pandas as pd
import numpy as np

from os import environ, listdir, makedirs, path
from matplotlib import pyplot as plt

from metricsdata import MetricsData

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

def bar_charts(algorithm_data: dict, algorithm : str, delta : bool = False):
    import numpy as np
    import matplotlib.pyplot as plt

    datasets = algorithm_data["datasets"]
    angles = algorithm_data["angles"]
    metric_values = algorithm_data["metric_values"]

    x = np.arange(len(angles))  
    num_bars_per_group = len(datasets) * len(next(iter(metric_values.values())).keys())
    bar_width = 0.8 / num_bars_per_group 

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, dataset in enumerate(datasets):
        for j, metric in enumerate(metric_values[dataset].keys()):
            values = metric_values[dataset][metric]
            if delta:
                values = np.gradient(values, angles)
            index_in_group = i * len(metric_values[dataset]) + j
            offset = (index_in_group - num_bars_per_group / 2) * bar_width + bar_width / 2
            ax.bar(x + offset, values, bar_width, label=f"{dataset} - {metric}")

    ax.set_xlabel("Number of Angles")
    ax.set_ylabel("Metric Value")
    ax.set_title("Metrics per Dataset and Angle")
    ax.set_xticks(x)
    ax.set_xticklabels(angles)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    
    folder = config.VISUALIZATION_PATH / algorithm
    makedirs(folder, exist_ok=True)
    fig.savefig("{0}/{1}bar_chart.png".format(folder, "delta_" if delta else ""))


if __name__ == "__main__":
    print("Start visualization")
    metrics_data = MetricsData()
    for algorithm, data in metrics_data.get_per_algorithm_data("Neighbor metrics"):
        print("Processing algorithm {}".format(algorithm))
        bar_charts(data, algorithm, delta=True)
        bar_charts(data, algorithm, delta=False)
        
            
