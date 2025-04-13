import json
import sys
import pathlib
import os
import pandas as pd
import numpy as np

from os import environ, listdir, makedirs, path
from matplotlib import pyplot as plt

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

def process_dataset(dataset_config):
    base_path = config.VISUALIZATION_PATH / dataset_config["experiment"] / dataset_config["name"]
    for evaluation in dataset_config["evaluations"]:
        evaluation_data = pd.read_csv(evaluation["path"], header = 0, index_col= 0)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for metric in evaluation_data:
            series = evaluation_data[metric]
            ax.scatter(series.index.to_numpy(), series.to_numpy(), label = metric)
        ax.set_title(evaluation["type"])
        ax.legend()
        makedirs(base_path, exist_ok=True)
        fig.savefig(base_path / "{} {}.png".format(evaluation["type"], dataset_config["algorithm"]))
        plt.close()


if __name__ == "__main__":
    print("Start visualization")
    for dataset_config in iterate_datasets():
        process_dataset(dataset_config)