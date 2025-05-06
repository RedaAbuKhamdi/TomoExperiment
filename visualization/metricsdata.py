import sys
import pandas as pd
import numpy as np

from os import environ, listdir, makedirs, path
from matplotlib import pyplot as plt

sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
import config
class MetricsData:
    def __init__(self):
        """
        Initialize the MetricsData object.
        
        Iterates over all the experiments, datasets and algorithms in the
        evaluation path and loads the metrics data for each of them.

        The data has the following structure:
        
        [
            {
                "experiment": "experiment_name",
                "name": "dataset_name",
                "algorithm": "algorithm_name",
                "evaluations": [
                    {
                        "type": "evaluation_type",
                        "path": "evaluation_path"
                    },
                    ...
                ]
            },
            ...
        ]
        
        :return: None
        """
        self.data = []
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
                    self.data.append(dataset_config)
        self.algorithms = None
        self.datasets = None
        
    def get_algorithms(self):
        if self.algorithms is None:
            self.algorithms = []
            for dataset_config in self.data:
                if dataset_config["algorithm"] not in self.algorithms:
                    self.algorithms.append(dataset_config["algorithm"])
        return self.algorithms

    def get_datasets(self):
        if self.datasets is None:
            self.datasets = []
            for dataset_config in self.data:
                if dataset_config["name"] not in self.datasets:
                    self.datasets.append(dataset_config["name"])
        return self.datasets
    
    def get_per_algorithm_data(self, evaluation_type : str):
        for algorithm in self.get_algorithms():
            algorithm_data = {
                "datasets": [],
                "metric_values": {}
            }
            for dataset_config in self.data:
                if dataset_config["algorithm"] == algorithm:
                    algorithm_data["datasets"].append(dataset_config["name"])
                    algorithm_data["metric_values"][dataset_config["name"]] = {}
                    for evaluation in dataset_config["evaluations"]:
                        if evaluation["type"] == evaluation_type:
                            evaluation_data = pd.read_csv(evaluation["path"], header = 0, index_col= 0)
                            if "angles" not in algorithm_data.keys():
                                algorithm_data["angles"] = evaluation_data.index.to_numpy()
                            for metric in evaluation_data:
                                algorithm_data["metric_values"][dataset_config["name"]][metric] = evaluation_data[metric].to_numpy()
            yield algorithm, algorithm_data
    
