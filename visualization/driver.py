import json
import os
import pandas as pd
import numpy as np

from os import environ, listdir, makedirs
from matplotlib import pyplot as plt


data_paths = json.loads(environ["paths"])
prefix = environ["prefix"]
plotData = {}

for dataset_path in data_paths:
    print(dataset_path)
    data = pd.read_csv(
        prefix + dataset_path + "/metrics.csv",
        header = 0, index_col= 0
    ).sort_index(axis = 1, key = lambda x : [int(el) for el in x])

    algorithm = dataset_path.split("/")[0]
    name = dataset_path.split("/")[-1]

    if name not in plotData.keys():
        plotData[name] = {}
    plotData[name][algorithm] = {}

    for metric, series in data.iterrows():
        plotData[name][algorithm][metric] = {
            "name": dataset_path,
            "data": series
        }

for dataset in plotData:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for algorithm in plotData[dataset]:
        for i, metric in enumerate(plotData[dataset][algorithm]):
            current_data = plotData[dataset][algorithm][metric]
            series = current_data["data"]
            number_of_angles = []
            for index in range(len(series)):
                name = current_data["name"].split("/")[2]
                reconstruction_scheme = current_data["name"].split("/")[1]
                with open( "./results/reconstructions/" + reconstruction_scheme + "/" + name + "/" + str(index) + "/settings.json", "r") as f:
                    settings = json.loads(f.read())
                    number_of_angles.append(len(settings["angles"]["values"]))
            ax.scatter(np.array(number_of_angles), series.to_numpy(), label = algorithm + "_" + metric)
    fig.set_dpi(400)
    plt.title(dataset)
    plt.legend()
    
    os.makedirs("./results/report/" , exist_ok=True)
    plt.savefig("./results/report/" + dataset + ".png")