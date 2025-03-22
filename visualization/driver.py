import json
import pandas as pd
import numpy as np

from os import environ, listdir, makedirs
from matplotlib import pyplot as plt


data_paths = json.loads(environ["paths"])
prefix = environ["prefix"]
plotData = {

}
for dataset_path in data_paths:
    data = pd.read_csv(
        prefix + dataset_path + "/metrics.csv",
        header = 0, index_col= 0
    ).sort_index(axis = 1, key = lambda x : [int(el) for el in x])
    algorithm = dataset_path.split("/")[0]
    if algorithm not in plotData.keys():
        plotData[algorithm] = {}
    folder = "./results/report/{0}".format(dataset_path)
    for metric, series in data.iterrows():
        if metric not in plotData[algorithm].keys():
            plotData[algorithm][metric] = []
        plotData[algorithm][metric].append({
            "name": dataset_path,
            "data": series
        })

for algorithm in plotData:
    for metric in plotData[algorithm]:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in range(len(plotData[algorithm][metric])):
            dataset = plotData[algorithm][metric][i]
            series = dataset["data"]
            number_of_angles = []
            for index in range(len(series)):
                name = dataset["name"].split("/")[2]
                reconstruction_scheme = dataset["name"].split("/")[1]
                with open( "./results/reconstructions/" + reconstruction_scheme + "/" + name + "/" + str(index) + "/settings.json", "r") as f:
                    settings = json.loads(f.read())
                    number_of_angles.append(len(settings["angles"]["values"]))
            ax.plot(np.array(number_of_angles), series.to_numpy(), label = dataset["name"].replace(algorithm + "/", ""))
        fig.set_dpi(400)
        plt.title(metric)
        plt.legend()
        plt.savefig("./results/report/" + algorithm + "_" + metric +".png")