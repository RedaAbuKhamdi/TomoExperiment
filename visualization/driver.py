import json
import pandas as pd

from os import environ, listdir, makedirs
from matplotlib import pyplot


data_paths = json.loads(environ["paths"])
prefix = environ["prefix"]
for dataset_path in data_paths:
    data = pd.read_csv(
        prefix + dataset_path + "/metrics.csv",
        header = 0, index_col= 0
    )
    folder = "./results/report/{0}".format(dataset_path)
    makedirs(folder, exist_ok=True)
    plot = data.T.plot(
        title="Metrics for {}".format(dataset_path.split("/")[-1]),
        xlabel = "Angle step",
        ylabel = "Metric value"
        )
    plot.get_figure().savefig(folder + "/" + "metrics_plot.png")