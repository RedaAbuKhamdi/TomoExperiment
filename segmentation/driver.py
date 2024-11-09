from imagedata import ImageData
import algorithms
import importlib
import json
from os import environ, listdir

data_paths = json.loads(environ["paths"])
prefix = environ["prefix"]
for path in data_paths:
    data = ImageData(prefix, path)
    for algorithm in algorithms.__all__:
        segmented, params = importlib.import_module("algorithms.{}".format(algorithm)).segment(data)
        data.save_result(segmented, params, algorithm)