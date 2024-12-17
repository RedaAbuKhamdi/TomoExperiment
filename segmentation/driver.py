from imagedata import ImageData
import algorithms
import importlib
import json
from os import environ, listdir, path

data_paths = json.loads(environ["paths"])
prefix = environ["prefix"]
for data_path in data_paths:
    data = ImageData(prefix, data_path)
    for algorithm in algorithms.__all__:
        if not (path.isdir(data.get_folder(algorithm))):
            segmented, params = importlib.import_module("algorithms.{}".format(algorithm)).segment(data)
            data.save_result(segmented, params, algorithm)