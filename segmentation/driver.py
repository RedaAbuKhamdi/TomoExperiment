from imagedata import ImageData
import algorithms
import importlib
import json
import re
from os import environ, listdir, path
def natural_sort_key(s, _nsre=re.compile(r'(\d+)')):
    return [int(text) if text.isdigit() else text.lower()
            for text in _nsre.split(s)]
data_paths = sorted(json.loads(environ["paths"]), key=natural_sort_key)
prefix = environ["prefix"]
for data_path in data_paths:
    data = ImageData(prefix, data_path)
    for algorithm in algorithms.__all__:
        if not (path.isdir(data.get_folder(algorithm))):
            segmented, params = importlib.import_module("algorithms.{}".format(algorithm)).segment(data)
            data.save_result(segmented, params, algorithm)