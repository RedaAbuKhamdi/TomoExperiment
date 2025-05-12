import tqdm
import algorithms
import importlib
import json
import re
import sys
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from os import environ, path, makedirs
from imagedata import ImageData

sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
import config

def natural_sort_key(s, _nsre=re.compile(r'(\d+)')):
    return [int(text) if text.isdigit() else text.lower()
            for text in _nsre.split(s)]

def run_segmentation(data_paths, used_algorithms):
    parameters = {}
    for index, data_path in enumerate(reversed(data_paths)):
        data = ImageData(data_path)
        print("Processing {0}".format(data.settings["name"]))
        if (data.settings["name"] not in parameters.keys()):
            parameters[data.settings["name"]] = {}
        for algorithm in used_algorithms.intersection(algorithms.__all__):
            saved_parameters = data.check_if_done(algorithm)
            if index == 0 and saved_parameters is not None:
                parameters[data.settings["name"]][algorithm] = saved_parameters
            if not (path.isdir(data.get_folder(algorithm))):
                has_parameters = algorithm in parameters[data.settings["name"]].keys()
                segmented, params = importlib.import_module("algorithms.{}".format(algorithm)).segment(data, parameters[data.settings["name"]][algorithm] if has_parameters else None)
                if not has_parameters:
                    parameters[data.settings["name"]][algorithm] = params
                data.save_result(segmented, params, algorithm)

data_paths = sorted(json.loads(environ["paths"]), key=natural_sort_key)
used_algorithms = set(json.loads(environ["algorithms"]))
experiment = environ["experiment"] == "True"
run_segmentation(data_paths, used_algorithms)