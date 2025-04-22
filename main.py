import json, subprocess, sys
from os import environ, listdir, getcwd, remove
from pathlib import Path
from shutil  import rmtree

import config
# Set your conda environment name here.
CONDA_ENV = "base"

dataset_paths = json.dumps(["./data/" + path for path in listdir("./data")])

with open("./experiments.json", "r") as f:
    settings = json.loads(f.read())

def list_reconstruction_paths():
    result = []
    folder = config.RECONSTRUCTION_PATH
    for experiment in folder.glob('*'):
        for dataset in (folder / experiment).glob('*'):
            for angle in (folder / experiment / dataset).glob('*'):
                result.append((folder / experiment / dataset / angle).as_posix())
    return json.dumps(result)

def list_results_paths(folder : Path):
    result = []
    for algorithm in folder.glob('*'):
        for experiment in (folder / algorithm).glob('*'):
            for dataset in (folder / algorithm / experiment).glob('*'):
                result.append((folder / algorithm / experiment / dataset).as_posix())
    return json.dumps(result)

def run_generation(dataset_paths, settings):
    env_gen = environ.copy()
    env_gen["paths"] = dataset_paths
    env_gen["angles"] = json.dumps(settings)
    return subprocess.run(
        ["conda", "run", "--live-stream", "-n", CONDA_ENV, "python", "-u", "./generation/driver.py"],
        env=env_gen,
        stdout=sys.stdout,
        stderr=sys.stderr,
        check=True
    )

def run_binarization():
    env_for_binarization = environ.copy()
    env_for_binarization["paths"] = list_reconstruction_paths()
    env_for_binarization["prefix"] = config.RECONSTRUCTION_PATH.as_posix()
    env_for_binarization["algorithms"] = json.dumps(["niblack_3d", "otsu", "brute"])
    env_for_binarization["USERPROFILE"] = getcwd()
    return subprocess.run(
        ["conda", "run", "--live-stream", "-n", CONDA_ENV, "python", "-u", "./segmentation/driver.py"],
        env=env_for_binarization,
        stdout=sys.stdout,
        stderr=sys.stderr,
        check=True
    )

def run_evaluation():
    env_for_evaluation = environ.copy()
    env_for_evaluation["paths"] = list_results_paths(config.SEGMENTATION_PATH)
    env_for_evaluation["USERPROFILE"] = getcwd()
    env_for_evaluation["prefix"] = config.SEGMENTATION_PATH.as_posix()
    return subprocess.run(
        ["conda", "run", "--live-stream", "-n", CONDA_ENV, "python", "-u", "./evaluation/driver.py"],
        env=env_for_evaluation,
        stdout=sys.stdout,
        stderr=sys.stderr,
        check=True
    )

def run_visualization():
    env_for_visualization = environ.copy()
    env_for_visualization["USERPROFILE"] = getcwd()
    return subprocess.run(
        ["conda", "run", "--live-stream", "-n", CONDA_ENV, "python", "-u", "./visualization/driver.py"],
        env=env_for_visualization,
        stdout=sys.stdout,
        stderr=sys.stderr,
        check=True
    )
def run_experiment(env : dict, path : str):
    return subprocess.run(
        ["conda", "run", "--live-stream", "-n", CONDA_ENV, "python", "-u", path],
        env=env,
        stdout=sys.stdout,
        stderr=sys.stderr,
        check=True
    )
def clean_up():
    folder = config.RECONSTRUCTION_PATH.as_posix()
    search = ["mean", "std"]
    for experiment in listdir(folder):
        for dataset in listdir("{}/{}".format(folder, experiment)):
            for angle in listdir("{}/{}/{}".format(folder, experiment, dataset)):
                for file in listdir("{}/{}/{}/{}".format(folder, experiment, dataset, angle)):
                    for s in search:
                        if s in file:
                            remove("{}/{}/{}/{}/{}".format(folder, experiment, dataset, angle, file))
def delete_algorithm(algorithm):
    for experiment in config.SEGMENTATION_PATH.iterdir():
        for dataset in experiment.iterdir():
            for angle in dataset.iterdir():
                for algo in angle.iterdir():
                    if algorithm in str(algo.as_posix()):
                        rmtree(algo)

if len(sys.argv) > 1:
    for i in range(len(settings)):
        if "1" in sys.argv[1]:
            run_generation(dataset_paths, settings[i])
        if "2" in sys.argv[1]:
            print("Running binarization")
            run_binarization()
        if "3" in sys.argv[1]:
            run_evaluation()
        if "4" in sys.argv[1]:
            run_visualization()
        if "clean" in sys.argv[1]:
            if len(sys.argv) > 2:
                delete_algorithm(sys.argv[2])
            else:
                clean_up()
