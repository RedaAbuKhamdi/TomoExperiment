import json, subprocess, sys
from os import environ, listdir, getcwd, remove

# Set your conda environment name here.
CONDA_ENV = "base"

dataset_paths = json.dumps(["./data/" + path for path in listdir("./data")])
reconstructions_directory = "./results/reconstructions/"
binarization_directory = "./results/binarizations/"
evaluation_directory = "./results/evaluation/"

with open("./experiments.json", "r") as f:
    settings = json.loads(f.read())

def list_reconstruction_paths():
    folder = reconstructions_directory
    result = []
    for experiment in listdir(folder):
        for dataset in listdir("{}/{}".format(folder, experiment)):
            for angle in listdir("{}/{}/{}".format(folder, experiment, dataset)):
                result.append("{}/{}/{}".format(experiment, dataset, angle))
    return json.dumps(result)

def list_paths_for_evaluation():
    folder = binarization_directory
    result = []
    for algorithm in listdir(folder):
        for experiment in listdir("{}/{}".format(folder, algorithm)):
            for dataset in listdir("{}/{}/{}".format(folder, algorithm, experiment)):
                result.append("{}/{}/{}".format(algorithm, experiment, dataset))
    return json.dumps(result)

def list_paths_for_visualization():
    folder = evaluation_directory
    result = []
    for algorithm in listdir(folder):
        for experiment in listdir("{}/{}".format(folder, algorithm)):
            for dataset in listdir("{}/{}/{}".format(folder, algorithm, experiment)):
                result.append("{}/{}/{}".format(algorithm, experiment, dataset))
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
    env_for_binarization["prefix"] = reconstructions_directory
    env_for_binarization["algorithms"] = json.dumps(["niblack_3d", "otsu"])
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
    env_for_evaluation["paths"] = list_paths_for_evaluation()
    env_for_evaluation["USERPROFILE"] = getcwd()
    env_for_evaluation["prefix"] = binarization_directory
    return subprocess.run(
        ["conda", "run", "--live-stream", "-n", CONDA_ENV, "python", "-u", "./evaluation/driver.py"],
        env=env_for_evaluation,
        stdout=sys.stdout,
        stderr=sys.stderr,
        check=True
    )

def run_visualization():
    env_for_visualization = environ.copy()
    env_for_visualization["paths"] = list_paths_for_visualization()
    env_for_visualization["USERPROFILE"] = getcwd()
    env_for_visualization["prefix"] = evaluation_directory
    return subprocess.run(
        ["conda", "run", "--live-stream", "-n", CONDA_ENV, "python", "-u", "./visualization/driver.py"],
        env=env_for_visualization,
        stdout=sys.stdout,
        stderr=sys.stderr,
        check=True
    )

def clean_up():
    folder = reconstructions_directory
    search = ["cumsum.npy", "square_cumsum.npy", "mean", "std"]
    for experiment in listdir(folder):
        for dataset in listdir("{}/{}".format(folder, experiment)):
            for angle in listdir("{}/{}/{}".format(folder, experiment, dataset)):
                for file in listdir("{}/{}/{}/{}".format(folder, experiment, dataset, angle)):
                    for s in search:
                        if s in file:
                            print("{}/{}/{}/{}/{}".format(folder, experiment, dataset, angle, file))
                            remove("{}/{}/{}/{}/{}".format(folder, experiment, dataset, angle, file))

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
            clean_up()
