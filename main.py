import json, subprocess, sys
from os import environ, listdir, getcwd
dataset_paths = json.dumps(["./data/" + path  for path in listdir("./data")])
reconstructions_directory = "./results/reconstructions/"
binarization_directory = "./results/binarizations/"
with open("./experiments.json", "r") as f:
    settings = json.loads(f.read())

def list_reconstruction_paths():
    folder = reconstructions_directory
    result = []
    for experiment in listdir(folder):
        for dataset in  listdir("{}/{}".format(folder, experiment)):
            for angle in listdir("{}/{}/{}".format(folder, experiment, dataset)):
                result.append("{}/{}/{}".format(experiment, dataset, angle))
    return json.dumps(result)

def list_paths_for_evaluation():
    folder = binarization_directory
    result = []
    for algorithm in listdir(folder):
        for experiment in  listdir("{}{}".format(folder, algorithm)):
            for dataset in listdir("{}/{}/{}".format(folder, algorithm, experiment)):
                result.append("{}/{}/{}".format(algorithm, experiment, dataset))     
    return json.dumps(result)

def run_generation(dataset_paths, settings):
    return subprocess.run(["python", "./generation/driver.py"], env={
        "paths": dataset_paths,
        "angles": json.dumps(settings)
    })

def run_binarization():
    env_for_binarization = environ.copy()
    env_for_binarization["paths"] = list_reconstruction_paths()
    env_for_binarization["prefix"] = reconstructions_directory
    env_for_binarization["USERPROFILE"] = getcwd()
    return subprocess.run(["python", "./segmentation/driver.py"], env=env_for_binarization)

def run_evaluation():
    env_for__evaluation = environ.copy()
    env_for__evaluation["paths"] = list_paths_for_evaluation()
    env_for__evaluation["USERPROFILE"] = getcwd()
    env_for__evaluation["prefix"] = binarization_directory
    return subprocess.run(["python", "./evaluation/driver.py"], env=env_for__evaluation)

if len(sys.argv) > 1:
    run_1 = "1" in sys.argv[1]
    run_2 = "2" in sys.argv[1]
    run_3 = "3" in sys.argv[1]
    for i in range(len(settings)):
        if run_1:
            run_generation(dataset_paths, settings[i])
        if run_2:
            run_binarization()
        if run_3:
            run_evaluation()