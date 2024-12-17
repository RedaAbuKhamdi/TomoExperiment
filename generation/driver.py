import json

import numpy as np

from typing import List
from os import environ, listdir, path

from dataset import Data
from reconstruction import Reconstruction
from math import pi

class GenerationExperiment:
    def __init__(self, data_path : str):
        self.data = Data([data_path + "/" + p for p in listdir(data_path)])
        self.parse_angle_settings()

    def parse_angle_settings(self):
        try:
            settings = json.loads(environ["angles"])
            self.angles_strategy = settings["strategy"]
            if "max_angles" in settings.keys():
                self.max_angles = settings["max_angles"]
            else:
                self.max_angles = 1000
            self.step = settings.get("step", "")
            print("step = ", self.step)
        except: 
            raise "Incorrect angle settings provided in enviroment variables"

    def binary_angles_stratery(self):
        step = pi / 2
        while 2*pi / step < self.max_angles:
            yield np.arange(0, 2*pi, step), step
            step /= 2
    
    def random_angles_strategy(self):
        number_of_angles = 4
        angles = np.array([0, pi / 2, pi, 3*pi / 2])
        full_angles = np.linspace(0, 2 * pi, self.max_angles, False)

        while True:
            missing_angles = np.setdiff1d(full_angles, angles)
            if missing_angles.size <= self.step:
                break
            chosen_angles = np.random.choice(missing_angles, self.step, False)
            angles = np.concatenate((angles, chosen_angles))
            yield angles.sort(), self.step
        

    def get_angle_generator(self):
        if self.angles_strategy == "binary":
            return self.binary_angles_stratery
        if self.angles_strategy == "random":
            return self.random_angles_strategy

    def run_single_experiment(self, angles : np.ndarray, experiment_number, step):
        settings = self.data.settings
        if (not path.isdir(Reconstruction.get_folder_name(self.data.settings["name"], experiment_number, self.angles_strategy))):
            reconstruction = Reconstruction(self.data)
            reconstruction.calculate_projection(
                settings["detector_spacing"],
                settings["detector_count"],
                angles
            )
            reconstruction.reconstruct(
                settings["iterations"],
                settings["algorithm"],
            )
            reconstruction.save_to_file(experiment_number, self.angles_strategy, angles, step)
            del reconstruction

    def run_experiments(self):
        i = 0
        angle_generator = self.get_angle_generator()
        for angles, step in angle_generator():
            self.run_single_experiment(angles, i, step)
            i += 1


data_paths = json.loads(environ["paths"])

for data_path in data_paths:
    experiment = GenerationExperiment(data_path)
    experiment.run_experiments()