# Import all the necessary libs
import astra
import os

import numpy as np

from dataset import Data
from skimage import io

class Reconstruction:
    def __init__(self, data : Data):
        self.data = data
        self.cfg =  None
        self.reconstruction = None
        self.algorithm_id = None
        self.sinogram = None
    
    def calculate_projection(
            self,
            det_spacing, 
            det_count,
            angles
    ):
        if (self.sinogram is not None):
            raise "Only one run is allowed per object. Create a new Reconstruction object"
        proj_geom = astra.create_proj_geom('parallel3d', 
                                        det_spacing["x"], 
                                        det_spacing["y"], 
                                        det_count["rows"], 
                                        det_count["columns"], 
                                        angles)
        sinogram_id = self.data.calculate_sinogram(None, proj_geom)
        
        phantom = astra.data3d.get(sinogram_id)
        self.noise = np.random.uniform(0.1, 0.5)
        phantom = phantom + np.random.normal(0, self.noise)
        astra.data3d.store(sinogram_id, phantom)

        self.sinogram = sinogram_id
        return sinogram_id

    def reconstruct(self, iterations : int,  algorithm : str):
        if self.sinogram is None:
            raise "Run calculate_projection method first!"
        self.rec_id = astra.data3d.create("-vol", self.data.data_geometry)
        cfg = astra.astra_dict(algorithm)
        cfg['ReconstructionDataId'] = self.rec_id 
        cfg['ProjectionDataId'] = self.sinogram
        algorithm_id = astra.algorithm.create(cfg)
        astra.algorithm.run(algorithm_id, iterations)
        astra.algorithm.delete(algorithm_id)
        return self.rec_id
    
    def get_folder_name(name, experiment_number : int, strategy : str):
        return "./results/reconstructions/{0}/{1}/{2}".format(strategy, name , experiment_number)

    def save_to_file(self, experiment_number : int, strategy : str, angles : np.array, step: str):
        if (self.rec_id is None):
            raise "No reconstruction has been run"
        reconstruction = astra.data3d.get(self.rec_id)
        folder = Reconstruction.get_folder_name(self.data.settings['name'], experiment_number, strategy)
        os.makedirs(folder, exist_ok=True)
        io.imsave("{0}/reconstruction_experiment.tiff".format(folder),  reconstruction)
        self.data.save_settings(folder, angles, strategy, step, self.noise)
    
    def clean_up(self):
        if self.rec_id is not None:
           astra.data3d.delete(self.rec_id)
        if self.sinogram is not None:
            astra.data3d.delete(self.sinogram)
    
    def __del__(self):
        self.clean_up()
        self.data = None

        