# Import all the necessary libs
import astra
import os

import numpy as np

from PIL import Image
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
        if self.data.volumetric:
            proj_geom = astra.create_proj_geom('parallel3d', 
                                            det_spacing["x"], 
                                            det_spacing["y"], 
                                            det_count["rows"], 
                                            det_count["columns"], 
                                            angles)
            sinogram_id = self.data.calculate_sinogram(None, proj_geom)
        else:
            proj_geom = astra.create_proj_geom('parallel', det_spacing, det_count, angles)
            proj_id = astra.create_projector('cuda', proj_geom, self.data.data_geometry)
            sinogram_id = self.data.calculate_sinogram(proj_id, proj_geom)
            astra.projector.delete(proj_id)
        self.sinogram = sinogram_id
        return sinogram_id

    def reconstruct(self, iterations : int,  algorithm : str):
        if self.sinogram is None:
            raise "Run calculate_projection method first!"
        self.rec_id = astra.data3d.create("-vol", self.data.data_geometry) if self.data.volumetric else astra.data2d.create("-vol", self.data.data_geometry)   
        cfg = astra.astra_dict(algorithm)
        cfg['ReconstructionDataId'] = self.rec_id 
        cfg['ProjectionDataId'] = self.sinogram
        algorithm_id = astra.algorithm.create(cfg)
        astra.algorithm.run(algorithm_id, iterations)
        astra.algorithm.delete(algorithm_id)
        return self.rec_id
    
    def get_folder_name(self, experiment_number : int, strategy : str):
        return "./results/reconstructions/{0}/{1}/{2}".format(strategy, self.data.settings["name"] , experiment_number)

    def save_to_file(self, experiment_number : int, strategy : str, angles : np.array, step: str):
        if (self.rec_id is None):
            raise "No reconstruction has been run"
        reconstruction = astra.data3d.get(self.rec_id) if self.data.volumetric else astra.data2d.get(self.rec_id)
        folder = self.get_folder_name(experiment_number, strategy)
        os.makedirs(folder, exist_ok=True)
        io.imsave("{0}/reconstruction_experiment.tiff".format(folder),  reconstruction)
        np.save("{0}/image.npy".format(folder), reconstruction, False)
        self.data.save_settings(folder, angles, strategy, step)
    
    def clean_up(self):
        if self.rec_id is not None:
            if self.data.volumetric:
                astra.data3d.delete(self.rec_id)
            else:
                astra.data2d.delete(self.rec_id)
        if self.sinogram is not None:
            if self.data.volumetric:
                astra.data3d.delete(self.sinogram)
            else:
                astra.data2d.delete(self.sinogram)
    
    def __del__(self):
        self.clean_up()
        self.data = None

        