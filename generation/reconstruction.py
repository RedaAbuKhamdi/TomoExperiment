# Import all the necessary libs
import astra
import os

import numpy as np

from dataset import Data
from skimage import io
import cupyx.scipy.ndimage as cpndi
import cupy as cp

def add_shading_and_noise_in_chunks(sino_id, sino_shape, noise_std, shade_frac=0.3, chunk_size=32, gaussian = None, shading_amp = None):
        """
        Reads your full sinogram from ASTRA, processes it in GPU‐backed chunks of `chunk_size` slices,
        then writes the final result back to ASTRA in one go.
        """
        D, H, W = sino_shape

        # 1) pull the entire sinogram into host RAM
        sino = astra.data3d.get(sino_id).astype(np.float32)

        # 2) compute its range once
        sino_min, sino_max = sino.min(), sino.max()
        sino_range = float(sino_max - sino_min)
        shading_amp = shading_amp if shading_amp is not None else shade_frac * sino_range
        sigma_y, sigma_x = H/4, W/4
        gaussian = gaussian if gaussian is not None else noise_std * sino_range

        # 3) process chunk by chunk
        for start in range(0, D, chunk_size):
            stop = min(start + chunk_size, D)
            sz = stop - start

            # move just this slab to GPU
            sino_gpu = cp.array(sino[start:stop], dtype=cp.float32)

            # a) low‐freq shading
            rnd = cp.random.random((sz, H, W), dtype=cp.float32)
            smooth = cpndi.gaussian_filter(rnd,
                                        sigma=(0, sigma_y, sigma_x),
                                        mode='reflect')
            # per‐slice normalize to [0,1]
            mn = smooth.min(axis=(1,2), keepdims=True)
            mx = smooth.max(axis=(1,2), keepdims=True)
            smooth = (smooth - mn) / (mx - mn + 1e-8)
            bias = smooth * shading_amp

            # b) fine noise
            noise = cp.random.normal(0, gaussian,
                                    size=(sz, H, W), dtype=cp.float32)

            # combine
            sino_gpu += bias
            sino_gpu += noise

            # copy that chunk back to host array
            sino[start:stop] = cp.asnumpy(sino_gpu)

            # free GPU memory immediately
            del sino_gpu, rnd, smooth, bias, noise
            cp._default_memory_pool.free_all_blocks()

        # 4) write full updated sinogram back into ASTRA
        astra.data3d.store(sino_id, sino)
        return noise_std * sino_range, shading_amp

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
        if self.sinogram is not None:
            raise RuntimeError("Only one run is allowed per object. Create a new Reconstruction object")

        # 1) build geometry & forward project (on GPU)
        proj_geom = astra.create_proj_geom(
            'parallel3d',
            det_spacing["x"],
            det_spacing["y"],
            det_count["rows"],
            det_count["columns"],
            angles
        )
        sino_id = self.data.calculate_sinogram(None, proj_geom)
        sino_shape = astra.data3d.get(sino_id).shape

        # compute absolute noise‐std (as before)
        print("Start")
        noise_level, shading_level = self.data.get_noise_level()

        if shading_level > 0 or noise_level > 0:
            gaussian = self.data.get_noise_value("gaussian")
            shading = self.data.get_noise_value("shading")
            # do chunked GPU shading+noise
            gaussian, shading = add_shading_and_noise_in_chunks(sino_id,
                                            sino_shape,
                                            noise_level,
                                            shading_level,
                                            chunk_size=32,
                                            gaussian = gaussian,
                                            shading_amp = shading)
            
            if gaussian > 0:
                self.data.set_noise_data("gaussian", gaussian)
            if shading > 0:
                self.data.set_noise_data("shading", shading)
        
        print("Finish")
        self.noise = noise_level
        self.sinogram = sino_id

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

        