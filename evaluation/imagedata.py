import json
import sys

import numpy as np
from skimage import io
from os import path, makedirs

sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from common.imagedata import ImageDataBase

class ImageData(ImageDataBase):
    def __init__(self,  path):
        super().__init__(path)
        self.strategy = path.split("/")[-4]