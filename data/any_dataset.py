import cv2
import json
import torch
import os.path
import numpy as np
import scipy.io as sio
from lib.core.config import cfg
import torchvision.transforms as transforms
from lib.utils.logging import setup_logging

logger = setup_logging(__name__)


class ANYDataset():
    def initialize(self, opt):
        self.data_size = 0

    def __len__(self):
        return self.data_size

    def name(self):
        return 'ANY'

