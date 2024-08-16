# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import json
import cv2
import random

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import pycocotools.mask as mask_util

class ObjectSegmentation:
    def __init__(self, model_name="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"):
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file(model_name))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_name)
        self.predictor = DefaultPredictor(self.cfg)

