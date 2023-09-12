import os
import json
import cv2
import numpy as np
from evaluator import Evaluator
from yolox.tools.demo import Predictor
from yolox.tools.demo import custom_image_demo
from yolox.models.yolox import YOLOX
import torch
from yolox.exp import get_exp
import time

DATA_PATH = "Test Set Mixed Final/train"
ANNOT_PATH = "Test Set Mixed Final/annotations/_annotations.coco.json"
MODEL_PATH = "YOLOX_outputs/yolox_tiny/BG220.pth"

exps = get_exp("exps/default/yolox_tiny.py", "yolox_tiny")
model = exps.get_model()
model.eval()
ckpt = torch.load(MODEL_PATH, map_location = 'cpu')
model.load_state_dict(ckpt["model"])
predictor = Predictor(model, exps)

custom_image_demo(predictor, "YOLOX_outputs/yolox_tiny/vis_res", DATA_PATH, "BG220_Mixed", True)

