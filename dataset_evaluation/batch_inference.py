import os
import json
import cv2
import numpy as np
from dataset_evaluation.evaluator import Evaluator
from tools.demo import Predictor
from tools.demo import image_demo
from yolox.models.yolox import YOLOX
import torch
from yolox.exp import get_exp
from exps.default.yolox_tiny import MyExp
import time

DATA_PATH = "./Test_Black/test"
ANNOT_PATH = "ID2TE_Black.json"
MODEL_PATH = "YOLOX_outputs/yolox_tiny/BG112.pth"

exps = get_exp("exps/default/yolox_tiny.py", "yolox_tiny")
model = exps.get_model()
model.eval()
ckpt = torch.load(MODEL_PATH, map_location = 'cpu')
model.load_state_dict(ckpt["model"])
predictor = Predictor(model, exps)

image_demo(predictor, "YOLOX_outputs/yolox_tiny/vis_res", DATA_PATH, 0, True)


