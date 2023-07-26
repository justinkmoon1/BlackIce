"""
TODO
이미지 인풋 사이즈가 맞는지
postprocessing 시 음수 bounding box 위치 정보가 안 나오도록
"""
import os
import json
import cv2
import numpy as np
from dataset_evaluation.evaluator import Evaluator
from tools.demo import Predictor
from yolox.models.yolox import YOLOX
import torch
from yolox.exp import get_exp
from exps.default.yolox_tiny import MyExp
def postprocessing(inference_results, ratio, input_shape, nms_thr=0.45, score_thr=0.3):
    predictions = demo_postprocess(inference_results, input_shape)[0]

    boxes = predictions[:, :4]
    scores = predictions[:, 4:5] * predictions[:, 5:]

    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
    boxes_xyxy /= ratio

    dets = multiclass_nms(boxes_xyxy, scores, nms_thr=nms_thr, score_thr=score_thr)
    
    return dets

def demo_postprocess(outputs, img_size, p6=False):

    grids = []
    expanded_strides = []

    if not p6:
        strides = [8, 16, 32]
    else:
        strides = [8, 16, 32, 64]

    hsizes = [img_size[0] // stride for stride in strides]
    wsizes = [img_size[1] // stride for stride in strides]

    for hsize, wsize, stride in zip(hsizes, wsizes, strides):
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride))

    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)
    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

    return outputs

def multiclass_nms(boxes, scores, nms_thr, score_thr, class_agnostic=True):
    """Multiclass NMS implemented in Numpy"""
    if class_agnostic:
        nms_method = multiclass_nms_class_agnostic
    else:
        nms_method = multiclass_nms_class_aware
    return nms_method(boxes, scores, nms_thr, score_thr) 

def multiclass_nms_class_agnostic(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy. Class-agnostic version."""
    cls_inds = scores.argmax(1)
    cls_scores = scores[np.arange(len(cls_inds)), cls_inds]

    valid_score_mask = cls_scores > score_thr
    if valid_score_mask.sum() == 0:
        return None
    valid_scores = cls_scores[valid_score_mask]
    valid_boxes = boxes[valid_score_mask]
    valid_cls_inds = cls_inds[valid_score_mask]
    keep = nms(valid_boxes, valid_scores, nms_thr)
    if keep:
        dets = np.concatenate(
            [valid_boxes[keep], valid_scores[keep, None], valid_cls_inds[keep, None]], 1
        )
    return dets


def nms(boxes, scores, nms_thr):
    """Single class NMS implemented in Numpy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep 

def get_trained_model(experiment, weights):
    model = experiment.get_model()

    model.cuda()

    model.eval()
    model.head.training = False
    model.training = False

    best_weights = torch.load(weights)
    model.load_state_dict(best_weights['model'])

    return model

# 경로 설정
DATA_PATH = "C:/Users/Justin Moon/BlackIce/blackice3.v1i.coco/train"
ANNOT_PATH = "C:/Users/Justin Moon/BlackIce/blackice3.v1i.coco/annot/_annotations.coco.json"
MODEL_PATH = "C:/Users/Justin Moon/BlackIce/YOLOX_outputs/yolox_tiny/best_ckpt.pth"

# data read
with open(ANNOT_PATH, 'r') as f:
    json_data = json.load(f)
img_list = os.listdir(DATA_PATH)

# evaluator 객체
evaluator = Evaluator(DATA_PATH, ANNOT_PATH, MODEL_PATH)

# model load
# model = YOLOX()
# saved_checkpoint = torch.load(MODEL_PATH)
# model.load_state_dict(saved_checkpoint, strict = False)
# model.eval()
exps = get_exp("C:/Users/Justin Moon/BlackIce/exps/default/yolox_tiny.py", "yolox_tiny")
model = exps.get_model()
model.eval()
ckpt = torch.load(MODEL_PATH, map_location = 'cpu')
model.load_state_dict(ckpt["model"])
#ckpt = torch.load(MODEL_PATH, map_location="gpu")
# load the model state dict
# model = get_trained_model(exps, MODEL_PATH)
# model.to(torch.device('cuda'))
#model.load_state_dict(ckpt["model"])
#exps = MyExp()


#model.to("cuda")

predictor = Predictor(model, exps)

# 모든 이미지 하나에 대한 gt, pred 비교해서 evaluator 객체에 저장하기
for i in range(len(img_list)):
    # if i > 0:
    #     break
    # gt file read
    actual_bbox, actual_class = evaluator.read_data(DATA_PATH + "/" + img_list[i], i)

    # model inference - 학습할 때 설정했던 거로 evaluate 하거나, 실제 application에서 쓸 thr 사용
    prediction_list, info = predictor.inference(DATA_PATH + "/" + img_list[i])
    #print(prediction_list.shape)
    #postprocessing(prediction_list, 1, (416, 416), 0.6, 0.5)

    prediction_bbox, prediction_class = evaluator.prediction_process(prediction_list, info)
    #print(actual_bbox, prediction_bbox)
    # 단일 이미지 하나에 대한 gt, pred 비교해서 evaluator 객체에 저장하기
    evaluator.put_data(prediction_bbox, prediction_class, actual_bbox, actual_class)


print(evaluator.cnts)
# 전체 metric 계산
#ap, raw_metric, f1_score = evaluator.get_results()


