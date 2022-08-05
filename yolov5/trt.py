import sys
sys.path.append('../')
from utils.utils import preproc, multiclass_nms, vis
from utils.utils import BaseEngine
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
from PIL import Image
import cv2
from utils import *
import time
import os


class Predictor(BaseEngine):
    def __init__(self, engine_path, efficientNMSPlugin = False, imgsz=(640,640)):
        super(Predictor, self).__init__(engine_path,efficientNMSPlugin)
        self.efficientNMSPlugin = efficientNMSPlugin
        self.imgsz = imgsz
        self.n_classes = 80
        self.class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                            'traffic light',
                            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
                            'sheep', 'cow',
                            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                            'suitcase', 'frisbee',
                            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
                            'surfboard',
                            'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
                            'apple',
                            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
                            'couch',
                            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                            'keyboard', 'cell phone',
                            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                            'teddy bear',
                            'hair drier', 'toothbrush']

if __name__ == '__main__':
    pred = Predictor(engine_path='../yolov5s.trt')
    img_path = '../src/2.jpg'
    pred.inference(img_path)
