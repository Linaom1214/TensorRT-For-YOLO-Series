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
    def __init__(self, engine_path , imgsz=(640,640)):
        super(Predictor, self).__init__(engine_path)
        self.imgsz = imgsz
        self.n_classes = 1
        self.class_names = ["gas_bottle"]

if __name__ == '__main__':
    pred = Predictor(engine_path='yolov5.trt')
    img_path = '../src/3.jpg'
    pred.inference(img_path)
