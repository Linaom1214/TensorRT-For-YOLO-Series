import sys
sys.path.append('../')
from utils.utils import preproc, vis
from utils.utils import BaseEngine
import numpy as np
import cv2
import time
import os


class Predictor(BaseEngine):
    def __init__(self, engine_path , imgsz=(416,416)):
        super(Predictor, self).__init__(engine_path)
        self.imgsz = imgsz
        self.n_classes = 1
        self.class_names = ["gas_bottle"]
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)


if __name__ == '__main__':
    pred = Predictor(engine_path='yolox.trt')
    img_path = '../src/3.jpg'
    pred.inference(img_path)
