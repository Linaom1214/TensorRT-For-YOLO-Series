import sys
sys.path.append('../')
from utils.utils import preproc, vis
from utils.utils import BaseEngine
import numpy as np
import cv2
import time
import os


class Predictor(BaseEngine):
    def __init__(self, engine_path, efficientNMSPlugin=False, imgsz=(640,640)):
        super(Predictor, self).__init__(engine_path, efficientNMSPlugin)
        self.efficientNMSPlugin = efficientNMSPlugin
        self.imgsz = imgsz
        self.n_classes = 80
        self.class_names = [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush' ]


if __name__ == '__main__':
    pred = Predictor(engine_path='./../yolov7-tiny-nms.trt', efficientNMSPlugin=True)  # efficientNMSPlugin only support yolov7
    img_path = '../src/3.jpg'
    origin_img = pred.inference(img_path, conf=0.3)
    cv2.imwrite("%s_yolov7.jpg" % os.path.splitext(
        os.path.split(img_path)[-1])[0], origin_img)
    pred.detect_video('../src/video1.mp4') # set 0 use a webcam
    pred.get_fps()
