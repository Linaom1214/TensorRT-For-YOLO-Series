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
    def inference(self, img_path):
        origin_img = cv2.imread(img_path)
        img, ratio = preproc(origin_img, self.imgsz, None, None)
        data = self.infer(img)
        predictions = np.reshape(data[-1], (1, -1, int(5+self.n_classes)))[0]
        dets = self.postprocess(predictions,ratio)
        if dets is not None:
            final_boxes, final_scores, final_cls_inds = dets[:,
                                                             :4], dets[:, 4], dets[:, 5]
            origin_img = vis(origin_img, final_boxes, final_scores, final_cls_inds,
                             conf=0.3, class_names=self.class_names)

        cv2.imwrite("%s_yolov5.jpg" % os.path.splitext(
            os.path.split(img_path)[-1])[0], origin_img)


if __name__ == '__main__':
    pred = Predictor(engine_path='yolov5.trt')
    img_path = '../imgs/3.jpg'
    for _ in range(5):
        t1 = time.time()
        pred.inference(img_path)
        print((time.time() - t1)*1000)
