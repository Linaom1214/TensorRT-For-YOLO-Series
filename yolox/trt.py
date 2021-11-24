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
    def inference(self, img_path):
        origin_img = cv2.imread(img_path)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        img, ratio = preproc(origin_img, self.imgsz, mean, std)
        data = self.infer(img)
        predictions = np.reshape(data, (1, -1, int(5+self.n_classes)))[0]
        dets = self.postprocess(predictions,ratio)
        if dets is not None:
            final_boxes, final_scores, final_cls_inds = dets[:,
                                                             :4], dets[:, 4], dets[:, 5]
            origin_img = vis(origin_img, final_boxes, final_scores, final_cls_inds,
                             conf=0.3, class_names=self.class_names)

        cv2.imwrite("%s_yolox.jpg" % os.path.splitext(
            os.path.split(img_path)[-1])[0], origin_img)


if __name__ == '__main__':
    pred = Predictor(engine_path='yolox.trt')
    img_path = '../imgs/3.jpg'
    for _ in range(5):
        t1 = time.time()
        pred.inference(img_path)
        print((time.time() - t1)*1000)
