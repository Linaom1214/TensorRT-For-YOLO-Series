import cv2
import torch
import random
import time
import numpy as np
import tensorrt as trt
from pathlib import Path
from collections import OrderedDict,namedtuple
import matplotlib.pyplot as plt


w = '/home/ikilbas/USA/yolo/ppe-testing-training/yolov7/tensorrt-python/yolov7_4.trt'
device = torch.device('cuda:0')
batch_size = 4

# Infer TensorRT Engine
Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
logger = trt.Logger(trt.Logger.INFO)
trt.init_libnvinfer_plugins(logger, namespace="")
with open(w, 'rb') as f, trt.Runtime(logger) as runtime:
    model = runtime.deserialize_cuda_engine(f.read())
bindings = OrderedDict()
for index in range(model.num_bindings):
    name = model.get_binding_name(index)
    dtype = trt.nptype(model.get_binding_dtype(index))
    shape = tuple(model.get_binding_shape(index))
    data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(device)
    bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
context = model.create_execution_context()


# warmup for 10 times
for _ in range(10):
    tmp = torch.randn(batch_size,3,640,640).to(device)
    binding_addrs['images'] = int(tmp.data_ptr())
    context.execute_v2(list(binding_addrs.values()))


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[1:3]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = np.stack([
        	cv2.resize(im[i_im], new_unpad, interpolation=cv2.INTER_LINEAR)
        	for i_im in range(len(im))
        ], axis=0)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = np.stack([ 
    	cv2.copyMakeBorder(im[i_im], top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    	for i_im in range(len(im))
    ], axis=0)
    return im, r, (dw, dh)

def postprocess(boxes,r,dwdh):
    dwdh = torch.tensor(dwdh*2).to(boxes.device)
    boxes -= dwdh
    boxes /= r
    return boxes

names = [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush' ]
colors = {name:[random.randint(0, 255) for _ in range(3)] for i,name in enumerate(names)}


def preprocess_img(img):
	img = img[..., ::-1]
	image = img.copy()
	image, ratio, dwdh = letterbox(image, auto=False)
	image = image.transpose((0, 3, 1, 2))
	image = np.ascontiguousarray(image)

	return image.astype(np.float32), ratio, dwdh


def draw_predictions(img, nums, boxes, scores, classes, ratio, dwdh):
	boxes = boxes[:nums[0]]
	scores = scores[:nums[0]]
	classes = classes[:nums[0]]

	for box,score,cl in zip(boxes, scores, classes):
	    box = postprocess(box,ratio,dwdh).round().int()
	    name = names[cl]
	    color = colors[name]
	    name += ' ' + str(round(float(score),3))
	    cv2.rectangle(img,box[:2].tolist(),box[2:].tolist(),color,2)
	    cv2.putText(img,name,(int(box[0]), int(box[1]) - 2),cv2.FONT_HERSHEY_SIMPLEX,0.75,color,thickness=2)
	return img


def predict(img):
	im, ratio, dwdh = preprocess_img(img)
	im = torch.from_numpy(im).to(device)
	im /= 255
	start = time.perf_counter()
	binding_addrs['images'] = int(im.data_ptr())
	context.execute_v2(list(binding_addrs.values()))
	# print(f'Cost {time.perf_counter()-start} s')

	nums = bindings['num_dets'].data
	boxes = bindings['det_boxes'].data
	scores = bindings['det_scores'].data
	classes = bindings['det_classes'].data
	return (nums, boxes, scores, classes), (ratio, dwdh)


if __name__ == '__main__':
	img = cv2.imread('test.png')
	img_batch = np.stack([img] * batch_size, axis=0)
	(nums, boxes, scores, classes), (ratio, dwdh) = predict(img_batch)

	res_pred = []
	for i, (num_s, box_s, score_s, class_s) in enumerate(zip(nums, boxes, scores, classes)):
		img_with_preds = draw_predictions(
			img_batch[i], num_s, box_s, score_s, class_s,
			ratio, dwdh
		)
		res_pred.append(img_with_preds)

	for i, res_img in enumerate(res_pred):
		cv2.imwrite(f'test_{i}.png', res_img)

