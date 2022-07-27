# YOLO系列 TensorRT Python/C++ 

## 支持
YOLOv7、YOLOv6、 YOLOX、 YOLOV5、

## 更新 
- 2022.7.8 支持YOLOV7 
- 2022.7.3 支持 TRT int8 post-training quantization 

## 准备TensorRT环境
`Python`
```
pip install --upgrade setuptools pip --user
pip install nvidia-pyindex
pip install --upgrade nvidia-tensorrt
pip install pycuda
```
`C++`

[By Docker](https://github.com/NVIDIA/TensorRT/blob/main/docker/ubuntu-20.04.Dockerfile)

## 快速上手

这个python的demo可以帮助你快速的理解这个项目 [Link](https://aistudio.baidu.com/aistudio/projectdetail/4263301?contributionType=1&shared=1)

## YOLOv7 [支持C++, Python]

![](yolov7/3_yolov7.jpg)

```shell
https://github.com/WongKinYiu/yolov7.git
```
修改代码:将 yolo.py 对应行修改如下：
https://github.com/WongKinYiu/yolov7/blob/5f1b78ad614b45c5a98e7afdd295e20033d5ad3c/models/yolo.py#L57 

```python
return x if self.training else (torch.cat(z, 1), ) if not self.export else (torch.cat(z, 1), x)
```

### 导出onnx
```shell
python models/export.py --weights ../yolov7.pt --grid
```

### 转化为TensorRT Engine 

```
python export.py -o onnx-name -e trt-name -p fp32/16/int8
```
### 测试

```
cd yolov7
python trt.py
```

### C++

C++ [Demo](yolov7/cpp/README.md)





## YOLOv6 [支持C++, Python]

| model |  input |  | FPS | Device | Language | 
| -------- | -------- | -------- | ------- | ------- | ------|
| yolov6s     | 640*640     | FP16     | 360FPS  | A100 | Python |
| yolov6s     | 640*640     | FP32     | 350FPS | A100| Python |
| yolov6s     | 640*640     | FP32     | 330FPS | 1080Ti | C++ |
| yolov6s     | 640*640     | FP32     | 300FPS | 1080Ti | Python |

[YOLOv6 bilibili](https://www.bilibili.com/video/BV1x3411w7T6?share_source=copy_web)

![](yolov6/3_yolov6.jpg)
```shell
git clone https://github.com/meituan/YOLOv6.git
```
### 导出onnx
```shell
python deploy/ONNX/export_onnx.py --weights yolov6s.pt --img 640 --batch 1
```

### 转化为TensorRT Engine 

```
python export.py -o onnx-name -e trt-name -p fp32/16/int8
```
### 测试

```
cd yolov6
python trt.py
```

### C++

C++ [Demo](yolov6/cpp/README.md)

## YOLOX [支持Python]
![](yolox/3_yolox.jpg)
### 导出ONNX

```
git clone https://github.com/Megvii-BaseDetection/YOLOX.git
```
```python

修改 export_onnx.py 为 model.head.decode_in_inference = True

修改 yolox/models/yolox_head.py文件

# [batch, n_anchors_all, 85]
# outputs = torch.cat(
#     [x.flatten(start_dim=2) for x in outputs], dim=2

# ).permute(0, 2, 1)
outputs = torch.cat(
    [x.view(-1,int(x.size(1)),int(x.size(2)*x.size(3))) for x in outputs], dim=2

).permute(0, 2, 1)

# outputs[..., :2] = (outputs[..., :2] + grids) * strides
# outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
# return outputs
xy =  (outputs[..., :2] + grids) * strides
wh = torch.exp(outputs[..., 2:4]) * strides
return torch.cat((xy, wh, outputs[..., 4:]), dim=-1)

```
```python
python3 tools/export_onnx.py --output-name yolox_s.onnx -n yolox-s -c yolox_s.pth
```
### 转化为TensorRT Engine 
```
python export.py -o onnx-name -e trt-name -p fp32/16/int8
```
### 测试

```
cd yolovx
python trt.py
```

## YOLOV5 [支持Python]
![](yolov5/3_yolov5.jpg)

### 导出ONNX

```
git clone https://github.com/ultralytics/yolov5.git
```

```python
python path/to/export.py --weights yolov5s.pt --include  onnx 
```

### 转化为TensorRT Engine 

```
python export.py -o onnx-name -e trt-name -p fp32/16/int8
```
### 测试

```
cd yolov5
python trt.py
```