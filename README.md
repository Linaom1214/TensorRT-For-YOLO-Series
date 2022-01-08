# YOLOX and  YOLOV5 TensorRT Python API 

![](imgs/3.jpg)

# YOLOX 
![](yolox/3_yolox.jpg)
## 导出ONNX

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
## 转化为TensorRT Engine 
```
python export_trt.py -m onnx-name -o trt-name
```
## 测试

```
cd yolovx
python trt.py
```

# YOLOV5
![](yolov5/3_yolov5.jpg)

## 导出ONNX

```
git clone https://github.com/ultralytics/yolov5.git
```

```python
python path/to/export.py --weights yolov5s.pt --include  onnx 
```

## 转化为TensorRT Engine 

```
python export_trt.py -m onnx-name -o trt-name
```
## 测试

```
cd yolov5
python trt.py
```

### 请重新或者ONNX 文件 并转化为TRT 引擎
### 权重文件获取 [Only for My]
[yolox](https://github.com/Linaom1214/tensorrt-python/releases/download/v1.0.0/yolox.trt)
[yolov5](https://github.com/Linaom1214/tensorrt-python/releases/download/v1.0.0/yolov5.trt)
