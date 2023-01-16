# YOLO Series TensorRT Python/C++ 
## [简体中文](README_CN.md)

## Support
[YOLOv8](https://v8docs.ultralytics.com/)、[YOLOv7](https://github.com/WongKinYiu/yolov7)、[YOLOv6](https://github.com/meituan/YOLOv6)、 [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)、 [YOLOV5](https://github.com/ultralytics/yolov5)、[YOLOv3](https://github.com/ultralytics/yolov3)

- [x] YOLOv8
- [x] YOLOv7
- [x] YOLOv6
- [x] YOLOX
- [x] YOLOv5
- [x] YOLOv3 

## Update 
- 2023.1.7 support YOLOv8
- 2022.11.29 fix some bug thanks @[JiaPai12138](https://github.com/JiaPai12138)
- 2022.8.13 rename reop、 public new version、 **C++ for end2end**
- 2022.8.11 nms plugin support ==> Now you can set --end2end flag while use `export.py` get a engine file  
- 2022.7.8 support YOLOv7 
- 2022.7.3 support TRT int8  post-training quantization 

##  Prepare TRT Env 
`Install via Python`
```
pip install --upgrade setuptools pip --user
pip install nvidia-pyindex
pip install --upgrade nvidia-tensorrt
pip install pycuda
```
`Install via  C++`

[By Docker](https://github.com/NVIDIA/TensorRT/blob/main/docker/ubuntu-20.04.Dockerfile)

## Try YOLOv8
### Install && Download [Weights](https://github.com/ultralytics/assets/)
```shell
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ ultralytics==0.0.59
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt 
```
### Export ONNX
```Python
from ultralytics import YOLO

model = YOLO("yolov8s.pt")
model.fuse()  
model.info(verbose=False)  # Print model information
model.export(format='onnx')  # TODO: 
```
### Generate TRT File 
```shell
python export.py -o yolov8n.onnx -e yolov8n.trt --end2end --v8
```
### Inference 
```shell
python trt.py -e yolov8n.trt  -i src/1.jpg -o yolov8n-1.jpg --end2end 
```

## Python Demo
<details><summary> <b>Expand</b> </summary>

1. [YOLOv5](##YOLOv5)
2. [YOLOx](##YOLOX)
3. [YOLOv6](##YOLOV6)
4. [YOLOv7](##YOLOv7)


## YOLOv5


```python
!git clone https://github.com/ultralytics/yolov5.git
```

```python
!wget https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5n.pt
```


```python
!python yolov5/export.py --weights yolov5n.pt --include onnx --simplify --inplace 
```

### include  NMS Plugin


```python
!python export.py -o yolov5n.onnx -e yolov5n.trt --end2end
```


```python
!python trt.py -e yolov5n.trt  -i src/1.jpg -o yolov5n-1.jpg --end2end 
```

###  exclude NMS Plugin


```python
!python export.py -o yolov5n.onnx -e yolov5n.trt 
```


```python
!python trt.py -e yolov5n.trt  -i src/1.jpg -o yolov5n-1.jpg 
```

## YOLOX 


```python
!git clone https://github.com/Megvii-BaseDetection/YOLOX.git
```


```python
!wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth
```


```python
!cd YOLOX && pip3 install -v -e . --user
```


```python
!cd YOLOX && python tools/export_onnx.py --output-name ../yolox_s.onnx -n yolox-s -c ../yolox_s.pth --decode_in_inference
```

### include  NMS Plugin


```python
!python export.py -o yolox_s.onnx -e yolox_s.trt --end2end
```


```python
!python trt.py -e yolox_s.trt  -i src/1.jpg -o yolox-1.jpg --end2end 
```

###  exclude NMS Plugin


```python
!python export.py -o yolox_s.onnx -e yolox_s.trt 
```


```python
!python trt.py -e yolox_s.trt  -i src/1.jpg -o yolox-1.jpg 
```

## YOLOv6 


```python
!wget https://github.com/meituan/YOLOv6/releases/download/0.1.0/yolov6s.onnx
```

### include  NMS Plugin


```python
!python export.py -o yolov6s.onnx -e yolov6s.trt --end2end
```


```python
!python trt.py -e yolov6s.trt  -i src/1.jpg -o yolov6s-1.jpg --end2end
```

###  exclude NMS Plugin


```python
!python export.py -o yolov6s.onnx -e yolov6s.trt 
```


```python
!python trt.py -e yolov6s.trt  -i src/1.jpg -o yolov6s-1.jpg 
```

## YOLOv7


```python
!git clone https://github.com/WongKinYiu/yolov7.git
```


```python
!wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt
```


```python
!pip install -r yolov7/requirements.txt
```


```python
!python yolov7/export.py --weights yolov7-tiny.pt --grid  --simplify
```

### include  NMS Plugin


```python
!python export.py -o yolov7-tiny.onnx -e yolov7-tiny.trt --end2end
```


```python
!python trt.py -e yolov7-tiny.trt  -i src/1.jpg -o yolov7-tiny-1.jpg --end2end
```

###  exclude NMS Plugin


```python
!python export.py -o yolov7-tiny.onnx -e yolov7-tiny-norm.trt
```


```python
!python trt.py -e yolov7-tiny-norm.trt  -i src/1.jpg -o yolov7-tiny-norm-1.jpg
```
</details>

### C++ Demo

support **NMS plugin**
show in [C++ Demo](cpp/README.MD)


## Citing 

If you use this repo in your publication, please cite it by using the following BibTeX entry.

```bibtex
@Misc{yolotrt2022,
  author =       {Jian Lin},
  title =        {YOLOTRT: tensorrt for yolo series},
  howpublished = {\url{[https://github.com/Linaom1214/TensorRT-For-YOLO-Series]}},
  year =         {2022}
}
```

## Sponsor

Buy me a cup of coffee

![](src/Sponsor.png)

