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
### infer 
```shell
python trt.py -e yolov8n.trt  -i src/1.jpg -o yolov8n-1.jpg --end2end 
```

##  Prepare TRT Env 
`Python`
```
pip install --upgrade setuptools pip --user
pip install nvidia-pyindex
pip install --upgrade nvidia-tensorrt
pip install pycuda
```
`C++`

[By Docker](https://github.com/NVIDIA/TensorRT/blob/main/docker/ubuntu-20.04.Dockerfile)

## Examples
![image](https://user-images.githubusercontent.com/60921095/203555073-91606059-f3b6-49c2-b821-c3fa4c14ac42.png)

### Python

show in `Examples.ipynb` <a href="https://github.com/Linaom1214/TensorRT-For-YOLO-Series/blob/main/Examples.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Jupyter Notebook"></a>


### C++ 

support **NMS plugin**
show in [C++ Demo](cpp/README.MD)


## Citing 

If you use this repo in your publication, please cite it by using the following BibTeX entry.

```bibtex
@Misc{yolotrt2022,
  author =       {Jian Lin},
  title =        {YOLOTRT: tensorrt for yolo series, nms plugin support},
  howpublished = {\url{[https://github.com/Linaom1214/TensorRT-For-YOLO-Series]}},
  year =         {2022}
}
```

## Old version
[v0.0.1](https://github.com/Linaom1214/TensorRT-For-YOLO-Series/releases/tag/v0.0.1)
[v0.0.2](https://github.com/Linaom1214/TensorRT-For-YOLO-Series/releases/tag/v0.0.2)

## Sponsor

Buy me a cup of coffee


<table><tr>
<td><img src=src/alipay.jpg border=0></td>
<td><img src=src/wechatpay.jpg border=0></td>
</tr></table>

