# YOLO Series TensorRT Python/C++ 
## [简体中文](README_CN.md)

## Support
YOLOv7、YOLOv6、 YOLOX、 YOLOV5

## Update 
- 2022.8.13 rename reop、 public new version、 **C++ for end2end**
- 2022.8.11 nms plugin support ==> Now you can set --end2end flag while use `export.py` get a engine file  
- 2022.7.8 support YOLOV7 
- 2022.7.3 support TRT int8  post-training quantization 

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

### Python

show in `Examples.ipynb` <a href="https://github.com/Linaom1214/TensorRT-For-YOLO-Series/blob/main/Examples.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Jupyter Notebook"></a>


### C++ 

support **NMS plugin**
show in [C++ Demo](cpp/README.MD)



## Old version
[v0.0.1](https://github.com/Linaom1214/TensorRT-For-YOLO-Series/releases/tag/v0.0.1)
[v0.0.2](https://github.com/Linaom1214/TensorRT-For-YOLO-Series/releases/tag/v0.0.2)



