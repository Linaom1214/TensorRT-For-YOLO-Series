# YOLO系列 TensorRT Python/C++ 

## 支持
YOLOv7、YOLOv6、 YOLOX、 YOLOV5

~Notice： **当前C++ demo 不支持端到端模型**~

## 更新 
- 2022.8.13 **重构仓库**
- 2022.8.11 **端到端导出支持, 更简洁的端到端导出方法**
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

## 如何上手

### Python

文件 `Examples.ipynb` 中提供了详细的Demo <a href="https://github.com/Linaom1214/TensorRT-For-YOLO-Series/blob/main/Examples.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Jupyter Notebook"></a>

### C++ 

~当前C++仅支持不包含NMS插件的模型~
[C++ Demo](cpp/README.MD)
