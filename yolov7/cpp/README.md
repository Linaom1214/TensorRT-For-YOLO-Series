# YOLOV7-TensorRT in C++

## Step 1: 准备TRT序列化引擎

```shell
https://github.com/WongKinYiu/yolov7.git
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

## Step 2: C++

Please follow the [TensorRT Installation Guide](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html) to install TensorRT.

And you should set the TensorRT path and CUDA path in CMakeLists.txt.

If you train your custom dataset, you may need to modify the value of `num_class`.

```c++
const int num_class = 80;
```

Install opencv with ```sudo apt-get install libopencv-dev``` (we don't need a higher version of opencv like v3.3+). 

build the demo:

```shell
mkdir build
cd build
cmake ..
make
```

Then run the demo:

```shell
./yolov7 ../model_trt.engine -i ../../../../assets/dog.jpg
```

or

```shell
./yolov7 <path/to/your/engine_file> -i <path/to/image>
```
