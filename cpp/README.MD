## With NMS Plugin

###  Generate trt File [More Details](https://github.com/Linaom1214/TensorRT-For-YOLO-Series/blob/main/README.md)
```shell
python export.py -o mdoel.onnx -e model.trt --end2end --v8 only for yolov8
```

- [x] YOLOv8
- [x] YOLOv7
- [x] YOLOv6
- [x] YOLOX
- [x] YOLOv5
- [x] YOLOv3 


**Edit code for your model**

```c++
auto in_dims = engine->getBindingDimensions(engine->getBindingIndex("image_arrays"));
```
**run**

```shell
cd end2end && mkdir build && build
cmake ..
make
./yolo -model_path  engine   -image_path xxx.jpg
```


## Without NMS Plugin
###  Generate trt File [More Details](https://github.com/Linaom1214/TensorRT-For-YOLO-Series/blob/main/README.md)

```shell
python export.py -o mdoel.onnx -e model.trt --end2end
```


- [ ] YOLOv8
- [x] YOLOv7
- [x] YOLOv6
- [x] YOLOX
- [x] YOLOv5
- [x] YOLOv3 


**Edit code for your model**

```c++
static const int INPUT_W = 640;
static const int INPUT_H = 640;
const char* INPUT_BLOB_NAME = "image_arrays";
const char* OUTPUT_BLOB_NAME = "outputs";
```
**run**

```shell
cd norm && mkdir build && build
cmake ..
make
./yolo ../model_trt.engine -i ../*.jpg
```
