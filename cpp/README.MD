## With NMS Plugin

**Edit code for you model**

```c++
const cv::Size& new_shape = cv::Size(640, 640),
static const int INPUT_H = 640;
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

**Edit code for you model**

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