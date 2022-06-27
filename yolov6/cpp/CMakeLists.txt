cmake_minimum_required(VERSION 2.6)

project(yolov6)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)
# add_definitions("-Wall -g")
find_package(CUDA REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# cudnn
# include_directories(/data/cuda/cuda-10.2/cudnn/v8.0.4/include)
# link_directories(/data/cuda/cuda-10.2/cudnn/v8.0.4/lib64)



# tensorrt
include_directories(/home/mat/anaconda3/envs/pytorch/lib/python3.8/site-packages/tensorrt/include)
link_directories(/home/mat/anaconda3/envs/pytorch/lib/python3.8/site-packages/tensorrt)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(yolov6 ${PROJECT_SOURCE_DIR}/yolov6.cpp)
target_link_libraries(yolov6 /home/mat/anaconda3/envs/pytorch/lib/python3.8/site-packages/tensorrt/libnvinfer.so.8)
target_link_libraries(yolov6 cudart)
target_link_libraries(yolov6 ${OpenCV_LIBS})

add_definitions(-O2 -pthread)



