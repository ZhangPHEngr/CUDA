CMAKE_MINIMUM_REQUIRED(VERSION 3.22.0)
PROJECT(CUDA_Demo VERSION 1.0.0)

#-------------------------工程设置----------------------
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++17 -Wall")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer")
#set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address")

#-------------------------工程依赖----------------------
set(CMAKE_CUDA_STANDARD 17)
find_package(CUDAToolkit REQUIRED)

#-------------------------项目结构-----------------------
INCLUDE_DIRECTORIES(common)

ADD_SUBDIRECTORY(00.gpu_info)               # GPU信息
ADD_SUBDIRECTORY(01.dim_idx)                # 线程索引
ADD_SUBDIRECTORY(02.mat_mul)                # 矩阵乘法
ADD_SUBDIRECTORY(03.stream_event)           # 多流
