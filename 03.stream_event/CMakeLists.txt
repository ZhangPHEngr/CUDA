CMAKE_MINIMUM_REQUIRED(VERSION 3.22.0)

PROJECT(demo_stream VERSION 1.0.0 LANGUAGES CXX CUDA)

FILE(GLOB SOURCES
        "*.hpp"
        "*.cu"
        "*.cuh"
        )

add_executable(demo_stream main.cpp ${SOURCES})
set_target_properties(demo_stream PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(demo_stream CUDA::cudart CUDA::cublas CUDA::cusparse CUDA::cusolver CUDA::nppc CUDA::nvToolsExt)
