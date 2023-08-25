CMAKE_MINIMUM_REQUIRED(VERSION 3.22.0)

PROJECT(demo_mat_mul VERSION 1.0.0 LANGUAGES CXX CUDA)

FILE(GLOB SOURCES
        "*.cpp"
        "*.hpp"
        "*.cu"
        "*.cuh"
        )

add_executable(demo_mat_mul_gpu main.cpp ${SOURCES})
set_target_properties(demo_mat_mul_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(demo_mat_mul_gpu CUDA::cudart CUDA::cublas CUDA::cusparse CUDA::cusolver CUDA::nppc CUDA::nvToolsExt)
