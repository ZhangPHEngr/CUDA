CMAKE_MINIMUM_REQUIRED(VERSION 3.22.0)

PROJECT(demo_dim_idx VERSION 1.0.0  LANGUAGES CXX CUDA)


add_executable(demo_dim_idx main.cpp cuda_index.cu)
set_target_properties(demo_dim_idx PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(demo_dim_idx CUDA::cudart CUDA::cublas CUDA::cusparse CUDA::cusolver CUDA::nppc CUDA::nvToolsExt)
