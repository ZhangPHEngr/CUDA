CMAKE_MINIMUM_REQUIRED(VERSION 3.22.0)

PROJECT(demo_gpu VERSION 1.0.0  LANGUAGES CXX CUDA)


add_executable(${PROJECT_NAME} main.cpp main.cpp)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(${PROJECT_NAME} CUDA::cudart CUDA::cublas CUDA::cusparse CUDA::cusolver CUDA::nppc CUDA::nvToolsExt)
