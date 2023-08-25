#include "cuda_runtime.h"
#include "logger.hpp"
#include <stdio.h>

__global__ void print_thread_idx_kernel() {
    int bSize = blockDim.z * blockDim.y * blockDim.x;

    int bIndex = blockIdx.z * gridDim.x * gridDim.y +
                 blockIdx.y * gridDim.x +
                 blockIdx.x;

    int tIndex = threadIdx.z * blockDim.x * blockDim.y +
                 threadIdx.y * blockDim.x +
                 threadIdx.x;

    int index = bIndex * bSize + tIndex;

    printf("block idx: %3d, thread idx in block: %3d, thread idx: %3d\n",
           bIndex, tIndex, index);
}


void print_idx_device(dim3 grid, dim3 block) {
    print_thread_idx_kernel<<<grid, block>>>();
    CUDA_CHECK(cudaDeviceSynchronize());
}
