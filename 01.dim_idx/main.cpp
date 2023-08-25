#include "cuda_index.cuh"
#include <cuda_runtime.h>
#include <iostream>


/*
 *  一个kernel函数 对应 一个grid
 *  dim3 grid(x, y, z) 用于定义该grid中有 x*y*z个block   先排满x再排满y最后排满z
 *  dim3 block(x, y, z) 用于定义该block中有 x*y*z个thread 先排满x再排满y最后排满z
 *
 *  所有block中的thread，也即该grid中的thread, 共享全局显存，常量显存，和纹理显存
 *  同一个block中的所有thread 共享一个share mem
 *  每个thread有自己的寄存器和local mem
 */

int main() {
    dim3 grid(2, 2);
    dim3 block(3, 4);
    print_idx_device(grid, block);
    return 0;
}