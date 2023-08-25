//
// Created by zph on 23-8-6.
//
#include "timer.hpp"
#include "logger.hpp"
#include "utils.hpp"
#include <string>
#include "cuda_mul.cuh"
#include "cpu_mul.hpp"

static char str[100];

int main() {
    Timer timer;

    int width = 1 << 10; // 1024
    int low = 0;
    int high = 1;
    int size = width * width;
    int blockSize = 16;
    bool statMem = true;

    //分配host内存
    float *h_matM = (float *) malloc(size * sizeof(float));
    float *h_matN = (float *) malloc(size * sizeof(float));
    float *h_matP = (float *) malloc(size * sizeof(float));
    float *d_matP = (float *) malloc(size * sizeof(float));

    //初始化矩阵
    int seed = 1;
    initMatrix(h_matM, size, low, high, seed);
    seed += 1;
    initMatrix(h_matN, size, low, high, seed);
    LOG("Input size is %d x %d", width, width);

    //cpu计算
    timer.start_cpu();
    MatmulOnHost(h_matM, h_matN, h_matP, width);
    timer.stop_cpu();
    timer.duration_cpu<Timer::ms>("matmul in cpu");

    //gpu 基于原子操作
    timer.start_gpu();
    MatmulAtomicAddOnDevice(h_matM, h_matN, d_matP, width);
    timer.stop_gpu();
    std::sprintf(str, "matmul in gpu with atomicAdd <<<(%d,%d), %d>>>", width, width, width);
    timer.duration_gpu(str);
    compareMat(h_matP, d_matP, size);

    //GPU general implementation <<<64, 16>>>
    timer.start_gpu();
    MatmulSharedOnDevice(h_matM, h_matN, d_matP, width, blockSize, statMem);
    timer.stop_gpu();
    std::sprintf(str, "matmul in gpu(with shared memory(static))<<<%d, %d>>>", width / blockSize, blockSize);
    timer.duration_gpu(str);
    compareMat(h_matP, d_matP, size);

    //GPU general implementation <<<64, 16>>>
    statMem = false;
    timer.start_gpu();
    MatmulSharedOnDevice(h_matM, h_matN, d_matP, width, blockSize, statMem);
    timer.stop_gpu();
    std::sprintf(str, "matmul in gpu(with shared memory(static))<<<%d, %d>>>", width / blockSize, blockSize);
    timer.duration_gpu(str);
    compareMat(h_matP, d_matP, size);


    return 0;
}
