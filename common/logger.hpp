#pragma once

#include <cuda_runtime.h>
#include <system_error>
#include <stdarg.h>

#define CUDA_CHECK(call)             __cudaCheck(call, __FILE__, __LINE__)
#define LAST_KERNEL_CHECK()          __kernelCheck(__FILE__, __LINE__)
#define LOG(...)                     __log_info(__VA_ARGS__)

inline static void __cudaCheck(cudaError_t err, const char *file, const int line) {
    if (err != cudaSuccess) {
        printf("ERROR: %s:%d, ", file, line);
        printf("CODE:%s, DETAIL:%s\n", cudaGetErrorName(err), cudaGetErrorString(err));
        exit(1);
    }
}

inline static void __kernelCheck(const char *file, const int line) {
    /*
     * 在编写CUDA是，错误排查非常重要，默认的cuda runtime API中的函数都会返回cudaError_t类型的结果，
     * 但是在写kernel函数的时候，需要通过cudaPeekAtLastError或者cudaGetLastError来获取错误
     */
    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        printf("ERROR: %s:%d, ", file, line);
        printf("CODE:%s, DETAIL:%s\n", cudaGetErrorName(err), cudaGetErrorString(err));
        exit(1);
    }
}

// 使用变参进行LOG的打印。比较推荐的打印log的写法
static void __log_info(const char *format, ...) {
    char msg[1000];
    va_list args;
    va_start(args, format);

    vsnprintf(msg, sizeof(msg), format, args);

    fprintf(stdout, "%s\n", msg);
    va_end(args);
}