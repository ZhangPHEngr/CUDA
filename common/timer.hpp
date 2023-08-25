#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <string>
#include "logger.hpp"

//https://renyili.org/post/std_chrono/
class Timer {
public:
    using s = std::ratio<1, 1>;
    using ms = std::ratio<1, 1000>;
    using us = std::ratio<1, 1000000>;
    using ns = std::ratio<1, 1000000000>;

public:
    Timer() {
        _timeElasped = 0;
        _cStart = std::chrono::high_resolution_clock::now();
        _cStop = std::chrono::high_resolution_clock::now();
        cudaEventCreate(&_gStart);
        cudaEventCreate(&_gStop);
    }

    ~Timer() {
        cudaFree(_gStart);
        cudaFree(_gStop);
    }

public:
    void start_cpu() {
        _cStart = std::chrono::high_resolution_clock::now();
    }

    void stop_cpu() {
        _cStop = std::chrono::high_resolution_clock::now();
    }

    void start_gpu() {
        cudaEventRecord(_gStart, 0);
    }

    void stop_gpu() {
        cudaEventRecord(_gStop, 0);
    }

    template<typename span>
    void duration_cpu(std::string msg);

    void duration_gpu(std::string msg);

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> _cStart;
    std::chrono::time_point<std::chrono::high_resolution_clock> _cStop;
    cudaEvent_t _gStart;
    cudaEvent_t _gStop;
    float _timeElasped;
};

template<typename span>
void Timer::duration_cpu(std::string msg) {
    std::string str;

    if (std::is_same<span, s>::value) { str = "s"; }
    else if (std::is_same<span, ms>::value) { str = "ms"; }
    else if (std::is_same<span, us>::value) { str = "us"; }
    else if (std::is_same<span, ns>::value) { str = "ns"; }

    std::chrono::duration<double, span> time = _cStop - _cStart;
    LOG("%-40s uses %.6lf %s", msg.c_str(), time.count(), str.c_str());
}

void Timer::duration_gpu(std::string msg) {
    CUDA_CHECK(cudaEventSynchronize(_gStart));
    CUDA_CHECK(cudaEventSynchronize(_gStop));
    cudaEventElapsedTime(&_timeElasped, _gStart, _gStop);

    // cudaDeviceSynchronize();
    // LAST_KERNEL_CHECK();

    LOG("%-60s uses %.6lf ms", msg.c_str(), _timeElasped);
}



