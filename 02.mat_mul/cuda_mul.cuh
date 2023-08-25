#include <cuda_runtime.h>

void MatmulAtomicAddOnDevice(float *M_host, float *N_host, float *P_host, int width);

void MatmulSharedOnDevice(float *M_host, float *N_host, float* P_host, int width, int blockSize, bool staticMem);


