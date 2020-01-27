#ifndef COMMON_H
#define COMMON_H

#include "hip/hip_runtime.h"

#define NUM_PROCS 8
#define HIP_IPC_HANDLE_SIZE_RCCL 64

#define HIPCHECK(cmd) do {                         \
  hipError_t e = cmd;                              \
  if( e != hipSuccess ) {                          \
    printf("Test HIP failure %s:%d '%s'\n",        \
        __FILE__,__LINE__,hipGetErrorString(e));   \
    return 1;                           \
  }                                                 \
} while(0)

__global__ void setData(int* data, int targetRank, size_t N)
{
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N)
        data[idx] = targetRank;
}

static unsigned int numElements = 1024;

const char* smName = "prototype";

const int NUM_HANDLES_TOTAL = NUM_PROCS * 2;

#endif