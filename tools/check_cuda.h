#ifndef CHECK_CUDA_H
#define CHECK_CUDA_H

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if(err != cudaSuccess){
        std::fprintf(stderr, "CUDA Error at: %s:%d: %s \n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        std::exit(1); \
    } \
} while(0)


#endif //CHECK_CUDA_H