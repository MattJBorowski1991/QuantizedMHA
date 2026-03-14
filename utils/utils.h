#pragma once

#include <cuda_runtime.h>

// Forward declarations for CUDA utility functions (implemented in utils.cu)

template<int TILE>
void launch_extract_mat(const float *mat_in, float *mat_out, 
                       int N, int d, int q_start, int k_start, 
                       int h, int d_per_head, cudaStream_t stream = 0);

template<int TILE>
void launch_concat_mat(float *mat_out, const float *mat_in, 
                      int N, int d, int q_start, int k_start, 
                      int h, int d_per_head, cudaStream_t stream = 0);
