#ifndef UTILS_CU
#define UTILS_CU

#include <cuda_runtime.h>

template<int TILE>
__global__ void extract_mat(const float* __restrict__ A, float* __restrict__ B, int row_off, int col_off, int M_A, int N_A, int M_B, int N_B){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < M_B && col < N_B){
        B[row * N_B + col] = A[(row_off + row) * N_A + (col_off + col)];
    }
}

template<int TILE>
__global__ void concat_mat(float* __restrict__ A, const float* __restrict__ B, int row_off, int col_off, int M_A, int N_A, int M_B, int N_B){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < M_B && col < N_B){
        A[(row + row_off) * N_A + (col + col_off)] = B[row * N_B + col];
    }
}

template<int TILE>
void launch_extract_mat(
    const float* A, float* B, 
    int row_off, int col_off, 
    int M_A, int N_A, int M_B, int N_B, cudaStream_t stream = 0){
    dim3 threads(TILE, TILE);
    dim3 blocks((N_B + TILE - 1) / TILE, (M_B + TILE - 1) / TILE);
    extract_mat<TILE><<<blocks, threads, 0, stream>>>(
        A, B, row_off, col_off, M_A, N_A, M_B, N_B);
}

template<int TILE>
void launch_concat_mat(
    float* A, const float* B, 
    int row_off, int col_off, 
    int M_A, int N_A, int M_B, int N_B, cudaStream_t stream = 0){
    dim3 threads(TILE, TILE);
    dim3 blocks((N_B + TILE - 1) / TILE, (M_B + TILE - 1) / TILE);
    concat_mat<TILE><<<blocks, threads, 0, stream>>>(
        A, B, row_off, col_off, M_A, N_A, M_B, N_B);
}

/**
 * RoPE Helper: Applied to Q and K rows.
 * Pairs elements at (k) and (k + d/2).
 */
static __device__ __forceinline__ void apply_rope(float* row, int pos, int d, float base = 10000.0f) {
    for (int k = 0; k < d / 2; k++) {
        // Compute rotation angle
        float theta = powf(base, -static_cast<float>(2 * k) / d);
        float angle = pos * theta;
        float sin_a, cos_a;
        sincosf(angle, &sin_a, &cos_a); // Faster than separate sin/cos calls

        float x = row[k];
        float y = row[k + d / 2];

        // Perform rotation
        row[k] = x * cos_a - y * sin_a;
        row[k + d / 2] = x * sin_a + y * cos_a;
    }
}

#endif //UTILS_CU