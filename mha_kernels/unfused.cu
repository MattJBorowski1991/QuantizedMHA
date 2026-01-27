#include <cuda_runtime.h>
#include <math.h>
#include "../include/config.h"
#include "../include/launchers.h"
#include "../utils.cu"

template<int TILE>
__global__ void mma_A_B(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    float alpha
)
{
    __shared__ float A_tile[TILE][TILE];
    __shared__ float B_tile[TILE][TILE];

    const int ty = threadIdx.y;
    const int tx = threadIdx.x;
    const int row = blockIdx.y * TILE + ty;
    const int col = blockIdx.x * TILE + tx;

    float sum = 0.0f;

    for(int tile_k = 0; tile_k < K; tile_k += TILE){
        if (row < M && (tile_k + tx) < K) {
            A_tile[ty][tx] = A[row * K + (tile_k + tx)];
        } else {
            A_tile[ty][tx] = 0.0f;
        }
        if ( (tile_k + ty) < K && col < N ){
            B_tile[ty][tx] = B[ (tile_k + ty) * N + col];
        }else{
            B_tile[ty][tx] = 0.0f;
        }
        __syncthreads();
        for(int kk = 0; kk < TILE && tile_k + kk < K; ++kk){
            sum += A_tile[ty][kk] * B_tile[kk][tx];
        }
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = sum;
}

template<int TILE>
__global__ void mma_A_Bt(
    const float* __restrict__ A,    // M x K
    const float* __restrict__ B,    // N x K => B^T is K x N
    float* __restrict__ C,
    int M, int N, int K,
    float alpha
)
{
    __shared__ float A_tile[TILE][TILE];
    __shared__ float Bt_tile[TILE][TILE];

    const int ty = threadIdx.y;
    const int tx = threadIdx.x;
    const int row = blockIdx.y * TILE + ty;
    const int col = blockIdx.x * TILE + tx;

    float sum = 0.0f;

    for(int tile_k = 0; tile_k < K; tile_k += TILE){
        if (row < M && (tile_k + tx) < K) {
            A_tile[ty][tx] = A[row * K + (tile_k + tx)];
        } else {
            A_tile[ty][tx] = 0.0f;
        }
        if (col < N && (tile_k + ty) < K) {
            Bt_tile[ty][tx] = B[col * K + (tile_k + ty)];
        } else {
            Bt_tile[ty][tx] = 0.0f;
        }
        __syncthreads();
        for(int kk = 0; kk < TILE && tile_k + kk < K; ++kk){
            sum += A_tile[ty][kk] * Bt_tile[kk][tx];
        }
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = alpha * sum; 
}

template<int TILE>
void launch_mma_A_B(const float *A, const float *B, float *C, int M, int N, int K, float alpha, bool isBtransposed, cudaStream_t stream = 0){

    dim3 threads(TILE,TILE);
    dim3 blocks(( N + TILE - 1 ) / TILE, (M + TILE - 1) / TILE);
    if(isBtransposed){
        mma_A_Bt<TILE><<<blocks, threads, 0, stream>>>(A, B, C, M, N, K, alpha);
    }else{
        mma_A_B<TILE><<<blocks, threads, 0, stream>>>(A, B, C, M, N, K, alpha);
    }
}

template <int TILE>
__global__ void softmax(const float *input, float *output, int N) {
    const unsigned fullMask = 0xFFFFFFFFu;
    int row = blockIdx.x;
    int tid = threadIdx.x;
    constexpr int threads = TILE * TILE;
    const float *row_in  = input  + row * N;
    float *row_out = output + row * N;

    // This implementation assumes blocks of at most 256 threads.
    __shared__ float sdata[threads];

    // compute local max over the row (stride by blockDim.x)
    float local_max = -INFINITY;
    for (int i = tid; i < N; i += threads) {
        float v = row_in[i];
        if (v > local_max) local_max = v;
    }
    sdata[tid] = local_max;
    __syncthreads();

    // tree-reduce in shared memory down to 32 lanes
    for (int s = threads / 2; s >= 32; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }

    // warp-level reduce using shuffle for the final 32 lanes
    if (tid < 32) {
        float v = sdata[tid];
        for (int offset = 16; offset > 0; offset >>= 1)
            v = fmaxf(v, __shfl_down_sync(fullMask, v, offset));
        if (tid == 0) sdata[0] = v;
    }
    __syncthreads();

    float max_val = sdata[0];

    // compute exponentials and local sums
    float local_sum = 0.0f;
    for (int i = tid; i < N; i += threads) {
        float e = expf(row_in[i] - max_val);
        row_out[i] = e;
        local_sum += e;
    }
    sdata[tid] = local_sum;
    __syncthreads();

    // reduce sums in shared memory down to 32 lanes
    for (int s = threads / 2; s >= 32; s >>= 1) {
        if (tid < s) sdata[tid] = sdata[tid] + sdata[tid + s];
        __syncthreads();
    }

    // warp-level sum reduction
    if (tid < 32) {
        float v = sdata[tid];
        for (int offset = 16; offset > 0; offset >>= 1)
            v += __shfl_down_sync(fullMask, v, offset);
        if (tid == 0) sdata[0] = v;
    }
    __syncthreads();

    float exp_sum = sdata[0];

    // normalize
    for (int i = tid; i < N; i += threads) row_out[i] = row_out[i] / exp_sum;
}

template <int TILE>
void launch_softmax(const float *input, float *output, int N, cudaStream_t stream = 0)
{
    int threadsPerBlock = TILE * TILE;
    int blocksPerGrid = N;
    softmax<TILE><<<blocksPerGrid, threadsPerBlock, 0, stream>>>(input, output, N);
}

MHA_SOLVE(      // safer here to use capture by value [=] here instead of no-capture [] as we do manual pointer math on the aux buffer: per_softmax
    [=](const float* q_s, const float* k_s, const float* v_s, float* out_s, int N, int d_head, float alpha, cudaStream_t stream, void* aux){
        size_t per_softmax = (size_t)N * N;
        float* softmax_in_s = (float*)aux;
        float* softmax_out_s = softmax_in_s + per_softmax;
        launch_mma_A_B<TILE>(q_s, k_s, softmax_in_s, N, N, d_head, alpha, true, stream);
        launch_softmax<TILE>(softmax_in_s, softmax_out_s, N, stream);
        launch_mma_A_B<TILE>(softmax_out_s, v_s, out_s, N, d_head, N, 1.0f, false, stream);
    },
    ((size_t)N * N * sizeof(float) * 2)  
    // AUX_BYTES are non-zero here as only unfused.cu needs extra DRAM (to store softmax_in & softmax_out). FA computes everything in SRAM and regs.
)