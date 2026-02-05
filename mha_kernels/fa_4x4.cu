#include <cuda_runtime.h>
#include <math.h>
#include <limits>
#include "../include/config.h"
#include "../include/launchers.h"
#include "../utils/utils.cu"

// Flash attention with Warp-level tiled matmul 
// 4x4 register-level tile per lane => 4 x 128 chunk per warp which is warp-strided for all columns of Q
// for this reason Q needs to have 128 columns (or mulitple of 128) for this to be most efficient solution
// implementation yields elapsed cycles of 7,200,000 on L4 for N=4096, d_model=2048, h = 32
// left for reference as it is not possible to build on it to implement TC WMMA (need 16 x d tile per warp)

#define THREADS_PER_WARP 32
#define WARPS_PER_BLOCK 16
#define FULL_MASK 0xffffffff

// Shared memory block matrix multiply (warp-tiled with register blocking)
// Each warp: 4 rows, each lane (32 lanes): 4 columns per iteration
// Accumulates into registers
template <bool add_to_output = false, int THREADS>
__device__ __forceinline__ void matmul_warp_tiled(
    const float* A,      // M x K
    const float* B,      // K x N
    float* C,        // M x N
    int M,
    int N,
    int K
) {
    constexpr int TILE_SIZE = 4;
    
    // Require dimensions to be multiples of TILE_SIZE for optimal unrolling
    assert(M % TILE_SIZE == 0 && "M must be a multiple of TILE_SIZE");
    assert(N % TILE_SIZE == 0 && "N must be a multiple of TILE_SIZE");
    assert(K % TILE_SIZE == 0 && "K must be a multiple of TILE_SIZE");
    
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    
    // Each warp handles TILE_SIZE rows
    for (int row_start = TILE_SIZE * warp_id; row_start < M; row_start += WARPS_PER_BLOCK * TILE_SIZE) {
        // Each lane handles TILE_SIZE columns
        for (int col_start = TILE_SIZE * lane_id; col_start < N; col_start += THREADS_PER_WARP * TILE_SIZE) {
            
            // Register accumulator (4x4)
            float acc[TILE_SIZE * TILE_SIZE];
            #pragma unroll
            for (int i = 0; i < TILE_SIZE * TILE_SIZE; i++) acc[i] = 0.0f;
            
            // Dot product loop
            for (int k_idx = 0; k_idx < K; k_idx++) {
                // Load A tile row
                float a_vals[TILE_SIZE];
                #pragma unroll
                for (int i = 0; i < TILE_SIZE; i++) {
                    a_vals[i] = A[(row_start + i) * K + k_idx];
                }
                
                // Load B tile column
                float b_vals[TILE_SIZE];
                #pragma unroll
                for (int j = 0; j < TILE_SIZE; j++) {
                    b_vals[j] = B[k_idx * N + col_start + j];
                }
                
                // Outer product
                #pragma unroll
                for (int i = 0; i < TILE_SIZE; i++) {
                    #pragma unroll
                    for (int j = 0; j < TILE_SIZE; j++) {
                        acc[i * TILE_SIZE + j] += a_vals[i] * b_vals[j];
                    }
                }
            }
            
            // Store to C
            #pragma unroll
            for (int i = 0; i < TILE_SIZE; i++) {
                #pragma unroll
                for (int j = 0; j < TILE_SIZE; j++) {
                    if constexpr (add_to_output) {
                        C[(row_start + i) * N + (col_start + j)] += acc[i * TILE_SIZE + j];
                    } else {
                        C[(row_start + i) * N + (col_start + j)] = acc[i * TILE_SIZE + j];
                    }
                }
            }
        }
    }
}

// TODO: one warp handles 1 row but should be handling 4 rows like in matmul_warp_tiled
// TODO: otherwise perf deficiencies: 
// (i) bank conflicts,
// (ii) poor cache locality - (softmax warp 0 computes row 0, matmul warp 0 reads rows 0-3 from different cache lines)
// (iii) load imbalance (some warps may finish earlier than others)

// Online softmax: scale scores, find max, compute softmax probs, update statistics, rescale output, accumulate O += P @ V
template<int THREADS>
__device__ __forceinline__ void online_softmax_and_accum_output(
    float* max_cur,
    const float* max_prev,
    float* sum_exp,
    float* scores,     // to be overwritten with softmax probs
    float* output,
    const float* values, // V (Bc x head_dim)
    int Br,
    int Bc,
    int head_dim,
    float sqrt_d
) {
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    // Process one query row per warp
    for (int q_row = warp_id; q_row < Br; q_row += WARPS_PER_BLOCK) {
        // Step 1: Find max in this KV block's scores for this query row and scale by 1/sqrt(d)
        float max_new = max_prev[q_row];
        for (int kv_col = lane_id; kv_col < Bc; kv_col += THREADS_PER_WARP) {
            float score_scaled = scores[q_row * Bc + kv_col] / sqrt_d;
            max_new = fmaxf(max_new, score_scaled);
            scores[q_row * Bc + kv_col] = score_scaled;
        }
        
        // Warp reduction to get global max for this query row
        #pragma unroll
        for (int shift = THREADS_PER_WARP / 2; shift >= 1; shift >>= 1) {
            max_new = fmaxf(max_new, __shfl_xor_sync(FULL_MASK, max_new, shift));
        }

        // Step 2: Compute exp(score - max) and accumulate sum with warp reduction
        float sum_new = 0.0f;
        for (int kv_col = lane_id; kv_col < Bc; kv_col += THREADS_PER_WARP) {
            float prob = expf(scores[q_row * Bc + kv_col] - max_new);
            scores[q_row * Bc + kv_col] = prob;  // Store softmax prob back
            sum_new += prob;
        }
        
        // Warp reduction to get global sum
        #pragma unroll
        for (int shift = THREADS_PER_WARP / 2; shift >= 1; shift >>= 1) {
            sum_new += __shfl_xor_sync(FULL_MASK, sum_new, shift);
        }

        // Step 3: Update max and sum statistics (only first lane writes)
        float exp_max_diff = expf(max_prev[q_row] - max_new);
        if (lane_id == 0) {
            max_cur[q_row] = max_new;
            sum_exp[q_row] = exp_max_diff * sum_exp[q_row] + sum_new;
        }
        
        // Step 4: Rescale output accumulator by exp(max_old - max_new)
        for (int d_idx = lane_id; d_idx < head_dim; d_idx += THREADS_PER_WARP) {
            output[q_row * head_dim + d_idx] *= exp_max_diff;
        }
    }
    
    __syncthreads();
    
    // Step 5: Accumulate O += (softmax probs) @ V
    matmul_warp_tiled<true>(scores, values, output, Br, head_dim, Bc);
}

template<int Br, int Bc, int THREADS>
__global__ void fa_kernel(
    const float* Q,    // [N, d]
    const float* K,    // [N, d]
    const float* V,    // [N, d]
    float* O,          // [N, d]
    const int N,       // Sequence length
    const int d,       // Dimension per head
    const float scale  // softmax_scale (1/sqrt(d) is passed as scale)
) {
    // One block per Br Q-rows
    const int q_block_idx = blockIdx.x;
    const int tid = threadIdx.x;

    // Compute allocation sizes
    int alloc_size = max(Br * Bc, Bc * d);
    
    // TODO fix alloc sizes below
    // Shared memory layout
    extern __shared__ float shared_mem[];
    float *output = shared_mem;                      // [Br x d]
    float *q_block = &shared_mem[alloc_size];        // [Br x d]
    float *kv_block = &shared_mem[2 * alloc_size];   // Holds K or V
    float *scores = &shared_mem[3 * alloc_size];     // Holds scores or probs
    float *sum_exp = &shared_mem[4 * alloc_size];    // [Br]
    float *max_statistics = &shared_mem[4 * alloc_size + Br]; // [2*Br]
    float *max_cur = &shared_mem[4 * alloc_size + 2 * Br];
    
    float* max_prev = max_statistics;
    float* max_cur_ptr = max_cur;

    int q_rows = min(Br, N - q_block_idx * Br);
    
    // Load Q block
    for (int idx = tid; idx < Br * d; idx += THREADS) {
        int row = idx / d;
        int col = idx % d;
        if (q_block_idx * Br + row < N) {
            q_block[idx] = Q[(q_block_idx * Br + row) * d + col];
        } else {
            q_block[idx] = 0.0f;
        }
    }
    
    // Initialize output, statistics
    for (int idx = tid; idx < q_rows * d; idx += THREADS) {
        output[idx] = 0.0f;
    }
    for (int idx = tid; idx < q_rows; idx += THREADS) {
        sum_exp[idx] = 0.0f;
        max_prev[idx] = -INFINITY;
    }
    __syncthreads();

    // Main loop over K,V blocks
    int num_kv_blocks = (N + Bc - 1) / Bc;
    for (int kv_block_idx = 0; kv_block_idx < num_kv_blocks; kv_block_idx++) {
        int kv_rows = min(Bc, N - kv_block_idx * Bc);
        
        // Load K (transposed)
        for (int idx = tid; idx < Bc * d; idx += THREADS) {
            int row = idx / d;
            int col = idx % d;
            int k_idx = kv_block_idx * Bc + row;
            if (k_idx < N) {
                kv_block[col * Bc + row] = K[k_idx * d + col];
            } else {
                kv_block[col * Bc + row] = 0.0f;
            }
        }
        __syncthreads();
        
        // Compute scores = Q @ K^T
        matmul_warp_tiled(q_block, kv_block, scores, q_rows, kv_rows, d);
        __syncthreads();
        
        // Load V
        for (int idx = tid; idx < Bc * d; idx += THREADS) {
            int row = idx / d;
            int col = idx % d;
            int k_idx = kv_block_idx * Bc + row;
            if (k_idx < N) {
                kv_block[idx] = V[k_idx * d + col];
            } else {
                kv_block[idx] = 0.0f;
            }
        }
        __syncthreads();
        
        // Online softmax + output accumulation
        float sqrt_d = sqrtf((float)d);
        online_softmax_and_accum_output(max_cur_ptr, max_prev, sum_exp, scores, output, kv_block, q_rows, kv_rows, d, sqrt_d);
        __syncthreads();
        
        // Swap statistics pointers for next iteration
        float* tmp = max_prev; max_prev = max_cur_ptr; max_cur_ptr = tmp;
    }

    // TODO: above one warp handles one row and below we do a block-stride across the full Q_block?
    // TODO: for d > THREADS below we do the same thing with multiple loop iterations

    // Epilogue: normalize output and write to global memory
    for (int idx = tid; idx < q_rows * d; idx += THREADS) {
        int row = idx / d;
        if (sum_exp[row] > 0.0f) {
            output[idx] /= sum_exp[row];
        }
    }
    __syncthreads();
    
    // Store to global memory
    for (int idx = tid; idx < q_rows * d; idx += THREADS) {
        int row = idx / d;
        int col = idx % d;
        if (q_block_idx * Br + row < N) {
            O[(q_block_idx * Br + row) * d + col] = output[idx];
        }
    }
}

template<int Br, int Bc, int THREADS>
void launch_fa(const float *Q, const float *K, const float *V, float *O, int N, int d, float alpha, cudaStream_t stream = 0)
{
    int BLOCKS = (N + Br - 1) / Br;
    int alloc_size = max(Br * Bc, Bc * d);
    size_t shared_mem = (4 * alloc_size + 3 * Br) * sizeof(float);      
    fa_kernel<Br, Bc, THREADS><<<BLOCKS, THREADS, shared_mem, stream>>>(Q, K, V, O, N, d, alpha);
}

extern "C" void solve(const float *Q, const float *K, const float *V, float *output, int N, int d_model, int h)
{   
    constexpr int THREADS = 512;
    auto fa_kernel = [](const float* q_s, const float* k_s, const float* v_s, float* out_s, int N, int d_head, float alpha, cudaStream_t stream, void* aux){
        (void)aux;
        launch_fa<Br, Bc, THREADS>(q_s, k_s, v_s, out_s, N, d_head, alpha, stream);
    };
    launch(Q, K, V, output, N, d_model, h, fa_kernel, 0);
}
