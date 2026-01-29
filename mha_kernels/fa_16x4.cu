#include <cuda_runtime.h>
#include <math.h>
#include <limits>
#include "../include/config.h"
#include "../include/launchers.h"
#include "../utils/utils.cu"

// TODO1 profile in NCU and optimize softmax so that elapsed cycles drop from 13.5m to c.a. 5m (results for unfused)
// TODO2 implement TC WMMA
// TODO3 implement Double Buffering

// Flash attention with Warp-level tiled matmul
// One warp handles 16 rows of Q, so that we can prepare the kernel to using TC WMMA (need 16x16 tiles)
// Since its best to have 4-8 warps per block => Br = 64 (or even better 128)
// for Br=Bc=64 and N=4096, d = 2048 the kernel fails silently due to SRAM overflow
// hence we set Br = 64, Bc = 32

#define FULL_MASK 0xffffffff
#define THREADS_PER_WARP 32

// Shared memory block matrix multiply (warp-tiled with register blocking)
// Each warp: 16 rows, each lane (32 lanes): 4 columns
// Accumulates 16x4 tiles into registers
template <bool add_to_output = false>
__device__ void matmul_warp_tiled(
    const float* mat_a,      // num_rows x num_shared_dim
    const float* mat_b,      // num_shared_dim x num_cols
    float* mat_c,            // num_rows x num_cols
    int num_rows,
    int num_cols,
    int num_shared_dim
) {
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int num_threads = blockDim.x;
    int num_warps = num_threads / THREADS_PER_WARP;

    constexpr int warp_rows = 16;
    constexpr int reg_tile_rows = 16;
    constexpr int reg_tile_cols = 4;
    
    // Each warp handles warp_rows (16) rows
    for (int row_start = warp_rows * warp_id; row_start < num_rows; row_start += num_warps * warp_rows) {
        int row_count = min(warp_rows, num_rows - row_start);
        
        // Each lane handles reg_tile_cols (4) columns
        for (int col_start = reg_tile_cols * lane_id; col_start < num_cols; col_start += THREADS_PER_WARP * reg_tile_cols) {
            int col_count = min(reg_tile_cols, num_cols - col_start);
            
            // Register accumulator (16x4)
            float acc[reg_tile_rows * reg_tile_cols];
            #pragma unroll
            for (int i = 0; i < reg_tile_rows * reg_tile_cols; i++) acc[i] = 0.0f;
            
            // Dot product loop
            for (int k_idx = 0; k_idx < num_shared_dim; k_idx++) {
                // Load A tile rows (all 16 rows for this lane)
                float a_vals[reg_tile_rows];
                #pragma unroll
                for (int i = 0; i < reg_tile_rows; i++) {
                    if (row_start + i < num_rows) {
                        a_vals[i] = mat_a[(row_start + i) * num_shared_dim + k_idx];
                    } else {
                        a_vals[i] = 0.0f;
                    }
                }
                
                // Load B tile columns (4 columns)
                float b_vals[reg_tile_cols];
                #pragma unroll
                for (int j = 0; j < reg_tile_cols; j++) {
                    b_vals[j] = mat_b[k_idx * num_cols + col_start + j];
                }
                
                // Outer product
                #pragma unroll
                for (int i = 0; i < reg_tile_rows; i++) {
                    #pragma unroll
                    for (int j = 0; j < reg_tile_cols; j++) {
                        acc[i * reg_tile_cols + j] += a_vals[i] * b_vals[j];
                    }
                }
            }
            
            // Store to C
            #pragma unroll
            for (int i = 0; i < row_count; i++) {
                #pragma unroll
                for (int j = 0; j < col_count; j++) {
                    if constexpr (add_to_output) {
                        mat_c[(row_start + i) * num_cols + (col_start + j)] += acc[i * reg_tile_cols + j];
                    } else {
                        mat_c[(row_start + i) * num_cols + (col_start + j)] = acc[i * reg_tile_cols + j];
                    }
                }
            }
        }
    }
}

// Online softmax: scale scores, find max, compute softmax probs, update statistics, rescale output, accumulate O += P @ V
__device__ void online_softmax_and_accum_output(
    float* max_cur,
    const float* max_prev,
    float* sum_exp,
    float* scores,     // Will be overwritten with softmax probs
    float* output,
    const float* values, // V (kv_block_size x head_dim)
    int q_block_size,
    int kv_block_size,
    int head_dim,
    float sqrt_head_dim
) {
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int num_threads = blockDim.x;
    int num_warps = num_threads / THREADS_PER_WARP;

    constexpr int warp_rows = 16;
    
    // Process warp_rows consecutive rows per warp
    for (int row_start = warp_rows * warp_id; row_start < q_block_size; row_start += num_warps * warp_rows) {
        int row_count = min(warp_rows, q_block_size - row_start);
        
        // Process each row assigned to this warp
        for (int row_idx = 0; row_idx < row_count; row_idx++) {
            int q_row = row_start + row_idx;
            
            // Step 1: Find max in this KV block's scores for this query row and scale by 1/sqrt(d)
            float max_new = max_prev[q_row];
            for (int kv_col = lane_id; kv_col < kv_block_size; kv_col += THREADS_PER_WARP) {
                float score_scaled = scores[q_row * kv_block_size + kv_col] / sqrt_head_dim;
                max_new = fmaxf(max_new, score_scaled);
                scores[q_row * kv_block_size + kv_col] = score_scaled;
            }
            
            // Warp reduction to get global max for this query row
            #pragma unroll
            for (int shift = THREADS_PER_WARP / 2; shift >= 1; shift >>= 1) {
                max_new = fmaxf(max_new, __shfl_xor_sync(FULL_MASK, max_new, shift));
            }

            // Step 2: Compute exp(score - max) and accumulate sum with warp reduction
            float sum_new = 0.0f;
            for (int kv_col = lane_id; kv_col < kv_block_size; kv_col += THREADS_PER_WARP) {
                float prob = expf(scores[q_row * kv_block_size + kv_col] - max_new);
                scores[q_row * kv_block_size + kv_col] = prob;  // Store softmax prob back
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
    }
    
    __syncthreads();
    
    // Step 5: Accumulate O += (softmax probs) @ V
    matmul_warp_tiled<true>(scores, values, output, q_block_size, head_dim, kv_block_size);
}

template<int Br, int Bc>
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
    const int num_threads = blockDim.x;

    // Compute allocation sizes
    int alloc_size = max(Br * Bc, Bc * d);
    
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
    for (int idx = tid; idx < Br * d; idx += num_threads) {
        int row = idx / d;
        int col = idx % d;
        if (q_block_idx * Br + row < N) {
            q_block[idx] = Q[(q_block_idx * Br + row) * d + col];
        } else {
            q_block[idx] = 0.0f;
        }
    }
    
    // Initialize output, statistics
    for (int idx = tid; idx < q_rows * d; idx += num_threads) {
        output[idx] = 0.0f;
    }
    for (int idx = tid; idx < q_rows; idx += num_threads) {
        sum_exp[idx] = 0.0f;
        max_prev[idx] = -INFINITY;
    }
    __syncthreads();

    // Main loop over K,V blocks
    int num_kv_blocks = (N + Bc - 1) / Bc;
    for (int kv_block_idx = 0; kv_block_idx < num_kv_blocks; kv_block_idx++) {
        int kv_rows = min(Bc, N - kv_block_idx * Bc);
        
        // Load K (transposed)
        for (int idx = tid; idx < Bc * d; idx += num_threads) {
            int row = idx / d;
            int col = idx % d;
            int k_idx = kv_block_idx * Bc + row;
            if (k_idx < N) {
                // Store transposed: kv_block[col * Bc + row] = K[k_idx * d + col]
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
        for (int idx = tid; idx < Bc * d; idx += num_threads) {
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
        float* tmp = max_prev;
        max_prev = max_cur_ptr;
        max_cur_ptr = tmp;
    }

    // Epilogue: normalize output and write to global memory
    for (int idx = tid; idx < q_rows * d; idx += num_threads) {
        int row = idx / d;
        if (sum_exp[row] > 0.0f) {
            output[idx] /= sum_exp[row];
        }
    }
    __syncthreads();
    
    // Store to global memory
    for (int idx = tid; idx < q_rows * d; idx += num_threads) {
        int row = idx / d;
        int col = idx % d;
        if (q_block_idx * Br + row < N) {
            O[(q_block_idx * Br + row) * d + col] = output[idx];
        }
    }
}

template<int Br, int Bc>
void launch_fa(
    const float *Q, const float *K, const float *V,
    float *O,
    int N, int d,
    float alpha,
    cudaStream_t stream = 0
)
{
    int num_blocks = (N + Br - 1) / Br;
    int alloc_size = max(Br * Bc, Bc * d);
    size_t shared_mem = (4 * alloc_size + 3 * Br) * sizeof(float);
    
    int threads_per_block = 512;
    
    fa_kernel<Br, Bc><<<num_blocks, threads_per_block, shared_mem, stream>>>(
        Q, K, V, O, N, d, alpha
    );
}

extern "C" void solve(const float *Q, const float *K, const float *V, float *output, int N, int d_model, int h)
{
    auto fa_kernel = [](const float* q_s, const float* k_s, const float* v_s, float* out_s, int N, int d_head, float alpha, cudaStream_t stream, void* aux){
        (void)aux;
        launch_fa<Br, Bc>(q_s, k_s, v_s, out_s, N, d_head, alpha, stream);
    };
    launch(Q, K, V, output, N, d_model, h, fa_kernel, 0);
}
