#include <cuda_runtime.h>
#include <math.h>
#include <limits>
#include <stdio.h>
#include "../include/config.h"
#include "../include/launchers.h"
#include "../utils/utils.cu"

// Flash attention with Warp-level tiled matmul with int8
// 4x4 register-level tile per lane => 4 x 128 chunk per warp which is warp-strided for all columns of Q
// for this reason Q needs to have 128 columns (or mulitple of 128) for this to be most efficient solution
// implementation yields elapsed cycles of 7,200,000 on L4 for N=4096, d_model=2048, h = 32

// Gameplan for implementing int8: 
// 0. Quantize float→int8 preprocessing: convert float inputs to int8 using scale/zero before main kernel
// 1. Dequantize on-the-fly in Q@K^T
// 2. Handle float x int8 for P@V matmul
// 3. Update SRAM layout - less mem needed for int8
// 4. Unchanged: softmax

#define FULL_MASK 0xffffffff
#define THREADS_PER_WARP 32

// Shared memory block matrix multiply (warp-tiled with register blocking)
// Each warp: 4 rows, each lane (32 lanes): 4 columns per iteration
// Accumulates into registers
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

    constexpr int TILE_SIZE = 4;
    
    // Each warp handles TILE_SIZE rows
    for (int row_start = TILE_SIZE * warp_id; row_start < num_rows; row_start += num_warps * TILE_SIZE) {
        int row_count = min(TILE_SIZE, num_rows - row_start);
        
        // Each lane handles TILE_SIZE columns
        for (int col_start = TILE_SIZE * lane_id; col_start < num_cols; col_start += THREADS_PER_WARP * TILE_SIZE) {
            int col_count = min(TILE_SIZE, num_cols - col_start);
            
            // Register accumulator (4x4)
            float acc[TILE_SIZE * TILE_SIZE];
            #pragma unroll
            for (int i = 0; i < TILE_SIZE * TILE_SIZE; i++) acc[i] = 0.0f;
            
            // Dot product loop
            for (int k_idx = 0; k_idx < num_shared_dim; k_idx++) {
                // Load A tile row
                float a_vals[TILE_SIZE];
                #pragma unroll
                for (int i = 0; i < TILE_SIZE; i++) {
                    a_vals[i] = mat_a[(row_start + i) * num_shared_dim + k_idx];
                }
                
                // Load B tile column
                float b_vals[TILE_SIZE];
                #pragma unroll
                for (int j = 0; j < TILE_SIZE; j++) {
                    b_vals[j] = mat_b[k_idx * num_cols + col_start + j];
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
            for (int i = 0; i < row_count; i++) {
                #pragma unroll
                for (int j = 0; j < col_count; j++) {
                    if constexpr (add_to_output) {
                        mat_c[(row_start + i) * num_cols + (col_start + j)] += acc[i * TILE_SIZE + j];
                    } else {
                        mat_c[(row_start + i) * num_cols + (col_start + j)] = acc[i * TILE_SIZE + j];
                    }
                }
            }
        }
    }
}

// int8 version: dequantizes on-the-fly during matmul
template <bool add_to_output = false>
__device__ void matmul_warp_tiled_int8(
    const int8_t* mat_a,      // int8: num_rows x num_shared_dim (Q)
    const int8_t* mat_b,      // int8: num_shared_dim x num_cols (K)
    float* mat_c,             // float: num_rows x num_cols
    int num_rows,
    int num_cols,
    int num_shared_dim,
    float a_scale, float a_zero,  // Q dequant params
    float b_scale, float b_zero    // K dequant params
) {
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int num_threads = blockDim.x;
    int num_warps = num_threads / THREADS_PER_WARP;

    constexpr int TILE_SIZE = 4;
    
    // Each warp handles TILE_SIZE rows
    for (int row_start = TILE_SIZE * warp_id; row_start < num_rows; row_start += num_warps * TILE_SIZE) {
        int row_count = min(TILE_SIZE, num_rows - row_start);
        
        // Each lane handles TILE_SIZE columns
        for (int col_start = TILE_SIZE * lane_id; col_start < num_cols; col_start += THREADS_PER_WARP * TILE_SIZE) {
            int col_count = min(TILE_SIZE, num_cols - col_start);
            
            // Register accumulator (4x4)
            float acc[TILE_SIZE * TILE_SIZE];
            #pragma unroll
            for (int i = 0; i < TILE_SIZE * TILE_SIZE; i++) acc[i] = 0.0f;
            
            // Dot product loop
            for (int k_idx = 0; k_idx < num_shared_dim; k_idx++) {
                // Load A tile row (int8, dequantize on-the-fly)
                float a_vals[TILE_SIZE];
                #pragma unroll
                for (int i = 0; i < TILE_SIZE; i++) {
                    int8_t a_int8 = mat_a[(row_start + i) * num_shared_dim + k_idx];
                    a_vals[i] = (a_int8 - a_zero) * a_scale;
                }
                
                // Load B tile column (int8, dequantize on-the-fly)
                float b_vals[TILE_SIZE];
                #pragma unroll
                for (int j = 0; j < TILE_SIZE; j++) {
                    int8_t b_int8 = mat_b[k_idx * num_cols + col_start + j];
                    b_vals[j] = (b_int8 - b_zero) * b_scale;
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
            for (int i = 0; i < row_count; i++) {
                #pragma unroll
                for (int j = 0; j < col_count; j++) {
                    if constexpr (add_to_output) {
                        mat_c[(row_start + i) * num_cols + (col_start + j)] += acc[i * TILE_SIZE + j];
                    } else {
                        mat_c[(row_start + i) * num_cols + (col_start + j)] = acc[i * TILE_SIZE + j];
                    }
                }
            }
        }
    }
}

// int8 matmul: float A (probs) × int8 B (values) -> float C (output)
template <bool add_to_output = false>
__device__ void matmul_float_int8(
    const float* mat_a,    // float: num_rows x num_shared_dim (probs)
    const int8_t* mat_b,   // int8: num_shared_dim x num_cols (V)
    float* mat_c,          // float: num_rows x num_cols (output)
    int num_rows,
    int num_cols,
    int num_shared_dim,
    float b_scale, float b_zero  // V dequant params
) {
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int num_threads = blockDim.x;
    int num_warps = num_threads / THREADS_PER_WARP;

    constexpr int TILE_SIZE = 4;
    
    // Each warp handles TILE_SIZE rows
    for (int row_start = TILE_SIZE * warp_id; row_start < num_rows; row_start += num_warps * TILE_SIZE) {
        int row_count = min(TILE_SIZE, num_rows - row_start);
        
        // Each lane handles TILE_SIZE columns
        for (int col_start = TILE_SIZE * lane_id; col_start < num_cols; col_start += THREADS_PER_WARP * TILE_SIZE) {
            int col_count = min(TILE_SIZE, num_cols - col_start);
            
            // Register accumulator (4x4)
            float acc[TILE_SIZE * TILE_SIZE];
            #pragma unroll
            for (int i = 0; i < TILE_SIZE * TILE_SIZE; i++) acc[i] = 0.0f;
            
            // Dot product loop
            for (int k_idx = 0; k_idx < num_shared_dim; k_idx++) {
                // Load A tile row (float)
                float a_vals[TILE_SIZE];
                #pragma unroll
                for (int i = 0; i < TILE_SIZE; i++) {
                    a_vals[i] = mat_a[(row_start + i) * num_shared_dim + k_idx];
                }
                
                // Load B tile column (int8, dequantize on-the-fly)
                float b_vals[TILE_SIZE];
                #pragma unroll
                for (int j = 0; j < TILE_SIZE; j++) {
                    int8_t b_int8 = mat_b[k_idx * num_cols + col_start + j];
                    b_vals[j] = (b_int8 - b_zero) * b_scale;
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
            for (int i = 0; i < row_count; i++) {
                #pragma unroll
                for (int j = 0; j < col_count; j++) {
                    if constexpr (add_to_output) {
                        mat_c[(row_start + i) * num_cols + (col_start + j)] += acc[i * TILE_SIZE + j];
                    } else {
                        mat_c[(row_start + i) * num_cols + (col_start + j)] = acc[i * TILE_SIZE + j];
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
    const int8_t* values, // V (kv_block_size x head_dim), int8
    int q_block_size,
    int kv_block_size,
    int head_dim,
    float sqrt_head_dim,
    float V_scale, float V_zero  // V dequant params
) {
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int num_threads = blockDim.x;
    int num_warps = num_threads / THREADS_PER_WARP;

    // Process one query row per warp
    for (int q_row = warp_id; q_row < q_block_size; q_row += num_warps) {
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
    
    __syncthreads();
    
    // Step 5: Accumulate O += (softmax probs) @ V (dequantize V on-the-fly)
    matmul_float_int8<true>(scores, values, output, q_block_size, head_dim, kv_block_size, V_scale, V_zero);
}

template<int Br, int Bc>
__global__ void fa_int8_kernel(
    const int8_t* Q,    // [N, d] int8
    const int8_t* K,    // [N, d] int8
    const int8_t* V,    // [N, d] int8
    float* O,           // [N, d] float
    const int N,        // Sequence length
    const int d,        // Dimension per head
    const float scale,  // softmax_scale (1/sqrt(d) is passed as scale)
    const float Q_scale, const float Q_zero,
    const float K_scale, const float K_zero,
    const float V_scale, const float V_zero
) {
    // One block per Br Q-rows
    const int q_block_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;

    // Shared memory layout with exact sizes (no wasteful max() allocations)
    extern __shared__ char shared_mem_char[];
    float* shared_mem_float = (float*)shared_mem_char;
    
    // Float allocations (exact sizes)
    float *output = shared_mem_float;                           // [Br x d]
    float *scores = &output[Br * d];                           // [Br x Bc]
    float *sum_exp = &scores[Br * Bc];                         // [Br]
    float *max_statistics = &sum_exp[Br];                      // [2*Br]
    
    // int8 allocations (after float data)
    int8_t* shared_mem_int8 = (int8_t*)&max_statistics[2 * Br];
    int8_t *q_block = shared_mem_int8;                         // [Br x d]
    int8_t *kv_block = &q_block[Br * d];                       // [Bc x d]
    
    float* max_prev = max_statistics;
    float* max_cur_ptr = &max_statistics[Br];

    int q_rows = min(Br, N - q_block_idx * Br);
    
    // Load Q block (int8)
    for (int idx = tid; idx < Br * d; idx += num_threads) {
        int row = idx / d;
        int col = idx % d;
        if (q_block_idx * Br + row < N) {
            q_block[idx] = Q[(q_block_idx * Br + row) * d + col];
        } else {
            q_block[idx] = 0;
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
        
        // Load K (transposed, int8)
        for (int idx = tid; idx < Bc * d; idx += num_threads) {
            int row = idx / d;
            int col = idx % d;
            int k_idx = kv_block_idx * Bc + row;
            if (k_idx < N) {
                // Store transposed: kv_block[col * Bc + row] = K[k_idx * d + col]
                kv_block[col * Bc + row] = K[k_idx * d + col];
            } else {
                kv_block[col * Bc + row] = 0;
            }
        }
        __syncthreads();
        
        // Compute scores = Q @ K^T (int8 @ int8 -> float)
        matmul_warp_tiled_int8(q_block, kv_block, scores, q_rows, kv_rows, d, Q_scale, Q_zero, K_scale, K_zero);
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
        online_softmax_and_accum_output(max_cur_ptr, max_prev, sum_exp, scores, output, kv_block, q_rows, kv_rows, d, sqrt_d, V_scale, V_zero);
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
void launch_fa_int8(
    const int8_t *Q, const int8_t *K, const int8_t *V,
    float *O,
    int N, int d,
    float alpha,
    float Q_scale, float Q_zero,
    float K_scale, float K_zero,
    float V_scale, float V_zero,
    cudaStream_t stream = 0
)
{
    int num_blocks = (N + Br - 1) / Br;
    // Shared memory: exact sizes only
    // Float: output[Br*d] + scores[Br*Bc] + sum_exp[Br] + max_statistics[2*Br]
    // Int8: q_block[Br*d] + kv_block[Bc*d]
    size_t shared_mem = (Br*d + Br*Bc + Br + 2*Br) * sizeof(float) + (Br*d + Bc*d) * sizeof(int8_t);
    
    int threads_per_block = 512;
    
    fa_int8_kernel<Br, Bc><<<num_blocks, threads_per_block, shared_mem, stream>>>(
        Q, K, V, O, N, d, alpha, Q_scale, Q_zero, K_scale, K_zero, V_scale, V_zero
    );
}

// Quantization kernel: float -> int8
__global__ void quantize_kernel(const float* input, int8_t* output, int numel, float scale, float zero) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        // Quantization: int8 = clamp(round((float / scale) + zero), -128, 127)
        float val = (input[idx] / scale) + zero;
        int int_val = (int)roundf(val);
        output[idx] = (int8_t)max(-128, min(127, int_val));
    }
}

extern "C" void solve(const float *Q, const float *K, const float *V, float *output, int N, int d_model, int h)
{
    // Create a kernel wrapper that quantizes per-head data and calls fa_int8
    auto fa_int8_kernel_wrapper = [](const float* q_s, const float* k_s, const float* v_s, float* out_s, int N_head, int d_head, float alpha, cudaStream_t stream, void* aux){
        (void)aux;
        constexpr int Br = 64;
        constexpr int Bc = 32;
        
        int num_blocks = (N_head + Br - 1) / Br;
        size_t shared_mem = (Br*d_head + Br*Bc + Br + 2*Br) * sizeof(float) + (Br*d_head + Bc*d_head) * sizeof(int8_t);
        int threads_per_block = 512;
        
        // Allocate temporary int8 arrays for this head
        int head_numel = N_head * d_head;
        int8_t *q_int8_head, *k_int8_head, *v_int8_head;
        cudaMalloc(&q_int8_head, head_numel * sizeof(int8_t));
        cudaMalloc(&k_int8_head, head_numel * sizeof(int8_t));
        cudaMalloc(&v_int8_head, head_numel * sizeof(int8_t));
        
        // Quantize per-head data to int8
        int threads = 256;
        int blocks = (head_numel + threads - 1) / threads;
        quantize_kernel<<<blocks, threads, 0, stream>>>(q_s, q_int8_head, head_numel, Q_SCALE, Q_ZERO);
        quantize_kernel<<<blocks, threads, 0, stream>>>(k_s, k_int8_head, head_numel, K_SCALE, K_ZERO);
        quantize_kernel<<<blocks, threads, 0, stream>>>(v_s, v_int8_head, head_numel, V_SCALE, V_ZERO);
        
        // Launch the int8 kernel
        fa_int8_kernel<Br, Bc><<<num_blocks, threads_per_block, shared_mem, stream>>>(
            q_int8_head, k_int8_head, v_int8_head, out_s, N_head, d_head, alpha,
            Q_SCALE, Q_ZERO, K_SCALE, K_ZERO, V_SCALE, V_ZERO
        );
        
        // Free temporary int8 arrays
        cudaFree(q_int8_head);
        cudaFree(k_int8_head);
        cudaFree(v_int8_head);
    };
    
    launch(Q, K, V, output, N, d_model, h, fa_int8_kernel_wrapper, 0);
}
