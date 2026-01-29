#include <cuda_runtime.h>
#include <math.h>
#include <limits>
#include "../include/config.h"
#include "../include/launchers.h"
#include "../utils/utils.cu"

// TODO: correctness check failed before profiling - FIX

// Flash attention with int8 quantization (SOTA workflow)
// - Inputs in float32, pre-quantized to int8
// - QK^T: int8 @ int8 → float32 (with upcast)
// - Softmax in float32 (accuracy-sensitive)
// - P @ V: float32 @ int8 → float32
// - Output in float32

#define FULL_MASK 0xffffffff
#define THREADS_PER_WARP 32

// Quantization kernel: float32 → int8
__global__ void quantize_kernel(
    const float* input,    // [N, d] float32
    int8_t* output,        // [N, d] int8
    int N, int d,
    float scale, float zero
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * d) {
        float val = input[idx];
        int8_t quantized = (int8_t)roundf((val - zero) / scale);
        output[idx] = quantized;
    }
}

// Dequantization helper: inline during matmul
__device__ inline float dequantize(int8_t val, float scale, float zero) {
    return ((float)val) * scale + zero;
}

// Shared memory block matrix multiply (int8 inputs → float32 output with upcast)
// Each warp: 4 rows, each lane (32 lanes): 4 columns per iteration
// Accumulates into registers as float32
template <bool add_to_output = false>
__device__ void matmul_warp_tiled(
    const int8_t* mat_a,      // num_rows x num_shared_dim (int8)
    const int8_t* mat_b,      // num_shared_dim x num_cols (int8)
    float* mat_c,             // num_rows x num_cols (float32 output)
    int num_rows,
    int num_cols,
    int num_shared_dim,
    float a_scale, float a_zero,
    float b_scale, float b_zero
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
            
            // Register accumulator (4x4) - float32 for numeric stability
            float acc[TILE_SIZE * TILE_SIZE];
            #pragma unroll
            for (int i = 0; i < TILE_SIZE * TILE_SIZE; i++) acc[i] = 0.0f;
            
            // Dot product loop with int8 inputs
            for (int k_idx = 0; k_idx < num_shared_dim; k_idx++) {
                // Load A tile row (int8, dequantize)
                float a_vals[TILE_SIZE];
                #pragma unroll
                for (int i = 0; i < TILE_SIZE; i++) {
                    a_vals[i] = dequantize(mat_a[(row_start + i) * num_shared_dim + k_idx], a_scale, a_zero);
                }
                
                // Load B tile column (int8, dequantize)
                float b_vals[TILE_SIZE];
                #pragma unroll
                for (int j = 0; j < TILE_SIZE; j++) {
                    b_vals[j] = dequantize(mat_b[k_idx * num_cols + col_start + j], b_scale, b_zero);
                }
                
                // Outer product (float32 accumulation)
                #pragma unroll
                for (int i = 0; i < TILE_SIZE; i++) {
                    #pragma unroll
                    for (int j = 0; j < TILE_SIZE; j++) {
                        acc[i * TILE_SIZE + j] += a_vals[i] * b_vals[j];
                    }
                }
            }
            
            // Store to C (float32)
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

// Online softmax: scale scores (float32), find max, compute softmax probs, update statistics, rescale output, accumulate O += P @ V
// P is float32 from softmax, V is int8 (dequantized on-the-fly)
__device__ void online_softmax_and_accum_output(
    float* max_cur,
    const float* max_prev,
    float* sum_exp,
    float* scores,         // Will be overwritten with softmax probs (float32)
    float* output,
    const int8_t* values,  // V (int8)
    int q_block_size,
    int kv_block_size,
    int head_dim,
    float sqrt_head_dim,
    float V_scale, float V_zero
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
    
    // Step 5: Accumulate O += (softmax probs) @ V (scores in float32, V in int8)
    // Reinterpret values as float* for matmul signature, dequantization happens inside
    matmul_warp_tiled<true>(
        (const int8_t*)scores, (const int8_t*)values, output, 
        q_block_size, head_dim, kv_block_size,
        1.0f, 0.0f, V_scale, V_zero  // scores already float32 (scale=1, zero=0), V is int8
    );
}

template<int Br, int Bc>
__global__ void fa_int8_kernel(
    const int8_t* Q,    // [N, d] int8 quantized
    const int8_t* K,    // [N, d] int8 quantized
    const int8_t* V,    // [N, d] int8 quantized
    float* O,           // [N, d] float output
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
    
    // Load Q block (already int8, store as-is in shared mem as float reinterpret for type matching)
    for (int idx = tid; idx < Br * d; idx += num_threads) {
        int row = idx / d;
        int col = idx % d;
        if (q_block_idx * Br + row < N) {
            q_block[idx] = (float)Q[(q_block_idx * Br + row) * d + col];  // Cast int8 to float for storage
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
        
        // Load K (transposed, int8 stored as float for type matching)
        for (int idx = tid; idx < Bc * d; idx += num_threads) {
            int row = idx / d;
            int col = idx % d;
            int k_idx = kv_block_idx * Bc + row;
            if (k_idx < N) {
                // Store transposed: kv_block[col * Bc + row] = K[k_idx * d + col]
                kv_block[col * Bc + row] = (float)K[k_idx * d + col];
            } else {
                kv_block[col * Bc + row] = 0.0f;
            }
        }
        __syncthreads();
        
        // Compute scores = Q @ K^T (both int8, output float32)
        matmul_warp_tiled(
            (const int8_t*)q_block, (const int8_t*)kv_block, scores, q_rows, kv_rows, d,
            Q_scale, Q_zero, K_scale, K_zero
        );
        __syncthreads();
        
        // Load V (int8 stored as float for type matching)
        for (int idx = tid; idx < Bc * d; idx += num_threads) {
            int row = idx / d;
            int col = idx % d;
            int k_idx = kv_block_idx * Bc + row;
            if (k_idx < N) {
                kv_block[idx] = (float)V[k_idx * d + col];
            } else {
                kv_block[idx] = 0.0f;
            }
        }
        __syncthreads();
        
        // Online softmax + output accumulation
        float sqrt_d = sqrtf((float)d);
        online_softmax_and_accum_output(
            max_cur_ptr, max_prev, sum_exp, scores, output, 
            (const int8_t*)kv_block, q_rows, kv_rows, d, sqrt_d,
            V_scale, V_zero
        );
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
    int alloc_size = max(Br * Bc, Bc * d);
    size_t shared_mem = (4 * alloc_size + 3 * Br) * sizeof(float);
    
    int threads_per_block = 512;
    
    fa_int8_kernel<Br, Bc><<<num_blocks, threads_per_block, shared_mem, stream>>>(
        Q, K, V, O, N, d, alpha, Q_scale, Q_zero, K_scale, K_zero, V_scale, V_zero
    );
}

extern "C" void solve(const float *Q_float, const float *K_float, const float *V_float, float *output, int N, int d_model, int h)
{
    int d_head = d_model / h;
    float alpha = 1.0f / sqrtf((float)d_head);
    const int nstreams = NSTREAMS;

    size_t per_q = (size_t)N * d_head;

    // Allocate quantized int8 buffers
    int8_t *q_int8, *k_int8, *v_int8;
    float *out;

    cudaMalloc(&q_int8, nstreams * per_q * sizeof(int8_t));
    cudaMalloc(&k_int8, nstreams * per_q * sizeof(int8_t));
    cudaMalloc(&v_int8, nstreams * per_q * sizeof(int8_t));
    cudaMalloc(&out, nstreams * per_q * sizeof(float));

    // Float temp buffers for input
    float *q_float, *k_float, *v_float;
    cudaMalloc(&q_float, nstreams * per_q * sizeof(float));
    cudaMalloc(&k_float, nstreams * per_q * sizeof(float));
    cudaMalloc(&v_float, nstreams * per_q * sizeof(float));

    cudaStream_t streams[NSTREAMS];
    for (int s = 0; s < nstreams; ++s) cudaStreamCreate(&streams[s]);

    for (int head = 0; head < h; ++head) {
        int curr_col = head * d_head;
        int s = head % nstreams;

        float *q_s = q_float + s * per_q;
        float *k_s = k_float + s * per_q;
        float *v_s = v_float + s * per_q;
        int8_t *q_int8_s = q_int8 + s * per_q;
        int8_t *k_int8_s = k_int8 + s * per_q;
        int8_t *v_int8_s = v_int8 + s * per_q;
        float *out_s = out + s * per_q;

        // Extract float inputs
        launch_extract_mat<TILE>(Q_float, q_s, 0, curr_col, N, d_model, N, d_head, streams[s]);
        launch_extract_mat<TILE>(K_float, k_s, 0, curr_col, N, d_model, N, d_head, streams[s]);
        launch_extract_mat<TILE>(V_float, v_s, 0, curr_col, N, d_model, N, d_head, streams[s]);

        // Pre-quantize Q, K, V to int8
        int threads = 256;
        int blocks = (N * d_head + threads - 1) / threads;
        quantize_kernel<<<blocks, threads, 0, streams[s]>>>(q_s, q_int8_s, N, d_head, Q_SCALE, Q_ZERO);
        quantize_kernel<<<blocks, threads, 0, streams[s]>>>(k_s, k_int8_s, N, d_head, K_SCALE, K_ZERO);
        quantize_kernel<<<blocks, threads, 0, streams[s]>>>(v_s, v_int8_s, N, d_head, V_SCALE, V_ZERO);

        // Launch FA with int8 inputs
        launch_fa_int8<Br, Bc>(
            (const int8_t*)q_int8_s, (const int8_t*)k_int8_s, (const int8_t*)v_int8_s,
            out_s, N, d_head, alpha,
            Q_SCALE, Q_ZERO, K_SCALE, K_ZERO, V_SCALE, V_ZERO,
            streams[s]
        );

        // Concat output
        launch_concat_mat<TILE>(output, out_s, 0, curr_col, N, d_model, N, d_head, streams[s]);
    }

    for (int s = 0; s < nstreams; ++s) cudaStreamSynchronize(streams[s]);
    for (int s = 0; s < nstreams; ++s) cudaStreamDestroy(streams[s]);

    cudaFree(q_int8);
    cudaFree(k_int8);
    cudaFree(v_int8);
    cudaFree(out);
    cudaFree(q_float);
    cudaFree(k_float);
    cudaFree(v_float);
}
