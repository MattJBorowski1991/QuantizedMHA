#include <cuda_runtime.h>
#include <math.h>
#include <assert.h>
#include <limits>
#include "../include/config.h"
#include "../include/launchers.h"
#include "../utils/utils.cu"
#include <stdio.h>

#include "mma.h"
#include <cuda_fp16.h>
using namespace nvcuda;

// Flash attention with Tensor Cores v1
// WMMA_M=16 rows of Q owned by 1 warp
// 1 warp owns a 16 x d chunk of Q and performs serial wmma, one 16x16 tile per iteration

#define THREADS_PER_WARP 32
#define WARPS_PER_TILE_ROW 2
#define WARP_TILE_ROWS 8
#define WARPS_PER_BLOCK (WARP_TILE_ROWS * WARPS_PER_TILE_ROW)
#define FULL_MASK 0xffffffff
#define PAD 0

//Tensor Core parameters
constexpr int WMMA_M = 8;
constexpr int WMMA_N = 32;
constexpr int WMMA_K = 16;

static_assert(Br == WMMA_M * WARP_TILE_ROWS, "Block size needs to equal number of warps times number of rows each warp handles");

// Bank conflict avoidance via XOR-based swizzling
// Spreads column bits to different banks, reducing conflicts on 32-bank shared memory
__device__ __forceinline__ int swizzle_index(int row, int col, int stride) {
    int linear = row * stride + col;
    int swizzle_bits = (col >> 4) & 15;  // Extract bits 4-7 of column
    return linear ^ swizzle_bits;  // XOR to spread bank distribution
}

template <bool add_to_output = false, int THREADS, int M, int N, int K>
__device__ __forceinline__ void wmma_A_B(
    const half* __restrict__ A,     // M x K (Q: Br x d, or P: Br x Bc)
    const half* __restrict__ B,    //  K x N (K^T: d x Bc, or V: Bc x d)
    float* C,                       //  M x N (scores: Br x Bc, or output: Br x d)
    float* c_scratch,               // Shared memory scratch for inter-warp communication
    int stride_A = 0,               // stride for A rows (0 = default K)
    int stride_B = 0,               // stride for B rows (0 = default N)
    int stride_C = 0                // stride for C rows (0 = default N)
){
    // Use default strides if not provided
    if (stride_A == 0) stride_A = K;
    if (stride_B == 0) stride_B = N;
    if (stride_C == 0) stride_C = N;

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    
    // NEW: Two warps share WMMA_M rows, split K dimension
    int warp_tile_row_id = warp_id / WARPS_PER_TILE_ROW;           // Which pair of warps (0, 1, 2, ...)
    int warp_tile_col_id = warp_id % WARPS_PER_TILE_ROW;      // Position within pair (0 = left half, 1 = right half)

    int warp_row = warp_tile_row_id * WMMA_M;
    if(warp_row >= M) return;
    
    // Determine K range for this warp
    int k_start = warp_tile_col_id * (K / WARPS_PER_TILE_ROW);
    int k_end = (warp_tile_col_id + 1) * (K / WARPS_PER_TILE_ROW);

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // Shared memory pointers for inter-warp accumulation
    // For each tile row: need to store the results of the left warp and the right warp and add them at the end
    // The below buffers are for the output of the wmma (MxN), i.e. each warp pair will produce: 
    // (i) left warp: WMMA_M x N partial results; (ii) right warp: WMMA x N partial results
    // => 2 x WMMA_M x N total per pair
    float* left_warp_res = c_scratch + warp_tile_row_id * (WMMA_M * WARPS_PER_TILE_ROW * N) + 0 * (WMMA_M * N);
    float* right_warp_res = c_scratch + warp_tile_row_id * (WMMA_M * WARPS_PER_TILE_ROW * N) + 1 * (WMMA_M * N);

    // Process each column chunk of N
    for(int b_col = 0; b_col < N; b_col += WMMA_N){
        // Initialize accumulator for THIS warp's K range
        wmma::fill_fragment(c_frag, 0.0f);

        // Accumulate over this warp's K range
        for(int k = k_start; k < k_end; k += WMMA_K){
            int a_col = k;
            int b_row = k;

            const half *a_ptr = A + warp_row * stride_A + a_col;
            const half *b_ptr = B + b_row * stride_B + b_col;   

            if(a_col + WMMA_K <= k_end && b_col + WMMA_N <= N){
                wmma::load_matrix_sync(a_frag, a_ptr, stride_A);
                wmma::load_matrix_sync(b_frag, b_ptr, stride_B);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
        }

        // Store this warp's partial result to shared memory
        if(warp_tile_col_id == 0){
            wmma::store_matrix_sync(left_warp_res, c_frag, N, wmma::mem_row_major);
        } else {
            wmma::store_matrix_sync(right_warp_res, c_frag, N, wmma::mem_row_major);
        }
        
        __syncthreads();  // Ensure both warps have stored their partial results

        // Left warp combines and stores final result
        if(warp_tile_col_id == 0){
            float* c_dst = C + warp_row * stride_C + b_col;
            
            if constexpr (add_to_output) {
                // Load existing C values
                wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_existing;
                wmma::load_matrix_sync(c_existing, c_dst, stride_C, wmma::mem_row_major);
                
                // Add left and right halfs
                for(int i = 0; i < c_existing.num_elements; ++i) {
                    c_existing.x[i] += (left_warp_res[i] + right_warp_res[i]);
                }
                
                wmma::store_matrix_sync(c_dst, c_existing, stride_C, wmma::mem_row_major);
            } else {
                // No existing output, just combine left and right halves
                wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_combined;
                wmma::fill_fragment(c_combined, 0.0f);
                
                // Element-wise addition of both halves
                for(int i = 0; i < c_combined.num_elements; ++i) {
                    c_combined.x[i] = left_warp_res[i] + right_warp_res[i];
                }
                
                wmma::store_matrix_sync(c_dst, c_combined, stride_C, wmma::mem_row_major);
            }
        }
    }
}

// Online softmax: WMMA_M rows per warp (matching matmul parallelism)
// Scale scores, find max, compute softmax probs, update statistics, rescale output, accumulate O += P @ V
template<int Br, int Bc, int THREADS, int Lc, int d, int stride_scores = 0, int stride_output = 0>
__device__ __forceinline__ void online_softmax_and_accum_output(
    float* max_cur,
    const float* max_prev,
    float* sum_exp,
    float* scores,     // to be overwritten with softmax probs, layout [Br x (Bc + PAD)]
    half* scores_fp16, // quantized scores [Br x (Bc + PAD)] 
    float* output,
    const half *values, // V (Bc x d)
    float inv_sqrt_d,
    float* c_scratch // Scratch buffer for inter-warp communication in P@V wmma_A_B
) {
    // Use padded strides
    const int sc_stride = (stride_scores == 0) ? (Bc + PAD) : stride_scores;
    const int out_stride = (stride_output == 0) ? (d + PAD) : stride_output;

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int warp_tile_row_id = warp_id / WARPS_PER_TILE_ROW;  // Which pair of warps
    int warp_tile_col_id = warp_id % WARPS_PER_TILE_ROW;  // Position within pair (0=left, 1=right)

    // Only LEFT warp (warp_tile_col_id==0) executes online_softmax
    // RIGHT warp (warp_tile_col_id==1) will participate in wmma_A_B P@V only
    if (warp_tile_col_id == 0) {
        // Process WMMA_M query rows assigned to this warp pair
        int row_start = warp_tile_row_id * WMMA_M;
        if (row_start < Br) {
        
        // Local arrays to track max, sum, and exp_max_diff for WMMA_M rows
        float max_new[WMMA_M], sum_new[WMMA_M], exp_max_diff[WMMA_M];
        #pragma unroll
        for (int i = 0; i < WMMA_M; i++) {
            max_new[i] = max_prev[row_start + i];
            sum_new[i] = 0.0f;
        }
        
        // Step 1: Find max in scores for each row and scale by 1/sqrt(d)
        for (int col_start = Lc * lane_id; col_start < Bc; col_start += THREADS_PER_WARP * Lc) {
            #pragma unroll
            for (int i = 0; i < WMMA_M; i++) {
                #pragma unroll
                for (int j = 0; j < Lc; j++) {
                    int row_idx = row_start + i;
                    int col_idx = col_start + j;
                    if (col_idx < Bc) {
                        int sidx = swizzle_index(row_idx, col_idx, sc_stride);
                        float score_scaled = scores[sidx] * inv_sqrt_d;
                        max_new[i] = fmaxf(max_new[i], score_scaled);
                        scores[sidx] = score_scaled;
                    }
                }
            }
        }
        
        // Warp reduction to get global max for each of the WMMA_M rows
        #pragma unroll
        for (int i = 0; i < WMMA_M; i++) {
            #pragma unroll
            for (int shift = THREADS_PER_WARP / 2; shift >= 1; shift >>= 1) {
                max_new[i] = fmaxf(max_new[i], __shfl_xor_sync(FULL_MASK, max_new[i], shift));
            }
        }

        // Step 2: Compute exp(score - max) and accumulate sum
        for (int col_start = Lc * lane_id; col_start < Bc; col_start += THREADS_PER_WARP * Lc) {
            #pragma unroll
            for (int i = 0; i < WMMA_M; i++) {
                #pragma unroll
                for (int j = 0; j < Lc; j++) {
                    int row_idx = row_start + i;
                    int col_idx = col_start + j;
                    int sidx = swizzle_index(row_idx, col_idx, sc_stride);

                    if (row_idx < Br && col_idx < Bc) {
                        //calculate scores, accumulate sum
                        float prob = expf(scores[sidx] - max_new[i]);
                        scores[sidx] = prob;
                        sum_new[i] += prob;

                        //quantize scores for later wmma_A_B
                        scores_fp16[sidx] = __float2half(prob);
                    }else{
                        // Unroll: scores[sidx] already written above
                        scores_fp16[sidx] = __float2half(0.0f);
                    }
                }
            }
        }
        
        // Warp reduction to get global sum for each row
        #pragma unroll
        for (int i = 0; i < WMMA_M; i++) {
            #pragma unroll
            for (int shift = THREADS_PER_WARP / 2; shift >= 1; shift >>= 1) {
                sum_new[i] += __shfl_xor_sync(FULL_MASK, sum_new[i], shift);
            }
        }

        // Step 3: Update max and sum statistics (only first lane writes)
        #pragma unroll
        for (int i = 0; i < WMMA_M; i++) {
            exp_max_diff[i] = expf(max_prev[row_start + i] - max_new[i]);
            if (lane_id == 0) {
                max_cur[row_start + i] = max_new[i];
                sum_exp[row_start + i] = exp_max_diff[i] * sum_exp[row_start + i] + sum_new[i];
            }
        }
        __syncthreads();  // Ensure all threads see updated sum_exp and max_cur
        
        // Step 4: Rescale output accumulator by exp(max_old - max_new) for each row
        for (int d_idx = lane_id; d_idx < d; d_idx += THREADS_PER_WARP) {
            #pragma unroll
            for (int i = 0; i < WMMA_M; i++) {
                int oidx = swizzle_index(row_start + i, d_idx, out_stride);
                output[oidx] *= exp_max_diff[i];
            }
        }
        }  // Close the (row_start < Br) if
    }  // Close the (warp_tile_col_id == 0) if
    
    __syncthreads();
    
    // Step 5: Accumulate O += (softmax probs) @ V (both warps in pair collaborate)
    // scores_fp16: Br x Bc with stride Bc+PAD
    // values: Bc x d with stride d+PAD
    // output: Br x d with stride d+PAD
    wmma_A_B<true, THREADS, Br, d, Bc>(scores_fp16, values, output, c_scratch, Bc + PAD, d + PAD, d + PAD);
    //previously: matmul_warp_tiled<true, THREADS, Wr, Lc>(scores, values, output, Br, d, Bc, Bc + 1, d, d);
}

template<int Br, int Bc, int THREADS, int d, int Lc>
__global__ void fa_kernel(
    const float* Q,    // [N, d]
    const float* K,    // [N, d]
    const float* V,    // [N, d]
    float* O,          // [N, d]
    const int N,       // Sequence length
    const float inv_sqrt_d  // 1/sqrt(d), pre-computed to save registers
) {
        
    assert(N % Br == 0 && "N must be a multiple of Br");
    assert(N % Bc == 0 && "N must be a multiple of Bc");

    // One block per Br of Q-rows
    const int q_block_idx = blockIdx.x;
    const int tid = threadIdx.x;

    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    int warp_tile_row_id = warp_id / WARPS_PER_TILE_ROW;  // Which pair of warps (for row distribution)
    
    // Shared memory layout with proper 16-byte alignment for tensor core operations
    // Use 1D arrays with stride-based indexing for wmma_A_B compatibility
    __shared__ float output[Br * (d + PAD)];
    __shared__ __align__(16) half q_block[Br * (d + PAD)];
    __shared__ __align__(16) half kt[d * (Bc + PAD)];
    __shared__ float scores[Br * (Bc + PAD)];
    __shared__ __align__(16) half scores_fp16[Br * (Bc + PAD)];
    __shared__ __align__(16) half values[Bc * (d + PAD)];
    __shared__ float sum_exp[Br];
    __shared__ float max_prev[Br];
    __shared__ float max_curr[Br];
    
    // Scratch buffer for inter-warp communication in wmma_A_B
    // Reused for both Q@K^T and P@V (sequential, non-overlapping operations)
    // Sized for max(d, Bc) to handle both Q@K^T (needs Bc) and P@V (needs d)
    __shared__ float c_scratch[2 * Br * ((d > Bc ? d : Bc) + PAD)];

    // Load Q block and quantize (warp pair processes its assigned row range)
    int row_start = warp_tile_row_id * WMMA_M;
    if (row_start < Br) {
        for (int col_start = Lc * lane_id; col_start < d; col_start += THREADS_PER_WARP * Lc) {
            #pragma unroll
            for (int i = 0; i < WMMA_M; i++) {
                #pragma unroll
                for (int j = 0; j < Lc; j++) {
                    int row = row_start + i;
                    int col = col_start + j;
                    if (q_block_idx * Br + row < N && col < d) {
                        q_block[row * (d + PAD) + col] = __float2half(Q[(q_block_idx * Br + row) * d + col]);
                    } else {
                        q_block[row * (d + PAD) + col] = __float2half(0.0f);
                    }
                }
            }
        }
    }
    
    // Initialize output, statistics (warp pair processes its assigned row range)
    row_start = warp_tile_row_id * WMMA_M;
    if (row_start < Br) {
        for (int col_start = Lc * lane_id; col_start < d; col_start += THREADS_PER_WARP * Lc) {
            #pragma unroll
            for (int i = 0; i < WMMA_M; i++) {
                #pragma unroll
                for (int j = 0; j < Lc; j++) {
                    int oidx = swizzle_index(row_start + i, col_start + j, d + PAD);
                    output[oidx] = 0.0f;
                }
            }
        }
    }
    for (int idx = lane_id; idx < Br; idx += THREADS_PER_WARP) {
        sum_exp[idx] = 0.0f;
        max_prev[idx] = 0.0f;  // Start at 0, not -INFINITY (first iteration will set it to actual max)
    }
    __syncthreads();

    // Main loop over K,V blocks
    int num_kv_blocks = (N + Bc - 1) / Bc;
    for (int kv_block_idx = 0; kv_block_idx < num_kv_blocks; kv_block_idx++) {
        
        // Initialize max_curr for this iteration
        // Warp pair warp_tile_row_id initializes rows [warp_tile_row_id*WMMA_M, (warp_tile_row_id+1)*WMMA_M)
        if (lane_id == 0) {
            for (int i = 0; i < WMMA_M; i++) {
                int row_idx = warp_tile_row_id * WMMA_M + i;
                if (row_idx < Br) {
                    max_curr[row_idx] = -INFINITY;
                }
            }
        }
        __syncthreads();
        
        // Load K & quantize (transposed with padding, warp pair may loop to cover all Bc rows)
        for (int row_start = warp_tile_row_id * WMMA_M; row_start < Bc; row_start += (WARPS_PER_BLOCK / WARPS_PER_TILE_ROW) * WMMA_M) {
            for (int col_start = Lc * lane_id; col_start < d; col_start += THREADS_PER_WARP * Lc) {
                #pragma unroll
                for (int i = 0; i < WMMA_M; i++) {
                    #pragma unroll
                    for (int j = 0; j < Lc; j++) {
                        int row = row_start + i;
                        int col = col_start + j;
                        int k_idx = kv_block_idx * Bc + row;
                        if (k_idx < N && col < d) {
                            kt[col * (Bc + PAD) + row] = __float2half(K[k_idx * d + col]);
                        } else {
                            kt[col * (Bc + PAD) + row] = __float2half(0.0f);
                        }
                    }
                }
            }
        }
        __syncthreads();
        
        // Compute scores = Q @ K^T = Br x Bc
        wmma_A_B<false, THREADS, Br, Bc, d>(q_block, kt, scores, c_scratch, d + PAD, Bc + PAD, Bc + PAD);
        //previously:  matmul_warp_tiled<false, THREADS, Wr, Lc>(q_block, kv_block, scores, Br, Bc, d, Bc + 1, Bc + 1);

        __syncthreads();
        
        // Load V (warp pair may loop to cover all Bc rows)
        for (int row_start = warp_tile_row_id * WMMA_M; row_start < Bc; row_start += (WARPS_PER_BLOCK / WARPS_PER_TILE_ROW) * WMMA_M) {
            for (int col_start = Lc * lane_id; col_start < d; col_start += THREADS_PER_WARP * Lc) {
                #pragma unroll
                for (int i = 0; i < WMMA_M; i++) {
                    #pragma unroll
                    for (int j = 0; j < Lc; j++) {
                        int row = row_start + i;
                        int col = col_start + j;
                        int k_idx = kv_block_idx * Bc + row;
                        if (k_idx < N && col < d) {
                            values[row * (d + PAD) + col] = __float2half(V[k_idx * d + col]);
                        } else {
                            values[row * (d + PAD) + col] = __float2half(0.0f);
                        }
                    }
                }
            }
        }
        __syncthreads();
        
        // Online softmax (Br x Bc) + output accumulation (Br x d)
        online_softmax_and_accum_output<Br, Bc, THREADS, Lc, d>(max_curr, max_prev, sum_exp, scores, scores_fp16, output, values, inv_sqrt_d, c_scratch);
        __syncthreads();
        
        // Copy max_curr to max_prev for next iteration (only for rows this warp pair processes)
        if (lane_id == 0) {
            for (int i = 0; i < WMMA_M; i++) {
                int row_idx = warp_tile_row_id * WMMA_M + i;
                if (row_idx < Br) {
                    max_prev[row_idx] = max_curr[row_idx];
                }
            }
        }
        __syncthreads();
    }

    // Epilogue: normalize output (warp pair processes its assigned row range)
    row_start = warp_tile_row_id * WMMA_M;
    if (row_start < Br) {
        for (int col_start = Lc * lane_id; col_start < d; col_start += THREADS_PER_WARP * Lc) {
            #pragma unroll
            for (int i = 0; i < WMMA_M; i++) {
                #pragma unroll
                for (int j = 0; j < Lc; j++) {
                    int row = row_start + i;
                    int col = col_start + j;
                    if (row < Br && col < d) {
                        int oidx = swizzle_index(row, col, d + PAD);
                        if (sum_exp[row] > 1e-10f) {
                            output[oidx] /= sum_exp[row];
                        } else {
                            output[oidx] = 0.0f;
                        }
                    }
                }
            }
        }
    }
    
    __syncthreads();
    
    // Store to global memory (warp pair processes its assigned row range)
    row_start = warp_tile_row_id * WMMA_M;
    if (row_start < Br) {
        for (int col_start = Lc * lane_id; col_start < d; col_start += THREADS_PER_WARP * Lc) {
            #pragma unroll
            for (int i = 0; i < WMMA_M; i++) {
                #pragma unroll
                for (int j = 0; j < Lc; j++) {
                    int row = row_start + i;
                    int col = col_start + j;
                    if (q_block_idx * Br + row < N && col < d) {
                        int oidx = swizzle_index(row, col, d + PAD);
                        O[(q_block_idx * Br + row) * d + col] = output[oidx];
                    }
                }
            }
        }
    }
}


template<int Br, int Bc, int THREADS, int d, int Lc>
void launch_fa(const float *Q, const float *K, const float *V, float *O, int N, cudaStream_t stream = 0)
{
    int BLOCKS = (N + Br - 1) / Br;
    float inv_sqrt_d = 1.0f / sqrtf((float)d);  // Pre-compute to save registers in kernel
    
    // Static __shared__ arrays are automatically allocated by compiler, no need to specify size
    fa_kernel<Br, Bc, THREADS, d, Lc><<<BLOCKS, THREADS, 0, stream>>>(Q, K, V, O, N, inv_sqrt_d);
}

extern "C" void solve(const float *Q, const float *K, const float *V, float *output, int N, int d_model, int h)
{   
    constexpr int THREADS = WARPS_PER_BLOCK * THREADS_PER_WARP;

    auto fa_kernel = [](const float* q_s, const float* k_s, const float* v_s, float* out_s, int N, const int d_param, float inv_sqrt_d_param, cudaStream_t stream, void* aux){
        (void)aux;
        (void)d_param;
        (void)inv_sqrt_d_param;
        launch_fa<Br, Bc, THREADS, d, Lc>(q_s, k_s, v_s, out_s, N, stream);
    };
    launch(Q, K, V, output, N, d_model, h, fa_kernel, 0);
}