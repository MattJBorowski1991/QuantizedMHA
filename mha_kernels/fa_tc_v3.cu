#include <cuda_runtime.h>
#include <math.h>
#include <assert.h>
#include <limits>
#include "../include/config.h"
#include "../include/launchers.h"
#include "../utils/utils.cu"
#include <stdio.h>
#include <climits>

#include "mma.h"
#include <cuda_fp16.h>
using namespace nvcuda;

// Flash attention with Tensor Cores and int8 quantization
// Supported int8 WMMA tile sizes: 8×8×32, 16×16×32, 16×8×32, 8×16×32. Only int32 output.
// 2 warps own 8xd of Q
// Extreme SRAM pressure with PAD=16 and extra buffers for int8 => had to decrease Br to 32 (from 64)

#define THREADS_PER_WARP 32
#define WARPS_PER_TILE_ROW 2
#define WARP_TILE_ROWS 4
#define WARPS_PER_BLOCK (WARP_TILE_ROWS * WARPS_PER_TILE_ROW)
#define THREADS (WARPS_PER_BLOCK * THREADS_PER_WARP)
#define FULL_MASK 0xffffffff
#define PAD 16

//Tensor Core parameters
constexpr int WMMA_M = 8;
constexpr int WMMA_N = 32;
constexpr int WMMA_K = 16;

static_assert(Br == WMMA_M * WARP_TILE_ROWS, "Block size needs to equal number of warps times number of rows each warp handles");

// asumme full_prec_in is provided either in DRAM (bool=1) or SRAM (bool=0)
// int8_out is always in SRAM
// tranposeInput for handling K^T

template<int M, int N, bool isInputinDram, bool transposeInput = false>
static __device__ __forceinline__ void fp32_to_int8sram(
    const float* __restrict__ fp32_in, // for inputs: Q: Nxd, K^T: N x d, P: Br x Bc, V: Bc x d)
    float* __restrict__ block_scales,
    int8_t* __restrict__ int8_out,
    int bid = 0  // Explicit block index (defaults to blockIdx.x if not provided)
){
    const int tid = threadIdx.x;
    int size = M * N;

    if (bid == 0) bid = blockIdx.x;  // Use provided bid or fall back to blockIdx.x

    const int lane_id = tid % THREADS_PER_WARP;
    const int warp_id = tid / THREADS_PER_WARP;

    //TODO: maybe put fp32_in into SRAM

    const float* block_in;
    if constexpr(isInputinDram){
        block_in = fp32_in + (size_t)bid * (size_t)size;
    }else{
        block_in = fp32_in;
    }
    int8_t* block_out = int8_out;

    //per-thread min and max calculated on the fly
    float local_min = INFINITY;
    float local_max = -INFINITY;
    for(int i = tid; i < size; i += THREADS){
        float v = block_in[i];
        local_min = fminf(local_min, v);
        local_max = fmaxf(local_max, v);
    }

    for(int off = THREADS_PER_WARP >> 1; off > 0; off >>= 1){
        float mmin = __shfl_down_sync(FULL_MASK, local_min, off);
        float mmax = __shfl_down_sync(FULL_MASK, local_max, off);
        local_min = fminf(local_min, mmin);
        local_max = fmaxf(local_max, mmax);
    }

    __shared__ float warp_maxs[WARPS_PER_BLOCK];
    __shared__ float warp_mins[WARPS_PER_BLOCK];


        if (lane_id == 0) {
        warp_maxs[warp_id] = local_max;
        warp_mins[warp_id] = local_min;
    }
    __syncthreads();

    float block_min = INFINITY;
    float block_max = -INFINITY;

    if (warp_id == 0) {
        // accumulate a subset of warp results per lane in warp 0
        for (int w = lane_id; w < WARPS_PER_BLOCK; w += THREADS_PER_WARP) {
            block_min = fminf(block_min, warp_mins[w]);
            block_max = fmaxf(block_max, warp_maxs[w]);
        }
        // intra-warp reduce to lane 0
        for (int off = THREADS_PER_WARP >> 1; off > 0; off >>= 1) {
            float mmin = __shfl_down_sync(FULL_MASK, block_min, off);
            float mmax = __shfl_down_sync(FULL_MASK, block_max, off);
            block_min = fminf(block_min, mmin);
            block_max = fmaxf(block_max, mmax);
        }

        if (lane_id == 0) {
            float sc = fmaxf(fmaxf(fabs(block_max), fabs(block_min)) / 127.0f, 1e-8f); 
            block_scales[bid] = sc;
        }
    }
    __syncthreads();

    //normalize to [-128, 127]
    float inv_sc = 1.0f / block_scales[bid];
    for(int i = tid; i < size; i += THREADS){

        float v;
        if constexpr(transposeInput){
            int row = i / N;
            int col = i % N;
            v = block_in[col * N + row];
        }else{
            v = block_in[i];
        }

        float scaled = v * inv_sc;
        int rounded = __float2int_rn(scaled);
        rounded = (rounded < -128) ? -128 : ((rounded > 127) ? 127 : rounded);
        block_out[i] = static_cast<int8_t>(rounded);
    }
}

//int32_in in SRAM, fp32_out in DRAM (P@V) or SRAM (scores=Q@K^T)
template<int M, int N, bool isOutputinDram>
static __device__ __forceinline__ void int32sram_to_fp32(
    const int* __restrict__ int32_in,
    float* __restrict__ block_scales_A,
    float* __restrict__ block_scales_B,
    float* __restrict__ fp32_out,
    int bid = 0  // Explicit block index (defaults to blockIdx.x if not provided)
){
    const int tid = threadIdx.x;
    int size = M * N;
    if (bid == 0) bid = blockIdx.x;  // Use provided bid or fall back to blockIdx.x

    const int* block_in = int32_in;
    float* block_out;

    if constexpr(isOutputinDram){
        block_out = fp32_out + bid * (size_t)size;
    }else{
        block_out = fp32_out;
    }

    for(int i = tid; i < size; i += THREADS){
        block_out[i] = (float)(block_in[i]) * block_scales_A[bid] * block_scales_B[bid];
    }
}

// assume A, B, C are in SRAM
template <bool add_to_output = false, int M, int N, int K>
static __device__ __forceinline__ void wmma_A_B(
    const int8_t* __restrict__ A,     // M x K (Q: Br x d, or P: Br x Bc)
    const int8_t* __restrict__ B,    //  K x N (K^T: d x Bc, or V: Bc x d)
    int* C,                       //  M x N (scores: Br x Bc, or output: Br x d)
    int* c_scratch,                // Shared memory scratch for inter-warp communication
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
    
    // Determine K range for this warp
    int k_start = warp_tile_col_id * (K / WARPS_PER_TILE_ROW);
    int k_end = (warp_tile_col_id + 1) * (K / WARPS_PER_TILE_ROW);

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, int8_t, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, int8_t, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int> c_frag;

    // Shared memory pointers for inter-warp accumulation (half-precision storage)
    // For each tile row: need to store the results of the left warp and the right warp and add them at the end
    // The below buffers are for the output of the wmma (MxN), i.e. each warp pair will produce: 
    // (i) left warp: WMMA_M x N partial results; (ii) right warp: WMMA x N partial results
    // => 2 x WMMA_M x N total per pair
    int* left_warp_res = c_scratch + warp_tile_row_id * (WMMA_M * WARPS_PER_TILE_ROW * N) + 0 * (WMMA_M * N);
    int* right_warp_res = c_scratch + warp_tile_row_id * (WMMA_M * WARPS_PER_TILE_ROW * N) + 1 * (WMMA_M * N);

    // Process each column chunk of N
    for(int b_col = 0; b_col < N; b_col += WMMA_N){
        // Initialize accumulator for THIS warp's K range
        wmma::fill_fragment(c_frag, 0);

        // Accumulate over this warp's K range
        for(int k = k_start; k < k_end; k += WMMA_K){
            int a_col = k;
            int b_row = k;

            const int8_t *a_ptr = A + warp_row * stride_A + a_col;
            const int8_t *b_ptr = B + b_row * stride_B + b_col;   

            if(a_col + WMMA_K <= k_end && b_col + WMMA_N <= N){
                wmma::load_matrix_sync(a_frag, a_ptr, stride_A);
                wmma::load_matrix_sync(b_frag, b_ptr, stride_B);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
        }

        // Store this warp's partial result to shared memory
        // TODO: quantize int32 -> int8 and later dequantize to save SRAM
        if(warp_tile_col_id == 0){
            for(int i = 0; i < c_frag.num_elements; ++i) {
                left_warp_res[i] = c_frag.x[i];
            }
        } else {
            for(int i = 0; i < c_frag.num_elements; ++i) {
                right_warp_res[i] = c_frag.x[i];
            }
        }
        
        __syncthreads();  // Ensure both warps have stored their partial results

        // Left warp combines and stores final result
        if(warp_tile_col_id == 0){
            int* c_dst = C + warp_row * stride_C + b_col;
            
            if constexpr (add_to_output) {
                // Load existing C values
                wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int> c_existing;
                wmma::load_matrix_sync(c_existing, c_dst, stride_C, wmma::mem_row_major);
                
                // Add left and right halves (convert from half to float)
                for(int i = 0; i < c_existing.num_elements; ++i) {
                    c_existing.x[i] += (left_warp_res[i] + right_warp_res[i]);
                }
                
                wmma::store_matrix_sync(c_dst, c_existing, stride_C, wmma::mem_row_major);
            } else {
                // No existing output, just combine left and right halves
                wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int> c_combined;
                wmma::fill_fragment(c_combined, 0);
                
                // Element-wise addition of both halves (convert from half to float)
                for(int i = 0; i < c_combined.num_elements; ++i) {
                    c_combined.x[i] = (left_warp_res[i] + right_warp_res[i]);
                }
                
                wmma::store_matrix_sync(c_dst, c_combined, stride_C, wmma::mem_row_major);
            }
        }
    }
}

// Online softmax: WMMA_M rows per warp (matching matmul parallelism)
// Scale scores, find max, compute softmax probs, update statistics, rescale output, accumulate O += P @ V
template<int Br, int Bc, int Lc, int d, int stride_scores = 0, int stride_output = 0>
__device__ __forceinline__ void online_softmax_and_accum_output(
    float* __restrict__ sum_exp,
    float* __restrict__ max_prev,
    float* __restrict__ max_curr,
    int* scores_int32,     // Q@K^T, layout [Br x (Bc + PAD)]
    int8_t* scores_int8, // quantized scores [Br x (Bc + PAD)]
    float* scores_fp32, // for the softmax calculations
    const float* __restrict__ block_scales_Q,              // for dequantization of scores: int32->fp32
    const float* __restrict__ block_scales_Kt,             // as above
    float* __restrict__ block_scales_P,          // for subsequent quantization of dequantized scores: fp32->int8 to get P_next
    int* temp_output_int32, //accumulation of wmma for P@V
    float* output,    // P@V
    const int8_t* values, // V (Bc x d)
    const float* scales_V,
    float inv_sqrt_d,
    int* c_scratch, // Scratch buffer for inter-warp communication in P@V wmma_A_B
    int kv_block_idx // KV block index for correct scale indexing
) {
    // Use padded strides
    const int sc_stride = (stride_scores == 0) ? (Bc + PAD) : stride_scores;
    const int out_stride = (stride_output == 0) ? (d + PAD) : stride_output;

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int bid = blockIdx.x;  // Q block index
    int warp_tile_row_id = warp_id / WARPS_PER_TILE_ROW;  // Which pair of warps
    int warp_tile_col_id = warp_id % WARPS_PER_TILE_ROW;  // Position within pair (0=left, 1=right)

    //read block_scales_Q and block_scales_Kt to dequantize
    int32sram_to_fp32<Br, Bc + PAD, false>(scores_int32, const_cast<float*>(block_scales_Q + bid), const_cast<float*>(block_scales_Kt + kv_block_idx), scores_fp32, 0);

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
                            float score_scaled = scores_fp32[row_idx * sc_stride + col_idx] * inv_sqrt_d;
                            max_new[i] = fmaxf(max_new[i], score_scaled);
                            scores_fp32[row_idx * sc_stride + col_idx] = score_scaled;
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
                        int idx = row_idx * sc_stride + col_idx;

                        if (row_idx < Br && col_idx < Bc) {
                            //calculate scores, accumulate sum
                            float prob = expf(scores_fp32[row_idx * sc_stride + col_idx] - max_new[i]);
                            scores_fp32[idx] = prob;
                            sum_new[i] += prob;
                        }else{
                            scores_fp32[idx] = 0.0f;
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
                    max_curr[row_start + i] = max_new[i];
                    sum_exp[row_start + i] = exp_max_diff[i] * sum_exp[row_start + i] + sum_new[i];
                }
            }
            __syncthreads();  // Ensure all threads see updated sum_exp and max_cur
            
            // Step 4: Rescale output accumulator by exp(max_old - max_new) for each row
            for (int d_idx = lane_id; d_idx < d; d_idx += THREADS_PER_WARP) {
                #pragma unroll
                for (int i = 0; i < WMMA_M; i++) {
                    output[(row_start + i) * out_stride + d_idx] *= exp_max_diff[i];
                }
            }
        }  // Close the (row_start < Br) if
    }  // Close the (warp_tile_col_id == 0) if
    
    __syncthreads();

    //write to block_scales_P when quantizing
    fp32_to_int8sram<Br, Bc + PAD, false, false>(scores_fp32, block_scales_P, scores_int8, kv_block_idx);
    
    // Step 5: Accumulate O += (softmax probs) @ V (both warps in pair collaborate)
    // scores_int8: Br x Bc with stride Bc+PAD
    // values: Bc x d with stride d+PAD
    // output: Br x d with stride d+PAD
    wmma_A_B<false, Br, d, Bc>(scores_int8, values, temp_output_int32, c_scratch, Bc + PAD, d + PAD, d + PAD);
    for(int i = tid; i < Br * (d + PAD); i += THREADS){
        output[i] += (float)temp_output_int32[i] * block_scales_P[kv_block_idx] * scales_V[kv_block_idx];
    }
}


template<int Br, int Bc, int d, int Lc>
__device__ __forceinline__ void init_output_and_stats(
    float* __restrict__ output,
    float* __restrict__ sum_exp,
    float* __restrict__ max_prev,
    float* __restrict__ max_curr
){  
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    int warp_tile_row_id = warp_id / WARPS_PER_TILE_ROW;
    int row_start = warp_tile_row_id * WMMA_M;

    // Initialize output, statistics (warp pair processes its assigned row range)
    if (row_start < Br) {
        for (int col_start = Lc * lane_id; col_start < d; col_start += THREADS_PER_WARP * Lc) {
            #pragma unroll
            for (int i = 0; i < WMMA_M; i++) {
                #pragma unroll
                for (int j = 0; j < Lc; j++) {
                    output[(row_start + i) * (d + PAD) + (col_start + j)] = 0.0f;
                }
            }
        }
    }
    for (int idx = lane_id; idx < Br; idx += THREADS_PER_WARP) {
        sum_exp[idx] = 0.0f;
        max_prev[idx] = 0.0f;  // Start at 0, not -INFINITY (first iteration will set it to actual max)
        max_curr[idx] = -INFINITY;
    }
    __syncthreads();
}

template<int Br, int Bc, int d, int Lc>
__global__ void fa_kernel(
    const float* Q,    // [N, d]
    const float* K,    // [N, d]
    const float* V,    // [N, d]
    float* block_scales,
    float* O,          // [N, d]
    const int N,       // Sequence length
    const int BLOCKS,  // Number of Q blocks (for scale indexing)
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
    constexpr int max_d_bc = (d > Bc ? d : Bc);  // Larger of d and Bc, used for unified buffer sizing
    
    // Shared memory layout with proper 16-byte alignment for tensor core operations
    // Use 1D arrays with stride-based indexing for wmma_A_B compatibility
    // Combined q_block and scores_int8 buffer (reused sequentially, both Br x (max_d_bc + PAD))
    // q_block: stores Q block, used in Q@K^T computation (lines ~355-406)
    // scores_int8: stores quantized softmax probabilities, used in P@V computation (lines ~457-461)
    // Safe reuse: q_block is unused after Q@K^T due to __syncthreads() at line 422
    __shared__ __align__(16) int8_t first_q_block_then_scores_int8[Br * (max_d_bc + PAD)];
    
    int8_t* q_block = first_q_block_then_scores_int8;
    int8_t* scores_int8 = first_q_block_then_scores_int8;
    
    __shared__ float output[Br * (d + PAD)];

    __shared__ union{
        float scores_fp32[Br * (Bc + PAD)];
        int scores_int32[Br * (Bc + PAD)];
    } scores;

    float* scores_fp32 = scores.scores_fp32;
    int* scores_int32 = scores.scores_int32;
    
    // Statistics arrays (separate instead of struct to simplify passing to helper functions)
    __shared__ float sum_exp[Br];
    __shared__ float max_prev[Br];
    __shared__ float max_curr[Br];
    
    // Shared buffer reused for both K^T and V (sequential, non-overlapping)
    // Safe allocation: (d + PAD) × (Bc + PAD) covers both access patterns
    // kt accessed as: kt[col * (Bc + PAD) + row] where col ∈ [0,d), row ∈ [0,Bc)
    // values accessed as: values[row * (d + PAD) + col] where row ∈ [0,Bc), col ∈ [0,d)
    __shared__ __align__(16) int8_t kv_buffer[(d + PAD) * (Bc + PAD)];
    
    // Convenience pointers for clarity (both point to same buffer at different times)
    int8_t* kt = kv_buffer;
    int8_t* values = kv_buffer; //combine/reuse buffers
    
    // Scratch buffer for inter-warp communication in wmma_A_B
    // Reused for both Q@K^T and P@V (sequential, non-overlapping operations)
    // Sized for max_d_bc to handle both Q@K^T (needs Bc) and P@V (needs d)
    // No padding needed: c_scratch is accessed via WMMA (bank-aware) then scalar reductions only
    __shared__ int c_scratch[2 * Br * max_d_bc];

    float *block_scales_Q = block_scales;
    float *block_scales_Kt = block_scales + BLOCKS;
    float *block_scales_V = block_scales + 2 * BLOCKS; 
    float *block_scales_P = block_scales + 3 * BLOCKS;
    __shared__ int temp_output_int32[Br * (d + PAD)];

    // Load Q block and quantize (warp pair processes its assigned row range)
    int row_start = warp_tile_row_id * WMMA_M;

    //quantize Q once (same for all kv_block_idx iterations)
    fp32_to_int8sram<Br, d + PAD, true, false>(Q, block_scales_Q, q_block);

    
    init_output_and_stats<Br, Bc, d, Lc>(output, sum_exp, max_prev, max_curr);

    // Main loop over K,V blocks
    int num_kv_blocks = (N + Bc - 1) / Bc;
    for (int kv_block_idx = 0; kv_block_idx < num_kv_blocks; kv_block_idx++) {
        // Compute pointers for this K,V block iteration
        const float *K_block = K + kv_block_idx * Bc * d;
        const float *V_block = V + kv_block_idx * Bc * d;
        
        // Load K and V blocks for this iteration with proper kv_block_idx offset
        fp32_to_int8sram<Bc, d + PAD, true, true>(K_block, block_scales_Kt, kt);
        fp32_to_int8sram<Bc, d + PAD, true, false>(V_block, block_scales_V, values);
        __syncthreads();
        
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
            
        // Compute scores = Q @ K^T = Br x Bc
        wmma_A_B<false, Br, Bc, d>(q_block, kt, scores_int32, c_scratch, d + PAD, d + PAD, Bc + PAD);
        __syncthreads();
        
        // Online softmax (Br x Bc) + output accumulation (Br x d)
        online_softmax_and_accum_output<Br, Bc, Lc, d, 0, 0>
        (sum_exp, max_prev, max_curr, scores_int32, scores_int8, scores_fp32, block_scales_Q, block_scales_Kt, block_scales_P, temp_output_int32, output, values, block_scales_V, inv_sqrt_d, c_scratch, kv_block_idx);
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
    } //end of kv_block_idx for loop iteration


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
                        if (sum_exp[row] > 1e-10f) {
                            output[row * (d + PAD) + col] /= sum_exp[row];
                        } else {
                            output[row * (d + PAD) + col] = 0.0f;
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
                        O[(q_block_idx * Br + row) * d + col] = output[row * (d + PAD) + col];
                    }
                }
            }
        }
    }
}


template<int Br, int Bc, int d, int Lc>
void launch_fa(const float *Q, const float *K, const float *V, float *O, int N, cudaStream_t stream = 0)
{
    int BLOCKS = (N + Br - 1) / Br;
    float inv_sqrt_d = 1.0f / sqrtf((float)d);  // Pre-compute to save registers in kernel

    float* d_block_scales = nullptr;
    cudaMalloc(&d_block_scales, 4 * BLOCKS * sizeof(float)); //One for each: Q, K, P, V
    
    // Static __shared__ arrays are automatically allocated by compiler, no need to specify size
    fa_kernel<Br, Bc, d, Lc><<<BLOCKS, THREADS, 0, stream>>>(Q, K, V, d_block_scales, O, N, BLOCKS, inv_sqrt_d);

    cudaFree(d_block_scales);
}

extern "C" void solve(const float *Q, const float *K, const float *V, float *output, int N, int d_model, int h)
{   
    auto fa_kernel = [](const float* q_s, const float* k_s, const float* v_s, float* out_s, int N, const int d_param, float inv_sqrt_d_param, cudaStream_t stream, void* aux){
        (void)aux;
        (void)d_param;
        (void)inv_sqrt_d_param;
        launch_fa<Br, Bc, d, Lc>(q_s, k_s, v_s, out_s, N, stream);
    };
    launch(Q, K, V, output, N, d_model, h, fa_kernel, 0);
}