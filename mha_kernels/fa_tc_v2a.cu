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
#define PAD 16

//Tensor Core parameters
constexpr int WMMA_M = 8;
constexpr int WMMA_N = 32;
constexpr int WMMA_K = 16;

static_assert(Br == WMMA_M * WARP_TILE_ROWS, "Block size needs to equal number of warps times number of rows each warp handles");


template <bool add_to_output = false, int THREADS, int M, int N, int K>
__device__ __forceinline__ void wmma_A_B(
    const half* __restrict__ A,     // M x K (Q: Br x d, or P: Br x Bc)
    const half* __restrict__ B,    //  K x N (K^T: d x Bc, or V: Bc x d)
    float* C,                       //  M x N (scores: Br x Bc, or output: Br x d)
    half* c_scratch,                // Shared memory scratch for inter-warp communication (half-precision)
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

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // Shared memory pointers for inter-warp accumulation (half-precision storage)
    // For each tile row: need to store the results of the left warp and the right warp and add them at the end
    // The below buffers are for the output of the wmma (MxN), i.e. each warp pair will produce: 
    // (i) left warp: WMMA_M x N partial results; (ii) right warp: WMMA x N partial results
    // => 2 x WMMA_M x N total per pair
    half* left_warp_res = c_scratch + warp_tile_row_id * (WMMA_M * WARPS_PER_TILE_ROW * N) + 0 * (WMMA_M * N);
    half* right_warp_res = c_scratch + warp_tile_row_id * (WMMA_M * WARPS_PER_TILE_ROW * N) + 1 * (WMMA_M * N);

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

        // Store this warp's partial result to shared memory (convert float to half)
        if(warp_tile_col_id == 0){
            for(int i = 0; i < c_frag.num_elements; ++i) {
                left_warp_res[i] = __float2half(c_frag.x[i]);
            }
        } else {
            for(int i = 0; i < c_frag.num_elements; ++i) {
                right_warp_res[i] = __float2half(c_frag.x[i]);
            }
        }
        
        __syncthreads();  // Ensure both warps have stored their partial results

        // Left warp combines and stores final result
        if(warp_tile_col_id == 0){
            float* c_dst = C + warp_row * stride_C + b_col;
            
            if constexpr (add_to_output) {
                // Load existing C values
                wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_existing;
                wmma::load_matrix_sync(c_existing, c_dst, stride_C, wmma::mem_row_major);
                
                // Add left and right halves (convert from half to float)
                for(int i = 0; i < c_existing.num_elements; ++i) {
                    c_existing.x[i] += (__half2float(left_warp_res[i]) + __half2float(right_warp_res[i]));
                }
                
                wmma::store_matrix_sync(c_dst, c_existing, stride_C, wmma::mem_row_major);
            } else {
                // No existing output, just combine left and right halves
                wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_combined;
                wmma::fill_fragment(c_combined, 0.0f);
                
                // Element-wise addition of both halves (convert from half to float)
                for(int i = 0; i < c_combined.num_elements; ++i) {
                    c_combined.x[i] = __half2float(left_warp_res[i]) + __half2float(right_warp_res[i]);
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
    void* stats_ptr,                // Statistics struct pointer (to avoid circular dependency)
    float* scores,     // to be overwritten with softmax probs, layout [Br x (Bc + PAD)]
    half* scores_fp16, // quantized scores [Br x (Bc + PAD)] 
    float* output,
    const half *values, // V (Bc x d)
    float inv_sqrt_d,
    half* c_scratch // Scratch buffer for inter-warp communication in P@V wmma_A_B (half-precision)
) {
    // Cast stats pointer back to the correct type
    // (Using void* to avoid forward declaration issues)
    struct Statistics {
        float sum_exp;
        float max_prev;
        float max_curr;
    };
    Statistics* stats = (Statistics*)stats_ptr;
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
            max_new[i] = stats[row_start + i].max_prev;
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
                        float score_scaled = scores[row_idx * sc_stride + col_idx] * inv_sqrt_d;
                        max_new[i] = fmaxf(max_new[i], score_scaled);
                        scores[row_idx * sc_stride + col_idx] = score_scaled;
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
                        float prob = expf(scores[row_idx * sc_stride + col_idx] - max_new[i]);
                        scores[idx] = prob;
                        sum_new[i] += prob;

                        //quantize scores for later wmma_A_B
                        scores_fp16[idx] = __float2half(prob);
                    }else{
                        scores[idx] = 0.0f;
                        scores_fp16[idx] = __float2half(0.0f);
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
            exp_max_diff[i] = expf(stats[row_start + i].max_prev - max_new[i]);
            if (lane_id == 0) {
                stats[row_start + i].max_curr = max_new[i];
                stats[row_start + i].sum_exp = exp_max_diff[i] * stats[row_start + i].sum_exp + sum_new[i];
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
    constexpr int max_d_bc = (d > Bc ? d : Bc);  // Larger of d and Bc, used for unified buffer sizing
    
    // Shared memory layout with proper 16-byte alignment for tensor core operations
    // Use 1D arrays with stride-based indexing for wmma_A_B compatibility
    // Combined q_block and scores_fp16 buffer (reused sequentially, both Br x (max_d_bc + PAD))
    // q_block: stores Q block, used in Q@K^T computation (lines ~355-406)
    // scores_fp16: stores quantized softmax probabilities, used in P@V computation (lines ~457-461)
    // Safe reuse: q_block is unused after Q@K^T due to __syncthreads() at line 422
    __shared__ __align__(16) half first_q_block_then_scores_fp16[Br * (max_d_bc + PAD)];
    half* q_block = first_q_block_then_scores_fp16;
    half* scores_fp16 = first_q_block_then_scores_fp16;
    
    __shared__ float output[Br * (d + PAD)];
    __shared__ float scores[Br * (Bc + PAD)];
    
    // Combined statistics struct to reduce SRAM usage (768 bytes saved)
    struct Statistics {
        float sum_exp;
        float max_prev;
        float max_curr;
    };
    __shared__ Statistics stats[Br];
    
    // Shared buffer reused for both K^T and V (sequential, non-overlapping)
    // Safe allocation: (d + PAD) × (Bc + PAD) covers both access patterns
    // kt accessed as: kt[col * (Bc + PAD) + row] where col ∈ [0,d), row ∈ [0,Bc)
    // values accessed as: values[row * (d + PAD) + col] where row ∈ [0,Bc), col ∈ [0,d)
    // kt is loaded, used for Q@K^T, then no longer needed before V is loaded
    __shared__ __align__(16) half kv_buffer[(d + PAD) * (Bc + PAD)];
    
    // Convenience pointers for clarity (both point to same buffer at different times)
    half* kt = kv_buffer;
    half* values = kv_buffer;
    
    // Scratch buffer for inter-warp communication in wmma_A_B (half-precision for 50% memory savings)
    // Reused for both Q@K^T and P@V (sequential, non-overlapping operations)
    // Sized for max_d_bc to handle both Q@K^T (needs Bc) and P@V (needs d)
    // No padding needed: c_scratch is accessed via WMMA (bank-aware) then scalar reductions only
    __shared__ half c_scratch[2 * Br * max_d_bc];

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
                    output[(row_start + i) * (d + PAD) + (col_start + j)] = 0.0f;
                }
            }
        }
    }
    for (int idx = lane_id; idx < Br; idx += THREADS_PER_WARP) {
        stats[idx].sum_exp = 0.0f;
        stats[idx].max_prev = 0.0f;  // Start at 0, not -INFINITY (first iteration will set it to actual max)
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
                    stats[row_idx].max_curr = -INFINITY;
                }
            }
        }
        __syncthreads();
        
        // Load K & quantize (transposed with padding, warp pair may loop to cover all Bc rows)
        // Transposed to row-major layout (matching V) for better bank distribution
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
                            kt[row * (d + PAD) + col] = __float2half(K[k_idx * d + col]);
                        } else {
                            kt[row * (d + PAD) + col] = __float2half(0.0f);
                        }
                    }
                }
            }
        }
        __syncthreads();
        
        // Compute scores = Q @ K^T = Br x Bc
        // kt now row-major with stride d + PAD for better bank distribution
        wmma_A_B<false, THREADS, Br, Bc, d>(q_block, kt, scores, c_scratch, d + PAD, d + PAD, Bc + PAD);
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
        online_softmax_and_accum_output<Br, Bc, THREADS, Lc, d, 0, 0>((void*)stats, scores, scores_fp16, output, values, inv_sqrt_d, c_scratch);
        __syncthreads();
        
        // Copy max_curr to max_prev for next iteration (only for rows this warp pair processes)
        if (lane_id == 0) {
            for (int i = 0; i < WMMA_M; i++) {
                int row_idx = warp_tile_row_id * WMMA_M + i;
                if (row_idx < Br) {
                    stats[row_idx].max_prev = stats[row_idx].max_curr;
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
                        if (stats[row].sum_exp > 1e-10f) {
                            output[row * (d + PAD) + col] /= stats[row].sum_exp;
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
