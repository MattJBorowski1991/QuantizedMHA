#include <cuda_runtime.h>
#include <math.h>
#include <assert.h>
#include <limits>
#include "../include/config.h"
#include "../include/launchers.h"
#include "../utils/utils.cu"

#include "mma.h"
#include <cuda_fp16.h>
using namespace nvcuda;

// Flash attention with Warp-level tiled matmul 
// Wr x Lc register-level tile per lane => 4 x (Lc x 32) chunk per warp which is warp-strided for all columns of Q

// Flash attention with Tensor Cores
// One warp owns a 16 x d chunk of Q and performs serial wmma, one 16x16 tile per iteration

#define THREADS_PER_WARP 32
#define WARPS_PER_BLOCK 8
#define FULL_MASK 0xffffffff


//Template calls: 
// Q@Kt: wmma_A_B<THREADS, Br, Bc, d>(Q, K^T, scores)
// P@V:  wmma_A_B<THREADS, Br, d, Bc>(P, V, output)
// TODO add padding

template <bool add_to_output = false, int THREADS, int M, int N, int K>
__device__ __forceinline__ void wmma_A_B(
    const half* __restrict__ A,     // M x K (Q: Br x d, or P: Br x Bc)
    const half* __restrict__ B,    //  K x N (K^T: d x Bc, or V: Bc x d)
    float* C,                       //  M x N (scores: Br x Bc, or output: Br x d)
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
    int lane_id = tid % 32;

    int tile_row = warp_id * WMMA_M;
    int tile_col = 0;          // For future versions: could be (warp_id % N_WARPS_COL) * WMMA_N for column distribution

    if(tile_row >= M) return;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // Process each column chunk of N, starting from tile_col
    for(int b_col = tile_col; b_col < N; b_col += WMMA_N){
        // Initialize accumulator for THIS column chunk
        wmma::fill_fragment(c_frag, 0.0f);

        // Accumulate over the K dimension (reduction dimension)
        for(int k = 0; k < K; k += WMMA_K){
            int a_col = k;
            int b_row = k;

            const half *a_ptr = A + tile_row * stride_A + a_col;
            const half *b_ptr = B + b_row * stride_B + b_col;   

            if(a_col + WMMA_K <= K && b_col + WMMA_N <= N){
                wmma::load_matrix_sync(a_frag, a_ptr, stride_A);
                wmma::load_matrix_sync(b_frag, b_ptr, stride_B);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
        }

        // If add_to_output is true, load existing C values and add them
        if constexpr (add_to_output) {
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_existing;
            float* c_dst = C + tile_row * stride_C + b_col;
            wmma::load_matrix_sync(c_existing, c_dst, stride_C, wmma::mem_row_major);
            
            // Element-wise addition
            for(int i = 0; i < c_frag.num_elements; ++i) {
                c_frag.x[i] += c_existing.x[i];
            }
        }

        // Store THIS column chunk result
        float* c_dst = C + tile_row * stride_C + b_col;
        wmma::store_matrix_sync(c_dst, c_frag, stride_C, wmma::mem_row_major);
    }
}


// // Shared memory block matrix multiply (warp-tiled)
// // Each warp: Wr rows, each lane (32 lanes): Lc columns per iteration
// // Accumulates into registers
// template <bool add_to_output = false, int THREADS, int Wr, int Lc>
// __device__ __forceinline__ void matmul_warp_tiled(
//     const float* A,      // M x K
//     const float* B,      // K x N
//     float* C,        // M x N
//     int M,
//     int N,
//     int K,
//     int stride_A = 0,  // stride for A rows (0 = default K)
//     int stride_B = 0,  // stride for B rows (0 = default N)
//     int stride_C = 0   // stride for C rows (0 = default N)
// ) {
    
//     // Require dimensions to be multiples of Wr/Lc for optimal unrolling
//     assert(M % Wr == 0 && "M must be a multiple of Wr");
//     assert(N % Lc == 0 && "N must be a multiple of Lc");
//     assert(K % Wr == 0 && "K must be a multiple of Wr");
    
//     // Use default stride if not provided
//     if (stride_A == 0) stride_A = K;
//     if (stride_B == 0) stride_B = N;
//     if (stride_C == 0) stride_C = N;

//     int tid = threadIdx.x;
//     int warp_id = tid / 32;
//     int lane_id = tid % 32;

    
//     // Each warp handles Wr rows
//     for (int row_start = Wr * warp_id; row_start < M; row_start += WARPS_PER_BLOCK * Wr) {
//         // Each lane handles Lc columns
//         for (int col_start = Lc * lane_id; col_start < N; col_start += THREADS_PER_WARP * Lc) {
            
//             // Register accumulator (Wr x Lc)
//             float acc[Wr * Lc];
//             #pragma unroll
//             for (int i = 0; i < Wr * Lc; i++) acc[i] = 0.0f;
            
//             // Dot product loop
//             for (int k_idx = 0; k_idx < K; k_idx++) {
//                 // Load A tile row
//                 float a_vals[Wr];
//                 #pragma unroll
//                 for (int i = 0; i < Wr; i++) {
//                     a_vals[i] = A[(row_start + i) * stride_A + k_idx];
//                 }
                
//                 // Load B tile column
//                 float b_vals[Wr];
//                 #pragma unroll
//                 for (int j = 0; j < Wr; j++) {
//                     b_vals[j] = B[k_idx * stride_B + col_start + j];
//                 }
                
//                 // Outer product
//                 #pragma unroll
//                 for (int i = 0; i < Wr; i++) {
//                     #pragma unroll
//                     for (int j = 0; j < Lc; j++) {
//                         acc[i * Lc + j] += a_vals[i] * b_vals[j];
//                     }
//                 }
//             }
            
//             // Store to C
//             #pragma unroll
//             for (int i = 0; i < Wr; i++) {
//                 #pragma unroll
//                 for (int j = 0; j < Lc; j++) {
//                     if constexpr (add_to_output) {
//                         C[(row_start + i) * stride_C + (col_start + j)] += acc[i * Lc + j];
//                     } else {
//                         C[(row_start + i) * stride_C + (col_start + j)] = acc[i * Lc + j];
//                     }
//                 }
//             }
//         }
//     }
// }

// Online softmax: Wr rows per warp (matching matmul parallelism)
// Scale scores, find max, compute softmax probs, update statistics, rescale output, accumulate O += P @ V
template<int Br, int Bc, int THREADS, int Wr, int Lc>
__device__ __forceinline__ void online_softmax_and_accum_output(
    float* max_cur,
    const float* max_prev,
    float* sum_exp,
    float* scores,     // to be overwritten with softmax probs, layout [Br x (Bc+1)]
    float* output,
    const half *values, // V (Bc x d)
    int head_dim,
    float inv_sqrt_d
) {
    // Compute scores_fp16 pointer from scores and buffer layout
    half* scores_fp16 = (half*)(scores + Br * (Bc + 1));  // [Br x Bc] (quantized scores)

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    // Process Wr query rows per warp (same as matmul_warp_tiled)
    for (int row_start = Wr * warp_id; row_start < Br; row_start += WARPS_PER_BLOCK * Wr) {
        
        // Local arrays to track max, sum, and exp_max_diff for Wr rows
        float max_new[Wr], sum_new[Wr], exp_max_diff[Wr];
        #pragma unroll
        for (int i = 0; i < Wr; i++) {
            max_new[i] = max_prev[row_start + i];
            sum_new[i] = 0.0f;
        }
        
        // Step 1: Find max in scores for each row and scale by 1/sqrt(d)
        for (int col_start = Lc * lane_id; col_start < Bc; col_start += THREADS_PER_WARP * Lc) {
            #pragma unroll
            for (int i = 0; i < Wr; i++) {
                #pragma unroll
                for (int j = 0; j < Lc; j++) {
                    int row_idx = row_start + i;
                    int col_idx = col_start + j;
                    if (col_idx < Bc) {
                        float score_scaled = scores[row_idx * (Bc + 1) + col_idx] * inv_sqrt_d;
                        max_new[i] = fmaxf(max_new[i], score_scaled);
                        scores[row_idx * (Bc + 1) + col_idx] = score_scaled;
                    }
                }
            }
        }
        
        // Warp reduction to get global max for each of the Wr rows
        #pragma unroll
        for (int i = 0; i < Wr; i++) {
            #pragma unroll
            for (int shift = THREADS_PER_WARP / 2; shift >= 1; shift >>= 1) {
                max_new[i] = fmaxf(max_new[i], __shfl_xor_sync(FULL_MASK, max_new[i], shift));
            }
        }

        // Step 2: Compute exp(score - max) and accumulate sum
        for (int col_start = Lc * lane_id; col_start < Bc; col_start += THREADS_PER_WARP * Lc) {
            #pragma unroll
            for (int i = 0; i < Wr; i++) {
                #pragma unroll
                for (int j = 0; j < Lc; j++) {
                    int row_idx = row_start + i;
                    int col_idx = col_start + j;
                    int idx = row_idx * Bc + col_idx;
                    int idx_pad = row_idx * (Bc + 1) + col_idx;

                    if (row_idx < Br && col_idx < Bc) {
                        //calculate scores, accumulate sum
                        float prob = expf(scores[row_idx * (Bc + 1) + col_idx] - max_new[i]);
                        scores[idx_pad] = prob;
                        sum_new[i] += prob;

                        //quantize scores for later wmma_A_B
                        scores_fp16[idx] = __float2half(prob);
                    }else{
                        scores[idx_pad] = 0.0f;
                        scores_fp16[idx] = __float2half(0.0f);
                    }
                }
            }
        }
        
        // Warp reduction to get global sum for each row
        #pragma unroll
        for (int i = 0; i < Wr; i++) {
            #pragma unroll
            for (int shift = THREADS_PER_WARP / 2; shift >= 1; shift >>= 1) {
                sum_new[i] += __shfl_xor_sync(FULL_MASK, sum_new[i], shift);
            }
        }

        // Step 3: Update max and sum statistics (only first lane writes)
        #pragma unroll
        for (int i = 0; i < Wr; i++) {
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
            for (int i = 0; i < Wr; i++) {
                output[(row_start + i) * d + d_idx] *= exp_max_diff[i];
            }
        }
    }
    
    __syncthreads();
    
    // Step 5: Accumulate O += (softmax probs) @ V

    wmma_A_B<true, THREADS, Br, d, Bc>(scores_fp16, values, output, Bc, d, d);
    //previously: matmul_warp_tiled<true, THREADS, Wr, Lc>(scores, values, output, Br, d, Bc, Bc + 1, d, 0);
}

template<int Br, int Bc, int THREADS, int d, int Wr, int Lc>
__global__ void fa_kernel(
    const float* Q,    // [N, d]
    const float* K,    // [N, d]
    const float* V,    // [N, d]
    float* O,          // [N, d]
    const int N,       // Sequence length
    const float scale,  // softmax_scale (1/sqrt(d) is passed as scale)
    const float inv_sqrt_d  // 1/sqrt(d), pre-computed to save registers
) {
    
    // Ensure configuration is valid
    static_assert(Br % (WARPS_PER_BLOCK * Wr) == 0 || Br >= WARPS_PER_BLOCK * Wr, 
                  "Br must be >= WARPS_PER_BLOCK * Wr to ensure all warps have rows to process");
    
    assert(N % Br == 0 && "N must be a multiple of Br");
    assert(N % Bc == 0 && "N must be a multiple of Bc");

    // One block per Br of Q-rows
    const int q_block_idx = blockIdx.x;
    const int tid = threadIdx.x;

    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    
    // Shared memory layout (exact allocation) - Now less SRAM needed thanks to FP16!!!
    extern __shared__ float shared_mem[];
    float *output = shared_mem;                                           // [Br x d]
    half *q_block = (half*)(shared_mem + Br*d);                                   // [Br x d]
    half *kv_block = (half*)(shared_mem + 2*Br*d);                                // [(Bc+1) x d]
    float *scores = shared_mem + 2*Br*d + (Bc+1)*d;                       // [Br x (Bc+1)]
    float *sum_exp = shared_mem + 2*Br*d + (Bc+1)*d + Br*(Bc+1);          // [Br]
    float* max_prev = shared_mem + 2*Br*d + (Bc+1)*d + Br*(Bc+1) + Br;    // [Br]
    float* max_curr = max_prev + Br;                                   // [Br]
    // Total SRAM required = Br×d + Br×d + (Bc+1)×d + Br×(Bc+1) + Br + Br + Br = (2×Br×d + (Bc+1)×d + Br×(Bc+1) + 3×Br)

    // Load Q block and quantize (warp/lane distribution for future scalability)
    for (int row_start = Wr * warp_id; row_start < Br; row_start += WARPS_PER_BLOCK * Wr) {
        for (int col_start = Lc * lane_id; col_start < d; col_start += THREADS_PER_WARP * Lc) {
            #pragma unroll
            for (int i = 0; i < Wr; i++) {
                #pragma unroll
                for (int j = 0; j < Lc; j++) {
                    int row = row_start + i;
                    int col = col_start + j;
                    if (q_block_idx * Br + row < N && col < d) {
                        q_block[row * d + col] = __float2half(Q[(q_block_idx * Br + row) * d + col]);
                    } else {
                        q_block[row * d + col] = __float2half(0.0f);
                    }
                }
            }
        }
    }
    
    // Initialize output, statistics (warp/lane distribution)
    for (int row_start = Wr * warp_id; row_start < Br; row_start += WARPS_PER_BLOCK * Wr) {
        for (int col_start = Lc * lane_id; col_start < d; col_start += THREADS_PER_WARP * Lc) {
            #pragma unroll
            for (int i = 0; i < Wr; i++) {
                #pragma unroll
                for (int j = 0; j < Lc; j++) {
                    output[(row_start + i) * d + (col_start + j)] = 0.0f;
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
        // Warp warp_id initializes rows [warp_id*Wr, (warp_id+1)*Wr)
        if (lane_id == 0) {
            for (int i = 0; i < Wr; i++) {
                int row_idx = warp_id * Wr + i;
                if (row_idx < Br) {
                    max_curr[row_idx] = -INFINITY;
                }
            }
        }
        __syncthreads();
        
        // Load K & quantize (transposed with padding, warp/lane distribution)
        for (int row_start = Wr * warp_id; row_start < Bc; row_start += WARPS_PER_BLOCK * Wr) {
            for (int col_start = Lc * lane_id; col_start < d; col_start += THREADS_PER_WARP * Lc) {
                #pragma unroll
                for (int i = 0; i < Wr; i++) {
                    #pragma unroll
                    for (int j = 0; j < Lc; j++) {
                        int row = row_start + i;
                        int col = col_start + j;
                        int k_idx = kv_block_idx * Bc + row;
                        if (k_idx < N && col < d) {
                            kv_block[col * (Bc + 1) + row] = __float2half(K[k_idx * d + col]);
                        } else {
                            kv_block[col * (Bc + 1) + row] = __float2half(0.0f);
                        }
                    }
                }
            }
        }
        __syncthreads();
        
        // Compute scores = Q @ K^T = Br x Bc
        wmma_A_B<false, THREADS, Br, Bc, d>(q_block, kv_block, scores, d, Bc + 1, Bc + 1);
        // previously: matmul_warp_tiled<false, THREADS, Wr, Lc>(q_block, kv_block, scores, Br, Bc, d, Bc + 1, Bc + 1);
        __syncthreads();
        
        // Load V (warp/lane distribution)
        for (int row_start = Wr * warp_id; row_start < Bc; row_start += WARPS_PER_BLOCK * Wr) {
            for (int col_start = Lc * lane_id; col_start < d; col_start += THREADS_PER_WARP * Lc) {
                #pragma unroll
                for (int i = 0; i < Wr; i++) {
                    #pragma unroll
                    for (int j = 0; j < Lc; j++) {
                        int row = row_start + i;
                        int col = col_start + j;
                        int k_idx = kv_block_idx * Bc + row;
                        if (k_idx < N && col < d) {
                            kv_block[row * d + col] = __float2half(V[k_idx * d + col]);
                        } else {
                            kv_block[row * d + col] = __float2half(0.0f);
                        }
                    }
                }
            }
        }
        __syncthreads();
        
        // Online softmax (Br x Bc) + output accumulation (Br x d)
        online_softmax_and_accum_output<Br, Bc, THREADS, Wr, Lc>(max_curr, max_prev, sum_exp, scores, output, kv_block, d, inv_sqrt_d);
        __syncthreads();
        
        // Copy max_curr to max_prev for next iteration (only for rows this warp processes)
        if (lane_id == 0) {
            for (int i = 0; i < Wr; i++) {
                int row_idx = warp_id * Wr + i;
                if (row_idx < Br) {
                    max_prev[row_idx] = max_curr[row_idx];
                }
            }
        }
        __syncthreads();
    }

    // Epilogue: normalize output and write to global memory (warp/lane distribution)
    for (int row_start = Wr * warp_id; row_start < Br; row_start += WARPS_PER_BLOCK * Wr) {
        for (int col_start = Lc * lane_id; col_start < d; col_start += THREADS_PER_WARP * Lc) {
            #pragma unroll
            for (int i = 0; i < Wr; i++) {
                #pragma unroll
                for (int j = 0; j < Lc; j++) {
                    int row = row_start + i;
                    int col = col_start + j;
                    if (row < Br && col < d) {
                        if (sum_exp[row] > 1e-10f) {  // Use small epsilon instead of > 0
                            output[row * d + col] /= sum_exp[row];
                        } else {
                            output[row * d + col] = 0.0f;  // Handle zero sum_exp case
                        }
                    }
                }
            }
        }
    }
    
    __syncthreads();
    
    // Store to global memory (warp/lane distribution)
    for (int row_start = Wr * warp_id; row_start < Br; row_start += WARPS_PER_BLOCK * Wr) {
        for (int col_start = Lc * lane_id; col_start < d; col_start += THREADS_PER_WARP * Lc) {
            #pragma unroll
            for (int i = 0; i < Wr; i++) {
                #pragma unroll
                for (int j = 0; j < Lc; j++) {
                    int row = row_start + i;
                    int col = col_start + j;
                    if (q_block_idx * Br + row < N && col < d) {
                        O[(q_block_idx * Br + row) * d + col] = output[row * d + col];
                    }
                }
            }
        }
    }
}

template<int Br, int Bc, int THREADS, int d, int Wr, int Lc>
void launch_fa(const float *Q, const float *K, const float *V, float *O, int N, float alpha, cudaStream_t stream = 0)
{

    int BLOCKS = (N + Br - 1) / Br;
    // Exact SRAM allocation: output + q_block + kv_block + scores + sum_exp + max_prev/max_cur + scores_fp16
    // output: Br*d (float), q_block: Br*d (half), kv_block: (Bc+1)*d (half), 
    // scores: Br*(Bc+1) (float), sum_exp: Br (float), max_prev: Br (float), max_curr: Br (float), scores_fp16: Br*Bc (half)
    size_t shared_mem = (2*Br*d + (Bc+1)*d + Br*(Bc+1) + 3*Br) * sizeof(float) + Br*Bc * sizeof(half);
    float inv_sqrt_d = 1.0f / sqrtf((float)d);  // Pre-compute to save registers in kernel
    
    fa_kernel<Br, Bc, THREADS, d, Wr, Lc><<<BLOCKS, THREADS, shared_mem, stream>>>(Q, K, V, O, N, alpha, inv_sqrt_d);

}

extern "C" void solve(const float *Q, const float *K, const float *V, float *output, int N, int d_model, int h)
{   
    constexpr int THREADS = WARPS_PER_BLOCK * THREADS_PER_WARP;

    auto fa_kernel = [](const float* q_s, const float* k_s, const float* v_s, float* out_s, int N, int d_head, float alpha, cudaStream_t stream, void* aux){
        (void)aux;
        launch_fa<Br, Bc, THREADS, d, Wr, Lc>(q_s, k_s, v_s, out_s, N, alpha, stream);
    };
    launch(Q, K, V, output, N, d_model, h, fa_kernel, 0);
}
