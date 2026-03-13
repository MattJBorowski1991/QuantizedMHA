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

// Flash attention with Tensor Cores and int8 quantization
// Supported int8 WMMA tile sizes: 8×8×32, 16×16×32, 16×8×32, 8×16×32. Only int32 output.
// WMMA_M=16 rows of Q owned by 1 warp
// 1 warp owns a 16 x d chunk of Q and performs serial wmma, one 16x16x16 tile per iteration

#define THREADS_PER_WARP 32
#define WARPS_PER_BLOCK 8
#define THREADS (WARPS_PER_BLOCK * THREADS_PER_WARP)
#define FULL_MASK 0xffffffff
#define PAD 0

//Tensor Core parameters
constexpr int WMMA_M = 8;
constexpr int WMMA_N = 32;
constexpr int WMMA_K = 16;

static_assert(Br == WMMA_M * WARPS_PER_BLOCK, "Block size needs to equal number of warps times number of rows each warp handles");

template<int M, int N, bool isInputInDram, bool transposeInput = false>
static __device__ __forceinline__ void fp32_to_int8sram(
    const float* __restrict__ fp32_in, //for inputs: Q: Nxd, K^T: Nxd, P: Br x Bc, V: Bc x d
    float* __restrict__ block_scales,
    int8_t* __restrict__ int8_out,
    int bid
){
    const int tid = threadIdx.x;
    int size = M * N;

    const int lane_id = tid % THREADS_PER_WARP;
    const int warp_id = tid / THREADS_PER_WARP;

    if(bid == 0) bid = blockIdx.x;

    //TODO: put fp32_in in SRAM depending on SRAM pressure

    const float* block_in;
    if constexpr(isInputInDram){
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

    if(lane_id == 0){
        warp_maxs[warp_id] = local_max;
        warp_mins[warp_id] = local_min;
    }
    __syncthreads();
    
    float block_min = INFINITY;
    float block_max = -INFINITY;
    __shared__ float inv_sc_shared;


    //TODO: replace with simple SRAM reduction

    if(warp_id == 0){

        for(int l = lane_id; l < WARPS_PER_BLOCK; l += THREADS_PER_WARP){
            block_min = fminf(block_min, warp_mins[l]);
            block_max = fmaxf(block_max, warp_maxs[l]);
        }

        for(int off = THREADS_PER_WARP >> 1; off > 0; off >>= 1){
            float mmin = __shfl_down_sync(FULL_MASK, block_min, off);
            float mmax = __shfl_down_sync(FULL_MASK, block_max, off);
            block_min = fminf(block_min, mmin);
            block_max = fmaxf(block_max, mmax);
        }

        if(lane_id == 0){
            // TODO: fix for blocks with all tiny values i.e. when sc=1e-8f. Now dequantization is not working for such blocks
            float sc = fmaxf(fmaxf(fabs(block_max), fabs(block_min)) / 127.0f, 1e-8f);
            block_scales[bid] = sc;
            inv_sc_shared = 1.0f / sc;
        }
    }
    __syncthreads();

    // threads read inv_sc_shared from sram

    float inv_sc = inv_sc_shared;
    for(int i = tid; i < size; i += THREADS){
        float v = block_in[i];
        float scaled = v * inv_sc;
        int rounded = __float2int_rn(scaled);
        rounded = (rounded < -128) ? -128 : ((rounded > 127) ? 127 : rounded);

        if constexpr(transposeInput){
            int row = i / N;
            int col = i % N;
            block_out[col * (M + PAD) + row] = static_cast<int8_t>(rounded);
        }else{
            block_out[i] = static_cast<int8_t>(rounded);
        }
    }
}

//int32_in in SRAM, fp32_out in DRAM (P@V) or SRAM (scores=Q@K^T)
template<int M, int N, bool isOutputInDram>
static __device__ __forceinline__ void int32sram_to_fp32(
    const int* __restrict__ int32_in,
    float* __restrict__ block_scales_A,
    float* __restrict__ block_scales_B,
    float* __restrict__ fp32_out,
    int bid
){
    const int tid = threadIdx.x;
    int size = M * N;

    const int* block_in = int32_in;

    float* block_out;
    if constexpr(isOutputInDram){
        block_out = fp32_out + (size_t)bid * (size_t)size;
    }else{
        block_out = fp32_out;
    }

    for(int i = tid; i < size; i += THREADS){
        float scale_a = block_scales_A[0];
        float scale_b = block_scales_B[0];
        float dequant = (float)block_in[i] * scale_a * scale_b;
        
        block_out[i] = dequant;
    }
    __syncthreads();
}

template <bool add_to_output = false, int M, int N, int K>
static __device__ __forceinline__ void wmma_A_B(
    const int8_t* __restrict__ A,     // M x K (Q: Br x d, or P: Br x Bc)
    const int8_t* __restrict__ B,    //  K x N (K^T: d x Bc, or V: Bc x d)
    int* C,                       //  M x N (scores: Br x Bc, or output: Br x d)
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

    int tile_row = warp_id * WMMA_M;
    int tile_col = 0;          // All warps process all column chunks independently

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, int8_t, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, int8_t, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int> c_frag;

    // Process each column chunk of N, starting from tile_col
    for(int b_col = tile_col; b_col < N; b_col += WMMA_N){
        // Initialize accumulator for THIS column chunk
        wmma::fill_fragment(c_frag, 0);

        // Accumulate over the K dimension (reduction dimension)
        for(int k = 0; k < K; k += WMMA_K){
            int a_col = k;
            int b_row = k;

            const int8_t *a_ptr = A + tile_row * stride_A + a_col;
            const int8_t *b_ptr = B + b_row * stride_B + b_col;   

            if(a_col + WMMA_K <= K && b_col + WMMA_N <= N){
                wmma::load_matrix_sync(a_frag, a_ptr, stride_A);
                wmma::load_matrix_sync(b_frag, b_ptr, stride_B);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
        }

        // If add_to_output is true, load existing C values and add them
        if constexpr (add_to_output) {
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int> c_existing;
            int* c_dst = C + tile_row * stride_C + b_col;
            wmma::load_matrix_sync(c_existing, c_dst, stride_C, wmma::mem_row_major);
            
            // Element-wise addition
            for(int i = 0; i < c_frag.num_elements; ++i) {
                c_frag.x[i] += c_existing.x[i];
            }
        }

        // Store THIS column chunk result with padded stride
        int* c_dst = C + tile_row * stride_C + b_col;
        wmma::store_matrix_sync(c_dst, c_frag, stride_C, wmma::mem_row_major);
        
        // NOTE: wmma operations are warp-synchronous, but for safety with multiple warps,
        // each warp completes its column independently. The kernel caller should 
        // __syncthreads() after this function returns.
    }
}

// Online softmax: WMMA_M rows per warp (matching matmul parallelism)
// Scale scores, find max, compute softmax probs, update statistics, rescale output, accumulate O += P @ V
template<int Br, int Bc, int Lc, int d, int stride_scores = 0, int stride_output = 0>
__device__ __forceinline__ void online_softmax_and_accum_output(
    float* __restrict__ max_cur,
    float* __restrict__ max_prev,
    float* __restrict__ sum_exp,
    int* scores_int32,      // Q@K^T, layout [Br x (Bc + PAD)]
    int8_t* scores_int8,    // quantized scores [Br x (Bc + PAD)]
    float* scores_fp32,     // for the softmax calculations 
    const float* __restrict__ block_scales_Q, // for dequantization of scores: int32->fp32
    const float* __restrict__ block_scales_Kt, // as above
    float* __restrict__ block_scales_P,         // for subsequent quantization of dequantized scores: : fp32->int8 to get P_next
    int* temp_output_int32, // for accumulation of wmma for P_i@V_i
    float* output,
    const int8_t* values, // V (Bc x d)
    const float* block_scales_V,
    float inv_sqrt_d,
    int kv_block_idx
) {
    // Use padded strides
    const int sc_stride = (stride_scores == 0) ? (Bc + PAD) : stride_scores;
    const int out_stride = (stride_output == 0) ? (d + PAD) : stride_output;

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int q_block_idx = blockIdx.x;

    //read block_scales_Q and block_scales_Kt to dequantize
    int32sram_to_fp32<Br, Bc + PAD, false>(scores_int32, const_cast<float*>(block_scales_Q + q_block_idx), const_cast<float*>(block_scales_Kt + kv_block_idx), scores_fp32, q_block_idx);
    __syncthreads();

    // Process WMMA_M query rows per warp (same as matmul_warp_tiled)
    for (int row_start = WMMA_M * warp_id; row_start < Br; row_start += WARPS_PER_BLOCK * WMMA_M) 
    {
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
                max_cur[row_start + i] = max_new[i];
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
    } //end for loop with row_start
    __syncthreads();

    fp32_to_int8sram<Br, Bc + PAD, false, false>(scores_fp32, block_scales_P, scores_int8, kv_block_idx);
    __syncthreads();

    // Step 5: Accumulate O += (softmax probs) @ V
    // scores_int8: Br x Bc with stride Bc+PAD
    // values: Bc x d with stride d+PAD
    // output: Br x d with stride d+PAD
    wmma_A_B<false, Br, d, Bc>(scores_int8, values, temp_output_int32, Bc + PAD, d + PAD, d + PAD);
    __syncthreads();

    for(int i = tid; i < Br * (d + PAD); i += THREADS){
        output[i] += (float)(temp_output_int32[i]) * block_scales_P[kv_block_idx] * block_scales_V[kv_block_idx];
    }
}

template <int Br, int Bc, int d, int Lc>
__device__ __forceinline__ void init_output_and_stats(
    float* __restrict__ output,
    float* __restrict__ sum_exp,
    float* __restrict__ max_prev,
    float* __restrict__ max_curr
){
    const int tid = threadIdx.x;
    const int warp_id = tid / THREADS_PER_WARP;
    const int lane_id = tid % THREADS_PER_WARP;

    int row_start = warp_id * WMMA_M;

    if(row_start < Br){
        for(int col_start = Lc * lane_id; col_start < d; col_start += THREADS_PER_WARP * Lc){
            #pragma unroll
            for(int i = 0; i < WMMA_M; i++){
                #pragma unroll
                for(int j = 0; j < Lc; j++){
                    output[(row_start + i) * (d + PAD) + col_start + j] = 0.0f;
                }
            }
        }
    }

    for(int rr = lane_id; rr < Br; rr += THREADS_PER_WARP){
        sum_exp[rr] = 0.0f;
        max_prev[rr] = 0.0f;
        max_curr[rr] = -INFINITY;
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
    const int N,       // Sequence length,
    const int BLOCKS,   // number of Q blocks for scale indexing
    const float inv_sqrt_d  // 1/sqrt(d), pre-computed to save registers
) {
    
    // Ensure configuration is valid
  
    assert(N % Br == 0 && "N must be a multiple of Br");
    assert(N % Bc == 0 && "N must be a multiple of Bc");

    // One block per Br of Q-rows
    const int q_block_idx = blockIdx.x;
    const int tid = threadIdx.x;

    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    // Shared memory layout with proper 16-byte alignment for tensor core operations
    // Use 1D arrays with stride-based indexing for wmma_A_B compatibility

    __shared__ __align__(16) int8_t q_block[Br * (d + PAD)];
    __shared__ __align__(16) int8_t kt[d * (Bc + PAD)];

    __shared__ float scores_fp32[Br * (Bc + PAD)];
    __shared__ int scores_int32[Br * (Bc + PAD)]; 
    __shared__ __align__(16) int8_t scores_int8[Br * (Bc + PAD)];

    __shared__ int temp_output_int32[Br * (Bc + PAD)];
    __shared__ float output[Br * (d + PAD)];

    __shared__ __align__(16) int8_t values[Bc * (d + PAD)];

    __shared__ float sum_exp[Br];
    __shared__ float max_prev[Br];
    __shared__ float max_curr[Br];

    float *block_scales_Q = block_scales;
    float *block_scales_Kt = block_scales + BLOCKS;
    float *block_scales_V = block_scales + 2 * BLOCKS;
    float *block_scales_P = block_scales + 3 * BLOCKS;

    int row_start = warp_id * WMMA_M;  // FIX: Use WMMA_M (tile height), not WMMA_N (tile width)

    //quantize Q once
    fp32_to_int8sram<Br, d + PAD, true, false>(Q, block_scales_Q, q_block, q_block_idx);
    
    init_output_and_stats<Br, Bc, d, Lc>(output, sum_exp, max_prev, max_curr);

    // Main loop over K,V blocks
    int num_kv_blocks = (N + Bc - 1) / Bc;
    for (int kv_block_idx = 0; kv_block_idx < num_kv_blocks; kv_block_idx++) {

        // pointers for this K,V block iteration
        const float *K_block = K + kv_block_idx * Bc * d;
        const float *V_block = V + kv_block_idx * Bc * d;

        fp32_to_int8sram<Bc, d + PAD, false, true>(K_block, block_scales_Kt, kt, kv_block_idx);
        fp32_to_int8sram<Bc, d + PAD, false, false>(V_block, block_scales_V, values, kv_block_idx);
        __syncthreads();
        
        // Initialize max_curr for this iteration
        // Warp warp_id initializes rows [warp_id*WMMA_M, (warp_id+1)*WMMA_M)
        if (lane_id == 0) {
            for (int i = 0; i < WMMA_M; i++) {
                int row_idx = warp_id * WMMA_M + i;
                if (row_idx < Br) {
                    max_curr[row_idx] = -INFINITY;
                }
            }
        }
        __syncthreads();

        // Compute scores = Q @ K^T = Br x Bc
        wmma_A_B<false, Br, Bc, d>(q_block, kt, scores_int32, d + PAD, Bc + PAD, Bc + PAD);
        __syncthreads();

        // Online softmax (Br x Bc) + output accumulation (Br x d)
        online_softmax_and_accum_output<Br, Bc, Lc, d>
        (max_curr, max_prev, sum_exp, scores_int32, scores_int8, scores_fp32, block_scales_Q, block_scales_Kt, block_scales_P, temp_output_int32, output, values, block_scales_V, inv_sqrt_d, kv_block_idx);
        __syncthreads();
        
        // Copy max_curr to max_prev for next iteration (only for rows this warp processes)
        if (lane_id == 0) {
            for (int i = 0; i < WMMA_M; i++) {
                int row_idx = warp_id * WMMA_M + i;
                if (row_idx < Br) {
                    max_prev[row_idx] = max_curr[row_idx];
                }
            }
        }
        __syncthreads();
    } //end of kv_block_idx for loop iteration

    // Epilogue: normalize output and write to global memory (warp/lane distribution)
    if(row_start < Br){
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
    
    // Store to global memory (warp/lane distribution)
    for (int row_start = WMMA_M * warp_id; row_start < Br; row_start += WARPS_PER_BLOCK * WMMA_M) {
        for (int col_start = Lc * lane_id; col_start < d; col_start += THREADS_PER_WARP * Lc) {
            #pragma unroll
            for (int i = 0; i < WMMA_M; i++) {
                #pragma unroll
                for (int j = 0; j < Lc; j++) {
                    int row = row_start + i;
                    int col = col_start + j;
                    if (q_block_idx * Br + row < N && col < d) {
                        float write_val = output[row * (d + PAD) + col];
                        int global_idx = (q_block_idx * Br + row) * d + col;
                        O[global_idx] = write_val;
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
    
    float *d_block_scales = nullptr;
    cudaMalloc(&d_block_scales, 4 * BLOCKS * sizeof(float)); //One for each: Q, K, P, V

    int sram_carveout = 100;
    cudaFuncSetAttribute((void*)fa_kernel<Br, Bc, d, Lc>, cudaFuncAttributePreferredSharedMemoryCarveout, sram_carveout);
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
