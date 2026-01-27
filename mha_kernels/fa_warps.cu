#include <cuda_runtime.h>
#include <math.h>
#include "../include/config.h"
#include "../include/launchers.h"
#include "../utils/utils.cu"

// Standard Flash-Attention 2: single-pass proper per-query-row online softmax

/**
 * FlashAttention-2 Forward Kernel with Warp Specialization
 * 1 block owns TILE_SIZE_Q (Br) Q rows.
 * Block has Br warps. Each warp owns 1 Q row.
 * Shared memory holds:
 *   - Br rows of Q.
 *   - Br rows of O
 *   - Bc rows of K, V (shared by all warps).
 */
template<int Br, int Bc>
__global__ void fa_warps(
    const float* Q,    // [N, d]
    const float* K,    // [N, d]
    const float* V,    // [N, d]
    float* O,          // [N, d]
    const int N,       // Sequence length
    const int d,       // Dimension per head
    const float scale  // softmax_scale (e.g., 1/sqrt(d))
) {
    const int warp_id = threadIdx.y;   // Warp/Row index within block [0, Br-1]
    const int lane_id = threadIdx.x;   // Lane ID [0, 31]
    
    // Global row index this warp is responsible for
    const int q_row_idx = blockIdx.x * Br + warp_id;
    
    // 1. Shared Memory for Tiles
    extern __shared__ float shared_mem[];
    // Layout: Q[Br, d] | K[Bc, d] | V[Bc, d] | O[Br, d]
    float* sQ = shared_mem;                         // [Br * d]
    float* sK = &shared_mem[Br * d];                // [Bc * d]
    float* sV = &shared_mem[Br * d + Bc * d];       // [Bc * d]
    float* sO = &shared_mem[Br * d + 2 * Bc * d];   // [Br * d]

    // Pointers for this warp's Q and O rows
    float* my_sQ = &sQ[warp_id * d];
    float* my_sO = &sO[warp_id * d];

    // 2. Local Registers
    float m_i = -1e38f; 
    float l_i = 0.0f;

    // 3. Load Q Tile (Br rows) - Each warp loads its own row
    if (q_row_idx < N) {
        for (int k = lane_id; k < d; k += 32) {
            my_sQ[k] = Q[q_row_idx * d + k];
            my_sO[k] = 0.0f; 
        }
        __syncwarp(); // Ensure load is visible before RoPE (if we parallelized it, but here just safety)
        
        // RoPE application: Serialized to lane 0 to avoid race condition
        // as apply_rope helper is designed for single-thread-per-row
        if (lane_id == 0) {
            apply_rope(my_sQ, q_row_idx, d);
        }
    }
    __syncthreads();

    // 4. Outer Loop: Stream K, V Tiles
    for (int j = 0; j < (N + Bc - 1) / Bc; j++) {
        
        // Load K and V into SRAM - ALL threads in block cooperate
        // Total threads = Br * 32. Total elements = Bc * d * 2.
        // We linearize the loading.
        int total_threads = blockDim.x * blockDim.y; // 32 * Br
        int tid = threadIdx.y * 32 + threadIdx.x;
        
        for (int i = tid; i < Bc * d; i += total_threads) {
             // Map linear index i to (row, col)
             int row = i / d;
             int col = i % d;
             int kv_row_global = j * Bc + row;
             if (kv_row_global < N) {
                 sK[i] = K[kv_row_global * d + col];
                 sV[i] = V[kv_row_global * d + col];
             }
        }
        __syncthreads();
        
        for (int r = tid; r < Bc; r += total_threads) {
             int kv_row_global = j * Bc + r;
             if (kv_row_global < N) {
                apply_rope(&sK[r * d], kv_row_global, d);
             }
        }
        __syncthreads();


        // 5. Compute Attention (Warp-Specialized)
        if (q_row_idx < N) {
            int kv_tile_size = min(Bc, N - j * Bc);
            for (int c = 0; c < kv_tile_size; c++) {
                // ... (Dot product & Softmax logic same as before, using my_sQ, my_sO) ...
                
                // Warp-level dot product
                float local_score = 0.0f;
                // Pre-fetch K row pointer
                float* curr_K_row = &sK[c * d];
                
                for (int k = lane_id; k < d; k += 32) {
                    local_score += my_sQ[k] * curr_K_row[k];
                }
                
                // Reduce
                float score = local_score;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1) {
                    score += __shfl_down_sync(0xFFFFFFFFu, score, offset);
                }
                score = __shfl_sync(0xFFFFFFFFu, score, 0); // Broadcast
                score *= scale;

                // Softmax update
                float m_next = fmaxf(m_i, score);
                float p_old = expf(m_i - m_next);
                float p_new = expf(score - m_next);
                
                // Update Accumulator
                float* curr_V_row = &sV[c * d];
                for (int k = lane_id; k < d; k += 32) {
                    my_sO[k] = (my_sO[k] * p_old) + (p_new * curr_V_row[k]);
                }
                
                l_i = (l_i * p_old) + p_new;
                m_i = m_next;
            }
        }
        __syncthreads();
    }

    // 6. Write to Global Memory
    if (q_row_idx < N) {
        for (int k = lane_id; k < d; k += 32) {
            O[q_row_idx * d + k] = my_sO[k] / l_i;
        }
    }
}

template<int Br, int Bc>
void launch_fa_warps(
    const float *Q, const float *K, const float *V,
    float *O,
    int N, int d,
    float alpha,
    cudaStream_t stream = 0
)
{
    // Grid: ceil(N / Br) blocks
    int num_blocks = (N + Br - 1) / Br;
    
    // Block: Br warps (Br * 32 threads)
    dim3 threads_per_block(32, Br); 
    
    // Shared Mem: (Q_tile + K_tile + V_tile + O_tile)
    // Q: Br*d, K: Bc*d, V: Bc*d, O: Br*d
    size_t shared_mem = (2 * Br + 2 * Bc) * d * sizeof(float);
    
    fa_warps<Br, Bc><<<num_blocks, threads_per_block, shared_mem, stream>>>(
        Q, K, V, O, N, d, alpha
    );
}


extern "C" void solve(const float *Q, const float *K, const float *V, float *output, int N, int d_model, int h)
{
    auto fa_warps_kernel = [](const float* q_s, const float* k_s, const float* v_s, float* out_s, int N, int d_head, float alpha, cudaStream_t stream, void* aux){
        (void)aux;
        launch_fa_warps<Br, Bc>(q_s, k_s, v_s, out_s, N, d_head, alpha, stream);
    };
    launch(Q, K, V, output, N, d_model, h, fa_warps_kernel, 0);
}
