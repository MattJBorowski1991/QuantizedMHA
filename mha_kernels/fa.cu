#include <cuda_runtime.h>
#include <math.h>
#include "../include/config.h"
#include "../include/launchers.h"
#include "../utils/utils.cu"

// Standard Flash-Attention 2: single-pass proper per-query-row online softmax

/**
 * FlashAttention-2 Forward Kernel with Fused RoPE
 * 1 block owns 1 Q_tile
 * 1 thread owns 1 row
 */
template<int Br, int Bc>
__global__ void fa(
    const float* Q,    // [N, d]
    const float* K,    // [N, d]
    const float* V,    // [N, d]
    float* O,          // [N, d]
    const int N,       // Sequence length
    const int d,       // Dimension per head
    const float scale  // softmax_scale (e.g., 1/sqrt(d))
) {
    const int q_tile_idx = blockIdx.x;
    const int tx = threadIdx.x; // BlockDim.x must be Br

    // 1. Shared Memory for Tiles
    extern __shared__ float shared_mem[];
    float* sQ = shared_mem;                   // [Br * d]
    float* sK = &shared_mem[Br * d];          // [Bc * d]
    float* sV = &shared_mem[Br * d + Bc * d]; // [Bc * d]

    // 2. Local Registers and shared memory for accumulator
    float m_i = -1e38f; 
    float l_i = 0.0f;
    float *row_O = &shared_mem[Br * d + 2 * Bc * d + tx * d];  // Use shared memory for O
    for (int k = 0; k < d; k++) row_O[k] = 0.0f;

    // 3. Load Q Tile into SRAM and Apply RoPE
    const int q_row_idx = q_tile_idx * Br + tx;
    if (q_row_idx < N) {
        for (int k = 0; k < d; k++) {
            sQ[tx * d + k] = Q[q_row_idx * d + k];
        }
        apply_rope(&sQ[tx * d], q_row_idx, d);
    }
    __syncthreads();

    // 4. Outer Loop: Stream K, V Tiles
    for (int j = 0; j < (N + Bc - 1) / Bc; j++) {
        
        // Load K and V into SRAM (Cooperative Load)
        for (int row = tx; row < Bc; row += Br) {
            const int kv_row_idx = j * Bc + row;
            if (kv_row_idx < N) {
                for (int k = 0; k < d; k++) {
                    sK[row * d + k] = K[kv_row_idx * d + k];
                    sV[row * d + k] = V[kv_row_idx * d + k];
                }
                apply_rope(&sK[row * d], kv_row_idx, d);
            }
        }
        __syncthreads();

        // 5. Compute Attention and Update Online Statistics
        if (q_row_idx < N) {
            // Inner Loop over current K,V tile
            int kv_tile_size = min(Bc, N - j * Bc);
            for (int c = 0; c < kv_tile_size; c++) {
                float score = 0.0f;
                for (int k = 0; k < d; k++) {
                    score += sQ[tx * d + k] * sK[c * d + k];
                }
                score *= scale;

                // Online Softmax Logic
                float m_next = fmaxf(m_i, score);
                float p_old = expf(m_i - m_next);
                float p_new = expf(score - m_next);

                for (int k = 0; k < d; k++) {
                    row_O[k] = (row_O[k] * p_old) + (p_new * sV[c * d + k]);
                }
                l_i = (l_i * p_old) + p_new;
                m_i = m_next;
            }
        }
        __syncthreads();
    }

    // 6. Final Normalization and Write to Global Memory
    if (q_row_idx < N) {
        for (int k = 0; k < d; k++) {
            O[q_row_idx * d + k] = row_O[k] / l_i;
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
    int threads_per_block = Br;
    size_t shared_mem = (2 * Br + 2 * Bc) * d * sizeof(float);  // Q, K, V, O accumulator tiles
    
    fa<Br, Bc><<<num_blocks, threads_per_block, shared_mem, stream>>>(
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
