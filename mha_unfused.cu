#include <cuda_runtime.h>
#include <math.h>


template<int TILE>
__global__ void mma_A_B(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    float alpha
)
{
    __shared__ float A_tile[TILE][TILE];
    __shared__ float B_tile[TILE][TILE];

    const int ty = threadIdx.y;
    const int tx = threadIdx.x;
    const int row = blockIdx.y * TILE + ty;
    const int col = blockIdx.x * TILE + tx;

    float sum = 0.0f;

    for(int tile_k = 0; tile_k < K; tile_k += TILE){
        if (row < M && (tile_k + tx) < K) {
            A_tile[ty][tx] = A[row * K + (tile_k + tx)];
        } else {
            A_tile[ty][tx] = 0.0f;
        }
        if ( (tile_k + ty) < K && col < N ){
            B_tile[ty][tx] = B[ (tile_k + ty) * N + col];
        }else{
            B_tile[ty][tx] = 0.0f;
        }
        __syncthreads();
        for(int kk = 0; kk < TILE && tile_k + kk < K; ++kk){
            sum += A_tile[ty][kk] * B_tile[kk][tx];
        }
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = sum;
}

template<int TILE>
__global__ void mma_A_Bt(
    const float* __restrict__ A,    // M x K
    const float* __restrict__ B,    // N x K => B^T is K x N
    float* __restrict__ C,
    int M, int N, int K,
    float alpha
)
{
    __shared__ float A_tile[TILE][TILE];
    __shared__ float Bt_tile[TILE][TILE];

    const int ty = threadIdx.y;
    const int tx = threadIdx.x;
    const int row = blockIdx.y * TILE + ty;
    const int col = blockIdx.x * TILE + tx;

    float sum = 0.0f;

    for(int tile_k = 0; tile_k < K; tile_k += TILE){
        if (row < M && (tile_k + tx) < K) {
            A_tile[ty][tx] = A[row * K + (tile_k + tx)];
        } else {
            A_tile[ty][tx] = 0.0f;
        }
        if (col < N && (tile_k + ty) < K) {
            Bt_tile[ty][tx] = B[col * K + (tile_k + ty)];
        } else {
            Bt_tile[ty][tx] = 0.0f;
        }
        __syncthreads();
        for(int kk = 0; kk < TILE && tile_k + kk < K; ++kk){
            sum += A_tile[ty][kk] * Bt_tile[kk][tx];
        }
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = alpha * sum; 
}

template<int TILE>
void launch_mma_A_B(const float *A, const float *B, float *C, int M, int N, int K, float alpha, bool isBtransposed, cudaStream_t stream = 0){

    dim3 threads(TILE,TILE);
    dim3 blocks(( N + TILE - 1 ) / TILE, (M + TILE - 1) / TILE);
    if(isBtransposed){
        mma_A_Bt<TILE><<<blocks, threads, 0, stream>>>(A, B, C, M, N, K, alpha);
    }else{
        mma_A_B<TILE><<<blocks, threads, 0, stream>>>(A, B, C, M, N, K, alpha);
    }
}



__global__ void extract_mat(const float *A, float *B, int row_off, int curr_col, int M_A, int N_A, int M_B, int N_B)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M_B && col < N_B)
    {
        B[row * N_B + col] = A[(row + row_off) * N_A + (col + curr_col)];
    }
}

__global__ void concat_mat(float *A, const float *B, int row_off, int curr_col, int M_A, int N_A, int M_B, int N_B)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M_B && col < N_B)
    {
        A[(row + row_off) * N_A + (col + curr_col)] = B[row * N_B + col];
    }
}

template <int TILE>
void launch_extract_mat(const float *A, float *B, int row_off, int curr_col, int M_A, int N_A, int M_B, int N_B, cudaStream_t stream = 0)
{
    dim3 threadsPerBlock(TILE, TILE);
    dim3 blocksPerGrid((N_B + TILE - 1) / TILE,
                       (M_B + TILE - 1) / TILE);

    extract_mat<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(A, B,
                                                              row_off, curr_col,
                                                              M_A, N_A, M_B, N_B);
}

template <int TILE>
void launch_concat_mat(float *A, const float *B, int row_off, int curr_col, int M_A, int N_A, int M_B, int N_B, cudaStream_t stream = 0)
{
    dim3 threadsPerBlock(TILE, TILE);
    dim3 blocksPerGrid((N_B + TILE - 1) / TILE,
                       (M_B + TILE - 1) / TILE);

    concat_mat<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(A, B,
                                                             row_off, curr_col,
                                                             M_A, N_A, M_B, N_B);
}

template <int TILE>
__global__ void softmax(const float *input, float *output, int N) {
    const unsigned fullMask = 0xFFFFFFFFu;
    int row = blockIdx.x;
    int tid = threadIdx.x;
    constexpr int threads = TILE * TILE;
    const float *row_in  = input  + row * N;
    float *row_out = output + row * N;

    // This implementation assumes blocks of at most 256 threads.
    __shared__ float sdata[threads];

    // compute local max over the row (stride by blockDim.x)
    float local_max = -INFINITY;
    for (int i = tid; i < N; i += threads) {
        float v = row_in[i];
        if (v > local_max) local_max = v;
    }
    sdata[tid] = local_max;
    __syncthreads();

    // tree-reduce in shared memory down to 32 lanes
    for (int s = threads / 2; s >= 32; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }

    // warp-level reduce using shuffle for the final 32 lanes
    if (tid < 32) {
        float v = sdata[tid];
        for (int offset = 16; offset > 0; offset >>= 1)
            v = fmaxf(v, __shfl_down_sync(fullMask, v, offset));
        if (tid == 0) sdata[0] = v;
    }
    __syncthreads();

    float max_val = sdata[0];

    // compute exponentials and local sums
    float local_sum = 0.0f;
    for (int i = tid; i < N; i += threads) {
        float e = expf(row_in[i] - max_val);
        row_out[i] = e;
        local_sum += e;
    }
    sdata[tid] = local_sum;
    __syncthreads();

    // reduce sums in shared memory down to 32 lanes
    for (int s = threads / 2; s >= 32; s >>= 1) {
        if (tid < s) sdata[tid] = sdata[tid] + sdata[tid + s];
        __syncthreads();
    }

    // warp-level sum reduction
    if (tid < 32) {
        float v = sdata[tid];
        for (int offset = 16; offset > 0; offset >>= 1)
            v += __shfl_down_sync(fullMask, v, offset);
        if (tid == 0) sdata[0] = v;
    }
    __syncthreads();

    float exp_sum = sdata[0];

    // normalize
    for (int i = tid; i < N; i += threads) row_out[i] = row_out[i] / exp_sum;
}

template <int TILE>
void launch_softmax(const float *input, float *output, int N, cudaStream_t stream = 0)
{
    int threadsPerBlock = TILE * TILE;
    int blocksPerGrid = N;
    softmax<TILE><<<blocksPerGrid, threadsPerBlock, 0, stream>>>(input, output, N);
}

// Apply RoPE (in-place) to Q and K: Q,K are [N x d], row-major
__global__ void apply_rope(float *Q, float *K, int N, int d)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N || col >= d) return;
    // only handle even indices and their following odd index
    if ((col & 1) == 0 && col + 1 < d) {
        int idx_even = row * d + col;
        int idx_odd = idx_even + 1;

        float q_e = Q[idx_even];
        float q_o = Q[idx_odd];
        float k_e = K[idx_even];
        float k_o = K[idx_odd];

        int i = col >> 1;
        float inv_freq = powf(10000.0f, -2.0f * (float)i / (float)d);
        float angle = (float)row * inv_freq;
        float c = cosf(angle);
        float s = sinf(angle);

        Q[idx_even] = q_e * c - q_o * s;
        Q[idx_odd]  = q_e * s + q_o * c;

        K[idx_even] = k_e * c - k_o * s;
        K[idx_odd]  = k_e * s + k_o * c;
    }
}

template<int TILE>
void launch_rope(float *Q, float *K, int N, int d, cudaStream_t stream = 0)
{
    dim3 threadsPerBlock(TILE, TILE);
    dim3 blocksPerGrid((d + TILE - 1) / TILE,
                       (N + TILE - 1) / TILE);
    apply_rope<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(Q, K, N, d);
}

extern "C" void solve(const float *Q, const float *K, const float *V, float *output, int N, int d_model, int h)
{   
    int d = d_model / h;
    float alpha = 1.0f / sqrtf((float)d);
    int curr_col;
    constexpr int TILE = 16;
    const int nstreams = 2;

    float *q, *k, *v, *softmax_in, *softmax_out, *o;

    size_t per_q = (size_t)N * d; //= per_k = per_v = per_o
    size_t per_softmax = (size_t)N * N;

    // you need to allocate more memory when you use streams!
    // NEVER share buffers accross streams, it creates races. 1 buffer = 1 stream (***).
    cudaMalloc(&q, nstreams * per_q * sizeof(float));
    cudaMalloc(&k, nstreams * per_q * sizeof(float));
    cudaMalloc(&v, nstreams * per_q * sizeof(float));
    cudaMalloc(&softmax_in, nstreams * per_softmax * sizeof(float));
    cudaMalloc(&softmax_out, nstreams * per_softmax * sizeof(float));
    cudaMalloc(&o, nstreams * per_q * sizeof(float));

    cudaStream_t streams[nstreams];
    for(int s = 0; s < nstreams; ++s) cudaStreamCreate(&streams[s]);

    for(int head = 0; head < h; ++head){
        curr_col = head * d;
        int s = head % nstreams;

        // stream s gets its own buffer !!! (***)
        float *q_s = q + s * per_q;                 // ALSO: CUDA GRAPHS REQUIRE FIXED DEVICE POINTERS AT CAPTURE TIME!!!
        float *k_s = k + s * per_q;                 // SCALARS CAN BE CHANGED BUT IT IS NOT BEST PRACTICE / VERY UNCOMMON
        float *v_s = v + s * per_q;                 // HENCE HERE CUDA GRAPHS WON'T WORK HERE UNLESS YOU DO 1 GRAPH PER HEAD
        float *softmax_in_s = softmax_in + s * per_softmax; // BUT THAT IS AN OVERKILL HERE => BETTER KEEP STREAMS-ONLY
        float *softmax_out_s = softmax_out + s * per_softmax;
        float *o_s = o + s * per_q;
        launch_extract_mat<TILE>(Q, q_s, 0, curr_col, N, d_model, N, d, streams[s]);
        launch_extract_mat<TILE>(K, k_s, 0, curr_col, N, d_model, N, d, streams[s]);
        launch_extract_mat<TILE>(V, v_s, 0, curr_col, N, d_model, N, d, streams[s]);
        // apply RoPE in-place on this stream's q/k before computing attention scores
        launch_rope<TILE>(q_s, k_s, N, d, streams[s]);
        launch_mma_A_B<TILE>(q_s, k_s, softmax_in_s, N, N, d, alpha, true, streams[s]);
        launch_softmax<TILE>(softmax_in_s, softmax_out_s, N, streams[s]);
        launch_mma_A_B<TILE>(softmax_out_s, v_s, o_s, N, d, N, 1.0f, false, streams[s]);
        launch_concat_mat<TILE>(output, o_s, 0, curr_col, N, d_model, N, d, streams[s]);
    }

    for(int s = 0; s < nstreams; ++s) cudaStreamSynchronize(streams[s]);
    for(int s = 0; s < nstreams; ++s) cudaStreamDestroy(streams[s]);


    cudaFree(q);
    cudaFree(k);
    cudaFree(v);
    cudaFree(softmax_in);
    cudaFree(softmax_out);
    cudaFree(o);

    }