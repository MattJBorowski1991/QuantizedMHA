#pragma once

#include <cuda_runtime.h>
#include <cmath>
#include "config.h"


// Macro to define the extern "C" solve function consistently across kernels.
// LAUNCHER_LAMBDA: capture-list + lambda body accepting (q_s, k_s, v_s, out_s, N, d_head, alpha, stream, aux)
// AUX_BYTES: size of per-stream auxiliary memory to allocate (0 if no aux memory needed)
#define MHA_SOLVE(LAUNCHER_LAMBDA, AUX_BYTES) \
extern "C" void solve(const float *Q, const float *K, const float *V, float *output, int N, int d_model, int h) \
{ \
    launch_all_heads_generic<Br, Bc>(Q, K, V, output, N, d_model, h, \
        LAUNCHER_LAMBDA, AUX_BYTES); \
}


// launch_all_heads_generic = template = Generic host-side launcher for per-head extraction/concat and kernel invocation.
// KernelLauncher is a callable=lfunctor=lambda (any object that you can invoke like a function) with signature:
//   void(const float* q, const float* k, const float* v, float* out, int N, int d_head, float alpha, cudaStream_t stream, void* aux)
// The last `aux` pointer is an optional per-stream auxiliary device buffer (1 stream = 1 buffer to prevent racing).

template<int Br, int Bc, typename KernelLauncher>
void launch_all_heads_generic(
    const float* Q, const float* K, const float* V,
    float* output,
    int N, int d_model, int h,
    KernelLauncher kernel_launch,
    size_t aux_bytes_per_stream = 0)
{
    int d_head = d_model / h;
    float alpha = 1.0f / sqrtf((float)d_head);
    const int nstreams = NSTREAMS;

    size_t per_q = (size_t)N * d_head;

    float *q, *k, *v, *out;

    cudaMalloc(&q, nstreams * per_q * sizeof(float));
    cudaMalloc(&k, nstreams * per_q * sizeof(float));
    cudaMalloc(&v, nstreams * per_q * sizeof(float));
    cudaMalloc(&out, nstreams * per_q * sizeof(float));

    cudaStream_t streams[NSTREAMS];
    for (int s = 0; s < nstreams; ++s) cudaStreamCreate(&streams[s]);

    // Optional auxiliary buffer (per-stream contiguous slices)
    void* aux = nullptr;
    if (aux_bytes_per_stream > 0) {
        cudaMalloc(&aux, nstreams * aux_bytes_per_stream);
    }

    for (int head = 0; head < h; ++head) {
        int curr_col = head * d_head;
        int s = head % nstreams;

        float *q_s = q + s * per_q;
        float *k_s = k + s * per_q;
        float *v_s = v + s * per_q;
        float *out_s = out + s * per_q;

        launch_extract_mat<TILE>(Q, q_s, 0, curr_col, N, d_model, N, d_head, streams[s]);
        launch_extract_mat<TILE>(K, k_s, 0, curr_col, N, d_model, N, d_head, streams[s]);
        launch_extract_mat<TILE>(V, v_s, 0, curr_col, N, d_model, N, d_head, streams[s]);

        // Compute pointer to per-stream aux slice (or nullptr)
        void* aux_ptr = nullptr;
        if (aux) aux_ptr = (char*)aux + s * aux_bytes_per_stream;

        // Invoke the provided kernel launcher for this head and stream
        kernel_launch(q_s, k_s, v_s, out_s, N, d_head, alpha, streams[s], aux_ptr);

        launch_concat_mat<TILE>(output, out_s, 0, curr_col, N, d_model, N, d_head, streams[s]);
    }

    for (int s = 0; s < nstreams; ++s) cudaStreamSynchronize(streams[s]);
    for (int s = 0; s < nstreams; ++s) cudaStreamDestroy(streams[s]);

    cudaFree(q);
    cudaFree(k);
    cudaFree(v);
    cudaFree(out);
    if (aux) cudaFree(aux);
}

