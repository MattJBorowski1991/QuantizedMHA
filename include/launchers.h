#pragma once

#include <cuda_runtime.h>
#include <cmath>
#include "config.h"
#include "../utils/utils.cu"


// Generic solve implementation template for all kernels.
// KernelFn: callable with signature void(float* q_s, float* k_s, float* v_s, float* out_s, 
//           int N, int d_head, float alpha, cudaStream_t stream, void* aux)
// aux_bytes_per_stream: size of per-stream auxiliary memory (0 if no aux memory needed)
template<typename KernelFn>
void launch(const float *Q, const float *K, const float *V, float *output, 
            int N, int d_model, int h, KernelFn kernel_fn, size_t aux_bytes_per_stream = 0)
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

        // Invoke the kernel-specific launcher
        kernel_fn(q_s, k_s, v_s, out_s, N, d_head, alpha, streams[s], aux_ptr);

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

