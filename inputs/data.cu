#include "data.h"
#include <cuda_runtime.h>
#include <vector>
#include <random>
#include "../tools/check_cuda.h"

void initialize_host_data(std::vector<float>& h_Q, std::vector<float>& h_K, std::vector<float>& h_V, int N, int d_model, bool use_random){
    int total_size = N * d_model;
    h_Q.resize(total_size);
    h_K.resize(total_size);
    h_V.resize(total_size);
    
    if (use_random) {
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dis(0.0f, 1.0f);
        for (int i = 0; i < total_size; ++i) {
            h_Q[i] = dis(gen);
            h_K[i] = dis(gen);
            h_V[i] = dis(gen);
        }
    } else {
        for (int i = 0; i < total_size; ++i) {
            h_Q[i] = 1.0f;
            h_K[i] = 1.0f;
            h_V[i] = 1.0f;
        }
    }
}

void allocate_and_copy_to_device(
    const std::vector<float>& h_Q, const std::vector<float>& h_K, const std::vector<float>& h_V,
    float*& d_Q, float*& d_K, float*& d_V, float*& d_output,
    int N, int d_model){
    int bytes = N * d_model * sizeof(float);
    CHECK_CUDA(cudaMalloc((void**)&d_Q, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_K, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_V, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_output, bytes));

    CHECK_CUDA(cudaMemcpy(d_Q, h_Q.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_K, h_K.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_V, h_V.data(), bytes, cudaMemcpyHostToDevice));
}

void cleanup_device_data(float* d_Q, float* d_K, float* d_V, float* d_output){
    if(d_Q) CHECK_CUDA(cudaFree(d_Q));
    if(d_K) CHECK_CUDA(cudaFree(d_K));
    if(d_V) CHECK_CUDA(cudaFree(d_V));
    if(d_output) CHECK_CUDA(cudaFree(d_output));
}