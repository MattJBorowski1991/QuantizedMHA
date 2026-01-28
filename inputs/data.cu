#include "data.h"
#include <cuda_runtime.h>
#include <vector>
#include <random>
#include <fstream>
#include <cstring>
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

bool save_inputs(const std::vector<float>& h_Q, const std::vector<float>& h_K, const std::vector<float>& h_V, 
                 const char* filename, int N, int d_model) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::fprintf(stderr, "Failed to open %s for writing\n", filename);
        return false;
    }
    
    // Write metadata
    file.write(reinterpret_cast<const char*>(&N), sizeof(int));
    file.write(reinterpret_cast<const char*>(&d_model), sizeof(int));
    
    // Write data
    size_t size = N * d_model;
    file.write(reinterpret_cast<const char*>(h_Q.data()), size * sizeof(float));
    file.write(reinterpret_cast<const char*>(h_K.data()), size * sizeof(float));
    file.write(reinterpret_cast<const char*>(h_V.data()), size * sizeof(float));
    
    file.close();
    std::printf("Saved input matrices to %s\n", filename);
    return true;
}

bool load_inputs(std::vector<float>& h_Q, std::vector<float>& h_K, std::vector<float>& h_V,
                 const char* filename, int N, int d_model) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        return false;  // File doesn't exist
    }
    
    // Read and verify metadata
    int saved_N, saved_d_model;
    file.read(reinterpret_cast<char*>(&saved_N), sizeof(int));
    file.read(reinterpret_cast<char*>(&saved_d_model), sizeof(int));
    
    if (saved_N != N || saved_d_model != d_model) {
        std::fprintf(stderr, "Input cache mismatch: expected N=%d d_model=%d but got N=%d d_model=%d\n",
                     N, d_model, saved_N, saved_d_model);
        file.close();
        return false;
    }
    
    // Read data
    size_t size = N * d_model;
    h_Q.assign(size, 0.0f);
    h_K.assign(size, 0.0f);
    h_V.assign(size, 0.0f);
    
    file.read(reinterpret_cast<char*>(h_Q.data()), size * sizeof(float));
    file.read(reinterpret_cast<char*>(h_K.data()), size * sizeof(float));
    file.read(reinterpret_cast<char*>(h_V.data()), size * sizeof(float));
    file.close();
    
    std::printf("Loaded input matrices from %s\n", filename);
    return true;
}