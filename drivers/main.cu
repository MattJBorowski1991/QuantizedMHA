#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <cstdio>
#include <vector>
#include <string>
#include <cstring>
#include <iostream>
#include "../inputs/data.h"
#include "../utils/verify.h"
#include "../tools/check_cuda.h"

// Forward declarations for kernel launchers (defined via MHA_SOLVE macro in respective .cu files)
extern "C" void launch(const float *Q, const float *K, const float *V, float *output, int N, int d_model, int h);

int main(int argc, char** argv){
    std::string kernel = "unfused";
    int warmup = 5;
    int runs = 10;
    bool use_random = false;

    for(int i = 1; i < argc; ++i){
        if(std::strncmp(argv[i], "--kernel=", 9) == 0) kernel = std::string(argv[i] + 9);
        else if(std::strcmp(argv[i], "-k") == 0 && i + 1 < argc) kernel = std::string(argv[++i]);
        else if(std::strncmp(argv[i], "--warmup=", 9) == 0) warmup = std::atoi(argv[i] + 9);
        else if(std::strncmp(argv[i], "--runs=", 7) == 0) runs = std::atoi(argv[i] + 7);
        else if(std::strcmp(argv[i], "--random") == 0) use_random = true;
        else if(std::strcmp(argv[i], "--help") == 0){
            std::printf("Usage: %s [--kernel=KERNEL] [--warmup=N] [--runs=M] [--random]\n", argv[0]);
            std::printf("  KERNEL options: unfused, fa, fa_warps, fa_tc, fa_int8\n");
            return 0;
        }
    }

    // Define MHA problem size
    const int N = 4096;
    const int d_model = 1024;
    const int h = 128;
    const int output_size = N * d_model;

    // Allocate host vectors
    std::vector<float> h_Q(output_size);
    std::vector<float> h_K(output_size);
    std::vector<float> h_V(output_size);
    std::vector<float> h_output(output_size);

    // Device pointers
    float *d_Q = nullptr, *d_K = nullptr, *d_V = nullptr, *d_output = nullptr;

    // Initialize host data (constant 1.0 for correctness check)
    std::printf("Initializing host data (constant values for correctness check)...\n");
    initialize_host_data(h_Q, h_K, h_V, N, d_model, false);
    allocate_and_copy_to_device(h_Q, h_K, h_V, d_Q, d_K, d_V, d_output, N, d_model);

    // Correctness check before profiling
    {
        std::printf("Running correctness check with kernel: %s\n", kernel.c_str());
        launch(d_Q, d_K, d_V, d_output, N, d_model, h);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(h_output.data(), d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost));

        std::vector<float> ref_output(output_size);
        std::printf("Computing CPU reference (multi-head attention with RoPE)...\n");
        cpu_reference(h_Q, h_K, h_V, ref_output, N, d_model, h);

        if(!verify_results(h_output, ref_output, 1e-3f, 1e-3f)){
            std::fprintf(stderr, "Correctness check FAILED. Aborting.\n");
            cleanup_device_data(d_Q, d_K, d_V, d_output);
            return 1;
        }
        std::printf("Correctness check PASSED.\n");
    }

    // Clean up device memory for profiling phase
    cleanup_device_data(d_Q, d_K, d_V, d_output);

    // Re-initialize with random data for profiling
    if (use_random) {
        std::printf("Reinitializing host data with random values for profiling...\n");
        initialize_host_data(h_Q, h_K, h_V, N, d_model, true);
    }
    allocate_and_copy_to_device(h_Q, h_K, h_V, d_Q, d_K, d_V, d_output, N, d_model);

    // Start profiling (for NCU and code-side timing)
    CHECK_CUDA(cudaProfilerStart());

    // Warmup iterations
    std::printf("Running %d warmup iterations...\n", warmup);
    for(int i = 0; i < warmup; ++i){
        launch(d_Q, d_K, d_V, d_output, N, d_model, h);
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // Profiling runs
    std::printf("Running %d profiling iterations...\n", runs);
    for(int r = 0; r < runs; ++r){
        launch(d_Q, d_K, d_V, d_output, N, d_model, h);
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // Stop profiling (for NCU)
    CHECK_CUDA(cudaProfilerStop());

    // Copy back final output
    CHECK_CUDA(cudaMemcpy(h_output.data(), d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost));

    // Cleanup
    cleanup_device_data(d_Q, d_K, d_V, d_output);

    std::printf("Profiling complete.\n");
    return 0;
}
