#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_fp16.h>
#include "../include/config.h"

// Simple error checking wrapper
#define checkCudaErrors(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Compile: 
// nvcc -O3 -gencode arch=compute_89,code=sm_89 -I. -I./include tests/test_fa_tc.cu -o bin/test_fa_tc 2>&1 | head -50


// Include wmma_A_B from fa_tc_v2.cu
#include "../mha_kernels/fa_tc_v2.cu"

// Compile-time constant for threads per block
// WARPS_PER_BLOCK and THREADS_PER_WARP are defined in fa_tc_v2.cu
// THREADS = WARPS_PER_BLOCK * THREADS_PER_WARP
constexpr int TEST_THREADS = WARPS_PER_BLOCK * THREADS_PER_WARP;

// Wrapper kernel to test wmma_A_B
template <int THREADS, int M, int N, int K>
__global__ void test_wmma_kernel(
    const half* A,
    const half* B,
    float* C
) {
    // Shared memory scratch buffer for inter-warp communication in wmma_A_B
    // Size: (M/WMMA_M/2) pairs * 2 tiles per pair * WMMA_M * N floats
    extern __shared__ float c_scratch[];
    
    // Strides must match actual memory layout with padding: A is M×K, B is K×N, C is M×N
    // K and N include padding, so stride = K for A, stride = N for B, stride = N for C
    wmma_A_B<false, THREADS, M, N, K>(A, B, C, c_scratch, K, N, N);
}

int main() {
    printf("========================================\n");
    printf("Testing wmma_A_B from fa_tc_v2.cu\n");
    printf("========================================\n\n");


    // Test configuration (using constants from fa_tc_v2.cu)
    const int M = Br;  // A: M x K (Br rows)
    const int K = d + PAD;  // A: M x K (d columns with padding)
    const int N = Bc + PAD;  // B: K x N (Bc columns with padding), C: M x N
    
    printf("BEFORE PADDING:\n");
    printf("  A: %d x %d\n", M, d);
    printf("  B: %d x %d\n", d, Bc);
    printf("  C: %d x %d\n\n", M, Bc);
    
    printf("AFTER PADDING (stride includes padding):\n");
    printf("  A: %d x %d (half precision, stride=%d)\n", M, K, K);
    printf("  B: %d x %d (half precision, stride=%d)\n", K, N, N);
    printf("  C: %d x %d (float, result, stride=%d)\n\n", M, N, N);
    printf("Expected result: Each element = %d (dot product of %d ones)\n\n", d, d);

    size_t size_A = M * K * sizeof(half);
    size_t size_B = K * N * sizeof(half);
    size_t size_C = M * N * sizeof(float);

    // Host memory
    half *h_A = (half*)malloc(size_A);
    half *h_B = (half*)malloc(size_B);
    float *h_C = (float*)malloc(size_C);

    // Initialize: valid columns = 1.0, padding columns = 0.0
    printf("Initializing matrices...\n");
    // A: M x K, first d cols = 1.0, last PAD cols = 0.0
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            if (j < d) {
                h_A[i * K + j] = __float2half(1.0f);
            } else {
                h_A[i * K + j] = __float2half(0.0f);
            }
        }
    }
    // B: K x N, valid region is d x Bc = 1.0, rest = 0.0
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            if (i < d && j < Bc) {
                h_B[i * N + j] = __float2half(1.0f);
            } else {
                h_B[i * N + j] = __float2half(0.0f);
            }
        }
    }
    for (int i = 0; i < M * N; i++) {
        h_C[i] = 0.0f;
    }

    // Device memory
    half *d_A, *d_B;
    float *d_C;
    checkCudaErrors(cudaMalloc(&d_A, size_A));
    checkCudaErrors(cudaMalloc(&d_B, size_B));
    checkCudaErrors(cudaMalloc(&d_C, size_C));

    // Copy to device
    checkCudaErrors(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_C, h_C, size_C, cudaMemcpyHostToDevice));

    // Launch kernel
    printf("Launching WMMA kernel...\n");
    constexpr int threads_per_block = TEST_THREADS;  // WARPS_PER_BLOCK * THREADS_PER_WARP
    printf("  Threads per block: %d (WARPS_PER_BLOCK=%d)\n", threads_per_block, WARPS_PER_BLOCK);
    
    // Calculate scratch buffer size: (WARPS_PER_BLOCK / 2) pairs * 2 tiles per pair * WMMA_M * N floats
    int num_pairs = WARPS_PER_BLOCK / 2;
    size_t scratch_size = num_pairs * 2 * WMMA_M * N * sizeof(float);
    printf("  Num pairs: %d, Scratch buffer size: %zu floats\n", num_pairs, scratch_size / sizeof(float));
    
    // Template parameter must be compile-time constant
    test_wmma_kernel<TEST_THREADS, M, N, K><<<1, threads_per_block, scratch_size>>>(d_A, d_B, d_C);
    checkCudaErrors(cudaDeviceSynchronize());

    // Copy result back
    checkCudaErrors(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));

    // Print results
    printf("\n========== Matrix A (all rows and cols) ==========\n");
    for (int i = 0; i < M; i++) {
        printf("Row %2d: ", i);
        for (int j = 0; j < K; j++) {
            if (j == d) printf("| ");  // Mark padding boundary
            printf("%.1f ", __half2float(h_A[i * K + j]));
        }
        printf("\n");
    }

    printf("\n========== Matrix B (all rows and cols) ==========\n");
    for (int i = 0; i < K; i++) {
        printf("Row %2d: ", i);
        for (int j = 0; j < N; j++) {
            if (j == Bc) printf("| ");  // Mark padding boundary
            printf("%.1f ", __half2float(h_B[i * N + j]));
        }
        printf("\n");
    }

    printf("\n========== Result C (A @ B) ==========\n");
    bool all_correct = true;
    int expected = d;  // Only valid columns contribute
    for (int i = 0; i < M; i++) {
        printf("Row %2d: ", i);
        for (int j = 0; j < N; j++) {
            float val = h_C[i * N + j];
            printf("%.1f ", val);
            // Check if result is approximately d (product of d ones)
            if (j < Bc && (val < expected - 0.5f || val > expected + 0.5f)) {
                all_correct = false;
            }
            // Padding region should be 0
            if (j >= Bc && val > 0.1f) {
                all_correct = false;
            }
        }
        printf("\n");
    }

    printf("\n========== Verification ==========\n");
    printf("Expected: Valid region (first %d cols) = %d, zero elsewhere\n", Bc, expected);
    if (all_correct) {
        printf("Result: PASS ✓\n");
    } else {
        printf("Result: FAIL ✗\n");
    }

    // Cleanup
    free(h_A);
    free(h_B);
    free(h_C);
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));

    return 0;
}
