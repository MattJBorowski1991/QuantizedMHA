#pragma once
#include <vector>

// Compute CPU reference multi-head attention with RoPE.
// Q, K, V: flattened row-major matrices of shape (N, d_model)
// output: preallocated vector of size N * d_model
void cpu_reference(const std::vector<float>& Q,
                   const std::vector<float>& K,
                   const std::vector<float>& V,
                   std::vector<float>& output,
                   int N, int d_model, int h);

// Compare two flat buffers element-wise within epsilon tolerance.
bool verify_results(const std::vector<float>& h_output,
                    const std::vector<float>& ref_output,
                    float epsilon = 1e-3f,
                    float rel_tol = 1e-3f);
// Save reference output to binary file
bool save_reference(const std::vector<float>& data, const char* filename, int N, int d_model);

// Load reference output from binary file (returns false if file doesn't exist)
bool load_reference(std::vector<float>& data, const char* filename, int N, int d_model);