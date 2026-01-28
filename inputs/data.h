#ifndef DATA_H
#define DATA_H

#include <vector>

void initialize_host_data(std::vector<float>& h_Q, std::vector<float>& h_K, std::vector<float>& h_V, int N, int d_model, bool use_random = false);

void allocate_and_copy_to_device(
    const std::vector<float>& h_Q, const std::vector<float>& h_K, const std::vector<float>& h_V,
    float*& d_Q, float*& d_K, float*& d_V, float*& d_output,
    int N, int d_model);

void cleanup_device_data(float* d_Q, float* d_K, float* d_V, float* d_output);

// Input caching for reproducible profiling
bool save_inputs(const std::vector<float>& h_Q, const std::vector<float>& h_K, const std::vector<float>& h_V, 
                 const char* filename, int N, int d_model);

bool load_inputs(std::vector<float>& h_Q, std::vector<float>& h_K, std::vector<float>& h_V,
                 const char* filename, int N, int d_model);

#endif //DATA_H