#include "verify.h"
#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <cstring>

static inline void apply_rope_cpu(float* row, int pos, int d, float base = 10000.0f) {
    // d assumed even
    for (int k = 0; k < d / 2; ++k) {
        float theta = std::pow(base, -static_cast<float>(2 * k) / d);
        float angle = pos * theta;
        float sin_a = std::sin(angle);
        float cos_a = std::cos(angle);

        float x = row[k];
        float y = row[k + d / 2];

        row[k] = x * cos_a - y * sin_a;
        row[k + d / 2] = x * sin_a + y * cos_a;
    }
}

void cpu_reference(const std::vector<float>& Q, const std::vector<float>& K, const std::vector<float>& V, std::vector<float>& output, int N, int d_model, int h){
    // Validate sizes
    if ((int)Q.size() != N * d_model || (int)K.size() != N * d_model || (int)V.size() != N * d_model) {
        std::cerr << "cpu_reference: input sizes do not match N*d_model\n";
        return;
    }

    output.assign(N * d_model, 0.0f);

    int d_head = d_model / h;
    float alpha = 1.0f / std::sqrt((float)d_head);

    // Temporary buffers
    std::vector<float> q_row(d_head);
    std::vector<float> k_row(d_head);
    std::vector<float> v_row(d_head);
    std::vector<float> scores(N);
    std::vector<float> softmax(N);

    for (int head = 0; head < h; ++head) {
        int col_off = head * d_head;

        // For each query position
        for (int i = 0; i < N; ++i) {
            // Progress display
            if (i % 512 == 0 || i == N - 1) {
                float progress = 100.0f * (head * N + i) / (h * N);
                std::printf("  CPU reference: %.1f%% (head %d/%d, pos %d/%d)\r", progress, head + 1, h, i + 1, N);
                std::fflush(stdout);
            }

            // load and apply RoPE to q_i
            for (int kk = 0; kk < d_head; ++kk) {
                q_row[kk] = Q[i * d_model + col_off + kk];
            }
            apply_rope_cpu(q_row.data(), i, d_head);

            // compute scores dot(q_i, k_j) for all j
            float max_score = -INFINITY;
            for (int j = 0; j < N; ++j) {
                // load and apply RoPE to k_j into k_row
                for (int kk = 0; kk < d_head; ++kk) {
                    k_row[kk] = K[j * d_model + col_off + kk];
                }
                apply_rope_cpu(k_row.data(), j, d_head);

                // dot product
                float s = 0.0f;
                for (int kk = 0; kk < d_head; ++kk) s += q_row[kk] * k_row[kk];
                s *= alpha;
                scores[j] = s;
                if (s > max_score) max_score = s;
            }

            // softmax (stable)
            float sum_exp = 0.0f;
            for (int j = 0; j < N; ++j) {
                float e = std::exp(scores[j] - max_score);
                softmax[j] = e;
                sum_exp += e;
            }
            for (int j = 0; j < N; ++j) softmax[j] /= sum_exp;

            // compute output row head slice: sum_j softmax[j] * v_j_head
            for (int kk = 0; kk < d_head; ++kk) v_row[kk] = 0.0f;
            for (int j = 0; j < N; ++j) {
                float w = softmax[j];
                for (int kk = 0; kk < d_head; ++kk) {
                    v_row[kk] += w * V[j * d_model + col_off + kk];
                }
            }

            // write back to output
            for (int kk = 0; kk < d_head; ++kk) {
                output[i * d_model + col_off + kk] = v_row[kk];
            }
        }
    }
    std::printf("\n");
}

bool save_reference(const std::vector<float>& data, const char* filename, int N, int d_model) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open " << filename << " for writing\n";
        return false;
    }
    
    // Write metadata
    file.write(reinterpret_cast<const char*>(&N), sizeof(int));
    file.write(reinterpret_cast<const char*>(&d_model), sizeof(int));
    
    // Write data
    size_t size = data.size();
    file.write(reinterpret_cast<const char*>(data.data()), size * sizeof(float));
    
    file.close();
    std::printf("Saved reference output to %s\n", filename);
    return true;
}

bool load_reference(std::vector<float>& data, const char* filename, int N, int d_model) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        return false;  // File doesn't exist, will compute instead
    }
    
    // Read and verify metadata
    int saved_N, saved_d_model;
    file.read(reinterpret_cast<char*>(&saved_N), sizeof(int));
    file.read(reinterpret_cast<char*>(&saved_d_model), sizeof(int));
    
    if (saved_N != N || saved_d_model != d_model) {
        std::cerr << "Reference cache mismatch: expected N=" << N << " d_model=" << d_model 
                  << " but got N=" << saved_N << " d_model=" << saved_d_model << "\n";
        file.close();
        return false;
    }
    
    // Read data
    data.assign(N * d_model, 0.0f);
    file.read(reinterpret_cast<char*>(data.data()), N * d_model * sizeof(float));
    file.close();
    
    std::printf("Loaded reference output from %s\n", filename);
    return true;
}

bool verify_results(const std::vector<float>& h_output, const std::vector<float>& ref_output, float epsilon, float rel_tol){
    if(h_output.size() != ref_output.size()){
        std::cerr << " Size mismatch: " << h_output.size() << " vs " << ref_output.size() << std::endl;
        return false;
    }

    for (size_t i = 0; i < h_output.size(); ++i){
        float a = h_output[i];
        float b = ref_output[i];
        if (!std::isfinite(a) || !std::isfinite(b)){
            std::cerr << "Non-finite value at index " << i << std::endl;
            return false;
        }
        float tol = std::max(epsilon, rel_tol * std::fabs(b));
        if (std::fabs(a - b) > tol) {
            std::cerr << "Mismatch at index: " << i << ": got=" << a << " ref=" << b << " tol=" << tol << "\n";
            return false;
        }
    }
    return true;
}