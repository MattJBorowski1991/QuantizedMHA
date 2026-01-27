#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <string>
#include <random>
#include <chrono>
#include <filesystem>
#include <iomanip>

namespace fs = std::filesystem;

static void write_f32_bin(const fs::path &p, const std::vector<float>& data){
    std::ofstream f(p, std::ios::binary);
    f.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
}
static void write_i8_bin(const fs::path &p, const std::vector<int8_t>& data){
    std::ofstream f(p, std::ios::binary);
    f.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(int8_t));
}
static void write_text(const fs::path &p, const std::string &s){ std::ofstream f(p); f<<s; }

static void softmax_rowwise(const std::vector<float>& in, std::vector<float>& out, int rows, int cols){
    out.resize(rows * cols);
    for(int r=0;r<rows;++r){
        const float* row = &in[r*cols];
        float* orow = &out[r*cols];
        float m = -INFINITY;
        for(int c=0;c<cols;++c) m = std::max(m, row[c]);
        float s = 0.0f;
        for(int c=0;c<cols;++c){ orow[c] = std::exp(row[c] - m); s += orow[c]; }
        if (s == 0.0f) s = 1.0f;
        for(int c=0;c<cols;++c) orow[c] /= s;
    }
}

// CPU version of RoPE (matches device apply_rope)
static void apply_rope_cpu(float* row, int pos, int d, float base = 10000.0f){
    for (int k = 0; k < d / 2; k++) {
        float theta = powf(base, -static_cast<float>(2 * k) / d);
        float angle = pos * theta;
        float sin_a = sinf(angle);
        float cos_a = cosf(angle);

        float x = row[k];
        float y = row[k + d / 2];

        row[k] = x * cos_a - y * sin_a;
        row[k + d / 2] = x * sin_a + y * cos_a;
    }
}

static std::vector<float> cpu_mha(const std::vector<float>& Q, const std::vector<float>& K, const std::vector<float>& V,
                                  int N, int d_model, int h, std::vector<float>* out_S=nullptr, std::vector<float>* out_P=nullptr){
    int d_head = d_model / h;
    float scale = 1.0f / std::sqrt((float)d_head);
    std::vector<float> out(N * d_model, 0.0f);
    if (out_S) out_S->assign(N*N*h, 0.0f); // store per-head S concatenated
    if (out_P) out_P->assign(N*N*h, 0.0f);
    for(int head=0; head<h; ++head){
        int col = head * d_head;
        // extract head slices
        std::vector<float> q(N * d_head), k(N * d_head), v(N * d_head);
        for(int i=0;i<N;i++) for(int j=0;j<d_head;j++){
            q[i*d_head + j] = Q[i*d_model + (col + j)];
            k[i*d_head + j] = K[i*d_model + (col + j)];
            v[i*d_head + j] = V[i*d_model + (col + j)];
        }
        // compute S = Q * K^T (N x N)
        std::vector<float> S(N * N, 0.0f);
        for(int i=0;i<N;++i) for(int j=0;j<N;++j){
            float s = 0.0f;
            for(int d=0; d<d_head; ++d) s += q[i*d_head + d] * k[j*d_head + d];
            S[i*N + j] = s * scale;
        }
        // softmax
        std::vector<float> P;
        softmax_rowwise(S, P, N, N);
        // O = P * V
        std::vector<float> O(N * d_head, 0.0f);
        for(int i=0;i<N;++i) for(int j=0;j<d_head;++j){
            float s = 0.0f;
            for(int kcol=0;kcol<N;++kcol) s += P[i*N + kcol] * v[kcol*d_head + j];
            O[i*d_head + j] = s;
        }
        // write back
        for(int i=0;i<N;i++) for(int j=0;j<d_head;j++) out[i*d_model + (col + j)] = O[i*d_head + j];
        if (out_S) std::copy(S.begin(), S.end(), out_S->begin() + head * N * N);
        if (out_P) std::copy(P.begin(), P.end(), out_P->begin() + head * N * N);
    }
    return out;
}

static void quantize_int8(const std::vector<float>& src, std::vector<int8_t>& dst, float scale, float zero_point){
    dst.resize(src.size());
    for(size_t i=0;i<src.size();++i){
        int q = (int)std::round(src[i] / scale + zero_point);
        if (q > 127) q = 127; if (q < -128) q = -128;
        dst[i] = (int8_t)q;
    }
}

int main(){
    // produce multiple cases: small, unaligned, medium, large, quant_small
    struct Case{ std::string name; int N; int d_model; int h; } cases[] = {
        {"small", 8, 32, 4},
        {"unaligned", 50, 64, 8},
        {"medium", 128, 512, 8},
        {"large", 256, 1024, 16},
        {"huge_1024", 1024, 128, 8},
        {"huge_2048", 2048, 128, 8},
        {"huge_4096", 4096, 128, 8},
        {"quant_small", 8, 32, 4}
    };

    for(const auto &c : cases){
        std::cout<<"Generating case: "<<c.name<<"\n";
        fs::path dir = fs::path("tests") / "golden" / c.name;
        fs::create_directories(dir);
        int N = c.N; int d_model = c.d_model; int h = c.h; int total = N * d_model;
        std::vector<float> Q(total), K(total), V(total);
        // realistic pseudo-random fill (normal distribution), deterministic via fixed seed
        std::mt19937 rng(12345 + N + d_model + h);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for(int i=0;i<N;i++) for(int j=0;j<d_model;j++){
            Q[i*d_model + j] = nd(rng) * 0.5f; // scale down a bit
            K[i*d_model + j] = nd(rng) * 0.5f;
            V[i*d_model + j] = nd(rng) * 0.5f;
        }
        // apply RoPE to Q and K per-head (match device behavior)
        int d_head = d_model / h;
        for(int i=0;i<N;++i){
            for(int head=0; head<h; ++head){
                int col = head * d_head;
                apply_rope_cpu(&Q[i*d_model + col], i, d_head);
                apply_rope_cpu(&K[i*d_model + col], i, d_head);
            }
        }

        // compute golden (may take some time for large cases)
        std::vector<float> S, P;
        auto t0 = std::chrono::high_resolution_clock::now();
        auto O = cpu_mha(Q,K,V,N,d_model,h, &S, &P);
        auto t1 = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();
        std::cout<<"  cpu_mha took "<<elapsed<<" s for N="<<N<<" d_model="<<d_model<<" h="<<h<<"\n";
        // write inputs and outputs
        write_f32_bin(dir / "Q.f32.bin", Q);
        write_f32_bin(dir / "K.f32.bin", K);
        write_f32_bin(dir / "V.f32.bin", V);
        write_f32_bin(dir / "O.f32.bin", O);
        write_f32_bin(dir / "S.f32.bin", S);
        write_f32_bin(dir / "P.f32.bin", P);
        // metadata
        std::ostringstream meta;
        meta<<"{\n";
        meta<<"  \"N\": "<<N<<",\n";
        meta<<"  \"d_model\": "<<d_model<<",\n";
        meta<<"  \"h\": "<<h<<"\n";
        meta<<"}\n";
        write_text(dir / "meta.json", meta.str());

        if (c.name.find("quant") != std::string::npos){
            // choose simple scales and zero points and write quantized inputs
            float q_scale = 0.05f, k_scale = 0.05f, v_scale = 0.05f;
            float q_zero = 0.0f, k_zero = 0.0f, v_zero = 0.0f;
            std::vector<int8_t> Qq, Kq, Vq;
            quantize_int8(Q, Qq, q_scale, q_zero);
            quantize_int8(K, Kq, k_scale, k_zero);
            quantize_int8(V, Vq, v_scale, v_zero);
            write_i8_bin(dir / "Q.int8.bin", Qq);
            write_i8_bin(dir / "K.int8.bin", Kq);
            write_i8_bin(dir / "V.int8.bin", Vq);
            std::ostringstream meta2;
            meta2<<"{\n";
            meta2<<"  \"N\": "<<N<<",\n";
            meta2<<"  \"d_model\": "<<d_model<<",\n";
            meta2<<"  \"h\": "<<h<<",\n";
            meta2<<"  \"q_scale\": "<<std::setprecision(8)<<q_scale<<",\n";
            meta2<<"  \"k_scale\": "<<std::setprecision(8)<<k_scale<<",\n";
            meta2<<"  \"v_scale\": "<<std::setprecision(8)<<v_scale<<",\n";
            meta2<<"  \"q_zero\": "<<q_zero<<",\n";
            meta2<<"  \"k_zero\": "<<k_zero<<",\n";
            meta2<<"  \"v_zero\": "<<v_zero<<"\n";
            meta2<<"}\n";
            write_text(dir / "meta_quant.json", meta2.str());
        }
    }

    std::cout<<"Done. Golden cases written under tests/golden/"<<"\n";
    return 0;
}
