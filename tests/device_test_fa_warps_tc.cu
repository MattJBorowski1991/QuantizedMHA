#include <iostream>
#include <vector>
#include <cassert>
#include <cuda_runtime.h>
#include <fstream>
#include <string>
#include "test_utils.h"

#include "../mha_kernels/fa_warps_tc.cu"

static void check_cuda(cudaError_t e, const char* msg){ if(e != cudaSuccess){ std::cerr<<msg<<": "<<cudaGetErrorString(e)<<"\n"; exit(1);} }

int main(){
    const std::string basedir = "tests/golden/small";
    auto read_f32 = [](const std::string &path, std::vector<float> &out){
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if(!f) { std::cerr<<"Failed to open "<<path<<"\n"; exit(1); }
        std::streamsize sz = f.tellg(); f.seekg(0, std::ios::beg);
        out.resize(sz / sizeof(float));
        f.read(reinterpret_cast<char*>(out.data()), sz);
    };
    std::vector<float> Q, K, V, refO;
    read_f32(basedir + "/Q.f32.bin", Q);
    read_f32(basedir + "/K.f32.bin", K);
    read_f32(basedir + "/V.f32.bin", V);
    read_f32(basedir + "/O.f32.bin", refO);
    int d_model = 32; int h = 4;
    int N = (int)(Q.size() / d_model);
    std::vector<float> O(refO.size());
    float *dQ, *dK, *dV, *dO;
    check_cuda(cudaMalloc(&dQ, Q.size()*sizeof(float)), "cudaMalloc dQ");
    check_cuda(cudaMalloc(&dK, K.size()*sizeof(float)), "cudaMalloc dK");
    check_cuda(cudaMalloc(&dV, V.size()*sizeof(float)), "cudaMalloc dV");
    check_cuda(cudaMalloc(&dO, O.size()*sizeof(float)), "cudaMalloc dO");
    check_cuda(cudaMemcpy(dQ, Q.data(), Q.size()*sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy H2D Q");
    check_cuda(cudaMemcpy(dK, K.data(), K.size()*sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy H2D K");
    check_cuda(cudaMemcpy(dV, V.data(), V.size()*sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy H2D V");
    solve(dQ, dK, dV, dO, N, d_model, h);
    check_cuda(cudaMemcpy(O.data(), dO, O.size()*sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy D2H O");
    float maxerr = 0.0f;
    for(size_t i=0;i<O.size() && i<refO.size();++i) maxerr = std::max(maxerr, std::fabs(O[i]-refO[i]));
    std::cout<<"device_test_fa_warps_tc: maxerr="<<maxerr<<"\n";
    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
    return 0;
}
