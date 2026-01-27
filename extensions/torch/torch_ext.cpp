#include <torch/extension.h>
#include <stdexcept>
#include <vector>

#include "../../include/launchers.h"

using torch::Tensor;

Tensor flash_solve(const Tensor &Q, const Tensor &K, const Tensor &V, int64_t d_model, int64_t num_heads) {
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda(), "Inputs must be CUDA tensors");

    auto Qc = Q.contiguous();
    auto Kc = K.contiguous();
    auto Vc = V.contiguous();

    int64_t q_elems = Qc.numel();
    TORCH_CHECK(q_elems % d_model == 0, "Q.numel() must be divisible by d_model");
    int N = static_cast<int>(q_elems / d_model);

    auto out = torch::empty_like(Qc);

    // call existing C ABI solve
    solve(reinterpret_cast<const float*>(Qc.data_ptr<float>()),
          reinterpret_cast<const float*>(Kc.data_ptr<float>()),
          reinterpret_cast<const float*>(Vc.data_ptr<float>()),
          reinterpret_cast<float*>(out.data_ptr<float>()),
          N, static_cast<int>(d_model), static_cast<int>(num_heads));

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_solve", &flash_solve, "FlashAttention solve (CUDA)");
}
