#include <torch/extension.h>
#include <stdexcept>
#include <vector>
#include <string>

#include "../../include/launchers.h"

using torch::Tensor;
namespace py = pybind11;

Tensor flash_solve(const Tensor &Q, const Tensor &K, const Tensor &V, 
                   int64_t d_model, int64_t num_heads, 
                   const std::string &kernel = "fa_tc_int8_b") {
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(Q.dtype() == torch::kFloat32, "Q must be float32");
    TORCH_CHECK(K.dtype() == torch::kFloat32, "K must be float32");
    TORCH_CHECK(V.dtype() == torch::kFloat32, "V must be float32");

    auto Qc = Q.contiguous();
    auto Kc = K.contiguous();
    auto Vc = V.contiguous();

    int64_t q_elems = Qc.numel();
    TORCH_CHECK(q_elems % d_model == 0, "Q.numel() must be divisible by d_model");
    int N = static_cast<int>(q_elems / d_model);

    auto out = torch::empty_like(Qc);

    // Route to specified kernel
    // Currently all kernels use the same solve() interface from launchers.h
    // In the future, different kernels can be routed to different launcher functions
    if (kernel != "fa_tc_int8_b") {
        TORCH_WARN("Kernel selection currently supports 'fa_tc_int8_b'; routing to default");
    }

    solve(reinterpret_cast<const float*>(Qc.data_ptr<float>()),
          reinterpret_cast<const float*>(Kc.data_ptr<float>()),
          reinterpret_cast<const float*>(Vc.data_ptr<float>()),
          reinterpret_cast<float*>(out.data_ptr<float>()),
          N, static_cast<int>(d_model), static_cast<int>(num_heads));

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_solve", &flash_solve, 
          "FlashAttention solve (CUDA)\n\n"
          "Args:\n"
          "  Q: Query tensor [N, d_model]\n"
          "  K: Key tensor [N, d_model]\n"
          "  V: Value tensor [N, d_model]\n"
          "  d_model: Model dimension\n"
          "  num_heads: Number of attention heads\n"
          "  kernel: Kernel variant (default: 'fa_tc_int8_b')",
          py::arg("Q"), py::arg("K"), py::arg("V"), 
          py::arg("d_model"), py::arg("num_heads"), 
          py::arg("kernel") = "fa_tc_int8_b");
}
