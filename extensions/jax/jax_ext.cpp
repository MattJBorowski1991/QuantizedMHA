#include <pybind11/pybind11.h>
#include <string>
#include <cstdint>
#include "../../include/launchers.h"

namespace py = pybind11;

// Simple pointer-based wrapper: accepts device pointer addresses (uint64)
// and forwards them to the existing `solve` entrypoint implemented in the
// kernel sources (mha_kernels/*.cu). The Python side will pass CuPy
// device pointers obtained from DLPack conversions.
void flash_solve(unsigned long long q_ptr,
                 unsigned long long k_ptr,
                 unsigned long long v_ptr,
                 unsigned long long out_ptr,
                 int N, int d_model, int num_heads,
                 const std::string &kernel = "fa_tc_int8_b")
{
    (void)kernel; // kernel selection is a build/runtime convention for now

    const float* Q = reinterpret_cast<const float*>(reinterpret_cast<void*>(q_ptr));
    const float* K = reinterpret_cast<const float*>(reinterpret_cast<void*>(k_ptr));
    const float* V = reinterpret_cast<const float*>(reinterpret_cast<void*>(v_ptr));
    float* O = reinterpret_cast<float*>(reinterpret_cast<void*>(out_ptr));

    // Call the existing C-linked entrypoint
    solve(Q, K, V, O, N, d_model, num_heads);
}

PYBIND11_MODULE(jax_ext, m) {
    m.doc() = "JAX / DLPack wrapper for QuantizedMHA CUDA kernels";
    m.def("flash_solve", &flash_solve,
          "Call CUDA `solve` with device pointers (uint64 addresses)",
          py::arg("q_ptr"), py::arg("k_ptr"), py::arg("v_ptr"), py::arg("out_ptr"),
          py::arg("N"), py::arg("d_model"), py::arg("num_heads"),
          py::arg("kernel") = "fa_tc_int8_b");
}
