#include <pybind11/pybind11.h>
#include <cstdint>

#include "../../include/launchers.h"

namespace py = pybind11;

void flash_solve_ptr(uint64_t q_ptr, uint64_t k_ptr, uint64_t v_ptr, uint64_t out_ptr, int N, int d_model, int num_heads) {
    const float* Q = reinterpret_cast<const float*>(q_ptr);
    const float* K = reinterpret_cast<const float*>(k_ptr);
    const float* V = reinterpret_cast<const float*>(v_ptr);
    float* O = reinterpret_cast<float*>(out_ptr);

    // call the existing C ABI
    solve(Q, K, V, O, N, d_model, num_heads);
}

PYBIND11_MODULE(jax_ext, m) {
    m.def("flash_solve_ptr", &flash_solve_ptr, "Launch solve via device pointers");
}
