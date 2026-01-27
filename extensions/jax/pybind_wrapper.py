import cupy as cp
import jax_ext as pybind_flash

def flash_solve_cupy(q: cp.ndarray, k: cp.ndarray, v: cp.ndarray, d_model: int, num_heads: int):
    assert q.dtype == cp.float32 and k.dtype == cp.float32 and v.dtype == cp.float32
    assert q.ndim == 2 and k.ndim == 2 and v.ndim == 2

    # ensure contiguous
    qc = cp.ascontiguousarray(q)
    kc = cp.ascontiguousarray(k)
    vc = cp.ascontiguousarray(v)

    N = qc.shape[0]
    out = cp.empty_like(qc)

    pybind_flash.flash_solve_ptr(int(qc.data.ptr), int(kc.data.ptr), int(vc.data.ptr), int(out.data.ptr), int(N), int(d_model), int(num_heads))

    return out
