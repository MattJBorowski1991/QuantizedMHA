# jax_ext (pybind/CuPy) — eager JAX helper
# jax_ext (pybind/CuPy) — eager JAX helper

Build

```bash
cd extensions/jax
python -m pip install --upgrade setuptools wheel pybind11 cupy
python setup.py build_ext --inplace
```

Test (what data and where)

- The smoke test uses the golden fixtures in `tests/golden/small/`:
	- `Q.f32.bin`, `K.f32.bin`, `V.f32.bin`, `O.f32.bin` (small case: d_model=32, num_heads=4).
- Run the test from the repo root:

```bash
python extensions/jax/tests/test_jax_bindings.py
```

What the test does

- Loads `Q,K,V` binaries, copies them to GPU via CuPy, calls the local `flash_solve_cupy` wrapper and compares the output to `O.f32.bin` (tolerance 1e-3).

Quick usage example

```python
import sys
sys.path.append('extensions/jax')            # from repo root
import pybind_wrapper as pbw
import cupy as cp

# q,k,v: CuPy arrays of shape (N, d_model), dtype=float32
q = cp.asarray(..., dtype=cp.float32)
k = cp.asarray(..., dtype=cp.float32)
v = cp.asarray(..., dtype=cp.float32)

out = pbw.flash_solve_cupy(q, k, v, d_model, num_heads)
# `out` is a CuPy array on GPU
```

Notes

- Make sure `cupy` matches your CUDA (install `cupy-cudaXX` for your CUDA).
- This wrapper is for eager/JAX-hosted usage only; for JIT integration implement an XLA custom-call.