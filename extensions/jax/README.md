````markdown
# jax_ext (JAX) — Quick DLPack/CuPy binding for QuantizedMHA

## Purpose

This folder contains a fast prototype that exposes the existing CUDA `solve()`
entrypoint to JAX via DLPack + CuPy. It is intended for rapid iteration and
profiling — it is NOT an XLA custom-call and therefore won't be fused by JAX.

## Build

Prerequisites: CUDA, a working NVCC, and PyTorch (the build uses
`torch.utils.cpp_extension.CUDAExtension` to compile CUDA sources).

```bash
# jax_ext (JAX) — Quick DLPack/CuPy binding for QuantizedMHA

## Purpose

This folder contains a fast prototype that exposes the existing CUDA `solve()`
entrypoint to JAX via DLPack + CuPy. It is intended for rapid iteration and
profiling — it is NOT an XLA custom-call and therefore won't be fused by JAX.

## Prerequisites

- CUDA toolkit and `nvcc` available on PATH
- PyTorch installed (required by the build helper used here)
- JAX with GPU support (install per the official JAX installation instructions)
- CuPy built for your CUDA version (e.g. `cupy-cuda11x` where `11x` matches your CUDA)
- Build tools: `pip`, `setuptools`, `wheel`. Optionally `pybind11` for headers.

Example (adjust for your CUDA / platform):
```bash
python -m pip install --upgrade pip setuptools wheel
# Install PyTorch (needed by the build helper) using your preferred method
# Install CuPy (choose package matching your CUDA version, example):
pip install cupy-cuda11x
# Install JAX (follow the official guide for the correct GPU wheel):
# see https://github.com/google/jax#installation
```

## Build

Build the extension in-place from this directory. Example (builds the `fa_tc_int8_b` kernel):
```bash
cd extensions/jax
KERNEL=fa_tc_int8_b python setup.py build_ext --inplace
```

The compiled module `jax_ext` (.so) will be copied into this directory by the
custom build extension.

## Runtime note (LD_LIBRARY_PATH)

If importing the compiled module fails at runtime due to missing shared libraries
(for example `libc10.so`), add PyTorch's library directory to `LD_LIBRARY_PATH`:

```bash
export LD_LIBRARY_PATH=$(python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"):$LD_LIBRARY_PATH
```

## Usage

```python
import sys
sys.path.insert(0, 'extensions/jax')
from jax_binding import flash_solve_jax

# q,k,v are JAX DeviceArrays on GPU (dtype float32)
out = flash_solve_jax(q, k, v, d_model, num_heads, kernel='fa_tc_int8_b')
```

## Test

Run the smoke test (requires GPU + JAX + CuPy installed):

```bash
cd extensions/jax
python tests/test_jax_bindings.py
```

## Notes

- This path uses DLPack → CuPy conversions. Keep data resident on device to
  avoid host/device copies.
- Gradients are not automatically wired. For training you must add a
  `jax.custom_vjp` wrapper on the Python side.
- For production/JIT fusion, implement an XLA custom-call (`jaxlib`) instead.
