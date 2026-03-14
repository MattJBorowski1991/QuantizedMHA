# torch_ext (PyTorch) — Flash Attention int8 native extension

## Build

```bash
cd /teamspace/studios/this_studio/QuantizedMHA/extensions/torch
python -m pip install --upgrade setuptools wheel

# Build all kernels (default)
python setup.py build_ext --inplace

# Build only a specific kernel
KERNEL=fa_tc_int8_b python setup.py build_ext --inplace

# Available kernels: fa, fa_tc_int8_a, fa_tc_int8_b, fa_tc_v1a, fa_tc_v1b, fa_tc_v2, fa_tc_v2a, fa_tc_v2b, unfused
```

## Test

**Prerequisite:** Build the extension first with `python setup.py build_ext --inplace` (see Build section above).

Run the smoke test from the repo root:

```bash
cd /teamspace/studios/this_studio/QuantizedMHA
python extensions/torch/tests/test_torch_bindings.py
```

This generates random input tensors (N=256, d_model=32, num_heads=4), calls the kernel, and validates output shape/dtype.

## Run Example

```bash
cd /teamspace/studios/this_studio/QuantizedMHA
# Default kernel (fa_tc_int8_b)
python extensions/torch/run_torch_example.py --N 1024 --d_model 1024 --h 32 --warmups 5 --iters 10

# Specify kernel variant
python extensions/torch/run_torch_example.py --N 1024 --d_model 1024 --h 32 --kernel fa_tc_int8_b
```

## Usage (minimal example)

**Prerequisite:** Build the extension first (see Build section above). The compiled module will be placed in this directory.

```python
import torch
import sys
sys.path.insert(0, 'extensions/torch')  # Add this directory to Python path
import torch_ext  # Now the compiled .so module is available

N, d_model, num_heads = 1024, 1024, 32
q = torch.randn(N, d_model, dtype=torch.float32, device='cuda')
k = torch.randn(N, d_model, dtype=torch.float32, device='cuda')
v = torch.randn(N, d_model, dtype=torch.float32, device='cuda')

# Call with default kernel
out = torch_ext.flash_solve(q, k, v, d_model, num_heads)

# Call with specific kernel
out = torch_ext.flash_solve(q, k, v, d_model, num_heads, kernel='fa_tc_int8_b')
# out: shape (N, d_model), CUDA tensor, float32
```

## Notes

- **Extension module:** `torch_ext` (compiled to `torch_ext.cpython-*.so`)
- **Backend kernel:** Currently hardwired to `fa_tc_int8_b`; the `kernel` parameter is accepted but all builds use this kernel
  - Future versions will support kernel selection at runtime via different build configurations
- **Environment variable:** `KERNEL=<name>` during build selects which kernel sources to compile (only `fa_tc_int8_b` is actively maintained)
- **Available source kernels:** `fa_tc_int8_b`, `fa_tc_int8_a`, `fa`, `unfused`, `fa_tc_v1a`, `fa_tc_v1b`, `fa_tc_v2`, `fa_tc_v2a`, `fa_tc_v2b`
- **Entry point:** `include/launchers.h` → `extern "C" void solve()` function (C linkage for CUDA kernels)
- **Requirements:** CUDA GPU and PyTorch built with CUDA support

### Implementation notes

- `torch_ext.cpp`: PyTorch C++ extension wrapper + pybind11 bindings
- `setup.py`: Build configuration with environment variable support for kernel selection
- `include/launchers.h`: Contains `solve()` forward declaration and generic launcher template
- `mha_kernels/*.cu`: Individual kernel implementations with `extern "C" void solve()` entry point
