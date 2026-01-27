# torch_ext (PyTorch) — minimal native extension

Build

```bash
cd extensions/torch
python -m pip install --upgrade setuptools wheel
python setup.py build_ext --inplace
```

Test (what data and where)

- Golden fixtures: `tests/golden/small/` contains `Q.f32.bin`, `K.f32.bin`, `V.f32.bin`, `O.f32.bin` (small case: d_model=32, num_heads=4).
- Run the smoke test from the repo root:

```bash
python extensions/torch/tests/test_torch_bindings.py
```

What the test does

- Loads `Q,K,V` binaries, copies them to GPU via PyTorch tensors, calls the built `torch_ext.flash_solve` extension, and compares the output to `O.f32.bin` (tolerance 1e-3).

Usage (quick)

```python
import sys
sys.path.append('extensions/torch')   # from repo root
import torch_ext
import torch

# q,k,v: torch.cuda.FloatTensor with shape (N, d_model)
q = torch.empty((N, d_model), dtype=torch.float32).cuda()
k = torch.empty_like(q).cuda()
v = torch.empty_like(q).cuda()

out = torch_ext.flash_solve(q, k, v, d_model, num_heads)
# out is a CUDA tensor
```

Notes

- The extension module built here is `torch_ext` and links to the C ABI in `include/launchers.h`.
