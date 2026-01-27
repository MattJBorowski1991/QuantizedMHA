# Examples

Minimal runnable examples demonstrating how to call the native extensions and perform warmup + timed loops.

Scripts:

- `run_torch_example.py` — uses `torch_ext`; creates random CUDA tensors, runs warmup and timed iterations, prints ms/iter, and saves the output.
- `run_jax_example.py` — uses `jax_ext` (CuPy wrapper); creates CuPy device arrays, runs warmup and timed iterations, prints ms/iter, and saves the output.

Usage (after building extensions):

```bash
python examples/run_torch_example.py --N 128 --d_model 512 --h 8
python examples/run_jax_example.py  --N 128 --d_model 512 --h 8
```

Notes:
- If the examples fail to import the extension, build it first under `extensions/torch` or `extensions/jax`.
- The example scripts try high-level `solve(Q,K,V)` calls first and fall back to pointer-based signatures if needed.
