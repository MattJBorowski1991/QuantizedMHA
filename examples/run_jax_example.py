#!/usr/bin/env python3
"""Minimal example: call `jax_ext`/CuPy wrapper, warmup + timed loop.

This script attempts to call `jax_ext.solve(Q, K, V)` (preferred) or
falls back to pointer-style calls if the extension exposes a lower-level API.
It uses CuPy arrays on the device.
"""
import time
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=128)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--h', type=int, default=8)
    parser.add_argument('--warmups', type=int, default=10)
    parser.add_argument('--iters', type=int, default=50)
    parser.add_argument('--save', type=str, default='examples/jax_output.npy')
    args = parser.parse_args()

    try:
        import cupy as cp
    except Exception:
        raise SystemExit('CuPy is required for this example (pip install cupy-cudaXX).')

    try:
        import jax_ext
    except Exception:
        print('Failed to import jax_ext. Build the extension first:')
        print('  cd extensions/jax && python setup.py build_ext --inplace')
        raise

    N, d_model, h = args.N, args.d_model, args.h
    Q = cp.random.randn(N, d_model).astype(cp.float32)
    K = cp.random.randn(N, d_model).astype(cp.float32)
    V = cp.random.randn(N, d_model).astype(cp.float32)

    def call_solve():
        try:
            return jax_ext.solve(Q, K, V)
        except TypeError:
            try:
                # fallback to pointer-based API
                return jax_ext.solve(int(Q.data.ptr), int(K.data.ptr), int(V.data.ptr), N, d_model, h)
            except Exception as e:
                raise RuntimeError('Unable to call jax_ext.solve with any known signature') from e

    # Warmup
    for _ in range(args.warmups):
        out = call_solve()

    # Timed loop
    t0 = time.time()
    for _ in range(args.iters):
        out = call_solve()
    cp.cuda.Stream.null.synchronize()
    dt = (time.time() - t0) / args.iters

    ms = dt * 1000.0
    print(f'Avg latency: {ms:.3f} ms/iter over {args.iters} iters')

    # Save output if it's array-like
    if hasattr(out, 'get'):
        np_out = cp.asnumpy(out)
        np.save(args.save, np_out)
        print('Saved example output to', args.save)


if __name__ == '__main__':
    main()
