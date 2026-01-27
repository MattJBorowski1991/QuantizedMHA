#!/usr/bin/env python3
"""Minimal example: call `torch_ext` MHA wrapper, warmup + timed loop.

This script attempts to call `torch_ext.solve(Q, K, V)` (preferred) or
falls back to pointer-style calls if the extension exposes a lower-level API.
"""
import time
import argparse
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=128)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--h', type=int, default=8)
    parser.add_argument('--warmups', type=int, default=10)
    parser.add_argument('--iters', type=int, default=50)
    parser.add_argument('--save', type=str, default='examples/torch_output.npy')
    args = parser.parse_args()

    try:
        import torch_ext
    except Exception as e:
        print('Failed to import torch_ext. Build the extension first:')
        print('  cd extensions/torch && python setup.py build_ext --inplace')
        raise

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        raise SystemExit('This example requires a CUDA GPU and PyTorch with CUDA.')

    N, d_model, h = args.N, args.d_model, args.h
    Q = torch.randn(N, d_model, device=device, dtype=torch.float32)
    K = torch.randn(N, d_model, device=device, dtype=torch.float32)
    V = torch.randn(N, d_model, device=device, dtype=torch.float32)

    # Try the high-level API first: torch_ext.solve(Q, K, V)
    def call_solve():
        try:
            return torch_ext.solve(Q, K, V)
        except TypeError:
            # Fallback: pointer-style (extension may expect raw pointers and ints)
            try:
                return torch_ext.solve(Q.data_ptr(), K.data_ptr(), V.data_ptr(), N, d_model, h)
            except Exception as e:
                raise RuntimeError('Unable to call torch_ext.solve with any known signature') from e

    # Warmup
    for _ in range(args.warmups):
        out = call_solve()
    torch.cuda.synchronize()

    # Timed loop
    t0 = time.time()
    for _ in range(args.iters):
        out = call_solve()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / args.iters

    ms = dt * 1000.0
    print(f'Avg latency: {ms:.3f} ms/iter over {args.iters} iters')

    # Save output if it's a tensor-like object
    if hasattr(out, 'cpu'):
        np_out = out.cpu().numpy()
        np.save(args.save, np_out)
        print('Saved example output to', args.save)


if __name__ == '__main__':
    main()
