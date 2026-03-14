#!/usr/bin/env python3
"""Minimal example: call `torch_ext` MHA wrapper with kernel selection, warmup + timed loop.
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
    parser.add_argument('--kernel', type=str, default='fa_tc_int8_b', 
                       help='Kernel variant to use')
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

    # Validate tensor shapes
    assert Q.shape == (N, d_model), f"Q shape {Q.shape} != ({N}, {d_model})"
    assert K.shape == (N, d_model), f"K shape {K.shape} != ({N}, {d_model})"
    assert V.shape == (N, d_model), f"V shape {V.shape} != ({N}, {d_model})"
    
    def call_solve():
        return torch_ext.flash_solve(Q, K, V, d_model, h, kernel=args.kernel)

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
    print(f'[{args.kernel}] Avg latency: {ms:.3f} ms/iter over {args.iters} iters')

    # Save output if it's a tensor-like object
    if hasattr(out, 'cpu'):
        np_out = out.cpu().numpy()
        np.save(args.save, np_out)
        print('Saved example output to', args.save)


if __name__ == '__main__':
    main()
