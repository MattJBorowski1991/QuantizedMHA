import os
import sys
import torch

# ensure local extension folder is on path
here = os.path.dirname(__file__)
ext_dir = os.path.abspath(os.path.join(here, '..'))
sys.path.insert(0, ext_dir)
import torch_ext as flash_ext

def test_flash_solve():
    """Test with random tensors (no golden data required)"""
    d_model = 32
    num_heads = 4
    N = 256
    kernel = 'fa_tc_int8_b'
    
    torch.manual_seed(42)
    q = torch.randn(N, d_model, dtype=torch.float32, device='cuda')
    k = torch.randn(N, d_model, dtype=torch.float32, device='cuda')
    v = torch.randn(N, d_model, dtype=torch.float32, device='cuda')

    # Call the extension with kernel parameter
    try:
        out = flash_ext.flash_solve(q, k, v, d_model, num_heads, kernel=kernel)
        assert out.shape == (N, d_model), f"Output shape {out.shape} != ({N}, {d_model})"
        assert out.dtype == torch.float32, f"Output dtype {out.dtype} != float32"
        assert out.is_cuda, "Output not on CUDA"
        print(f"[{kernel}] Output stats: min={out.min():.6f}, max={out.max():.6f}, mean={out.mean():.6f}")
    except Exception as e:
        raise RuntimeError(f"flash_solve call failed: {e}") from e

if __name__ == '__main__':
    test_flash_solve()
    print('ok')
