import os
import sys
import numpy as np
import torch

# ensure local extension folder is on path
here = os.path.dirname(__file__)
ext_dir = os.path.abspath(os.path.join(here, '..'))
sys.path.insert(0, ext_dir)
import torch_ext as flash_ext

# repo root
REPO_ROOT = os.path.abspath(os.path.join(here, '..', '..', '..'))
GOLDEN = os.path.join(REPO_ROOT, 'tests', 'golden', 'small')

def load_bin(path, dmodel):
    arr = np.fromfile(path, dtype=np.float32)
    assert arr.size % dmodel == 0
    return arr.reshape(-1, dmodel)

def test_flash_solve():
    d_model = 32
    num_heads = 4

    q = load_bin(os.path.join(GOLDEN, 'Q.f32.bin'), d_model)
    k = load_bin(os.path.join(GOLDEN, 'K.f32.bin'), d_model)
    v = load_bin(os.path.join(GOLDEN, 'V.f32.bin'), d_model)
    o_ref = load_bin(os.path.join(GOLDEN, 'O.f32.bin'), d_model)

    q_t = torch.from_numpy(q).cuda()
    k_t = torch.from_numpy(k).cuda()
    v_t = torch.from_numpy(v).cuda()

    out = flash_ext.flash_solve(q_t, k_t, v_t, d_model, num_heads)

    out_cpu = out.cpu().numpy().reshape(o_ref.shape)
    assert np.allclose(out_cpu, o_ref, atol=1e-3, rtol=1e-3), "Mismatch vs golden"

if __name__ == '__main__':
    test_flash_solve()
    print('ok')
