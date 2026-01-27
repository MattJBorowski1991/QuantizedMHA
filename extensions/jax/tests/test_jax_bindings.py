import os
import sys
import numpy as np
import cupy as cp

# ensure the built extension and wrapper are importable
here = os.path.dirname(__file__)
ext_dir = os.path.abspath(os.path.join(here, '..'))
sys.path.insert(0, ext_dir)
import pybind_wrapper as pbw

# repo root (three levels up from this test file)
REPO_ROOT = os.path.abspath(os.path.join(here, '..', '..', '..'))
GOLDEN = os.path.join(REPO_ROOT, 'tests', 'golden', 'small')

def load_bin(path, dmodel):
    arr = np.fromfile(path, dtype=np.float32)
    assert arr.size % dmodel == 0
    return arr.reshape(-1, dmodel)

def test_jax_eager():
    d_model = 32
    num_heads = 4

    q = load_bin(os.path.join(GOLDEN, 'Q.f32.bin'), d_model)
    k = load_bin(os.path.join(GOLDEN, 'K.f32.bin'), d_model)
    v = load_bin(os.path.join(GOLDEN, 'V.f32.bin'), d_model)
    o_ref = load_bin(os.path.join(GOLDEN, 'O.f32.bin'), d_model)

    q_c = cp.asarray(q)
    k_c = cp.asarray(k)
    v_c = cp.asarray(v)

    out = pbw.flash_solve_cupy(q_c, k_c, v_c, d_model, num_heads)

    out_cpu = cp.asnumpy(out).reshape(o_ref.shape)
    assert np.allclose(out_cpu, o_ref, atol=1e-3, rtol=1e-3)

if __name__ == '__main__':
    test_jax_eager()
    print('ok')
