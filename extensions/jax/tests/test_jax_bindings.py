import os
import sys

import jax
import jax.numpy as jnp

here = os.path.dirname(__file__)
ext_dir = os.path.abspath(os.path.join(here, '..'))
sys.path.insert(0, ext_dir)
from jax_binding import flash_solve_jax


def test_flash_solve():
    d_model = 32
    num_heads = 4
    N = 256
    kernel = 'fa_tc_int8_b'

    # Require a GPU device for this smoke test
    gpus = [d for d in jax.devices() if d.platform == 'gpu']
    if not gpus:
        print('No GPU device found; skipping test.')
        return
    dev = gpus[0]

    key = jax.random.PRNGKey(0)
    q = jax.device_put(jax.random.normal(key, (N, d_model), dtype=jnp.float32), device=dev)
    k = jax.device_put(jax.random.normal(key, (N, d_model), dtype=jnp.float32), device=dev)
    v = jax.device_put(jax.random.normal(key, (N, d_model), dtype=jnp.float32), device=dev)

    out = flash_solve_jax(q, k, v, d_model, num_heads, kernel=kernel)
    assert out.shape == (N, d_model)
    assert out.dtype == jnp.float32
    print('ok')


if __name__ == '__main__':
    test_flash_solve()
