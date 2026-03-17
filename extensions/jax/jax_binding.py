"""Python helper to call the compiled `jax_ext` module from JAX via DLPack + CuPy.

This module implements a thin conversion layer:
- JAX DeviceArray -> DLPack -> CuPy
- call compiled extension with CuPy device pointers
- CuPy -> DLPack -> JAX DeviceArray (returned)

This mirrors the quick-prototype pattern (fast to iterate, not XLA-fused).
"""
import os
import sys

import jax
import jax.dlpack as jdlpack
import cupy as cp

# Ensure compiled extension shared object is importable from this folder
_here = os.path.dirname(__file__)
if _here not in sys.path:
    sys.path.insert(0, _here)

import jax_ext


def flash_solve_jax(q, k, v, d_model, num_heads, kernel='fa_tc_int8_b'):
    """Call the CUDA `solve` entrypoint using JAX DeviceArrays.

    Args:
        q, k, v: JAX DeviceArray on GPU with shape [N, d_model], dtype float32
        d_model: model dimension (int)
        num_heads: number of heads (int)
        kernel: kernel name (string) — currently passed through but build-time selection applies

    Returns:
        JAX DeviceArray (on the same device) with the output.
    """
    # Convert JAX -> CuPy via DLPack (zero-copy where supported).
    # Be robust across JAX versions: prefer jax.dlpack.to_dlpack, else
    # fall back to array method `.to_dlpack()` or the DLPack protocol `__dlpack__()`.
    def _jax_to_dlpack(arr):
        if hasattr(jdlpack, 'to_dlpack'):
            return jdlpack.to_dlpack(arr)
        if hasattr(arr, 'to_dlpack'):
            return arr.to_dlpack()
        if hasattr(arr, '__dlpack__'):
            return arr.__dlpack__()
        raise RuntimeError('Cannot convert JAX array to DLPack; upgrade JAX or use a compatible version')

    # Use the modern cupy.from_dlpack API (lowercase) to avoid deprecation warnings
    q_cp = cp.from_dlpack(_jax_to_dlpack(q))
    k_cp = cp.from_dlpack(_jax_to_dlpack(k))
    v_cp = cp.from_dlpack(_jax_to_dlpack(v))

    # Ensure float32 and contiguous layout
    q_cp = cp.ascontiguousarray(q_cp.astype(cp.float32, copy=False))
    k_cp = cp.ascontiguousarray(k_cp.astype(cp.float32, copy=False))
    v_cp = cp.ascontiguousarray(v_cp.astype(cp.float32, copy=False))

    out_cp = cp.empty_like(q_cp)

    # Call compiled extension with device pointers
    jax_ext.flash_solve(
        q_cp.data.ptr,
        k_cp.data.ptr,
        v_cp.data.ptr,
        out_cp.data.ptr,
        int(q_cp.shape[0]),
        int(d_model),
        int(num_heads),
        kernel,
    )

    # Convert result back to JAX DeviceArray via DLPack.
    # Pass the CuPy array directly; CuPy implements the DLPack protocol
    # via `__dlpack__`/`__dlpack_device__`, which `jax.dlpack.from_dlpack`
    # accepts.
    return jdlpack.from_dlpack(out_cp)
