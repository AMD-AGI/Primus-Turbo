###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.
###############################################################################

"""Numerical vmap coverage, including the variable-K weight-gradient path."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from primus_turbo.jax.lax.grouped_gemm import compute_group_offs, grouped_gemm
from primus_turbo.jax.primitive._batching import enable_x64
from tests.jax.test_utils import compute_snr


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("spmd", [False, True])
def test_offsets_vmap_preserves_int64(axis, spmd):
    lengths = jnp.array([[2, 0, 3], [0, 4, 1]], jnp.int32)
    fn = jax.vmap(compute_group_offs, in_axes=axis, spmd_axis_name="stage" if spmd else None)
    with enable_x64(False):
        out = fn(jnp.moveaxis(lengths, 0, axis))
    assert out.dtype == jnp.int64
    np.testing.assert_array_equal(out, [[0, 2, 2, 5], [0, 0, 4, 5]])


@pytest.mark.parametrize("trans_b", [False, True])
@pytest.mark.parametrize("a_axis,b_axis", [(0, 0), (1, None), (None, 0)])
def test_grouped_gemm_vmap_forward_backward(trans_b, a_axis, b_axis):
    key_a, key_b, key_ct = jax.random.split(jax.random.key(17), 3)
    # Unequal groups exercise offset reconstruction without relying on divisibility.
    a = (0.1 * jax.random.normal(key_a, (2, 192, 128), jnp.float32)).astype(jnp.bfloat16)
    b = (0.1 * jax.random.normal(key_b, (2, 2, 128, 128), jnp.float32)).astype(jnp.bfloat16)
    if trans_b:
        b = b.swapaxes(-1, -2)
    cotangent = (0.1 * jax.random.normal(key_ct, (2, 192, 128), jnp.float32)).astype(jnp.bfloat16)
    with enable_x64(True):
        lengths = jnp.array([64, 128], jnp.int64)

        def kernel(x, w):
            return grouped_gemm(x, w, lengths, transB=trans_b)

        def reference(x, w):
            w = w.swapaxes(-1, -2) if trans_b else w
            return jnp.concatenate((x[:64] @ w[0], x[64:] @ w[1]), axis=0)

        args = (
            a[0] if a_axis is None else jnp.moveaxis(a, 0, a_axis),
            b[0] if b_axis is None else jnp.moveaxis(b, 0, b_axis),
        )
        mapped = jax.vmap(kernel, in_axes=(a_axis, b_axis))
        ref_mapped = jax.vmap(reference, in_axes=(a_axis, b_axis))
        out, pullback = jax.vjp(mapped, *args)
        expected, ref_pullback = jax.vjp(ref_mapped, *args)
        gradients, reference_gradients = pullback(cotangent), ref_pullback(cotangent)
    np.testing.assert_allclose(
        np.asarray(out, dtype=np.float32), np.asarray(expected, dtype=np.float32), rtol=0.01, atol=0.01
    )
    assert compute_snr(expected, out) > 45
    for actual, wanted in zip(gradients, reference_gradients):
        assert compute_snr(wanted, actual) > 45
        np.testing.assert_allclose(
            np.asarray(actual, dtype=np.float32), np.asarray(wanted, dtype=np.float32), rtol=0.01, atol=0.01
        )
