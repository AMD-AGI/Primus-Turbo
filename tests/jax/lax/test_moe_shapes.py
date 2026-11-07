###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.
###############################################################################

"""Trace-only MoE shape contracts; these do not validate RDMA execution."""

import jax
import jax.numpy as jnp
import pytest

from primus_turbo.jax.core.low_precision import float8_e4m3
from primus_turbo.jax.lax.moe import moe_dispatch_combine as mdc


@pytest.mark.parametrize("ep_size", [2, 16])
@pytest.mark.parametrize("scale_shape", [None, (5,), (5, 2)])
def test_dispatch_cached_combine_shapes(monkeypatch, ep_size, scale_shape):
    internode = ep_size > 8
    monkeypatch.setattr(
        mdc,
        "_frozen",
        mdc.FrozenRuntimeState(mdc.MODE_PER_PROCESS, 1, ep_size, internode, 32, 80 if internode else 0),
    )
    config = mdc.Config(32, 8, 256, 8, 128)
    shape = jax.ShapeDtypeStruct
    x = shape((5, 256), jnp.bfloat16 if scale_shape is None else float8_e4m3)
    payload = x if scale_shape is None else (x, shape(scale_shape, jnp.float32))
    idx, weights = shape((5, 2), jnp.int32), shape((5, 2), jnp.float32)
    recv, recv_idx, recv_weights, handle = jax.eval_shape(
        lambda x, i, w: mdc.moe_dispatch(x, i, w, ep_size * 4, config=config), payload, idx, weights
    )
    main = recv if scale_shape is None else recv[0]
    assert main.shape == (5 * ep_size, 256)
    assert main.dtype == x.dtype
    assert recv_idx.shape == recv_weights.shape == (5 * ep_size, 2)
    assert recv_idx.dtype == jnp.int32 and recv_weights.dtype == jnp.float32
    if scale_shape is not None:
        assert recv[1].shape == (5 * ep_size,) + scale_shape[1:]
    assert len(handle) == (10 if internode else 6)
    if internode:
        assert handle[7].shape == (5 * ep_size, 80)
        assert handle[7].dtype == jnp.uint8
        assert handle[8].shape == (5, 2)
        assert handle[9].shape == (5 * ep_size, 8)
    else:
        assert handle[0].shape == (2, 2)
        assert handle[1].shape == (2, 16)
        assert handle[4].shape == (5, 2)
    # The cached dispatch used by combine's VJP must preserve the receive layout.
    cached, cached_idx, cached_weights, cached_handle = jax.eval_shape(
        lambda x, h: mdc._moe_dispatch_impl(x, handle=h, config=config), payload, handle
    )
    assert jax.tree.structure(cached) == jax.tree.structure(recv)
    assert jax.tree.leaves(cached) == jax.tree.leaves(recv)
    assert cached_idx is cached_weights is cached_handle is None
    # Combine accepts BF16 expert outputs even when dispatch used FP8 transport.
    expert_output = shape(main.shape, jnp.bfloat16)
    combined = jax.eval_shape(lambda x, h: mdc.moe_combine(x, h, config=config), expert_output, handle)
    assert combined.shape == (5, 256) and combined.dtype == jnp.bfloat16
    combined_x, combined_weights = jax.eval_shape(
        lambda x, h, w: mdc._moe_combine_impl(x, h, topk_weights=w, config=config),
        expert_output,
        handle,
        recv_weights,
    )
    assert combined_x == combined
    assert combined_weights.shape == (5, 2)


@pytest.mark.parametrize("bias_kind", ["single", "pair"])
def test_intranode_combine_bias_shape(monkeypatch, bias_kind):
    monkeypatch.setattr(mdc, "_frozen", mdc.FrozenRuntimeState(mdc.MODE_PER_PROCESS, 1, 2, False, 32, 0))
    shape = jax.ShapeDtypeStruct
    config = mdc.Config(32, 8, 256, 8, 128)
    x = shape((10, 256), jnp.bfloat16)
    h = (
        shape((2, 2), jnp.int32),
        shape((2, 16), jnp.int32),
        shape((2, 16), jnp.int32),
        shape((10,), jnp.int32),
        shape((5, 2), jnp.bool_),
        shape((5, 2), jnp.int32),
    )

    def call(x, handle):
        bias = jnp.zeros((256,), jnp.bfloat16)
        return mdc._moe_combine_impl(
            x, handle, bias=bias if bias_kind == "single" else (bias, bias), config=config
        )

    out, weights = jax.eval_shape(call, x, h)
    assert out.shape == (5, 256) and weights.shape == (0,)


@pytest.mark.parametrize("ep_size", [2, 16])
def test_dispatch_combine_vjp_shape(monkeypatch, ep_size):
    monkeypatch.setattr(
        mdc,
        "_frozen",
        mdc.FrozenRuntimeState(mdc.MODE_PER_PROCESS, 1, ep_size, ep_size > 8, 32, 80 if ep_size > 8 else 0),
    )
    config = mdc.Config(32, 8, 256, 8, 128)

    def loss(x, weights):
        idx = jnp.zeros((5, 2), jnp.int32)
        recv, _, recv_weights, handle = mdc.moe_dispatch(x, idx, weights, ep_size * 4, config=config)
        combined = mdc.moe_combine(recv, handle, config=config)
        return combined.astype(jnp.float32).sum() + recv_weights.sum()

    x = jax.ShapeDtypeStruct((5, 256), jnp.bfloat16)
    weights = jax.ShapeDtypeStruct((5, 2), jnp.float32)
    dx, dw = jax.eval_shape(jax.grad(loss, argnums=(0, 1)), x, weights)
    assert dx == x and dw == weights
    # Lowering is deliberately not compiled/executed: no fictitious peer buffers.
    ir = str(jax.jit(jax.grad(loss, argnums=(0, 1))).lower(x, weights).compiler_ir())
    prefix = "moe_internode_" if ep_size > 8 else "moe_"
    for op in ("dispatch", "cached_dispatch", "combine"):
        assert f"{prefix}{op}_per_process" in ir
    assert "has_side_effect = true" in ir
    if ep_size > 8:
        assert "source_meta_bytes" not in ir
