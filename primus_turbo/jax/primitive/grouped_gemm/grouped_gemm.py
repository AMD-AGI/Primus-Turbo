###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from functools import partial

import jax
import jax.numpy as jnp
from jax.core import ShapedArray
from jax.extend.core import Primitive
from jax.interpreters import xla

from primus_turbo.jax._C import get_ck_grouped_gemm_workspace_size
from primus_turbo.jax.primitive import ABSTRACT_EVAL_TABLE, IMPL_TABLE, LOWERING_TABLE

# ----------------------------------------
# Step-1: Primitive Define
# ----------------------------------------
ck_grouped_gemm_p = Primitive("ck_grouped_gemm")
ck_grouped_gemm_p.multiple_results = True

ck_grouped_gemm_variable_k_p = Primitive("ck_grouped_gemm_variable_k")
ck_grouped_gemm_variable_k_p.multiple_results = True

compute_group_offs_p = Primitive("compute_group_offs")
compute_group_offs_p.multiple_results = False


# ----------------------------------------
# Step-2: Impl
# ----------------------------------------
IMPL_TABLE[ck_grouped_gemm_p] = partial(xla.apply_primitive, ck_grouped_gemm_p)
IMPL_TABLE[ck_grouped_gemm_variable_k_p] = partial(xla.apply_primitive, ck_grouped_gemm_variable_k_p)
IMPL_TABLE[compute_group_offs_p] = partial(xla.apply_primitive, compute_group_offs_p)


# ----------------------------------------
# Step-3: Abstract eval
# ----------------------------------------
def _grouped_gemm_abstract_eval(a, b, group_lens, group_offs, transA, transB, num_cu):
    assert a.dtype == b.dtype, "dtype mismatch between a and b"

    m = a.shape[1] if transA else a.shape[0]
    n = b.shape[1] if transB else b.shape[2]

    group_num = group_lens.shape[0]
    ws_size = get_ck_grouped_gemm_workspace_size(group_num)

    out_aval = ShapedArray((m, n), a.dtype)
    ws_aval = ShapedArray((ws_size,), jnp.uint8)

    return (out_aval, ws_aval)


ABSTRACT_EVAL_TABLE[ck_grouped_gemm_p] = _grouped_gemm_abstract_eval


def _compute_group_offs_abstract_eval(group_lens):
    bs = group_lens.shape[0]
    return ShapedArray((bs + 1,), group_lens.dtype)


ABSTRACT_EVAL_TABLE[compute_group_offs_p] = _compute_group_offs_abstract_eval


def _grouped_gemm_variable_k_abstract_eval(a, b, group_lens, group_offs, transA, transB, num_cu):
    assert a.dtype == b.dtype, "dtype mismatch between a and b"
    assert transA == True and transB == False, "Only transA=True, transB=False supported"

    bs = group_lens.shape[0]
    m = a.shape[1] if transA else a.shape[0]
    n = b.shape[0] if transB else b.shape[1]

    ws_size = get_ck_grouped_gemm_workspace_size(bs)

    out_aval = ShapedArray((bs, m, n), a.dtype)
    ws_aval = ShapedArray((ws_size,), jnp.uint8)

    return (out_aval, ws_aval)


ABSTRACT_EVAL_TABLE[ck_grouped_gemm_variable_k_p] = _grouped_gemm_variable_k_abstract_eval


# ----------------------------------------
# Step-4: JIT Lowering
# ----------------------------------------
LOWERING_TABLE[ck_grouped_gemm_p] = jax.ffi.ffi_lowering("ck_grouped_gemm")

LOWERING_TABLE[ck_grouped_gemm_variable_k_p] = jax.ffi.ffi_lowering("ck_grouped_gemm_variable_k")

LOWERING_TABLE[compute_group_offs_p] = jax.ffi.ffi_lowering("compute_group_offs")


# ----------------------------------------
# Step-5: batching
# ----------------------------------------
# Grouped-GEMM batching for pipeline parallelism.  The plain-vmap rule merges the
# batch into the leading (token / group / K) dim so one kernel call covers the
# whole batch; the SPMD-aware rule dynamic-slices the local stage instead.  See
# primus_turbo.jax.primitive._batching for the shared helpers and rationale.
from jax.interpreters import batching

from primus_turbo.jax.primitive._batching import (
    broadcast_to_batch,
    make_grouped_gemm_spmd_rule,
)

# ---- compute_group_offs ------------------------------------------------------


def _compute_group_offs_batch_rule(args, axes):
    """Batched `cumsum` (prepended with 0).  Computed in pure JAX rather than by
    scanning the primitive, since it is a cheap, shape-stable index computation.

    The C++ kernel hard-requires int64 for `group_offs`, so the cast is wrapped in
    `enable_x64()` to survive a non-x64 trace context.
    """
    (group_lens,) = args
    (axis,) = axes
    if axis is None:
        return compute_group_offs_p.bind(group_lens), None
    if axis != 0:
        group_lens = jnp.moveaxis(group_lens, axis, 0)
    with jax.experimental.enable_x64():
        group_lens = group_lens.astype(jnp.int64)
        last_axis = group_lens.ndim - 1
        zeros = jnp.zeros(group_lens.shape[:-1] + (1,), dtype=jnp.int64)
        cum = jax.lax.cumsum(group_lens, axis=last_axis)
        out = jnp.concatenate([zeros, cum], axis=last_axis).astype(jnp.int64)
    return out, 0


batching.primitive_batchers[compute_group_offs_p] = _compute_group_offs_batch_rule
# cumsum is cheap and shape-stable, so the SPMD path needs no dynamic-slice; the
# fancy batcher (required once spmd_axis_name is set) just reuses the plain rule.
batching.fancy_primitive_batchers[compute_group_offs_p] = (
    lambda axis_data, args, axes: _compute_group_offs_batch_rule(args, axes)
)


# ---- ck_grouped_gemm ---------------------------------------------------------


def _ck_grouped_gemm_merge_rule(args, axes, **kwargs):
    """Plain-vmap rule: merge the batch into the leading token dim, run one
    grouped-GEMM, then un-merge.  `group_offs` is recomputed from the merged
    `group_lens` (int64, per the kernel contract)."""
    a, b, group_lens, group_offs = args
    a_ax, b_ax, gl_ax, go_ax = axes
    if all(ax is None for ax in axes):
        return ck_grouped_gemm_p.bind(*args, **kwargs), (None, None)

    bs = next(arg.shape[ax] for arg, ax in zip(args, axes) if ax is not None)
    a_b = broadcast_to_batch(a, a_ax, bs)  # (B, m_total, k)
    b_b = broadcast_to_batch(b, b_ax, bs)  # (B, G, k, n)
    gl_b = broadcast_to_batch(group_lens, gl_ax, bs)  # (B, G)

    a_merged = a_b.reshape((-1,) + a_b.shape[2:])  # (B*m_total, k)
    b_merged = b_b.reshape((-1,) + b_b.shape[2:])  # (B*G, k, n)
    with jax.experimental.enable_x64():
        gl_merged = gl_b.reshape(-1).astype(jnp.int64)
        zeros = jnp.zeros((1,), dtype=jnp.int64)
        go_merged = jnp.concatenate([zeros, jax.lax.cumsum(gl_merged, axis=0)], axis=0).astype(jnp.int64)

    out_merged, ws = ck_grouped_gemm_p.bind(a_merged, b_merged, gl_merged, go_merged, **kwargs)
    out = out_merged.reshape(a_b.shape[:2] + (out_merged.shape[-1],))
    return (out, ws), (0, None)


batching.primitive_batchers[ck_grouped_gemm_p] = _ck_grouped_gemm_merge_rule
batching.fancy_primitive_batchers[ck_grouped_gemm_p] = make_grouped_gemm_spmd_rule(
    ck_grouped_gemm_p, _ck_grouped_gemm_merge_rule
)


# ---- ck_grouped_gemm_variable_k ---------------------------------------------


def _ck_grouped_gemm_variable_k_merge_rule(args, axes, **kwargs):
    """Plain-vmap rule for the variable-K kernel (groups vary along the K axis;
    output is (G, m, n) per call).  Merge the batch into the leading K dim, run
    once, then un-merge."""
    a, b, group_lens, group_offs = args
    a_ax, b_ax, gl_ax, go_ax = axes
    if all(ax is None for ax in axes):
        return ck_grouped_gemm_variable_k_p.bind(*args, **kwargs), (None, None)

    bs = next(arg.shape[ax] for arg, ax in zip(args, axes) if ax is not None)
    a_b = broadcast_to_batch(a, a_ax, bs)  # (B, k_total, m)
    b_b = broadcast_to_batch(b, b_ax, bs)  # (B, k_total, n)
    gl_b = broadcast_to_batch(group_lens, gl_ax, bs)  # (B, G)

    a_merged = a_b.reshape((-1,) + a_b.shape[2:])  # (B*k_total, m)
    b_merged = b_b.reshape((-1,) + b_b.shape[2:])  # (B*k_total, n)
    with jax.experimental.enable_x64():
        gl_merged = gl_b.reshape(-1).astype(jnp.int64)
        zeros = jnp.zeros((1,), dtype=jnp.int64)
        go_merged = jnp.concatenate([zeros, jax.lax.cumsum(gl_merged, axis=0)], axis=0).astype(jnp.int64)

    out_merged, ws = ck_grouped_gemm_variable_k_p.bind(a_merged, b_merged, gl_merged, go_merged, **kwargs)
    out = out_merged.reshape((bs, -1) + out_merged.shape[1:])
    return (out, ws), (0, None)


batching.primitive_batchers[ck_grouped_gemm_variable_k_p] = _ck_grouped_gemm_variable_k_merge_rule
batching.fancy_primitive_batchers[ck_grouped_gemm_variable_k_p] = make_grouped_gemm_spmd_rule(
    ck_grouped_gemm_variable_k_p, _ck_grouped_gemm_variable_k_merge_rule
)


__all__ = [
    "ck_grouped_gemm_p",
    "ck_grouped_gemm_variable_k_p",
    "compute_group_offs_p",
]
