###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Shared JAX batching rules for running Primus-Turbo primitives under
MaxText pipeline parallelism.

MaxText builds its pipeline-parallel layer stack by wrapping each layer in
``nn.vmap(spmd_axis_name="stage")`` composed with an outer ``shard_map``. For
our custom primitives to trace through that stack, every primitive must
register a batching rule in BOTH:

  * ``batching.primitive_batchers``        - plain ``jax.vmap`` (no spmd axis);
  * ``batching.fancy_primitive_batchers``  - the ``axis_data``-aware variant,
    which JAX *requires* whenever ``spmd_axis_name`` is set (it raises
    otherwise).

The SPMD-aware path is also a memory win. Under the vmap-of-shard_map
composition JAX's ``_shard_map_batch`` divides ``axis_data.size`` by the mesh
stage dim (typically landing at 1 per device), so we can dynamic-slice the
single local stage via ``axis_index(spmd_name)`` and broadcast the result back.
XLA folds that dynamic-index + broadcast pair into metadata in shard_map manual
mode, producing a tighter HLO than the scan/reshape-of-batch-1 fallback
(empirically ~30 GB less HBM temp on DS3-671B sgd-pp8 pdbs=8).

When there is no ``spmd_axis_name`` the fancy rule degrades to the plain rule.
"""

import jax
import jax.numpy as jnp

__all__ = [
    "broadcast_to_batch",
    "spmd_name_or_none",
    "make_ipc_batchers",
    "make_grouped_gemm_spmd_rule",
]


def broadcast_to_batch(arg, ax, batch_size):
    """Move ``ax`` of ``arg`` to position 0, or broadcast a new leading dim of
    ``batch_size`` when ``ax`` is None (the arg is not batched)."""
    if ax is None:
        return jnp.broadcast_to(jnp.expand_dims(arg, 0), (batch_size,) + arg.shape)
    if ax != 0:
        arg = jnp.moveaxis(arg, ax, 0)
    return arg


def spmd_name_or_none(axis_data):
    """Return the (single) ``spmd_axis_name`` as a str, or None when batching
    without one."""
    spmd = getattr(axis_data, "spmd_name", None)
    if isinstance(spmd, (list, tuple)):
        return spmd[0] if spmd else None
    return spmd


def make_ipc_batchers(prim, num_outputs):
    """Build ``(plain, fancy)`` batchers for a cross-rank IPC primitive.

    IPC primitives (dispatch / combine) carry per-rank state that cannot be
    merged across the batch axis, so the plain rule scans the primitive over the
    batch one microbatch at a time. The fancy rule takes the SPMD dynamic-slice
    fast path described in the module docstring, falling back to the scan when
    there is no ``spmd_axis_name``.
    """
    none_dims = (None,) * num_outputs
    batched_dims = (0,) * num_outputs

    def plain_rule(args, axes, **kwargs):
        if all(ax is None for ax in axes):
            return prim.bind(*args, **kwargs), none_dims
        bs = next(a.shape[ax] for a, ax in zip(args, axes) if ax is not None)
        args_b = [broadcast_to_batch(a, ax, bs) for a, ax in zip(args, axes)]

        def body(_, slice_args):
            return None, prim.bind(*slice_args, **kwargs)

        _, outputs = jax.lax.scan(body, None, tuple(args_b))
        return outputs, batched_dims

    def fancy_rule(axis_data, args, axes, **kwargs):
        spmd_name = spmd_name_or_none(axis_data)
        if spmd_name is None:
            return plain_rule(args, axes, **kwargs)
        if all(ax is None for ax in axes):
            return prim.bind(*args, **kwargs), none_dims
        bs = axis_data.size
        args_b = [broadcast_to_batch(a, ax, bs) for a, ax in zip(args, axes)]
        local_idx = jax.lax.axis_index(spmd_name)
        args_local = [jax.lax.dynamic_index_in_dim(a, local_idx, axis=0, keepdims=False) for a in args_b]
        out_local = prim.bind(*args_local, **kwargs)
        stacked = tuple(jnp.broadcast_to(o[None], (bs,) + o.shape) for o in out_local)
        return stacked, batched_dims

    return plain_rule, fancy_rule


def make_grouped_gemm_spmd_rule(prim, plain_rule, int64_arg_indices=(2, 3)):
    """Build the fancy batcher for a grouped-GEMM primitive (4 inputs ``a, b,
    group_lens, group_offs``; 2 outputs ``out, workspace``).

    Falls back to ``plain_rule`` (the per-primitive reshape-merge rule) when
    there is no ``spmd_axis_name``. On the SPMD path it dynamic-slices the local
    stage and re-casts the int64-required args (the C++ kernel hard-requires
    int64 for ``group_lens`` / ``group_offs``) inside ``enable_x64`` so the cast
    survives a non-x64 trace context.
    """

    def fancy_rule(axis_data, args, axes, **kwargs):
        spmd_name = spmd_name_or_none(axis_data)
        if spmd_name is None:
            return plain_rule(args, axes, **kwargs)
        if all(ax is None for ax in axes):
            return prim.bind(*args, **kwargs), (None, None)
        bs = axis_data.size
        args_b = [broadcast_to_batch(a, ax, bs) for a, ax in zip(args, axes)]
        local_idx = jax.lax.axis_index(spmd_name)
        args_local = [jax.lax.dynamic_index_in_dim(a, local_idx, axis=0, keepdims=False) for a in args_b]
        with jax.experimental.enable_x64():
            for i in int64_arg_indices:
                args_local[i] = args_local[i].astype(jnp.int64)
        out_local, ws_local = prim.bind(*args_local, **kwargs)
        out = jnp.broadcast_to(out_local[None], (bs,) + out_local.shape)
        ws = jnp.broadcast_to(ws_local[None], (bs,) + ws_local.shape)
        return (out, ws), (0, 0)

    return fancy_rule
