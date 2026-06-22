###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Correctness + torch.compile + multi-backend dispatch test for the
``primus_turbo::mega_moe_prologue_impl`` custom op (world=1 self-loopback).

The op is the ``torch.compile``-traceable / autotune-dispatchable wrapper around
the FlyDSL prologue kernel. We drive it over the ``MegaMoePrologue`` workspace
buffers and validate the resulting dispatch tables exactly against the same torch
reference used by ``test_mega_moe_prolugue.py``, then re-run under ``torch.compile``
and through the explicit-autotune path.

Run inside dev_primus:
  PYTHONPATH=<...>/Primus-Turbo python -m pytest -s \
      tests/pytorch/ops/test_mega_moe_prologue_impl.py
"""

import os
import sys

import pytest
import torch

# reuse the workspace + torch reference checker from the FlyDSL kernel test
sys.path.insert(0, os.path.dirname(__file__))

try:
    from test_mega_moe_prolugue import _check_correctness, _make_routing, _reference

    import primus_turbo.pytorch.kernels.mega_moe.mega_moe_prologue_impl  # noqa: F401  (registers op)
    from primus_turbo.flydsl.mega.mega_moe_prologue import MegaMoePrologue
    from primus_turbo.pytorch.core.backend import BackendType

    _HAVE_FLYDSL = True
except Exception as _e:  # flydsl only present inside the dev container
    _HAVE_FLYDSL = False
    _IMPORT_ERR = _e


# tensor kwargs the custom op consumes (== workspace _kernel_args, minus topk pair)
_OP_BUFFER_KEYS = (
    "send_local",
    "within_expert_counter",
    "c_buffer_ptrs",
    "signal_ptrs",
    "origin_rank_ptrs",
    "origin_slot_ptrs",
    "start_per_expert",
    "source_offset_per_expert",
    "pool_base",
    "destination",
    "start",
    "count",
    "source_offset_out",
    "tile_to_group",
    "expected",
    "source_tokens",
    "source_topk_slot",
    "source_weight",
    "zero_topk_weights",
    "origin_rank",
    "origin_slot",
    "meta_scalars",
    "grid_barrier_state",
    "token_rank_table",
    "dedup_src_row_ptrs",
    "dedup_src_row",
    "source_dedup",
    "profile",
)


def _op_kwargs(cfg, *, autotune=False):
    return dict(
        num_tokens=cfg["num_tokens"],
        num_topk=cfg["num_topk"],
        num_experts=cfg["num_experts"],
        world_size=cfg["world_size"],
        rank=cfg["rank"],
        block_m=cfg["block_m"],
        pool_capacity=cfg["pool_capacity"],
        default_backend=BackendType.FLYDSL.value,
        dedup=cfg["dedup"],
        no_cpu_sync=cfg["no_cpu_sync"],
        autotune=autotune,
    )


def _op_call(wp, topk_idx, topk_w, *, autotune=False):
    """Invoke the custom op over a workspace's buffers (world=1, FLYDSL default)."""
    bufs = [wp.buffers[k] for k in _OP_BUFFER_KEYS]
    torch.ops.primus_turbo.mega_moe_prologue_impl(
        topk_idx, topk_w, *bufs, **_op_kwargs(wp.config, autotune=autotune)
    )


def _fresh_workspace(num_tokens, num_topk, num_experts, block_m):
    total_pairs = num_tokens * num_topk
    pool_capacity = total_pairs + num_experts * block_m
    pool_capacity = ((pool_capacity + block_m - 1) // block_m) * block_m
    wp = MegaMoePrologue.allocate(
        num_tokens=num_tokens,
        num_topk=num_topk,
        num_experts=num_experts,
        pool_capacity=pool_capacity,
        world_size=1,
        rank=0,
        block_m=block_m,
    )
    return wp, pool_capacity


@pytest.mark.skipif(not _HAVE_FLYDSL, reason="flydsl not importable (run inside dev_primus)")
@pytest.mark.parametrize("num_topk", [6, 8])
@pytest.mark.parametrize("drop_frac", [0.0, 0.1])
@pytest.mark.parametrize("with_weight", [True, False])
def test_custom_op_matches_reference(num_topk, drop_frac, with_weight):
    """Custom op (FLYDSL default) builds tables identical to the torch reference."""
    torch.cuda.set_device(0)
    num_tokens, num_experts, block_m = 4096, 32, 256
    topk_idx, topk_w = _make_routing(num_tokens, num_topk, num_experts, drop_frac=drop_frac)
    wp, pool_capacity = _fresh_workspace(num_tokens, num_topk, num_experts, block_m)

    # with_weight=False exercises the topk_w=None branch (kernel uses zero weights)
    _op_call(wp, topk_idx, topk_w if with_weight else None)
    torch.cuda.synchronize()

    ref = _reference(
        topk_idx,
        topk_w if with_weight else torch.zeros_like(topk_w),
        num_tokens,
        num_topk,
        num_experts,
        block_m,
    )
    _check_correctness(wp, ref, num_tokens, num_experts, block_m, pool_capacity)


@pytest.mark.skipif(not _HAVE_FLYDSL, reason="flydsl not importable (run inside dev_primus)")
def test_custom_op_torch_compile():
    """fullgraph=True with buffers as real graph inputs -> op genuinely compiles."""
    torch.cuda.set_device(0)
    num_tokens, num_topk, num_experts, block_m = 4096, 8, 32, 256
    topk_idx, topk_w = _make_routing(num_tokens, num_topk, num_experts)
    wp, pool_capacity = _fresh_workspace(num_tokens, num_topk, num_experts, block_m)
    kw = _op_kwargs(wp.config)

    def run(idx, w, *bufs):
        torch.ops.primus_turbo.mega_moe_prologue_impl(idx, w, *bufs, **kw)
        # read a mutated buffer back through the graph to force dependency tracking
        return wp.buffers["count"].clone()

    bufs = [wp.buffers[k] for k in _OP_BUFFER_KEYS]
    compiled = torch.compile(run, fullgraph=True)
    got = compiled(topk_idx, topk_w, *bufs)
    torch.cuda.synchronize()

    ref = _reference(topk_idx, topk_w, num_tokens, num_topk, num_experts, block_m)
    assert torch.equal(got.cpu().to(torch.int64), ref["counts"]), "compiled op count mismatch"
    _check_correctness(wp, ref, num_tokens, num_experts, block_m, pool_capacity)
    print("[torch.compile fullgraph] count matches reference")


@pytest.mark.skipif(not _HAVE_FLYDSL, reason="flydsl not importable (run inside dev_primus)")
def test_custom_op_dedup():
    """dedup=True path: secondaries are flagged, tables still match the reference."""
    torch.cuda.set_device(0)
    num_tokens, num_topk, num_experts, block_m = 4096, 8, 32, 256
    topk_idx, topk_w = _make_routing(num_tokens, num_topk, num_experts)
    total_pairs = num_tokens * num_topk
    pool_capacity = ((total_pairs + num_experts * block_m + block_m - 1) // block_m) * block_m
    wp = MegaMoePrologue.allocate(
        num_tokens=num_tokens,
        num_topk=num_topk,
        num_experts=num_experts,
        pool_capacity=pool_capacity,
        world_size=1,
        rank=0,
        block_m=block_m,
        dedup=True,
    )

    _op_call(wp, topk_idx, topk_w)
    torch.cuda.synchronize()

    # world=1: every top-k pair of a token shares dest-rank 0 -> 1 primary + (K-1) secondaries
    used = int((topk_idx >= 0).sum())
    secondaries = int(wp.buffers["source_dedup"][:used].sum())
    assert (
        secondaries == used - num_tokens
    ), f"dedup secondaries {secondaries} != expected {used - num_tokens}"
    ref = _reference(topk_idx, topk_w, num_tokens, num_topk, num_experts, block_m)
    _check_correctness(wp, ref, num_tokens, num_experts, block_m, pool_capacity)
    print("[dedup] secondaries flagged + tables match reference")


@pytest.mark.skipif(not _HAVE_FLYDSL, reason="flydsl not importable (run inside dev_primus)")
def test_custom_op_autotune_path():
    """Explicit autotune=True path runs (dispatcher tune -> default fallback) and is correct."""
    torch.cuda.set_device(0)
    num_tokens, num_topk, num_experts, block_m = 4096, 8, 32, 256
    topk_idx, topk_w = _make_routing(num_tokens, num_topk, num_experts)
    wp, pool_capacity = _fresh_workspace(num_tokens, num_topk, num_experts, block_m)

    _op_call(wp, topk_idx, topk_w, autotune=True)
    torch.cuda.synchronize()

    ref = _reference(topk_idx, topk_w, num_tokens, num_topk, num_experts, block_m)
    _check_correctness(wp, ref, num_tokens, num_experts, block_m, pool_capacity)
    print("[autotune] tables match reference")


def main():
    if not _HAVE_FLYDSL:
        raise SystemExit(f"flydsl not importable: {_IMPORT_ERR}")
    torch.cuda.set_device(0)
    test_custom_op_matches_reference(8, 0.0, True)
    test_custom_op_matches_reference(8, 0.0, False)  # topk_w=None branch
    print("[op vs reference] PASS")
    test_custom_op_torch_compile()
    print("[torch.compile] PASS")
    test_custom_op_dedup()
    print("[dedup] PASS")
    test_custom_op_autotune_path()
    print("[autotune] PASS")
    print("[mega_moe_prologue_impl] ALL PASS")


if __name__ == "__main__":
    main()
