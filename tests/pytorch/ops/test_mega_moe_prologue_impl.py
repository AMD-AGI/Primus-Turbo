###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Correctness + torch.compile + multi-backend dispatch test for the
``primus_turbo::mega_moe_prologue_impl`` custom op (EP, cross-rank).

The op is the ``torch.compile``-traceable / autotune-dispatchable wrapper around
the FlyDSL prologue kernel. We drive it over a single symmetric allocation
(``SymmBuffer``) across ``world`` ranks and validate the resulting dispatch
tables against the same EP reference used by ``test_mega_moe_prolugue.py``, then
re-run under ``torch.compile`` and through the explicit-autotune path.

Run inside dev_primus (>=2 GPUs):
  PYTHONPATH=<...>/Primus-Turbo python \
      tests/pytorch/ops/test_mega_moe_prologue_impl.py --num-processes 8
"""

from __future__ import annotations

import argparse
import os
import sys

import pytest
import torch
import torch.distributed as dist

# reuse the EP reference + table checker + routing from the FlyDSL kernel test
sys.path.insert(0, os.path.dirname(__file__))

try:
    from test_mega_moe_prolugue import _check_tables, _ep_reference, _make_routing

    import primus_turbo.pytorch.kernels.mega_moe.mega_moe_prologue_impl  # noqa: F401  (registers op)
    from primus_turbo.pytorch.core.backend import BackendType
    from primus_turbo.pytorch.ops.moe.mega_moe_fused import get_symm_buffer_for_mega_moe

    _HAVE_FLYDSL = True
except Exception as _e:  # flydsl only present inside the dev container
    _HAVE_FLYDSL = False
    _IMPORT_ERR = _e


# tensor inputs the custom op consumes (positional, after the topk pair); the plan
# output tables are RETURNED by the op (not passed in)
_OP_BUFFER_KEYS = (
    "buffer_base",
    "buffer_offsets",
    "origin_rank",
    "origin_slot",
    "meta_scalars",
    "grid_barrier_state",
    "profile",
    "scoreboard",
    "barrier_local",
)


def _op_kwargs(symm, *, autotune=False):
    return dict(
        num_tokens=symm.num_tokens,
        num_topk=symm.num_topk,
        num_experts=symm.num_experts,
        world_size=symm.world,
        rank=symm.rank,
        block_m=symm.block_m,
        pool_capacity=symm.pool_capacity,
        default_backend=BackendType.FLYDSL.value,
        no_cpu_sync=True,
        autotune=autotune,
    )


def _op_call(symm, topk_idx, topk_w, *, autotune=False):
    """Invoke the custom op; returns (plan, tile_to_group, expected)."""
    bufs = [getattr(symm, k) for k in _OP_BUFFER_KEYS]
    return torch.ops.primus_turbo.mega_moe_prologue_impl(
        topk_idx, topk_w, *bufs, **_op_kwargs(symm, autotune=autotune)
    )


def _validate(symm, ret, topk_idx, topk_w, group, world, rank):
    all_idx = [torch.empty_like(topk_idx) for _ in range(world)]
    dist.all_gather(all_idx, topk_idx, group=group)
    all_idx = [t.cpu() for t in all_idx]
    ref = _ep_reference(
        all_idx, rank, symm.num_topk, symm.num_experts, world, symm.block_m, symm.pool_capacity
    )
    w = topk_w if topk_w is not None else torch.zeros_like(topk_idx, dtype=torch.float32)
    # the op returns: plan (7 tensors) + tile_to_group + expected
    plan, tile_to_group, expected = ret[0], ret[1], ret[2]
    _check_tables(
        symm,
        ref,
        plan,
        tile_to_group,
        expected,
        topk_idx,
        w,
        symm.num_topk,
        symm.num_experts,
        world,
        rank,
        symm.block_m,
        symm.pool_capacity,
    )


def _run(local_rank, world, args):
    ip = os.getenv("MASTER_ADDR", "127.0.0.1")
    port = int(os.getenv("MASTER_PORT", "8413"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", init_method=f"tcp://{ip}:{port}", world_size=world, rank=local_rank)
    torch.set_default_device("cuda")
    group = dist.new_group(list(range(world)))
    rank = dist.get_rank()

    T, K, E, BM = args.num_tokens, args.num_topk, args.num_experts, args.block_m
    symm = get_symm_buffer_for_mega_moe(
        group,
        num_experts=E,
        num_max_tokens_per_rank=T,
        num_topk=K,
        hidden=256,
        intermediate_hidden=256,
        block_m=BM,
    )

    # ---- 1) custom op (FLYDSL default) builds tables identical to the reference ----
    for with_weight in (True, False):  # False exercises the topk_w=None branch
        topk_idx, topk_w = _make_routing(T, K, E, seed=100 + rank, drop_frac=args.drop_frac)
        symm.group.barrier()
        ret = _op_call(symm, topk_idx, topk_w if with_weight else None)
        torch.cuda.synchronize()
        symm.group.barrier()
        _validate(symm, ret, topk_idx, topk_w if with_weight else None, group, world, rank)
    if rank == 0:
        print("[op vs reference] PASS (both weight branches)")

    # ---- 2) torch.compile fullgraph: op genuinely compiles + returns its plan tables ----
    topk_idx, topk_w = _make_routing(T, K, E, seed=100 + rank)
    kw = _op_kwargs(symm)

    def run(idx, w, *bufs):
        out = torch.ops.primus_turbo.mega_moe_prologue_impl(idx, w, *bufs, **kw)
        return out[0][2].clone()  # plan[2] = count

    bufs = [getattr(symm, k) for k in _OP_BUFFER_KEYS]
    symm.group.barrier()
    got = torch.compile(run, fullgraph=True)(topk_idx, topk_w, *bufs)
    torch.cuda.synchronize()
    symm.group.barrier()
    ret = _op_call(symm, topk_idx, topk_w)  # eager re-run for full table validation
    _validate(symm, ret, topk_idx, topk_w, group, world, rank)
    assert got.shape[0] == E, "compiled op count shape"
    if rank == 0:
        print("[torch.compile fullgraph] PASS")

    # ---- 3) explicit autotune path runs (dispatcher tune -> default fallback) ----
    topk_idx, topk_w = _make_routing(T, K, E, seed=100 + rank)
    symm.group.barrier()
    ret = _op_call(symm, topk_idx, topk_w, autotune=True)
    torch.cuda.synchronize()
    symm.group.barrier()
    _validate(symm, ret, topk_idx, topk_w, group, world, rank)
    if rank == 0:
        print("[autotune] PASS")
        print("[mega_moe_prologue_impl] ALL PASS")

    dist.barrier(group)
    dist.destroy_process_group()


_WORLD = 8


def _build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-processes", type=int, default=_WORLD)
    ap.add_argument("--num-tokens", type=int, default=4096)
    ap.add_argument("--num-topk", type=int, default=8)
    ap.add_argument("--num-experts", type=int, default=32)
    ap.add_argument("--block-m", type=int, default=256)
    ap.add_argument("--drop-frac", type=float, default=0.0)
    return ap


@pytest.mark.skipif(not _HAVE_FLYDSL, reason="flydsl not importable (run inside dev_primus)")
@pytest.mark.skipif(torch.cuda.device_count() < _WORLD, reason=f"needs {_WORLD} GPUs for EP{_WORLD}")
@pytest.mark.parametrize("drop_frac", [0.0, 0.1])
def test_mega_moe_prologue_impl(drop_frac):
    args = _build_argparser().parse_args(["--drop-frac", str(drop_frac)])
    torch.multiprocessing.spawn(_run, args=(_WORLD, args), nprocs=_WORLD)


def main():
    args = _build_argparser().parse_args()
    if not _HAVE_FLYDSL:
        raise SystemExit(f"flydsl not importable: {_IMPORT_ERR}")
    torch.multiprocessing.spawn(_run, args=(args.num_processes, args), nprocs=args.num_processes)


if __name__ == "__main__":
    main()
