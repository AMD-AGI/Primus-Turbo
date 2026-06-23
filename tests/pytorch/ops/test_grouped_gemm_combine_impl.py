###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Correctness + torch.compile + multi-backend dispatch test for the
``primus_turbo::grouped_gemm_combine_impl`` custom op (world=1 self-loopback).

The op is the ``torch.compile``-traceable / autotune-dispatchable wrapper around
the FlyDSL fused grouped GEMM + combine PUSH kernel. We drive the two-role path
(grouped GEMM into the caller-owned ``l2y`` + combine PUSH of every row into a
local combine buffer reached through the self peer-pointer table) and validate
against a torch grouped-GEMM reference, then re-run under ``torch.compile`` and the
explicit-autotune path.

``l2y`` is pre-filled with the GEMM reference before each call: the GEMM rewrites
it with the (bf16-identical) product, so the concurrent combine PUSH reads the
correct values regardless of fused GEMM->combine L2Y read ordering -> a
deterministic single-rank check. The 3-role topk reduce needs uncached symmetric
signal memory + the prologue's token-major origin_slot, so it is covered by the EP
integration test, not here.

Run inside dev_primus:
  PYTHONPATH=<...>/Primus-Turbo python -m pytest -s \
      tests/pytorch/ops/test_grouped_gemm_combine_impl.py
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))

try:
    from test_mega_moe_dispatch_combine_grouped_gemm import _uncached_i32

    import primus_turbo.pytorch.kernels.mega_moe.grouped_gemm_combine_impl  # noqa: F401
    from primus_turbo.pytorch.core.backend import BackendType

    _HAVE_FLYDSL = True
except Exception as _e:  # flydsl only present inside the dev container
    _HAVE_FLYDSL = False
    _IMPORT_ERR = _e


# ---------------------------------------------------------------------------
# Two-role test data (world=1): pool of `n_blocks` BM-row blocks, one expert each.
# out_features is a multiple of the combine step (512) -> no masked-tail path.
# ---------------------------------------------------------------------------
def _make_data(n_blocks, BM, N, K, *, seed=7, device="cuda"):
    torch.manual_seed(seed)
    G = n_blocks  # one group per block
    pool_capacity = n_blocks * BM
    act = (torch.randn(pool_capacity, K, device=device) / 8).bfloat16()
    weight = (torch.randn(G, N, K, device=device) / 8).bfloat16()  # NT: [G, N, K]
    tile_to_group = torch.arange(n_blocks, dtype=torch.int32, device=device)
    num_tile_blocks = torch.tensor([n_blocks], dtype=torch.int32, device=device)
    # every row occupied (origin rank 0), pushed to its own slot
    origin_rank = torch.zeros(pool_capacity, dtype=torch.int32, device=device)
    origin_slot = torch.arange(pool_capacity, dtype=torch.int32, device=device)
    return dict(
        act=act,
        weight=weight,
        tile_to_group=tile_to_group,
        num_tile_blocks=num_tile_blocks,
        origin_rank=origin_rank,
        origin_slot=origin_slot,
        pool_capacity=pool_capacity,
        N=N,
        G=G,
    )


def _ref_l2y(d, BM):
    """Grouped NT GEMM reference: block i -> act[block] @ weight[i].T (fp32 acc)."""
    act, weight = d["act"].float(), d["weight"].float()
    l2y = torch.empty(d["pool_capacity"], d["N"], device=act.device, dtype=torch.float32)
    for i in range(d["G"]):
        rows = slice(i * BM, (i + 1) * BM)
        l2y[rows] = act[rows] @ weight[i].T
    return l2y


def _alloc_io(d, BM):
    """Uncached scoreboard + caller-owned l2y (pre-filled = ref). The combine PUSH
    destination is allocated + returned by the op itself (self-push, world=1)."""
    sb_l2 = _uncached_i32(int(d["num_tile_blocks"].item()))
    l2y = _ref_l2y(d, BM).bfloat16().contiguous()  # GEMM rewrites with ~identical bf16
    return sb_l2, l2y


def _cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


@pytest.mark.skipif(not _HAVE_FLYDSL, reason="flydsl not importable (run inside dev_primus)")
@pytest.mark.parametrize("n_blocks", [2, 4])
def test_op_matches_reference(n_blocks):
    """Custom op: l2y == grouped-GEMM ref, and combine PUSH lands every row."""
    torch.cuda.set_device(0)
    BM = BN = 256
    K, N = 2048, 2048
    d = _make_data(n_blocks, BM, N, K)
    sb_l2, l2y = _alloc_io(d, BM)
    ref = _ref_l2y(d, BM)

    comb_buf, _gate = torch.ops.primus_turbo.grouped_gemm_combine_impl(
        d["act"],
        d["weight"],
        l2y,
        d["tile_to_group"],
        sb_l2,
        d["origin_rank"],
        d["origin_slot"],
        d["num_tile_blocks"],
        None,
        None,
        d["pool_capacity"],
        BackendType.FLYDSL.value,
        layout="nt",
        BM=BM,
        BN=BN,
    )
    torch.cuda.synchronize()

    assert _cos(l2y, ref) > 0.999, f"l2y cos {_cos(l2y, ref)} too low"
    # origin_slot[row]=row -> combine_output[row] holds l2y[row]; combine must reproduce ref
    assert _cos(comb_buf, ref) > 0.999, f"combine PUSH cos {_cos(comb_buf, ref)} too low"
    print(
        f"[op vs ref] n_blocks={n_blocks} l2y cos={_cos(l2y, ref):.6f} "
        f"combine cos={_cos(comb_buf, ref):.6f}"
    )


@pytest.mark.skipif(not _HAVE_FLYDSL, reason="flydsl not importable (run inside dev_primus)")
def test_op_torch_compile():
    """fullgraph=True with l2y as a real graph input -> the op genuinely compiles."""
    torch.cuda.set_device(0)
    BM = BN = 256
    K, N, n_blocks = 2048, 2048, 4
    d = _make_data(n_blocks, BM, N, K)
    sb_l2, l2y = _alloc_io(d, BM)
    ref = _ref_l2y(d, BM)

    def run(act, weight, l2y, tile_to_group, origin_rank, origin_slot):
        comb_buf, _gate = torch.ops.primus_turbo.grouped_gemm_combine_impl(
            act,
            weight,
            l2y,
            tile_to_group,
            sb_l2,
            origin_rank,
            origin_slot,
            d["num_tile_blocks"],
            None,
            None,
            d["pool_capacity"],
            BackendType.FLYDSL.value,
            layout="nt",
            BM=BM,
            BN=BN,
        )
        return l2y.clone(), comb_buf

    compiled = torch.compile(run, fullgraph=True)
    out, comb_buf = compiled(
        d["act"], d["weight"], l2y, d["tile_to_group"], d["origin_rank"], d["origin_slot"]
    )
    torch.cuda.synchronize()

    assert _cos(out, ref) > 0.999, f"compiled l2y cos {_cos(out, ref)} too low"
    assert _cos(comb_buf, ref) > 0.999, f"compiled combine cos {_cos(comb_buf, ref)} too low"
    print(f"[torch.compile fullgraph] l2y cos={_cos(out, ref):.6f}")


@pytest.mark.skipif(not _HAVE_FLYDSL, reason="flydsl not importable (run inside dev_primus)")
def test_op_autotune_path():
    """Explicit autotune=True: kernel-internal num_combine_cu sweep runs and stays correct."""
    torch.cuda.set_device(0)
    BM = BN = 256
    K, N, n_blocks = 2048, 2048, 4
    d = _make_data(n_blocks, BM, N, K)
    sb_l2, l2y = _alloc_io(d, BM)
    ref = _ref_l2y(d, BM)

    torch.ops.primus_turbo.grouped_gemm_combine_impl(
        d["act"],
        d["weight"],
        l2y,
        d["tile_to_group"],
        sb_l2,
        d["origin_rank"],
        d["origin_slot"],
        d["num_tile_blocks"],
        None,
        None,
        d["pool_capacity"],
        BackendType.FLYDSL.value,
        layout="nt",
        BM=BM,
        BN=BN,
        autotune=True,
    )
    torch.cuda.synchronize()

    assert _cos(l2y, ref) > 0.999, f"autotune l2y cos {_cos(l2y, ref)} too low"
    print(f"[autotune] l2y cos={_cos(l2y, ref):.6f}")


def main():
    if not _HAVE_FLYDSL:
        raise SystemExit(f"flydsl not importable: {_IMPORT_ERR}")
    torch.cuda.set_device(0)
    test_op_matches_reference(2)
    test_op_matches_reference(4)
    print("[op vs reference] PASS")
    test_op_torch_compile()
    print("[torch.compile] PASS")
    test_op_autotune_path()
    print("[autotune] PASS")
    print("[grouped_gemm_combine_impl] ALL PASS")


if __name__ == "__main__":
    main()
