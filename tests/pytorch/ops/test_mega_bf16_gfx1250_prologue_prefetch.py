###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""A prefetched Mega dispatch prologue (prefetch_dispatch_prologue) gives what an inline one does."""

import os
import socket

import pytest
import torch
import torch.distributed as dist

import primus_turbo.pytorch  # noqa: F401
from primus_turbo.pytorch.core.utils import is_gfx1250

pytestmark = pytest.mark.skipif(not is_gfx1250(), reason="gfx1250 kernel")


@pytest.fixture(scope="module")
def ep1_group():
    owned = not dist.is_initialized()
    if owned:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ["MASTER_PORT"] = str(port)
        dev = torch.device("cuda", torch.cuda.current_device())
        dist.init_process_group("nccl", rank=0, world_size=1, device_id=dev)
    yield dist.group.WORLD
    if owned:
        dist.destroy_process_group()


@pytest.fixture
def prologue():
    from primus_turbo.flydsl.mega.gfx1250 import dispatch_prologue_kernel as m

    yield m
    m._prefetched = None


def _inputs(num_tokens, hidden, inter, num_experts, num_topk, seed):
    torch.manual_seed(seed)
    w1 = (torch.randn(num_experts, 2 * inter, hidden, device="cuda") / hidden**0.5).to(torch.bfloat16)
    x = torch.randn(num_tokens, hidden, device="cuda", dtype=torch.bfloat16)
    topk_w, topk_idx = torch.randn(num_tokens, num_experts, device="cuda").softmax(-1).topk(num_topk, dim=-1)
    return x, w1, topk_idx, topk_w


# tables the prologue writes in full; the others (e.g. dispatched_token_idx) have unwritten tails
_FULL_TABLES = (
    "expert_send_dst_rank",
    "expert_send_count",
    "expert_send_offset",
    "num_tokens_per_expert",
    "num_tokens_per_expert_prefix",
    "num_tile_blocks",
    "combine_recv_dst_rank",
    "combine_recv_start_row",
    "combine_recv_fold_count",
)


def _real_rows(handle):
    prefix = handle.num_tokens_per_expert_prefix.tolist()
    counts = handle.num_tokens_per_expert.tolist()
    return torch.cat([torch.arange(p, p + c, device="cuda") for p, c in zip(prefix, counts)])


def _by_route(handle):
    """The real pool rows, ordered by the route each holds: the prologue places an expert's rows with
    atomics, so their order within the expert can change from call to call."""
    rows = _real_rows(handle)
    order = torch.argsort(handle.pool_row_to_route[rows].long())
    return rows[order]


def _same(a, a_weights, b, b_weights):
    """Same routing tables and weights (rows matched by route); returns both calls' rows in that order."""
    for name in _FULL_TABLES:
        assert torch.equal(getattr(a, name), getattr(b, name)), name
    rows_a, rows_b = _by_route(a), _by_route(b)
    for name in ("pool_row_to_route", "pool_row_to_recv_token"):
        assert torch.equal(getattr(a, name)[rows_a], getattr(b, name)[rows_b]), name
    assert torch.equal(a_weights[rows_a], b_weights[rows_b])
    return rows_a, rows_b


@pytest.mark.parametrize("num_tokens", [1024, 333, 1])
def test_prefetched_prologue_matches_inline(ep1_group, prologue, num_tokens):
    from primus_turbo.flydsl.mega.gfx1250.bf16.dispatch_gemm_kernel import (
        dispatch_gemm_nt_gfx1250,
    )

    x, w1, topk_idx, topk_w = _inputs(num_tokens, 7168, 2048, 32, 8, seed=num_tokens)
    handle, weights = prologue.run_dispatch_prologue(x, w1, ep1_group, topk_idx, topk_w)
    out = dispatch_gemm_nt_gfx1250(x, w1, handle)
    for _ in range(3):
        prologue.prefetch_dispatch_prologue(x, w1, ep1_group, topk_idx, topk_w)
        busy = torch.randn(4096, 4096, device="cuda") @ torch.randn(
            4096, 4096, device="cuda"
        )  # queued between
        got, got_weights = prologue.run_dispatch_prologue(x, w1, ep1_group, topk_idx, topk_w)
        del busy
        rows, got_rows = _same(handle, weights, got, got_weights)
        assert torch.equal(out[rows], dispatch_gemm_nt_gfx1250(x, w1, got)[got_rows])


def test_prefetch_must_match_its_call(ep1_group, prologue):
    x, w1, topk_idx, topk_w = _inputs(512, 7168, 2048, 32, 8, seed=7)
    prologue.prefetch_dispatch_prologue(x, w1, ep1_group, topk_idx, topk_w)
    with pytest.raises(RuntimeError, match="was not used"):
        prologue.prefetch_dispatch_prologue(x, w1, ep1_group, topk_idx, topk_w)
    other = topk_idx.flip(0)
    with pytest.raises(RuntimeError, match="other arguments"):
        prologue.run_dispatch_prologue(x, w1, ep1_group, other, topk_w)
    assert prologue._prefetched is None
    # an inline call still works after the failed one
    prologue.run_dispatch_prologue(x, w1, ep1_group, topk_idx, topk_w)


def test_stage1_with_prefetch_is_bitwise_identical(ep1_group, prologue):
    from primus_turbo.pytorch.ops.moe.fused_mega_moe import (
        fused_mega_moe_stage1,
        prefetch_fused_mega_moe_dispatch,
    )

    x, w1, topk_idx, topk_w = _inputs(2048, 7168, 2048, 32, 8, seed=11)
    l1, weights, handle = fused_mega_moe_stage1(x, topk_idx, topk_w, w1, ep1_group)
    assert prefetch_fused_mega_moe_dispatch(x, topk_idx, topk_w, w1, ep1_group)
    got_l1, got_weights, got_handle = fused_mega_moe_stage1(x, topk_idx, topk_w, w1, ep1_group)
    assert prologue._prefetched is None
    rows, got_rows = _same(handle, weights, got_handle, got_weights)
    assert torch.equal(l1[rows], got_l1[got_rows])
