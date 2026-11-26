###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The gfx1250 fused Mega dispatch + L1 GEMM on one GPU (EP1), against torch.

EP1 still runs the whole protocol -- prologue tables, comm workgroups pushing into the
symmetric dispatch buffer, the per-expert flag epoch, GEMM workgroups waiting on it and
gathering A through TDM -- only every peer is this rank. Each case runs several calls with
fresh tokens and routing, so a GEMM that read the previous call's buffer or flag would fail.
"""

import os
import socket

import pytest
import torch
import torch.distributed as dist

import primus_turbo.pytorch  # noqa: F401
from primus_turbo.pytorch.core.utils import is_gfx1250

pytestmark = pytest.mark.skipif(not is_gfx1250(), reason="gfx1250 kernel")

_CALLS = 4


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


@pytest.mark.parametrize(
    "num_tokens, hidden, inter, num_experts, num_topk, num_empty",
    [
        (1024, 7168, 2048, 32, 8, 0),  # DeepSeek-V3 GateUp, EP1 slice of the expert set
        (333, 7168, 2048, 32, 8, 5),  # ragged token count, five experts receive nothing
        (2048, 2048, 1408, 16, 4, 0),  # N = 2816 = 11 tiles, a different hidden
        (1, 7168, 2048, 32, 8, 0),  # one token: every live tile is mostly padding
        # GPT-OSS: 5760-byte rows end mid 256-byte line (64-wide k-tiles), so a row's edge shares a line
        # with its neighbour's; a stale read of that line shows from the 2nd call on
        (1024, 2880, 2880, 32, 4, 0),
    ],
)
def test_dispatch_gemm_nt_ep1(ep1_group, num_tokens, hidden, inter, num_experts, num_topk, num_empty):
    from primus_turbo.flydsl.mega.gfx1250.bf16.dispatch_gemm_kernel import (
        dispatch_gemm_nt_gfx1250,
    )
    from primus_turbo.flydsl.mega.gfx1250.dispatch_prologue_kernel import (
        run_dispatch_prologue,
    )

    torch.manual_seed(num_tokens + num_empty)
    w1 = (torch.randn(num_experts, 2 * inter, hidden, device="cuda") / hidden**0.5).to(torch.bfloat16)
    for _ in range(_CALLS):
        x = torch.randn(num_tokens, hidden, device="cuda", dtype=torch.bfloat16)
        scores = torch.randn(num_tokens, num_experts, device="cuda")
        scores[:, num_experts - num_empty :] = float("-inf")
        topk_w, topk_idx = scores.softmax(-1).topk(num_topk, dim=-1)
        handle, _ = run_dispatch_prologue(x, w1, ep1_group, topk_idx, topk_w)
        out = dispatch_gemm_nt_gfx1250(x, w1, handle)

        prefix = handle.num_tokens_per_expert_prefix.tolist()
        counts = handle.num_tokens_per_expert.tolist()
        assert counts == torch.bincount(topk_idx.flatten(), minlength=num_experts).tolist()
        token = handle.pool_row_to_recv_token.long()
        for e in range(num_experts):
            rows = slice(prefix[e], prefix[e] + counts[e])
            ref = x.float()[token[rows]] @ w1[e].float().T
            torch.testing.assert_close(out[rows].float(), ref, rtol=1.6e-2, atol=1e-2, msg=f"expert {e}")


def test_combine_buffer_capacity_is_checked():
    """The combine buffer's 32-bit offsets bound the tokens per rank (host-only: the check where the heap is sized,
    for the bf16 and the MXFP8 Mega): DeepSeek-V3's top-8 and hidden 7168 take 37,376 tokens, not one more."""
    from primus_turbo.flydsl.mega.gfx1250.symm_buffer import check_combine_capacity

    check_combine_capacity(37376, 8, 7168)
    check_combine_capacity(65536, 8, 2048)
    for tokens in (37377, 65536):
        with pytest.raises(ValueError, match="at most 37376 tokens per rank at top-8, hidden 7168"):
            check_combine_capacity(tokens, 8, 7168)


def test_dispatch_only_ep1(ep1_group):
    """The dispatch-only launch (the comm role without the GEMM role) lands every token at its receive row."""
    from primus_turbo.flydsl.mega.gfx1250.bf16.dispatch_gemm_kernel import (
        dispatch_only_gfx1250,
    )
    from primus_turbo.flydsl.mega.gfx1250.dispatch_prologue_kernel import (
        run_dispatch_prologue,
    )

    num_tokens, hidden, num_experts, num_topk = 333, 2048, 16, 4
    torch.manual_seed(num_tokens)
    w1 = torch.empty(num_experts, 512, hidden, device="cuda", dtype=torch.bfloat16)  # only its shape is read
    for _ in range(2):
        x = torch.randn(num_tokens, hidden, device="cuda", dtype=torch.bfloat16)
        topk_w, topk_idx = (
            torch.randn(num_tokens, num_experts, device="cuda").softmax(-1).topk(num_topk, dim=-1)
        )
        handle, _ = run_dispatch_prologue(x, w1, ep1_group, topk_idx, topk_w)
        received = dispatch_only_gfx1250(x, handle)  # at EP1 token t's receive row is t
        assert torch.equal(received[:num_tokens].view(torch.int16), x.view(torch.int16))


@pytest.mark.parametrize(
    "num_tokens, num_experts, num_topk, num_empty",
    [(333, 8, 4, 2), (8192, 32, 8, 0)],
)
def test_dispatch_prologue_pool_order_ep1(ep1_group, num_tokens, num_experts, num_topk, num_empty):
    """Each part of an expert's pool segment (the rows the combine folds a token's routes into, then the others)
    lists its tokens in increasing order, and so does each send list: the tables depend on the routing alone, not
    on the grid the prologue was tuned to or on the order its workgroups ran in, so the wgrads, which contract over
    the pool rows, are the same in every run."""
    from primus_turbo.flydsl.mega.gfx1250.dispatch_prologue_kernel import (
        run_dispatch_prologue,
    )

    torch.manual_seed(num_tokens + num_empty)
    x = torch.randn(num_tokens, 1024, device="cuda", dtype=torch.bfloat16)
    w1 = torch.empty(num_experts, 512, 1024, device="cuda", dtype=torch.bfloat16)  # only its shape is read
    scores = torch.randn(num_tokens, num_experts, device="cuda")
    scores[:, num_experts - num_empty :] = float("-inf")
    topk_w, topk_idx = scores.softmax(-1).topk(num_topk, dim=-1)
    tables = []
    for _ in range(2):
        handle, weights = run_dispatch_prologue(x, w1, ep1_group, topk_idx, topk_w)
        start = handle.combine_recv_start_row.tolist()
        fold = handle.combine_recv_fold_count.tolist()
        counts = handle.num_tokens_per_expert.tolist()
        token = handle.pool_row_to_recv_token.tolist()
        for e in range(num_experts):
            for lo, hi in ((start[e], start[e] + fold[e]), (start[e] + fold[e], start[e] + counts[e])):
                assert token[lo:hi] == sorted(set(token[lo:hi])), (
                    f"expert {e}: pool rows {lo}..{hi} out of order"
                )
        send_offset = handle.expert_send_offset.tolist()
        send_count = handle.expert_send_count.tolist()
        sent = handle.dispatched_token_idx.tolist()
        for e in range(num_experts):
            part = sent[send_offset[e] : send_offset[e] + send_count[e]]
            assert part == sorted(set(part)), f"expert {e}: send list out of order"
        num_sent = sum(send_count)
        tables.append(
            [handle.pool_row_to_recv_token, handle.pool_row_to_route, handle.recv_token_to_pool_rows, weights]
            + [handle.dispatched_token_idx[:num_sent]]
        )
    for a, b, what in zip(
        *tables,
        (
            "pool_row_to_recv_token",
            "pool_row_to_route",
            "recv_token_to_pool_rows",
            "dispatch weights",
            "dispatched_token_idx",
        ),
    ):
        assert torch.equal(a, b), f"{what} differs between two calls"


# the plain schedule first, then issue_first alone, then every option on (lds_cosched needs issue_first)
_SCHEDULES = [
    dict(issue_first=0, lds_cosched=0, wmma_b2b=0),
    dict(issue_first=1, lds_cosched=0, wmma_b2b=0),
    dict(issue_first=1, lds_cosched=1, wmma_b2b=1),
]


@pytest.mark.parametrize("b_nn", [False, True], ids=["nt", "nn"])
def test_dispatch_gemm_scheduling_is_bit_identical(ep1_group, b_nn):
    """``issue_first``, ``lds_cosched`` and ``wmma_b2b`` reorder each k-tile's loads and matrix ops, not any
    output's k-step sequence: every schedule gives the bits of the plain one, for the forward L1 (NT) and
    the backward's dispatch of dy against w2 (``b_nn``)."""
    from primus_turbo.flydsl.mega.gfx1250.bf16.dispatch_gemm_kernel import (
        dispatch_gemm_nt_gfx1250,
    )
    from primus_turbo.flydsl.mega.gfx1250.dispatch_prologue_kernel import (
        run_dispatch_prologue,
    )

    num_tokens, hidden, inter, num_experts, num_topk = 1024, 7168, 2048, 32, 8
    torch.manual_seed(0)
    x = torch.randn(num_tokens, hidden, device="cuda", dtype=torch.bfloat16)
    topk_w, topk_idx = torch.randn(num_tokens, num_experts, device="cuda").softmax(-1).topk(num_topk, dim=-1)
    w1 = (torch.randn(num_experts, 2 * inter, hidden, device="cuda") / hidden**0.5).to(torch.bfloat16)
    handle, _ = run_dispatch_prologue(x, w1, ep1_group, topk_idx, topk_w)
    w = w1
    if b_nn:
        w = (torch.randn(num_experts, hidden, inter, device="cuda") / hidden**0.5).to(torch.bfloat16)
    # the padding rows of each expert's pool block are undefined
    prefix = handle.num_tokens_per_expert_prefix.tolist()
    counts = handle.num_tokens_per_expert.tolist()
    rows = torch.cat([torch.arange(p, p + c, device="cuda") for p, c in zip(prefix, counts)])
    outs = [dispatch_gemm_nt_gfx1250(x, w, handle, b_nn=b_nn, **opts)[rows] for opts in _SCHEDULES]
    for opts, out in zip(_SCHEDULES[1:], outs[1:]):
        assert torch.equal(out, outs[0]), f"{opts} changed the result"


@pytest.mark.parametrize(
    "tile_n, issue_first, requested, resolved",
    [
        (256, 1, -1, 1),  # the default tile: 24 fragment reads for 32 matrix ops
        (256, 0, -1, 0),  # co-scheduling needs issue_first
        (128, 1, -1, 0),  # 16 reads for 16 matrix ops
        (256, 1, 0, 0),
    ],
)
def test_lds_cosched_auto(tile_n, issue_first, requested, resolved):
    """-1, the default of dispatch_gemm and gemm_combine, turns lds_cosched on where it can run."""
    from primus_turbo.flydsl.mega.gfx1250.bf16.gemm_nt_tile import resolve_lds_cosched

    assert resolve_lds_cosched(requested, issue_first, 256, tile_n, 4, 2) == resolved


@pytest.mark.parametrize("tile_n, issue_first", [(128, 1), (256, 0)])
def test_lds_cosched_rejects_what_it_cannot_schedule(tile_n, issue_first):
    """An explicit lds_cosched=1 the tile cannot take raises, under python -O too."""
    from primus_turbo.flydsl.mega.gfx1250.bf16.gemm_nt_tile import resolve_lds_cosched

    with pytest.raises(ValueError, match="lds_cosched=1"):
        resolve_lds_cosched(1, issue_first, 256, tile_n, 4, 2)
