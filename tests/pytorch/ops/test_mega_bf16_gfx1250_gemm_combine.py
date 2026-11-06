###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The gfx1250 fused Mega L2 GEMM + combine on one GPU (EP1), against torch.

EP1 still runs the whole protocol -- GEMM workgroups bumping per-pool-block combine flags,
combine workgroups summing each token's local rows and pushing them into the symmetric combine
buffer, the reduce summing the primary routes -- only every peer is this rank. Each case runs
several calls with fresh tokens and routing, so a call that read the previous call's buffer or
flag epoch would fail.
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
        (1024, 7168, 2048, 32, 8, 0),  # DeepSeek-V3 Down, EP1 slice of the expert set
        (333, 7168, 2048, 32, 8, 5),  # ragged token count, five experts receive nothing
        (2048, 2048, 1408, 16, 4, 0),  # N = 2048 = 8 tiles, a different hidden and K
        (1, 7168, 2048, 32, 8, 0),  # one token: almost every pool block is empty
    ],
)
def test_gemm_combine_nt_ep1(ep1_group, num_tokens, hidden, inter, num_experts, num_topk, num_empty):
    from primus_turbo.flydsl.mega.gfx1250.bf16.gemm_combine_kernel import (
        gemm_combine_nt_gfx1250,
    )
    from primus_turbo.flydsl.mega.gfx1250.dispatch_prologue_kernel import (
        run_dispatch_prologue,
    )

    torch.manual_seed(num_tokens + num_empty)
    w1 = torch.empty(num_experts, 2 * inter, hidden, device="cuda", dtype=torch.bfloat16)  # shape only
    w2 = (torch.randn(num_experts, hidden, inter, device="cuda") / inter**0.5).to(torch.bfloat16)
    for _ in range(_CALLS):
        x = torch.randn(num_tokens, hidden, device="cuda", dtype=torch.bfloat16)
        scores = torch.randn(num_tokens, num_experts, device="cuda")
        scores[:, num_experts - num_empty :] = float("-inf")
        topk_w, topk_idx = scores.softmax(-1).topk(num_topk, dim=-1)
        handle, _ = run_dispatch_prologue(x, w1, ep1_group, topk_idx, topk_w)
        act = torch.randn(handle.pool_row_to_route.numel(), inter, device="cuda", dtype=torch.bfloat16)
        out = gemm_combine_nt_gfx1250(act, w2, handle, topk_idx)

        # the kernel stages each route's L2 row in bf16 before summing the routes
        prefix = handle.num_tokens_per_expert_prefix.tolist()
        counts = handle.num_tokens_per_expert.tolist()
        token = handle.pool_row_to_recv_token.long()
        ref = torch.zeros(num_tokens, hidden, device="cuda")
        for e in range(num_experts):
            rows = slice(prefix[e], prefix[e] + counts[e])
            if counts[e]:
                l2 = (act[rows].float() @ w2[e].float().T).to(torch.bfloat16).float()
                ref.index_add_(0, token[rows], l2)
        torch.testing.assert_close(out.float(), ref, rtol=1e-2, atol=3e-2)


# the plain schedule first, then issue_first alone, then every option on (lds_cosched needs issue_first)
_SCHEDULES = [
    dict(issue_first=0, lds_cosched=0, wmma_b2b=0),
    dict(issue_first=1, lds_cosched=0, wmma_b2b=0),
    dict(issue_first=1, lds_cosched=1, wmma_b2b=1),
]


@pytest.mark.parametrize("b_nn", [False, True], ids=["nt", "nn"])
def test_gemm_combine_scheduling_is_bit_identical(ep1_group, b_nn):
    """``issue_first``, ``lds_cosched`` and ``wmma_b2b`` reorder each k-tile's loads and matrix ops, not any
    output's k-step sequence: every schedule gives the bits of the plain one, for the forward L2 (NT) and
    the backward's L1 dgrad against w1 (``b_nn``)."""
    from primus_turbo.flydsl.mega.gfx1250.bf16.gemm_combine_kernel import (
        gemm_combine_nt_gfx1250,
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
    # NT: act [pool rows, inter] against w2 [experts, hidden, inter]; NN: grad_l1 [pool rows, 2 * inter]
    # against w1 [experts, 2 * inter, hidden]
    k, w = 2 * inter, w1
    if not b_nn:
        k = inter
        w = (torch.randn(num_experts, hidden, inter, device="cuda") / inter**0.5).to(torch.bfloat16)
    act = torch.randn(handle.pool_row_to_route.numel(), k, device="cuda", dtype=torch.bfloat16)
    # the output may live in a buffer the next call reuses
    outs = [
        gemm_combine_nt_gfx1250(act, w, handle, topk_idx, b_nn=b_nn, **opts).clone() for opts in _SCHEDULES
    ]
    for opts, out in zip(_SCHEDULES[1:], outs[1:]):
        assert torch.equal(out, outs[0]), f"{opts} changed the result"


@pytest.mark.parametrize("b_nn", [False, True], ids=["nt", "nn"])
@pytest.mark.parametrize("num_tokens, num_empty", [(1024, 0), (333, 5), (1, 0)])
def test_gemm_combine_persist_is_bit_identical(ep1_group, b_nn, num_tokens, num_empty):
    """The persistent GEMM workgroups compute every tile as the one-tile-per-workgroup launch does, with
    the gate gradient (``b_nn``) and over pools with empty experts or almost no rows."""
    from primus_turbo.flydsl.mega.gfx1250.bf16.gemm_combine_kernel import (
        gemm_combine_nt_gfx1250,
    )
    from primus_turbo.flydsl.mega.gfx1250.dispatch_prologue_kernel import (
        run_dispatch_prologue,
    )

    hidden, inter, num_experts, num_topk = 7168, 2048, 32, 8
    torch.manual_seed(num_tokens + num_empty)
    w1 = (torch.randn(num_experts, 2 * inter, hidden, device="cuda") / hidden**0.5).to(torch.bfloat16)
    w2 = (torch.randn(num_experts, hidden, inter, device="cuda") / inter**0.5).to(torch.bfloat16)
    k, w = (2 * inter, w1) if b_nn else (inter, w2)
    for _ in range(_CALLS):
        x = torch.randn(num_tokens, hidden, device="cuda", dtype=torch.bfloat16)
        scores = torch.randn(num_tokens, num_experts, device="cuda")
        scores[:, num_experts - num_empty :] = float("-inf")
        topk_w, topk_idx = scores.softmax(-1).topk(num_topk, dim=-1)
        handle, _ = run_dispatch_prologue(x, w1, ep1_group, topk_idx, topk_w)
        rows = handle.pool_row_to_route.numel()
        act = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16)
        kw = dict(b_nn=b_nn)
        if b_nn:
            kw["grad_gate"] = torch.randn(rows, device="cuda")
        # the outputs may live in buffers the next call reuses
        outs = [
            [t.clone() for t in gemm_combine_nt_gfx1250(act, w, handle, topk_idx, persist=p, **kw)]
            if b_nn
            else [gemm_combine_nt_gfx1250(act, w, handle, topk_idx, persist=p, **kw).clone()]
            for p in (False, True)
        ]
        for ref, got in zip(*outs):
            assert torch.equal(got, ref), "the persistent GEMM changed the result"
