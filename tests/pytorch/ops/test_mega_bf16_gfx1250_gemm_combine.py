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
    from primus_turbo.flydsl.mega.gfx1250.dispatch_prologue_kernel import (
        run_dispatch_prologue,
    )
    from primus_turbo.flydsl.mega.gfx1250.bf16.gemm_combine_kernel import (
        gemm_combine_nt_gfx1250,
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
