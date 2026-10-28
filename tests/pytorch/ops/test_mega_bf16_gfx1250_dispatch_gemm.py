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
