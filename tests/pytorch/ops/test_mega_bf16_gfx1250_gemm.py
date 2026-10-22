###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Accuracy of the gfx1250 Mega GEMM tile, gathered and dense, NT and NN, against per-expert torch
matmul.

The pool is built the way Mega lays it out: each expert's rows padded to a 256-row pool block,
padding rows indexed -1, some experts empty, the same dispatched token feeding several pool
rows (a token routed to two local experts), and trailing pool blocks past ``num_tile_blocks``
that must not be written.
"""

import pytest
import torch

import primus_turbo.pytorch  # noqa: F401
from primus_turbo.pytorch.core.utils import is_gfx1250

pytestmark = pytest.mark.skipif(not is_gfx1250(), reason="gfx1250 tile")

_BLOCK_M = 256


def _pool(group_lens, num_tokens, seed):
    """(row_idx, tile_to_expert, num_live_blocks, pool_rows): Mega's pool tables for group_lens."""
    g = torch.Generator().manual_seed(seed)
    row_idx, tile_to_expert = [], []
    for expert, n in enumerate(group_lens):
        blocks = (n + _BLOCK_M - 1) // _BLOCK_M
        rows = torch.randint(0, num_tokens, (n,), generator=g, dtype=torch.int32)
        pad = torch.full((blocks * _BLOCK_M - n,), -1, dtype=torch.int32)
        row_idx.append(torch.cat([rows, pad]))
        tile_to_expert += [expert] * blocks
    live = len(tile_to_expert)
    dead = 2  # trailing blocks the grid covers but num_tile_blocks excludes
    row_idx.append(torch.zeros(dead * _BLOCK_M, dtype=torch.int32))
    tile_to_expert += [0] * dead
    return torch.cat(row_idx), torch.tensor(tile_to_expert, dtype=torch.int32), live, (live + dead) * _BLOCK_M


def _reference(a_rows, b, tile_to_expert, live):
    out = torch.empty(live * _BLOCK_M, b.shape[1], dtype=torch.float32, device=b.device)
    for t in range(live):
        rows = slice(t * _BLOCK_M, (t + 1) * _BLOCK_M)
        out[rows] = a_rows[rows].float() @ b[int(tile_to_expert[t])].float().T
    return out


@pytest.mark.parametrize(
    "N, K, group_lens",
    [
        (4096, 7168, [300, 0, 1024, 37, 512]),  # DeepSeek-V3 GateUp
        (7168, 2048, [256, 700, 0, 1]),  # DeepSeek-V3 Down
        (2880, 2048, [129, 600]),  # ragged N: 2880 = 11 * 256 + 64
        (5760, 2880, [300, 0, 77]),  # GPT-OSS GateUp: K = 2880 takes 64-wide k-tiles, ragged N
    ],
)
@pytest.mark.parametrize("mode", ["gather16", "gather32", "dense"])
@pytest.mark.parametrize("b_layout", ["nt", "nn"])
def test_mega_nt_tile(N, K, group_lens, mode, b_layout):
    from primus_turbo.flydsl.mega.bf16_gfx1250.gemm_nt_tile import mega_grouped_gemm_nt_gfx1250

    torch.manual_seed(0)
    G = len(group_lens)
    num_tokens = 1500
    row_idx, tile_to_expert, live, pool_rows = _pool(group_lens, num_tokens, seed=1)
    b = (torch.randn(G, N, K, device="cuda") / K**0.5).to(torch.bfloat16)
    if mode == "dense":
        a = torch.randn(pool_rows, K, device="cuda").to(torch.bfloat16)
        a_rows = a
        idx_arg, idx_bits = None, 0
    else:
        a = torch.randn(num_tokens, K, device="cuda").to(torch.bfloat16)
        a_rows = a[row_idx.clamp(min=0).cuda().long()]
        idx_arg, idx_bits = row_idx.cuda(), int(mode[len("gather") :])
    out = torch.full((pool_rows, N), float("nan"), dtype=torch.bfloat16, device="cuda")
    b_nn = b_layout == "nn"
    mega_grouped_gemm_nt_gfx1250(
        a,
        b.transpose(1, 2).contiguous() if b_nn else b,  # nn stores B as [G, K, N]
        out,
        tile_to_expert.cuda(),
        torch.tensor([live], dtype=torch.int32, device="cuda"),
        idx_arg,
        idx_bits=idx_bits,
        b_nn=b_nn,
    )
    torch.cuda.synchronize()

    ref = _reference(a_rows, b, tile_to_expert, live)
    valid = row_idx[: live * _BLOCK_M] >= 0 if mode != "dense" else torch.ones(live * _BLOCK_M, dtype=torch.bool)
    valid = valid.cuda()
    got = out[: live * _BLOCK_M].float()
    torch.testing.assert_close(got[valid], ref[valid], rtol=1.6e-2, atol=1e-2)
    assert bool(out[live * _BLOCK_M :].isnan().all()), "a block past num_tile_blocks was written"
