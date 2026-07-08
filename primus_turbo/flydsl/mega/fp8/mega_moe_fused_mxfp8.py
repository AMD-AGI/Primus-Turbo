###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Mega MoE forward with mxfp8 compute.

Comm modes for the cross-rank dispatch:

* ``comm="fp8_fused"`` (DEFAULT, fastest): a 3-stage single-kernel pipeline
  (``dispatch_grouped_gemm_mxfp8_cleanpush(preshuffle_role=True)``): the comm role CLEAN-pushes
  pre-quantized fp8 tokens + raw E8M0 scales cross-rank (coalesced, XGMI-saturating); a
  dedicated preshuffle role transposes each pool-block's A-scale raw->broadcast ONCE
  (non-redundant); the L1 grouped mxfp8 GEMM consumes the broadcast scale -- comm / preshuffle
  / gemm all overlapped and gated per pool-block by the sys-scope scoreboard (a sentinel hands
  the block off from the preshuffle role to the gemm). Beats the decoupled path and the bf16
  fused kernel. Cross-rank/cross-XCD visibility is carried by device-scope L2 write-back /
  invalidate fences in-kernel (no host sync + standalone L2 invalidate).
* ``comm="fp8_fused_qip"`` (diagnostic): the older quant-in-push fused kernel
  (``dispatch_grouped_gemm_mxfp8``) -- the comm role quantizes bf16 x AND writes the broadcast
  scale over XGMI (scattered), which exposes the comm; kept for comparison only.
* ``comm="fp8"`` (decoupled, validated ~23 dB): quantize tokens to fp8 +
  E8M0 scales on the source rank and PUSH fp8 over XGMI into the peer ``pool_fp8`` /
  ``pool_scale`` (half the dispatch bytes vs bf16). The L1 grouped mxfp8 GEMM reads
  ``pool_fp8`` directly. Because ``pool_fp8`` / ``pool_scale`` live in the CACHED main
  heap (unlike the uncached ``comb``), the peer-written tokens need an L2 invalidate
  (``buffer_inv``) before the reader; a host sync after the invalidate ensures it
  retires before the L1 GEMM's read.
* ``comm="bf16"`` (milestone-1, validated ~23 dB): reuse the bf16 dispatch (PUSH bf16
  tokens into the pool), then quantize the bf16 pool locally before L1.

Both then run: L1 grouped mxfp8 GEMM -> SwiGLU -> re-quantize act -> L2 grouped mxfp8
GEMM -> bf16 combine PUSH + weighted top-k reduce. Combine stays bf16. Forward-only
(backward handled separately in bf16 for this phase).
"""

import torch

from primus_turbo.flydsl.mega.dispatch_grouped_gemm_bf16_kernel import (
    dispatch_grouped_gemm_bf16,
)
from primus_turbo.flydsl.mega.dispatch_prologue_kernel import dispatch_prologue
from primus_turbo.flydsl.mega.fp8.dispatch_fp8_push_kernel import (
    dispatch_fp8_push_launch,
    l2_invalidate_all,
)
from primus_turbo.flydsl.mega.fp8.dispatch_grouped_gemm_mxfp8_kernel import (
    dispatch_grouped_gemm_mxfp8,
    dispatch_grouped_gemm_mxfp8_cleanpush,
)
from primus_turbo.flydsl.mega.fp8.grouped_gemm_mxfp8_kernel import (
    grouped_gemm_mxfp8_flydsl_kernel,
)
from primus_turbo.flydsl.mega.fp8.quant import (
    quantize_grouped_weight_mxfp8,
    quantize_rowwise_mxfp8,
)
from primus_turbo.flydsl.mega.grouped_gemm_combine_bf16_kernel import (
    combine_only,
    topk_reduce_only,
)
from primus_turbo.flydsl.mega.swiglu_kernel import swiglu
from primus_turbo.flydsl.mega.symm_buffer import get_symm_buffer_for_mega_moe

_HANDLE_GROUP_OFFS = 10  # handle[10] = group_offs (int64 [G+1], BLOCK_M-padded pool boundaries)


def mega_moe_fused_mxfp8_forward(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    group,
    *,
    block_m: int = 256,
    block_n: int = 256,
    num_combine_cu: int | None = None,
    comm: str = "fp8_fused",
) -> torch.Tensor:
    """Fused mega MoE forward, mxfp8 compute. Returns y[T, H] bf16.

    ``x`` [T, H] bf16 source tokens; ``topk_idx``/``topk_weights`` [T, K]; ``w1`` [G, 2I, H]
    and ``w2`` [G, H, I] this rank's expert weights (bf16). ``comm`` selects the cross-rank
    dispatch: ``fp8_fused`` (default, 3-stage clean-push + preshuffle-role + gemm pipeline),
    ``fp8_fused_qip`` (quant-in-push diagnostic), ``fp8`` (decoupled), or ``bf16``.
    Requires H % 1024 == 0 (fp8 push) / % 256 (weight N), I % 1024 == 0 (SwiGLU), K % 128.
    """
    assert x.dtype == torch.bfloat16
    assert w1.dtype == torch.bfloat16 and w2.dtype == torch.bfloat16
    # forward-only: detach (mxfp8 quant of an autograd Parameter trips the quant op)
    w1 = w1.detach()
    w2 = w2.detach()
    G = w1.shape[0]
    world = group.size()
    T, H = x.shape
    I = w1.shape[1] // 2
    K = topk_idx.shape[-1]
    topk_idx64 = topk_idx.to(torch.int64)
    l1 = None  # set by the fused fp8 dispatch+L1 path; else computed after the comm branch

    if comm in ("fp8_fused", "fp8_fused_qip"):
        # milestone-2 FUSED: one kernel pushes fp8 tokens + E8M0 scales cross-rank AND computes
        # the L1 grouped mxfp8 GEMM, gated per pool-block by the sys-scope scoreboard (no host
        # sync + L2 invalidate: the scoreboard acquire carries the peer-write visibility).
        symm = get_symm_buffer_for_mega_moe(
            group, num_experts=G * world, num_max_tokens_per_rank=T, num_topk=K,
            hidden=H, intermediate_hidden=I, block_m=block_m, block_n=block_n, use_mxfp8=True,
        )
        sym_layout = symm.make_sym_layout()
        handle = tuple(
            dispatch_prologue(
                topk_idx64, topk_weights, sym_layout=sym_layout, num_tokens=T, num_topk=K,
                num_experts=G * world, world_size=world, rank=symm.rank, experts_per_rank=G,
                block_m=block_m, num_max_pool_tokens=symm.num_max_pool_tokens,
            )
        )
        w1q, w1s = quantize_grouped_weight_mxfp8(w1)
        if comm == "fp8_fused":
            # DEFAULT fused: quant tokens once, CLEAN push the raw fp8 + E8M0 (coalesced,
            # XGMI-saturating), a dedicated preshuffle role transposes each block_m's A-scale
            # raw->broadcast ONCE (non-redundant), then the preshuffled L1 GEMM consumes it --
            # 3-stage comm/preshuffle/gemm pipeline, all overlapped (fastest fp8 fused path).
            xq, xs = quantize_rowwise_mxfp8(x)
            l1 = dispatch_grouped_gemm_mxfp8_cleanpush(
                xq, xs, w1q, w1s, handle, sym_layout, symm, BM=block_m, BN=block_n,
                preshuffle_role=True,
            )
        else:  # fp8_fused_qip: quant-in-push (diagnostic; the comm role quantizes bf16 x AND
            # writes broadcast scale over XGMI -- slower comm, kept for comparison only).
            l1 = dispatch_grouped_gemm_mxfp8(
                x, w1q, w1s, handle, sym_layout, symm, BM=block_m, BN=block_n
            )
    elif comm == "fp8":
        # quantize-then-push: fp8 tokens + E8M0 scales cross-rank into pool_fp8/pool_scale
        symm = get_symm_buffer_for_mega_moe(
            group, num_experts=G * world, num_max_tokens_per_rank=T, num_topk=K,
            hidden=H, intermediate_hidden=I, block_m=block_m, block_n=block_n, use_mxfp8=True,
        )
        sym_layout = symm.make_sym_layout()
        handle = tuple(
            dispatch_prologue(
                topk_idx64, topk_weights, sym_layout=sym_layout, num_tokens=T, num_topk=K,
                num_experts=G * world, world_size=world, rank=symm.rank, experts_per_rank=G,
                block_m=block_m, num_max_pool_tokens=symm.num_max_pool_tokens,
            )
        )
        xq, xs = quantize_rowwise_mxfp8(x)
        dispatch_fp8_push_launch(xq, xs, handle, sym_layout, symm.num_max_pool_tokens, world)
        torch.cuda.synchronize()
        group.barrier()  # cross-rank PUSH completion before reading the pool
        # pool_fp8/pool_scale are in the CACHED main heap; invalidate L2 so the L1 GEMM
        # re-fetches the peer-written tokens from DRAM (comb works barrier-only as it is uncached).
        # The sync ensures the invalidate retires before the L1 GEMM's (first-call) JIT + read.
        l2_invalidate_all()
        torch.cuda.synchronize()
        a_fp8, a_scale = symm.pool_fp8, symm.pool_scale
    else:  # bf16 comm (milestone-1): push bf16 tokens, then quantize the pool locally
        _l1, _pool, _wr, handle = dispatch_grouped_gemm_bf16(
            x, w1, group, handle=None, topk_idx=topk_idx64, topk_weights=topk_weights,
            layout="nt", BM=block_m, BN=block_n,
        )
        symm = get_symm_buffer_for_mega_moe()
        a_fp8, a_scale = quantize_rowwise_mxfp8(symm.pool)

    group_offs = handle[_HANDLE_GROUP_OFFS]

    # L1 up/gate (mxfp8) -> SwiGLU -> L2 down (mxfp8). The fused path already produced l1.
    if l1 is None:
        w1q, w1s = quantize_grouped_weight_mxfp8(w1)
        l1 = grouped_gemm_mxfp8_flydsl_kernel(
            a_fp8, a_scale, w1q, w1s, group_offs, out_dtype=torch.bfloat16
        )
    act = swiglu(l1)
    aq, as_ = quantize_rowwise_mxfp8(act)
    w2q, w2s = quantize_grouped_weight_mxfp8(w2)
    l2 = grouped_gemm_mxfp8_flydsl_kernel(aq, as_, w2q, w2s, group_offs, out_dtype=torch.bfloat16)
    symm.l2_token_buffer.copy_(l2)

    # combine (bf16): cross-rank PUSH of l2_token_buffer then weighted top-k reduce
    torch.cuda.synchronize()
    group.barrier()
    if comm == "fp8":
        # combine reads origin_rank/origin_slot (cached main heap, written cross-rank by the
        # prologue); L1/L2 re-cached them, so invalidate L2 before the combine PUSH reads them.
        l2_invalidate_all()
    combine_only(group, BM=block_m, num_combine_cu=num_combine_cu)
    torch.cuda.synchronize()
    group.barrier()
    symm.barrier_local.zero_()

    y = torch.empty((int(symm.num_tokens), H), dtype=torch.bfloat16, device=x.device)
    topk_reduce_only(
        y, symm.comb, symm.barrier_local,
        topk_idx.to(torch.int32).contiguous().view(-1), symm.num_tokens_per_rank,
        int(symm.combine_slots), topk=int(symm.num_topk), num_experts=int(symm.num_experts),
        rank=int(symm.rank), topk_weights=topk_weights.to(torch.float32).contiguous().view(-1),
    )
    return y
