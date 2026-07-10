###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Mega MoE forward with mxfp8 compute.

Comm modes for the cross-rank dispatch:

* ``comm="fp8_fused"`` (DEFAULT, fastest): a 3-stage single-kernel pipeline
  (``dispatch_grouped_gemm_mxfp8``): the comm role CLEAN-pushes pre-quantized fp8 tokens + raw
  E8M0 scales cross-rank (coalesced, XGMI-saturating); a dedicated preshuffle role transposes
  each pool-block's A-scale raw->broadcast ONCE (non-redundant); the L1 grouped mxfp8 GEMM
  consumes the broadcast scale -- comm / preshuffle / gemm all overlapped and gated per
  pool-block by the sys-scope scoreboard (a sentinel hands the block off from the preshuffle
  role to the gemm). Beats the decoupled path and the bf16 fused kernel. Cross-rank/cross-XCD
  visibility is carried by device-scope L2 write-back / invalidate fences in-kernel.
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
)
from primus_turbo.flydsl.mega.fp8.grouped_gemm_combine_mxfp8_kernel import (
    grouped_gemm_combine_mxfp8,
)
from primus_turbo.flydsl.mega.fp8.grouped_gemm_mxfp8_kernel import (
    grouped_gemm_mxfp8_flydsl_kernel,
)
from primus_turbo.flydsl.mega.fp8.quant import (
    quantize_grouped_weight_mxfp8,
    quantize_rowwise_mxfp8,
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
    return_aux: bool = False,
):
    """Fused mega MoE forward, mxfp8 compute. Returns y[T, H] bf16.

    ``x`` [T, H] bf16 source tokens; ``topk_idx``/``topk_weights`` [T, K]; ``w1`` [G, 2I, H]
    and ``w2`` [G, H, I] this rank's expert weights (bf16). ``comm`` selects the cross-rank
    dispatch: ``fp8_fused`` (default, 3-stage clean-push + preshuffle-role + gemm pipeline),
    ``fp8`` (decoupled), or ``bf16``.
    Requires H % 1024 == 0 (fp8 push) / % 256 (weight N), I % 1024 == 0 (SwiGLU), K % 128.

    ``return_aux`` (fp8_fused only): also return the backward intermediates the mega mxfp8
    autograd Function needs -- ``(y, {handle, l1, dispatch_weights})`` -- where
    ``dispatch_weights`` is the per-pool-row routing weight (``symm.weight_recv_buf`` cloned
    before the L2 combine reuses the buffer). ``w1``/``w2`` are NOT detached in this mode
    (the Function owns the graph)."""
    assert x.dtype == torch.bfloat16
    assert w1.dtype == torch.bfloat16 and w2.dtype == torch.bfloat16
    if return_aux:
        assert comm == "fp8_fused", "return_aux is only wired for comm='fp8_fused'"
    else:
        # forward-only: detach (mxfp8 quant of an autograd Parameter trips the quant op)
        w1 = w1.detach()
        w2 = w2.detach()
    _aux_dispatch_weights = None
    G = w1.shape[0]
    world = group.size()
    T, H = x.shape
    I = w1.shape[1] // 2
    K = topk_idx.shape[-1]
    topk_idx64 = topk_idx.to(torch.int64)
    l1 = None  # set by the fused fp8 dispatch+L1 path; else computed after the comm branch

    if comm == "fp8_fused":
        # FUSED (3-stage pipeline): quant tokens ONCE, then one kernel CLEAN-pushes the raw fp8
        # + E8M0 cross-rank (coalesced), a preshuffle role transposes each pool-block's A-scale
        # raw->broadcast once (non-redundant), and the L1 grouped mxfp8 GEMM consumes it --
        # comm/preshuffle/gemm overlapped, gated per pool-block by the sys-scope scoreboard
        # (no host sync + L2 invalidate: the scoreboard acquire + device-scope fences carry
        # peer-write visibility). Fastest fp8 fused path.
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
        xq, xs = quantize_rowwise_mxfp8(x)
        # zero the scoreboard (cross-rank, barrier-bracketed) so the per-pool-block sentinel
        # handoff (preshuffle role -> gemm role) starts clean before any peer signals it.
        torch.cuda.synchronize()
        group.barrier()
        symm.scoreboard.zero_()
        torch.cuda.synchronize()
        group.barrier()
        l1 = dispatch_grouped_gemm_mxfp8(
            xq, xs, w1q, w1s, handle, sym_layout, symm, BM=block_m, BN=block_n
        )
        if return_aux:
            # per-pool-row routing weight (prologue-scattered); clone before L2 reuses the buffer
            _aux_dispatch_weights = symm.weight_recv_buf.clone()
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

    # FUSED L2: one 3-role kernel does the grouped mxfp8 L2 GEMM -> cross-rank combine PUSH ->
    # weighted top-k reduce (mirror of the L1 fused kernel: compute -> comm). Init the L2
    # scoreboard (0) + reduce flags (-1, the "wait" state) cross-rank before the kernel signals
    # them, then it returns the per-token output directly.
    torch.cuda.synchronize()
    group.barrier()
    symm.sb_l2.zero_()
    symm.barrier_local.fill_(-1)
    torch.cuda.synchronize()
    group.barrier()
    y = grouped_gemm_combine_mxfp8(
        aq, as_, w2q, w2s, handle, group,
        topk_indices=topk_idx64, topk_weights=topk_weights.to(torch.float32),
        BM=block_m, BN=block_n, num_combine_cu=(num_combine_cu if num_combine_cu is not None else 64),
    )
    if return_aux:
        return y, {"handle": handle, "l1": l1, "dispatch_weights": _aux_dispatch_weights}
    return y
