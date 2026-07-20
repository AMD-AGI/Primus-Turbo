###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Trainable mega MoE with MXFP8 forward + partial-fp8 backward (autograd Function).

This is the fp8 sibling of the bf16 ``mega_moe_fused`` (``mega_moe_fused.py``). Unlike bf16 it does
NOT go through the ``pytorch/kernels/mega_moe`` custom_op / AutoKernelDispatcher layer: the
orchestration is inlined here directly on the FlyDSL fp8 kernels, because the fp8 path carries
state the custom_op schema can't hold -- optional caller-supplied weight-prequant tuples
(``w1_fp8`` / ``w2_fp8`` / ``w2t_fp8`` / ``w1t_fp8``), reuse of the forward's live symmetric buffer
in backward, and host ``synchronize()`` + ``group.barrier()`` rendezvous.

Status: **forward wired** (L1 fused mxfp8 dispatch+fc1 NT -> SwiGLU bf16 -> L2 fp8 combine).
Backward is the next port step (see ``/perf_apps/xiaoming/Primus-Turbo``
``primus_turbo/pytorch/ops/moe/mega_moe_fused_mxfp8.py``):
  * STEP1 dispatch(dy)+fc2 dgrad (NN) MXFP8 -> grad_swiglu + rowwise-fp8 dispatched-dy pool.
  * STEP2 SwiGLU^T (bf16), re-inject routing weight, gate grad.
  * dW2   variable-K wgrad (MXFP8), a-operand requant-fused from the STEP1 fp8 pool.
  * STEP3 fc1 dgrad (fp8 GEMM) + combine/reduce (bf16, byte-bound).
  * dW1   variable-K wgrad (MXFP8), LOCAL -- reuses the FORWARD-dispatched fc1-input pool.

The bf16 ``w1`` / ``w2`` stay the differentiable inputs; the ``*_fp8`` args are non-diff derived
weight preps a stateful holder (``MegaMoEFP8``) maintains version-keyed and hands in.
"""

from typing import Optional, Tuple

import torch
from torch.distributed import ProcessGroup

from primus_turbo.flydsl.mega.fp8 import (
    dispatch_grouped_gemm_mxfp8,
    dispatch_prologue,
    get_symm_buffer_for_mega_moe,
    grouped_gemm_combine_fp8,
    prepare_w2_fp8,
    quantize_grouped_weight_mxfp8_cached,
    swiglu,
)

__all__ = [
    "MegaMoEFusedFP8Function",
    "mega_moe_fused_fp8",
    "prepare_w1t_dgrad_fp8",
    "prepare_w2t_dgrad_fp8",
]

_BWD_NOT_PORTED = (
    "mega MoE fp8 backward not ported yet (forward works); next port step. Reference: "
    "/perf_apps/xiaoming/Primus-Turbo/primus_turbo/pytorch/ops/moe/mega_moe_fused_mxfp8.py"
)


def _host_rendezvous(group) -> None:
    """Cross-rank publish barrier: drain this rank's GPU work, then all-rank barrier, so a
    scoreboard/flag reset is visible on every peer before any rank signals it. (Full mode;
    the source op gates these behind PT_MEGA_BARRIER_MODE -- kept always-on here for safety.)"""
    torch.cuda.synchronize()
    group.barrier()


_W2_PREP_ATTR = "_mega_fp8_w2_prep"


def _w2_fp8_cached(w2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Version-keyed cache of :func:`prepare_w2_fp8` (fc2 weight quant + scale preshuffle + int8
    flat) stashed ON the weight tensor: re-prep only when ``w2`` changes (``optim.step`` bumps
    ``_version``), reused across a grad-accum window. The op-layer analog of the w1 cache
    (``quantize_grouped_weight_mxfp8_cached``); the combine kernel stays pure compute."""
    v = getattr(w2, "_version", 0)
    ent = getattr(w2, _W2_PREP_ATTR, None)
    if ent is None or ent[0] != v:
        with torch.no_grad():
            out = prepare_w2_fp8(w2)
        ent = (v, out)
        setattr(w2, _W2_PREP_ATTR, ent)
    return ent[1]


def prepare_w2t_dgrad_fp8(w2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Grouped mxfp8 quant of ``w2^T`` (``[G,I,H]``) for the backward STEP1 fc2-dgrad NT-reuse GEMM.

    Returns ``(w2tq [G,I,H] fp8, w2ts [G,I,H//32] raw E8M0)``. STATIC weight prep -- a stateful
    holder (``MegaMoEFP8``) runs it once per ``w2._version`` and passes it via the op's ``w2t_fp8``
    arg so the transpose+quant never runs inside backward.
    """
    raise NotImplementedError(_BWD_NOT_PORTED)


def prepare_w1t_dgrad_fp8(w1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Grouped mxfp8 quant of ``w1^T`` (``[G,H,2I]``) for the backward STEP3 fc1-dgrad NT-reuse.

    Returns ``(w1tq [G,H,2I] fp8, w1ts [G,H,2I//32] raw E8M0)``. Mirrors
    :func:`prepare_w2t_dgrad_fp8`; owned version-keyed by ``MegaMoEFP8`` and passed via ``w1t_fp8``.
    """
    raise NotImplementedError(_BWD_NOT_PORTED)


class MegaMoEFusedFP8Function(torch.autograd.Function):
    """Fused mega MoE, MXFP8 forward + fp8-dW1/dW2 backward. Joins the autograd graph."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        group: ProcessGroup,
        block_m: int,
        block_n: int,
    ) -> torch.Tensor:
        # w1 / w2 are the bf16 differentiable weights. Their mxfp8 quant is maintained INSIDE this op
        # by a version-keyed cache -- w1 via ``quantize_grouped_weight_mxfp8_cached`` (caches the fp8
        # on the weight tensor, keyed by ``w1._version``) and w2 via the combine's own version-keyed
        # cache -- so a weight is re-quantized only when it actually changes (``optim.step`` bumps
        # ``_version``) and reused across a grad-accum window. No caller-supplied prequant tuples.
        assert x.dtype == torch.bfloat16, f"x must be bf16, got {x.dtype}"
        assert w1.dtype == torch.bfloat16 and w2.dtype == torch.bfloat16, "w1/w2 must be bf16"
        num_topk = topk_idx.shape[-1]
        topk_idx = topk_idx.to(torch.int64)
        ctx.set_materialize_grads(False)
        G, world = w1.shape[0], group.size()
        T, H = x.shape
        I, K = w1.shape[1] // 2, num_topk

        # ── L1: fused mxfp8 dispatch + fc1 (token quant folded inside via the bf16-x path) ──
        symm = get_symm_buffer_for_mega_moe(
            group, num_experts=G * world, num_max_tokens_per_rank=T, num_topk=K,
            hidden=H, intermediate_hidden=I, block_m=block_m, block_n=block_n, use_mxfp8=True,
        )
        sym_layout = symm.make_sym_layout()
        handle = tuple(
            dispatch_prologue(
                topk_idx, topk_weights, sym_layout=sym_layout, num_tokens=T, num_topk=K,
                num_experts=G * world, world_size=world, rank=symm.rank, experts_per_rank=G,
                block_m=block_m, num_max_pool_tokens=symm.num_max_pool_tokens,
            )
        )
        w1q, w1s = quantize_grouped_weight_mxfp8_cached(w1)  # version-keyed cache on w1._version
        # publish scoreboard=0 cross-rank before the L1 comm signals (per-pool-block sentinel handoff)
        _host_rendezvous(group)
        symm.scoreboard.zero_()
        _host_rendezvous(group)
        l1 = dispatch_grouped_gemm_mxfp8(
            x, None, w1q, w1s, handle, sym_layout, symm, BM=block_m, BN=block_n
        )

        act = swiglu(l1)

        # w2 fp8 prep (quant + scale preshuffle), version-keyed here at the op layer -- the combine
        # is a pure-compute kernel and takes the prepared weight in (symmetric with w1 at L1).
        w2_fp8 = _w2_fp8_cached(w2)

        # ── L2: fp8 combine (fp8 GEMM + mxfp8 epilogue + fp8 PUSH + bf16-out dequant reduce) ──
        _host_rendezvous(group)
        symm.sb_l2.zero_()
        symm.barrier_local.fill_(-1)
        _host_rendezvous(group)
        y = grouped_gemm_combine_fp8(
            act, w2_fp8, list(handle), group,
            topk_indices=topk_idx, topk_weights=topk_weights.to(torch.float32),
            BM=block_m, BN=block_n, num_combine_cu=48,
        )
        return y

    @staticmethod
    @torch.no_grad()
    def backward(ctx, grad_y: Optional[torch.Tensor]) -> Tuple[Optional[torch.Tensor], ...]:
        # PORT: STEP1 fp8 fork; dW2 + dW1 MXFP8 variable-K (LOCAL); STEP3 fc1-dgrad(fp8)+combine(bf16).
        # grads align with forward inputs (x, topk_idx, topk_weights, w1, w2, group, block_m,
        # block_n) -> return 8: (dx, None, grad_topk_weights, dW1, dW2, None, None, None). The
        # backward will maintain its own version-keyed w1^T / w2^T dgrad quant (prepare_w1t/w2t_dgrad_fp8),
        # same as forward -- no ctx-passed prequant. NOTE: forward does NOT save_for_backward yet;
        # add the saves when wiring backward.
        raise NotImplementedError(_BWD_NOT_PORTED)


def mega_moe_fused_fp8(
    group: ProcessGroup,
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    *,
    block_m: int = 256,
    block_n: int = 256,
) -> torch.Tensor:
    """One fully fused mega MoE forward (MXFP8) that joins autograd; backward fp8-izes dW1/dW2.

    fp8 sibling of :func:`primus_turbo.pytorch.ops.moe.mega_moe_fused.mega_moe_fused`. Pass the bf16
    ``w1`` / ``w2`` weights directly -- the op maintains their mxfp8 quant internally with a
    version-keyed cache (re-quantized only on ``optim.step``), so there are no weight-prequant args.
    """
    return MegaMoEFusedFP8Function.apply(
        x, topk_idx, topk_weights, w1, w2, group, block_m, block_n,
    )
