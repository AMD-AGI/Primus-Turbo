###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Trainable mega MoE with MXFP8 forward + partial-fp8 backward (autograd Function).

Forward (inlined here, mirrors the bf16 `MegaMoEFusedFunction.forward`): L1 = the fused mxfp8
dispatch+fc1 kernel (`dispatch_grouped_gemm_mxfp8`: cross-rank fp8 PUSH + preshuffle + grouped
mxfp8 NT GEMM) -> SwiGLU -> L2 = the fp8 combine (`grouped_gemm_combine_fp8`: fp8 GEMM + CShuffle
mxfp8-quant epilogue + fp8 XGMI PUSH + fp8-dequant weighted reduce).
Backward (conjugate via the Dispatch<->Combine duality, mirrors the bf16
`MegaMoEFusedFunction.backward`):
  * STEP1 (dispatch(dy) + fc2 dgrad, NN): **MXFP8** fork -- emits ``grad_swiglu`` + the dispatched-dy
    pool in native rowwise-fp8 (the dW2 ``a`` operand). STEP3 (fc1 dgrad + combine) + dW1: bf16.
  * dW2 (fc2 weight grad, variable-K over the pool): **MXFP8** -- the ``a`` operand
    (``dispatch_l2_grad``) is requant-fused colwise DIRECTLY from the STEP1 rowwise-fp8 pool (no
    bf16 round-trip, LDS-transposed coalesced write); ``act_weighted`` is colwise-quantized; then
    the mxfp8 variable-K wgrad (`grouped_gemm_fp8_variable_k_impl`, MX_BLOCKWISE, autotuned).
    dW2 is the large GEMM (H*I*pool) -> the main backward win (~1.04x bf16 balanced).

``_DW2_FP8_FORMAT`` selects the dW2 wgrad fp8 encoding (default E5M2 for gradient range;
E4M3 measured higher-SNR at DSv3 magnitudes -- gated by dW2 SNR + few-step training loss)."""

import os
from typing import Optional, Tuple

import torch
from torch.distributed import ProcessGroup

from primus_turbo.flydsl.mega.dispatch_prologue_kernel import dispatch_prologue
from primus_turbo.flydsl.mega.fp8.dispatch_grouped_gemm_mxfp8_kernel import (
    dispatch_grouped_gemm_mxfp8,
)
from primus_turbo.flydsl.mega.fp8.dispatch_grouped_gemm_mxfp8_bwd_kernel import (
    dispatch_grouped_gemm_mxfp8_bwd,
)
from primus_turbo.flydsl.mega.fp8.grouped_gemm_combine_fp8_kernel import (
    grouped_gemm_combine_fp8,
)
from primus_turbo.flydsl.mega.fp8.quant import (
    quantize_grouped_weight_mxfp8,
    quantize_grouped_weight_mxfp8_cached,
)
from primus_turbo.flydsl.mega.fp8.quant_flydsl import quantize_rowwise_mxfp8_flydsl
from primus_turbo.flydsl.mega.swiglu_kernel import swiglu, swiglu_backward
from primus_turbo.flydsl.mega.symm_buffer import get_symm_buffer_for_mega_moe
from primus_turbo.pytorch.core.backend import BackendType
from primus_turbo.pytorch.core.low_precision import (
    ScalingGranularity,
    float8_e4m3,  # noqa: F401  (alternative dW2 encoding; see _DW2_FP8_FORMAT)
    float8_e5m2,
)
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_fp8_impl import (
    grouped_gemm_fp8_variable_k_impl,
)
from primus_turbo.pytorch.kernels.mega_moe.dispatch_grouped_gemm_impl import (
    dispatch_grouped_gemm_impl,
)
from primus_turbo.pytorch.kernels.mega_moe.grouped_gemm_combine_impl import (
    grouped_gemm_combine_impl,
)
from primus_turbo.flydsl.mega.fp8.quant_colwise_trans_flydsl import (
    colwise_grouped_meta,
    colwise_quant_mxfp8_grouped_flydsl,
    colwise_requant_mxfp8_grouped_fp8in_flydsl,
)

__all__ = ["MegaMoEFusedMxfp8Function", "mega_moe_fused_mxfp8"]

_DW2_FP8_FORMAT = float8_e5m2  # dW2 wgrad encoding (E5M2 = grad range; flip to float8_e4m3 to compare)
_MXFP8_BLOCK = 32
_HANDLE_GROUP_LENS = 9
_HANDLE_GROUP_OFFS = 10

# DIAGNOSTIC: PT_MEGA_BARRIER_MODE selects how many host torch.cuda.synchronize()+group.barrier()
# rendezvous bracket the scoreboard/flag resets (the cross-rank "publish the reset before any peer
# signals it" barriers):
#   "full"    (default): all 4 (A: pre-scoreboard-zero, B: post-zero/pre-L1-signal,
#                                C: pre-L2-reset,       D: post-reset/pre-L2-signal).
#   "reduced": keep only the 2 load-bearing ones (B + D). A and C are redundant given D + the
#              reduce spin already order the previous forward's cross-rank ops before this reset.
#   "none":    skip all 4 (NOT correctness-safe cross-rank; floor probe only, deadlocks the reduce).
_BARRIER_MODE = os.environ.get("PT_MEGA_BARRIER_MODE", "full")


def _host_rendezvous(group, *, load_bearing):
    if _BARRIER_MODE == "none":
        return
    if _BARRIER_MODE == "reduced" and not load_bearing:
        return
    torch.cuda.synchronize()
    group.barrier()


def _mxfp8_variable_k_wgrad(a_fp8, b_bf16, group_lens, group_offs):
    """dW2 = a^T @ b (variable-K over the pool/contraction axis) in MXFP8. Returns
    [G, H, b.shape[1]] bf16.

    ``a_fp8`` (a-branch producer fusion) is the STEP1 rowwise-fp8 pool ``(pool_fp8 [P,H],
    pool_scale [P,H//32])``: requant it colwise DIRECTLY from fp8 (fused dequant->requant, no bf16
    ``dispatch_l2_grad`` round-trip, LDS-transposed coalesced write). ``b`` (act_weighted, bf16) is
    colwise-quantized with the dedicated FlyDSL colwise-transpose quant. Both emit ONLY the
    transposed operand the wgrad needs (a_t + raw E8M0); the metadata is shared. Then the FlyDSL
    mxfp8 variable-K grouped GEMM (autotuned config). NOTE: colwise_grouped_meta does one
    total_M_pad D2H."""
    meta = colwise_grouped_meta(group_lens, group_offs)
    pool_fp8, pool_scale = a_fp8
    a_t, a_ts, lens_pc, offs_pc = colwise_requant_mxfp8_grouped_fp8in_flydsl(
        pool_fp8, pool_scale, _DW2_FP8_FORMAT, meta=meta
    )
    b_t, b_ts, _, _ = colwise_quant_mxfp8_grouped_flydsl(b_bf16, _DW2_FP8_FORMAT, meta=meta)
    return grouped_gemm_fp8_variable_k_impl(
        a_t, b_t,
        a_ts.view(torch.float8_e8m0fnu), b_ts.view(torch.float8_e8m0fnu),
        lens_pc.to(torch.int64), offs_pc.to(torch.int64),
        trans_a=False, trans_b=False, trans_c=False,
        out_dtype=torch.bfloat16, granularity=ScalingGranularity.MX_BLOCKWISE.value,
        num_cu=None, default_backend=BackendType.FLYDSL.value,
    )


_W2T_FP8_CACHE: dict = {}


def _w2t_mxfp8_cached(w2):
    """Grouped mxfp8 quant of ``w2^T`` (``[G,I,H]``) for the fp8 STEP1 NT-reuse dgrad; cached (w2 is
    a static weight, so the transpose + quant run once per weight version)."""
    key = (w2.data_ptr(), int(w2._version))
    v = _W2T_FP8_CACHE.get(key)
    if v is None:
        # bench-exact C++ grouped weight quant (matches the validated fork bench); the cached flydsl
        # path is a separate follow-up once STEP1 is validated.
        v = quantize_grouped_weight_mxfp8(w2.transpose(1, 2).contiguous())  # [G,I,H]
        _W2T_FP8_CACHE.clear()
        _W2T_FP8_CACHE[key] = v
    return v


def _mxfp8_step1_dispatch_dgrad(dy, w2, group, handle, block_m, block_n):
    """Backward STEP1 in fp8: fp8 dispatch(dy) + fp8 fc2 dgrad (the optimized bwd fork).  Returns
    ``(grad_swiglu bf16, pool_fp8_handle)`` where ``pool_fp8_handle = (pool_fp8 [P,H] E4M3,
    pool_scale [P,H//32] E8M0)`` is the dispatched-``dy`` pool (the dW2 ``a`` operand,
    ``dispatch_l2_grad``, in its native rowwise-fp8 form).  The fork emits only ``grad_swiglu``;
    dW2 requants the pool COLWISE directly from fp8 (a-branch fusion, no bf16 round-trip).  Needs
    the mxfp8 symm live (the forward's global buffer) and the same cross-rank scoreboard gate the
    forward L1 uses."""
    symm = get_symm_buffer_for_mega_moe()  # live buffer from the forward
    sym_layout = symm.make_sym_layout()
    dyq, dys = quantize_rowwise_mxfp8_flydsl(dy)
    dys = dys.view(torch.float8_e8m0fnu)
    w2tq, w2ts = _w2t_mxfp8_cached(w2)
    # publish scoreboard=0 cross-rank before the fp8 comm signals (mirror the forward L1 B-gate).
    _host_rendezvous(group, load_bearing=False)
    symm.scoreboard.zero_()
    _host_rendezvous(group, load_bearing=True)
    grad_swiglu = dispatch_grouped_gemm_mxfp8_bwd(
        dyq, dys, w2tq, w2ts, handle, sym_layout, symm, BM=block_m, BN=block_n,
    )
    P, H = symm.pool_fp8.shape
    pool_fp8_handle = (symm.pool_fp8, symm.pool_scale.reshape(P, H // 32))
    return grad_swiglu, pool_fp8_handle


class MegaMoEFusedMxfp8Function(torch.autograd.Function):
    """Fused mega MoE, MXFP8 forward + fp8-dW2 backward. Joins the autograd graph."""

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
        w1_fp8: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        w2_fp8: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        # w1_fp8 / w2_fp8: optional caller-supplied (MegaMoEFP8-owned, version-maintained) fp8 weight
        # prep -- w1_fp8=(w1q, w1s) grouped quant for L1 dispatch; w2_fp8=(weight_flat, b_sp) the
        # combine's fully-prepared w2 (quant + scale preshuffle). If None (standalone op use) they
        # are prepared here. bf16 w1/w2 remain the differentiable inputs (backward).
        assert x.dtype == torch.bfloat16
        assert w1.dtype == torch.bfloat16 and w2.dtype == torch.bfloat16
        num_tokens = x.shape[0]
        num_topk = topk_idx.shape[-1]
        topk_idx = topk_idx.to(torch.int64)
        ctx.set_materialize_grads(False)
        G, world, (T, H), I, K = w1.shape[0], group.size(), x.shape, w1.shape[1] // 2, num_topk
        save_bwd = any(ctx.needs_input_grad)

        # ── L1: fused mxfp8 dispatch + fc1 (one 3-role kernel: cross-rank clean-push raw fp8 +
        # E8M0 -> preshuffle role -> preshuffled grouped mxfp8 NT GEMM; scoreboard-gated, comm
        # hidden under the MFMA GEMM). Cross-rank/cross-XCD visibility via in-kernel L2 fences.
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
        # w1 fp8 quant: caller-supplied (module-owned, version-maintained) or quantize here.
        w1q, w1s = w1_fp8 if w1_fp8 is not None else quantize_grouped_weight_mxfp8_cached(w1)
        # x (activation) rowwise mxfp8 via the FlyDSL kernel (~near-HBM-peak; combine's act uses it
        # too). scale viewed e8m0 to match the generic quantize_rowwise_mxfp8 contract.
        xq, xs = quantize_rowwise_mxfp8_flydsl(x)
        xs = xs.view(torch.float8_e8m0fnu)
        # zero the scoreboard (cross-rank, barrier-bracketed) so the per-pool-block sentinel
        # handoff (preshuffle role -> gemm role) starts clean before any peer signals it.
        _host_rendezvous(group, load_bearing=False)  # A: pre-zero (redundant given D + reduce-spin)
        symm.scoreboard.zero_()
        _host_rendezvous(group, load_bearing=True)  # B: publish scoreboard=0 before L1 comm signals
        l1 = dispatch_grouped_gemm_mxfp8(
            xq, xs, w1q, w1s, handle, sym_layout, symm, BM=block_m, BN=block_n
        )
        # per-pool-row routing weight (prologue-scattered); clone before L2 reuses the buffer
        dispatch_weights = symm.weight_recv_buf.clone() if save_bwd else None

        act = swiglu(l1)

        # ── L2: fp8 combine (fp8 GEMM + CShuffle mxfp8-quant epilogue + fp8 XGMI PUSH + fp8-dequant
        # weighted top-k reduce). Init the L2 scoreboard (0) + reduce flags (-1) cross-rank first.
        _host_rendezvous(group, load_bearing=False)  # C: pre-reset (redundant given D + reduce-spin)
        symm.sb_l2.zero_()
        symm.barrier_local.fill_(-1)
        _host_rendezvous(group, load_bearing=True)  # D: publish sb_l2=0/flags=-1 before L2 signals
        y = grouped_gemm_combine_fp8(
            act, w2, list(handle), group,
            topk_indices=topk_idx, topk_weights=topk_weights.to(torch.float32),
            BM=block_m, BN=block_n, num_combine_cu=48, w2_fp8=w2_fp8,
        )

        if save_bwd:
            ctx.group = group
            ctx.num_tokens = num_tokens
            ctx.num_topk = num_topk
            ctx.block_m = block_m
            ctx.block_n = block_n
            ctx.handle_len = len(handle)
            ctx.save_for_backward(*handle, x, l1, dispatch_weights, w1, w2, topk_idx)
        return y

    @staticmethod
    @torch.no_grad()
    def backward(ctx, grad_y: Optional[torch.Tensor]) -> Tuple[Optional[torch.Tensor], ...]:
        """Conjugate of the mxfp8 forward. STEP1 fp8 fork; STEP3/dW1 bf16; dW2 MXFP8 (a-branch fused)."""
        if grad_y is None:
            return (None,) * 8
        saved = ctx.saved_tensors
        handle = tuple(saved[: ctx.handle_len])
        (saved_x, l1_out, dispatch_weights_in_buf, w1, w2, topk_idx) = saved[ctx.handle_len :]

        group_lens = handle[_HANDLE_GROUP_LENS]
        group_offs = handle[_HANDLE_GROUP_OFFS]
        num_tokens, num_topk = ctx.num_tokens, ctx.num_topk
        dy = grad_y.contiguous().to(torch.bfloat16)

        # STEP 1 (fp8 fork): dispatch(dy) + fc2 dgrad. Returns grad_swiglu + the dispatched-dy pool
        # in its native rowwise-fp8 form -- the dW2 `a` operand (dispatch_l2_grad), requant colwise
        # directly from fp8 (a-branch fusion, no bf16 round-trip).
        grad_swiglu, dispatch_l2_grad_fp8 = _mxfp8_step1_dispatch_dgrad(
            dy, w2, ctx.group, handle, ctx.block_m, ctx.block_n,
        )

        # STEP 2 (bf16): SwiGLU^T (re-inject routing weight) + gate grad
        grad_l1, grad_gate, act_weighted = swiglu_backward(
            grad_swiglu, l1_out, scale=dispatch_weights_in_buf, return_gate=True, return_act_w=True,
        )

        # dW2 (MXFP8): dispatch_l2_grad^T @ act_weighted (variable-K wgrad); the `a` operand is
        # requant-fused directly from the STEP1 rowwise-fp8 pool.
        dW2 = _mxfp8_variable_k_wgrad(dispatch_l2_grad_fp8, act_weighted, group_lens, group_offs)

        # STEP 3 (bf16): L1 dgrad + combine PUSH + dx reduce + grad_gate scatter
        dx, grad_topk_weights_flat = grouped_gemm_combine_impl(
            grad_l1, w1, list(handle), BackendType.FLYDSL.value,
            topk_indices=topk_idx.contiguous().view(-1), topk_weights=None,
            grad_gate=grad_gate, num_combine_cu=16, num_reduce_cu=0,
            layout="nn", BM=ctx.block_m, BN=ctx.block_n,
        )

        # dW1 (bf16): pool(x)^T @ grad_l1 (variable-K TN wgrad; re-dispatch saved x)
        dW1, _, _, _ = dispatch_grouped_gemm_impl(
            saved_x, grad_l1, ctx.group, BackendType.FLYDSL.value,
            handle=handle, layout="tn", trans_c=True, num_dispatch_cu=16,
        )

        grad_topk_weights = grad_topk_weights_flat.view(num_tokens, num_topk)
        # grads for (x, topk_idx, topk_weights, w1, w2, group, block_m, block_n, w1_fp8, w2_fp8);
        # w1_fp8/w2_fp8 are non-differentiable derived inputs -> None.
        return (
            dx,
            None,
            grad_topk_weights,
            dW1.to(w1.dtype),
            dW2.to(w2.dtype),
            None,
            None,
            None,
            None,
            None,
        )


def mega_moe_fused_mxfp8(
    group: ProcessGroup,
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    *,
    block_m: int = 256,
    block_n: int = 256,
    w1_fp8: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    w2_fp8: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    """One fully fused mega MoE forward (MXFP8) that joins autograd; backward fp8-izes dW2.

    ``w1_fp8`` / ``w2_fp8`` (optional): caller-owned grouped mxfp8 weight quant ``(w_fp8, w_scale)``
    -- pass these when a stateful holder (``MegaMoEFP8``) maintains the quantized weights
    (version-keyed) so the forward skips re-quantizing; leave ``None`` to quantize inside."""
    return MegaMoEFusedMxfp8Function.apply(
        x, topk_idx, topk_weights, w1, w2, group, block_m, block_n, w1_fp8, w2_fp8
    )
