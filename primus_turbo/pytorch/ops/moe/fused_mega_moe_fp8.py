###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Mega MoE autograd ops with an MXFP8 forward and a partial-fp8 backward.

The fp8 counterpart of ``fused_mega_moe``, with the same two entry points:

  * ``fused_mega_moe_fp8`` -- one fully fused op (dispatch+fc1+SwiGLU+fc2+combine).
  * ``fused_mega_moe_fp8_stage1`` / ``fused_mega_moe_fp8_stage2`` -- the same math split into two
    autograd edges at ``l1`` (pre-SwiGLU), so w1 and w2 can be separate DDP gradient boundaries.
    Stage state rides a ``_Fp8StageState`` side channel rather than the ops' args/returns.

Pass ``w1`` / ``w2`` as the high-precision weights; their mxfp8 quant is maintained inside the
impls, keyed on ``w._version``. The op is NOT CUDA-graph capturable.

An expert shape off the fp8 kernels' grid (see ``fp8_padded_dims``; GPT-OSS's 2880) runs
zero-padded: the weight preps pad the weights, and the ops here pad x / dy and slice y / dx / dW.
"""

from typing import Optional

import torch
import torch.nn.functional as F
from torch.distributed import ProcessGroup

from primus_turbo.flydsl.utils.glu_activation import GLUActivation
from primus_turbo.pytorch.kernels.fused_mega_moe import (
    fused_mega_moe_stage1_backward_fp8_impl,
    fused_mega_moe_stage1_forward_fp8_impl,
    fused_mega_moe_stage2_backward_fp8_impl,
    fused_mega_moe_stage2_forward_fp8_impl,
)
from primus_turbo.pytorch.kernels.fused_mega_moe.mega_moe_fp8_weights import (
    fp8_padded_dims,
    pad_b1_fp8_dims,
    unpad_b1_grad,
    unpad_w1_grad,
    unpad_w2_grad,
)


def _pad_cols(t: torch.Tensor, cols: int) -> torch.Tensor:
    return t if t.shape[-1] == cols else F.pad(t, (0, cols - t.shape[-1]))


def _slice_cols(t: torch.Tensor, cols: int) -> torch.Tensor:
    return t if t.shape[-1] == cols else t[:, :cols].contiguous()


class _GatheredExpertBiasFunction(torch.autograd.Function):
    """``y + sum_k p_k * bias[topk_idx_k]`` on the token's own rank, the fc2-bias term of the experts.

    The fp8 combine quantizes each expert row before it leaves the expert's rank, so the bias is
    added after the reduce instead, where the routing weights are at hand: the EP-sharded bias is
    small enough ([E, H]) to all-gather whole, and the term is one ``[T, E] @ [E, H]`` with the
    routing weights scattered dense. Its backward reduce-scatters the bias gradient back to the
    experts' owners and adds ``dy . bias[e]`` to each routing weight's gradient."""

    @staticmethod
    def forward(ctx, y, topk_idx, topk_weights, bias, group):
        bias_full = torch.empty(
            (bias.shape[0] * group.size(), bias.shape[1]), dtype=bias.dtype, device=bias.device
        )
        torch.distributed.all_gather_into_tensor(bias_full, bias.contiguous(), group=group)
        bias_full = bias_full.float()
        probs = torch.zeros((y.shape[0], bias_full.shape[0]), dtype=torch.float32, device=y.device)
        probs.scatter_(1, topk_idx, topk_weights.float())
        ctx.save_for_backward(topk_idx, probs, bias_full)
        ctx.group, ctx.bias_dtype = group, bias.dtype
        return torch.addmm(y.float(), probs, bias_full).to(y.dtype)

    @staticmethod
    @torch.no_grad()
    def backward(ctx, dy):
        topk_idx, probs, bias_full = ctx.saved_tensors
        dy32 = dy.float()
        d_topk_weights = (dy32 @ bias_full.t()).gather(1, topk_idx)
        d_bias_full = probs.t() @ dy32
        d_bias = torch.empty(
            (d_bias_full.shape[0] // ctx.group.size(), d_bias_full.shape[1]),
            dtype=torch.float32,
            device=dy.device,
        )
        torch.distributed.reduce_scatter_tensor(d_bias, d_bias_full, group=ctx.group)
        # grads for (y, topk_idx, topk_weights, bias, group)
        return dy, None, d_topk_weights, d_bias.to(ctx.bias_dtype), None


# This op file exports only its own final API (the autograd Function + its wrapper). Everything else
# (the per-stage ``_mxfp8_*`` helpers, ``_DW_FP8_FORMAT``, and the ``prepare_w1t/w2t_dgrad_fp8``
# weight-prep) lives in the kernels layer -- callers import those from
# ``primus_turbo.pytorch.kernels.fused_mega_moe`` directly (weight prep lives in
# ``mega_moe_fp8_weights``, the per-stage helpers in the stage modules), not via
# this file.
__all__ = [
    "FusedMegaMoEFP8Stage1Function",
    "FusedMegaMoEFP8Stage2Function",
    "fused_mega_moe_fp8_stage1",
    "fused_mega_moe_fp8_stage2",
]


class _Fp8StageState:
    """Side channel carrying non-differentiable fp8 operands between stage1 and stage2.

    The bf16 split threads everything through op args/returns, but the fp8 backward cannot: its
    fused SwiGLU^T emits grad_l1 ONLY as the two quantized operands the L1 dgrad and dW1 consume
    ``((q_row, a_sp), (q_col, s_col))``, uint8/int32 tensors of a different shape than ``l1``, which
    no gradient slot accepts. So stage2.backward parks them here, returns ``None`` on the ``l1``
    slot, and stage1.backward picks them up -- safe because autograd's topological order runs
    stage2.backward strictly first. ``grad_gate`` still rides the ``dispatch_weights`` gradient slot,
    the same channel bf16 uses.

    The forward half (``pool_x_colwise`` / ``colwise_meta``, produced by stage1's pool requant) rides
    along because stage2's dual-quant and dW2 need the same grouped meta.

    One instance per stage1 call; both stages hold it on their ``ctx`` until their backward is done.
    """

    __slots__ = (
        "pool_x_colwise",
        "colwise_meta",
        "grad_l1_rowwise_fp8",
        "grad_l1_colwise_fp8",
    )

    def __init__(self):
        self.pool_x_colwise = None
        self.colwise_meta = None
        self.grad_l1_rowwise_fp8 = None
        self.grad_l1_colwise_fp8 = None


# Each backward drops the per-layer state it consumed instead of letting the graph hold it until the
# whole step's backward is over, so the same forward cannot be backpropagated twice.
_SECOND_BACKWARD = (
    "fused_mega_moe_fp8: this forward was already backpropagated; retain_graph is not supported"
)


class FusedMegaMoEFP8Stage1Function(torch.autograd.Function):
    """Stage1 gate-up (MXFP8): dispatch + fc1, owns w1. Output is pre-SwiGLU ``l1``."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        w1: torch.Tensor,
        group: ProcessGroup,
        state: _Fp8StageState,
    ):
        with torch.profiler.record_function("mega_moe_fp8_stage1_forward"):
            assert x.dim() == 2 and x.is_cuda and x.dtype == torch.bfloat16, (
                f"x must be 2D bf16 CUDA, got {tuple(x.shape)}/{x.dtype}"
            )
            assert w1.is_cuda and w1.dim() == 3 and w1.dtype == torch.bfloat16, (
                "w1 must be a 3D bf16 CUDA tensor"
            )

            # int64 end-to-end (combine reads topk i64)
            topk_idx = topk_idx.to(torch.int64)
            ctx.set_materialize_grads(False)
            save_bwd = any(ctx.needs_input_grad)
            hidden, inter = w1.shape[2], w1.shape[1] // 2
            padded_hidden, _ = fp8_padded_dims(hidden, inter)

            (
                l1,
                dispatch_weights,
                handle,
                pool_x_colwise,
                colwise_meta,
            ) = fused_mega_moe_stage1_forward_fp8_impl(
                _pad_cols(x, padded_hidden),
                w1,
                group,
                topk_idx,
                topk_weights,
                save_bwd=save_bwd,
            )
            state.pool_x_colwise = pool_x_colwise
            state.colwise_meta = colwise_meta

            if save_bwd:
                # Unlike bf16, dW1 contracts over the pool requant stage1 already produced, so x
                # is NOT saved -- only w1 is. Non-tensor / non-graph state goes on ctx.
                ctx.group = group
                ctx.state = state
                ctx.handle = handle
                ctx.topk_idx = topk_idx
                ctx.num_tokens = x.shape[0]
                ctx.num_topk = topk_idx.shape[-1]
                ctx.hidden, ctx.inter = hidden, inter
                ctx.save_for_backward(w1)
            # handle tensors are non-differentiable index/table tensors
            ctx.mark_non_differentiable(*handle)
            return (l1, dispatch_weights, *handle)

    @staticmethod
    @torch.no_grad()
    def backward(
        ctx, grad_l1: Optional[torch.Tensor], grad_dispatch_weights: Optional[torch.Tensor], *grad_handle
    ):
        """dx / grad_topk_weights + dW1. ``grad_l1`` is always None -- the real grad_l1 arrives
        pre-quantized through ``ctx.state``; ``grad_dispatch_weights`` couriers ``grad_gate``."""
        with torch.profiler.record_function("mega_moe_fp8_stage1_backward"):
            state, handle = ctx.state, ctx.handle
            if state is None:
                raise RuntimeError(_SECOND_BACKWARD)
            ctx.state = ctx.handle = None
            grad_l1_rowwise_fp8, grad_l1_colwise_fp8 = state.grad_l1_rowwise_fp8, state.grad_l1_colwise_fp8
            pool_x_colwise, colwise_meta = state.pool_x_colwise, state.colwise_meta
            state.grad_l1_rowwise_fp8 = state.grad_l1_colwise_fp8 = None
            state.pool_x_colwise = state.colwise_meta = None
            if grad_l1_rowwise_fp8 is None:  # stage2.backward never ran -> nothing to do
                return (None,) * 6
            (w1,) = ctx.saved_tensors

            dx, grad_topk_weights, dW1 = fused_mega_moe_stage1_backward_fp8_impl(
                grad_l1_rowwise_fp8,
                grad_l1_colwise_fp8,
                grad_dispatch_weights,
                pool_x_colwise,
                colwise_meta,
                w1,
                handle,
                ctx.group,
                ctx.topk_idx,
                ctx.num_tokens,
                ctx.num_topk,
            )

            dx = _slice_cols(dx, ctx.hidden)
            dW1 = unpad_w1_grad(dW1, ctx.hidden, ctx.inter)
            # grads for (x, topk_idx, topk_weights, w1, group, state)
            return dx, None, grad_topk_weights, dW1.to(w1.dtype), None, None


class FusedMegaMoEFP8Stage2Function(torch.autograd.Function):
    """Stage2 gate-down (MXFP8): SwiGLU + fc2 + combine, owns w2. Output is the MoE result y."""

    @staticmethod
    def forward(
        ctx,
        l1: torch.Tensor,
        dispatch_weights: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        w2: torch.Tensor,
        fc1_bias: Optional[torch.Tensor],
        group: ProcessGroup,
        state: _Fp8StageState,
        activation: Optional[GLUActivation],
        *handle,
    ) -> torch.Tensor:
        with torch.profiler.record_function("mega_moe_fp8_stage2_forward"):
            assert w2.is_cuda and w2.dim() == 3 and w2.dtype == torch.bfloat16, (
                "w2 must be a 3D bf16 CUDA tensor"
            )
            handle = tuple(handle)
            hidden, inter = w2.shape[1], w2.shape[2]
            fc1_bias_run = None if fc1_bias is None else pad_b1_fp8_dims(fc1_bias, l1.shape[1] // 2)

            y = fused_mega_moe_stage2_forward_fp8_impl(
                l1,
                w2,
                handle,
                group,
                topk_idx,
                topk_weights,
                activation,
                fc1_bias_run,
            )
            y = _slice_cols(y, hidden)

            ctx.set_materialize_grads(False)
            if any(ctx.needs_input_grad):
                ctx.group = group
                ctx.state = state
                ctx.activation = activation
                ctx.handle = handle
                ctx.hidden, ctx.inter = hidden, inter
                # dispatch_weights is unused in forward; saved only as the SwiGLU^T scale in backward
                ctx.save_for_backward(l1, dispatch_weights, w2, fc1_bias_run)
            return y

    @staticmethod
    @torch.no_grad()
    def backward(ctx, grad_y: Optional[torch.Tensor]):
        """dW2 + grad_gate (on the ``dispatch_weights`` slot). The quantized grad_l1 pair goes to
        stage1 through ``ctx.state``, so the ``l1`` slot returns None."""
        with torch.profiler.record_function("mega_moe_fp8_stage2_backward"):
            state, handle = ctx.state, ctx.handle
            if state is None:
                raise RuntimeError(_SECOND_BACKWARD)
            # stage1's ctx still holds both for its own backward
            ctx.state = ctx.handle = None
            n_in = 9 + len(handle)
            if grad_y is None:
                return (None,) * n_in
            l1, dispatch_weights, w2, fc1_bias_run = ctx.saved_tensors

            (
                grad_l1_rowwise_fp8,
                grad_l1_colwise_fp8,
                grad_gate,
                dW2,
                d_fc1_bias,
            ) = fused_mega_moe_stage2_backward_fp8_impl(
                _pad_cols(grad_y, fp8_padded_dims(ctx.hidden, ctx.inter)[0]),
                l1,
                dispatch_weights,
                w2,
                handle,
                ctx.group,
                state.colwise_meta,
                ctx.activation,
                fc1_bias_run,
            )
            state.grad_l1_rowwise_fp8 = grad_l1_rowwise_fp8
            state.grad_l1_colwise_fp8 = grad_l1_colwise_fp8
            dW2 = unpad_w2_grad(dW2, ctx.hidden, ctx.inter)
            if d_fc1_bias is not None:
                d_fc1_bias = unpad_b1_grad(d_fc1_bias, ctx.inter)

            # grads for (l1, dispatch_weights, topk_idx, topk_weights, w2, fc1_bias, group, state,
            # activation, *handle)
            return (
                None,
                grad_gate,
                None,
                None,
                dW2.to(w2.dtype),
                d_fc1_bias,
                None,
                None,
                None,
                *((None,) * len(handle)),
            )


def fused_mega_moe_fp8_stage1(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    w1: torch.Tensor,
    group: ProcessGroup,
):
    """Stage1 gate-up (MXFP8). Returns ``(l1, dispatch_weights, handle, state)`` to feed stage2.

    ``dispatch_weights``, ``handle`` and ``state`` are opaque forward state; pass them straight into
    :func:`fused_mega_moe_fp8_stage2`. Pass ``w1`` directly -- its mxfp8 quant is maintained
    internally by a version-keyed cache.

    After the dispatch prologue the forward reads the real row counts back once (a D2H, like
    DeepEP's notify), so every tensor indexed by dispatched row -- ``l1``, the dW1 operand, the
    handle's row maps and the per-call intermediates of both stages, forward and backward -- is
    allocated at this rank's real rows. That sync also keeps the op out of CUDA graphs.
    """
    state = _Fp8StageState()
    l1, dispatch_weights, *handle = FusedMegaMoEFP8Stage1Function.apply(
        x,
        topk_idx,
        topk_weights,
        w1,
        group,
        state,
    )
    return l1, dispatch_weights, tuple(handle), state


def fused_mega_moe_fp8_stage2(
    l1: torch.Tensor,
    dispatch_weights: torch.Tensor,
    handle: tuple,
    state: _Fp8StageState,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    w2: torch.Tensor,
    group: ProcessGroup,
    activation: Optional[GLUActivation] = None,
    fc1_bias: Optional[torch.Tensor] = None,
    fc2_bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Stage2 gate-down (MXFP8). Consumes stage1's forward state; returns y.

    Stage2 owns the gated activation, so ``activation`` (``None`` = SiLU-SwiGLU) is given here only.
    The optional expert biases follow Megatron's TEGroupedMLP, as in the bf16 ``fused_mega_moe_stage2``:
    ``fc1_bias`` [E_local, 2I] before the activation, ``fc2_bias`` [E_local, H] scaled by the routing
    weight, i.e. ``y = sum_k p_k (act_k @ w2^T + fc2_bias)``.
    """
    y = FusedMegaMoEFP8Stage2Function.apply(
        l1,
        dispatch_weights,
        topk_idx,
        topk_weights,
        w2,
        fc1_bias,
        group,
        state,
        activation,
        *handle,
    )
    if fc2_bias is not None:
        y = _GatheredExpertBiasFunction.apply(y, topk_idx.to(torch.int64), topk_weights, fc2_bias, group)
    return y
