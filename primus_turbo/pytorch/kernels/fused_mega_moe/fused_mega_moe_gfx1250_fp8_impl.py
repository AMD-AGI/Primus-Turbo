###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The Mega MoE stages in MXFP8 on gfx1250.

The bf16 Mega's protocol (``fused_mega_moe_gfx1250_impl``: the dispatch prologue, the per-expert dispatch flags,
the combine and reduce) with E4M3 rows + raw E8M0 scale rows on the wire, quantized once on the token's rank, and
the stock gfx1250 MX kernels in between: the quantizers (``mxfp8_quant_gfx1250``) and the grouped / variable-K
GEMMs (``gemm_mxfp8_kernel_gfx1250``) of the routed-expert grouped path. A dispatched row is gathered into pool
order with its codes and scales kept and its column copy re-quantized (``grouped_gather_fp8_quant_mxfp8``). The
pool's padding rows are zero, so one offsets table, the prologue's 256-aligned ``num_tokens_per_expert_prefix``,
serves every grouped kernel, and a GEMM row comes out as the grouped path computes it from the same operands.
The combine stays bf16. Dispatch and GEMM run one after the other: the transfer does not overlap the GEMM at EP > 1.

Forward: send x, gather + column copy (dW1's, saved), fc1, GLU with the routing weight + quant (fc2's rows, dW2's
columns), fc2, combine. Backward: send dy, gather + column copy (dW2's), fc2 dgrad, GLU backward + quant + the
routing-weight gradient, dW2, fc1 dgrad + combine, dW1; a wgrad adds into the given ``main_grad`` (beta = 1).
"""

import math
from typing import Optional

import torch

from primus_turbo.flydsl.gemm.gemm_mxfp8_kernel_gfx1250 import (
    grouped_gemm_mxfp8_gfx1250,
    grouped_gemm_mxfp8_variable_k_gfx1250,
)
from primus_turbo.flydsl.mega.gfx1250.bf16.dispatch_gemm_kernel import dispatch_only_gfx1250
from primus_turbo.flydsl.mega.gfx1250.bf16.gemm_combine_kernel import combine_only_gfx1250
from primus_turbo.flydsl.mega.gfx1250.dispatch_prologue_kernel import (
    DispatchHandle,
    run_dispatch_prologue,
)
from primus_turbo.flydsl.mega.gfx1250.symm_buffer import (
    TOKEN_ALIGNMENT,
    align,
    check_combine_capacity,
)
from primus_turbo.flydsl.quantization import mxfp8_quant_gfx1250 as Q
from primus_turbo.flydsl.utils.glu_activation import GLUActivation
from primus_turbo.pytorch.kernels.fused_mega_moe.mega_moe_fp8_weights import _version_keyed_weight_prep

# The quantizer's ``layout`` for a column copy: row-major codes as a wgrad's A operand (MX_A), 16x16 blocks as its
# B operand (MX_B). Row copies are A operands.
_COLS_A, _COLS_B = 0, Q.SHUF_COL


# local experts the dispatch-only launch's waiting workgroup has a lane for (``dispatch_gemm_kernel``)
_MAX_EXPERTS_PER_RANK = 256


def check(w1: torch.Tensor, *activation) -> None:
    """Raise, before any cross-rank work, what the stages would refuse later: an expert shape off the grid (whole
    16-byte E8M0 scale rows on the wire: hidden % 512; the GLU's packed row scales: intermediate % 128; there is no
    padding fallback on gfx1250), more than 256 experts on this rank and, when given, an activation ``glu_form``
    refuses."""
    experts, hidden = w1.shape[0], w1.shape[2]
    if hidden % 512 or w1.shape[1] % 256:
        raise ValueError(
            f"MXFP8 MegaMoE on gfx1250 needs hidden % 512 == 0 and intermediate % 128 == 0 (no padding fallback), "
            f"got w1 {tuple(w1.shape)}: hidden {hidden}, intermediate {w1.shape[1] / 2:g}"
        )
    if experts > _MAX_EXPERTS_PER_RANK:
        raise ValueError(
            f"MXFP8 MegaMoE on gfx1250 serves at most {_MAX_EXPERTS_PER_RANK} experts per rank, got {experts}"
        )
    if activation:
        glu_form(*activation)


def glu_form(activation) -> tuple:
    """``(activation, clamp_limit)`` of the quantizers' GLU for a Mega ``GLUActivation``: SiLU-SwiGLU, plain or with
    Megatron's clamp (``GLUActivation.swiglu``). Any other spec raises -- the stages compute the model's activation
    exactly or not at all -- and so does ``None``, the bf16 kernels' default (gate clamped on both sides)."""
    a = GLUActivation() if activation is None else GLUActivation(*activation)
    lo, hi, up = (None if v is None or math.isinf(v) else float(v) for v in a[2:])
    if a.alpha == 1.0 and a.glu_offset == 0.0 and lo is None and hi == up:
        if up is not None and up <= 0:
            raise ValueError(f"MXFP8 MegaMoE on gfx1250: the SwiGLU clamp limit must be positive, got {up}")
        return "silu", up
    raise NotImplementedError(
        f"MXFP8 MegaMoE on gfx1250 computes SiLU-SwiGLU, plain or with Megatron's clamp "
        f"(GLUActivation.swiglu(limit)); got {a}"
    )


def _weights(w: torch.Tensor) -> tuple:
    """``(W, its scales, Wᵀ, its scales)`` of an expert weight ``[E, N, K]`` in the GEMM layout (32x32 blocks, packed
    scales, 16x16 codes: the grouped path's ``MX_W``), refreshed once per optimizer step
    (``advance_weight_generation``)."""
    from primus_turbo.pytorch.core.low_precision import MXFP8_BLOCK_SIZE, ScalingGranularity
    from primus_turbo.pytorch.ops.quantization import quantize_fp8_with_trans
    from primus_turbo.pytorch.ops.utils import MX_W

    def prep(t):
        return quantize_fp8_with_trans(
            t,
            torch.float8_e4m3fn,
            ScalingGranularity.MX_BLOCKWISE,
            block_size=MXFP8_BLOCK_SIZE,
            scaling_recipe=MX_W,
            scaling_recipe_for_trans=MX_W,
        )

    return _version_keyed_weight_prep(w, "_mega_mx1250_prep", prep)


def _send(t: torch.Tensor, handle: DispatchHandle, cols_layout: int, cols: bool = True) -> tuple:
    """Quantize ``t [T, H]`` by rows, dispatch them to the experts' ranks and gather the rows that landed here
    into pool order: ``(rows, scales, columns, scales)`` (the columns None without ``cols``)."""
    q, s = Q.quant_mxfp8_rows(t.contiguous())
    rq, rs = dispatch_only_gfx1250(q, handle, x_scale=s)
    out = Q.grouped_gather_fp8_quant_mxfp8(
        rq,
        rs,
        handle.pool_row_to_recv_token,
        handle.num_tokens_per_expert,
        handle.num_tokens_per_expert_prefix,
        total_m=handle.pool_row_to_route.numel(),
        cols=cols,
        layout=cols_layout,
    )
    return out[:4]


def _grouped(a, sa, w, sw, handle: DispatchHandle) -> torch.Tensor:
    """``out[p] = a[p] @ w[e(p)]^T`` over the pool rows (bf16): expert rows, A rows and output rows all at the pool's
    offsets (whole 256-row blocks, so 32-row aligned as the A layout needs)."""
    offs = handle.num_tokens_per_expert_prefix
    return grouped_gemm_mxfp8_gfx1250(a, sa, w, sw, offs, offs, handle.pool_row_to_route.numel())


def _wgrad(a, b, handle: DispatchHandle, out: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """``dW[e] = a[:, T_e] @ b[:, T_e]^T`` over the column copies (expert e's pool tokens; padding columns are 0),
    added into ``out`` when given (and None returned)."""
    dw = grouped_gemm_mxfp8_variable_k_gfx1250(
        a[0], a[1], b[0], b[1], handle.num_tokens_per_expert_prefix, out=out, accumulate=out is not None
    )
    return None if out is not None else dw


def stage1_forward(x, w1, group, topk_idx, topk_weights, save_bwd: bool = True):
    """dispatch + fc1 -> ``(l1 [P, 2I] bf16, dispatch_weights [P] f32, handle, x_col, None)``, the gfx950 fp8
    stage1's contract; ``x_col`` (the pool's column copy and its scales, dW1's operand) None without ``save_bwd``."""
    check(w1)
    if x.dtype != torch.bfloat16 or x.shape[1] != w1.shape[2]:
        raise ValueError(f"x must be bf16 [T, {w1.shape[2]}], got {x.dtype} {tuple(x.shape)}")
    recv_rows = group.size() * align(x.shape[0], TOKEN_ALIGNMENT)  # the receive buffer the pool gather reads
    if recv_rows * x.shape[1] >= 2**31:
        raise ValueError(
            f"MXFP8 MegaMoE on gfx1250 addresses the receive buffer with 32-bit offsets: ranks x align(tokens, "
            f"{TOKEN_ALIGNMENT}) x hidden must stay below 2**31, got {recv_rows} rows x {x.shape[1]}"
        )
    check_combine_capacity(x.shape[0], topk_idx.shape[-1], x.shape[1])
    handle, dispatch_weights = run_dispatch_prologue(x, w1, group, topk_idx, topk_weights)
    xr, sxr, xc, sxc = _send(x, handle, _COLS_B, cols=save_bwd)
    W1, sW1, _, _ = _weights(w1)
    l1 = _grouped(xr, sxr, W1, sW1, handle)
    return l1, dispatch_weights, handle, ((xc, sxc) if save_bwd else None), None


def stage2_forward(l1, w2, handle, topk_idx, dispatch_weights, activation, save_bwd: bool = True):
    """GLU (routing weight inside) + quant, fc2, combine -> ``(y [T, H], act_col)``, ``act_col`` (the activation's
    column copy and its scales) dW2's operand, None (and not built) without ``save_bwd``."""
    act, clamp = glu_form(activation)
    handle = DispatchHandle(*handle)
    ar, sar, ac, sac, *_ = Q.grouped_glu_quant_mxfp8(
        l1,
        dispatch_weights,
        handle.num_tokens_per_expert,
        handle.num_tokens_per_expert_prefix,
        act,
        clamp,
        layout=_COLS_B,
        cols=save_bwd,
    )
    W2, sW2, _, _ = _weights(w2)
    y = combine_only_gfx1250(_grouped(ar, sar, W2, sW2, handle), handle, topk_idx)
    return y, ((ac, sac) if save_bwd else None)


def stage2_backward(grad_y, l1, dispatch_weights, w2, handle, act_col, activation, dw2_out=None):
    """dy dispatch + fc2 dgrad + GLU backward + quant + dW2 -> ``(grad_l1 rows, grad_l1 columns, grad_gate [P],
    dW2)``, each copy with its scales; dW2 None when added into ``dw2_out``, or when ``act_col`` is None (the
    forward kept no column copy: w2 needs no gradient), and then no dy column copy is built either."""
    act, clamp = glu_form(activation)
    handle = DispatchHandle(*handle)
    dw2 = act_col is not None
    dr, sdr, dc, sdc = _send(grad_y.to(torch.bfloat16), handle, _COLS_A, cols=dw2)
    _, _, W2t, sW2t = _weights(w2)
    grad_gate, gr, sgr, gc, sgc, *_ = Q.grouped_dglu_quant_mxfp8(
        _grouped(dr, sdr, W2t, sW2t, handle),
        l1,
        dispatch_weights,
        handle.num_tokens_per_expert,
        handle.num_tokens_per_expert_prefix,
        act,
        clamp,
        layout=_COLS_A,
    )
    return (gr, sgr), (gc, sgc), grad_gate, (_wgrad((dc, sdc), act_col, handle, dw2_out) if dw2 else None)


def stage1_backward(grad_l1_row, grad_l1_col, grad_gate, x_col, w1, handle, topk_idx, dw1_out=None):
    """fc1 dgrad + combine with the routing-weight gradient + dW1 -> ``(dx [T, H], grad_topk_weights [T, topk],
    dW1)``; dW1 None when added into ``dw1_out``."""
    handle = DispatchHandle(*handle)
    _, _, W1t, sW1t = _weights(w1)
    dx_pool = _grouped(grad_l1_row[0], grad_l1_row[1], W1t, sW1t, handle)
    dx, grad_topk_weights = combine_only_gfx1250(dx_pool, handle, topk_idx, grad_gate=grad_gate)
    return dx, grad_topk_weights, _wgrad(grad_l1_col, x_col, handle, dw1_out)
