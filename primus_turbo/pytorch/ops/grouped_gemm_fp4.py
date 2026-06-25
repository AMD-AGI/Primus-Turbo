###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""MXFP4 grouped GEMM (MX_BLOCKWISE, Triton backend, gfx950).

Mirrors :class:`FP8GroupedGemmMXFunc` but for E2M1 FP4. The MXFP4 training
recipe (https://arxiv.org/pdf/2509.25149) keeps the forward operands plain MX
and applies a 16-point Random Hadamard Transform (RHT) to the backward operands
(grad_out + the cached col-wise transposes). Because the Hadamard is orthogonal
it cancels inside each GEMM **only when both contracted operands share it**, so
the recipes are paired exactly like the dense ``gemm_fp4``:

    fwd   : C    = A_row(rht=F)         @ B_row(rht=F)^T          (contract K)
    dgrad : dA   = gradO_row(rht=T)     @ B_col(rht=T)^T          (contract N)
    wgrad : dB   = gradO_col(rht=T)     @ A_col(rht=T)^T          (contract M_g)

Quantization reuses the dense ``quantize_fp4`` kernel (which already implements
RHT / SR / 2D-block / fp4 packing) composed per direction:
  * row-wise operands (fwd A, dgrad grad_out) quantize the whole activation —
    K-blocks never cross per-group M boundaries, so no padding is needed;
  * col-wise operands (wgrad A / grad_out) need per-group M zero-padding to 128
    so the variable-K reduction is block-aligned and never mixes groups. We
    build a 128-aligned padded activation and col-quantize it once (32-blocks and
    16-pt RHT both divide 128, so no cross-group contamination).

A future fused C++ ``grouped_quantize_mxfp4_dual`` can replace the Python
composition without touching the GEMM kernels.
"""

from typing import List, Tuple, Union

import torch

from primus_turbo.pytorch.core.low_precision import (
    MXFP4_BLOCK_SIZE,
    Float4QuantConfig,
    Format,
    ScalingGranularity,
    ScalingRecipe,
    check_mxfp4_support,
    float4_e2m1fn_x2,
)
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_fp4_impl import (
    grouped_gemm_fp4_impl,
    grouped_gemm_fp4_variable_k_impl,
)
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_utils import (
    group_offs_from_lens,
)
from primus_turbo.pytorch.ops.quantization import quantize_fp4

__all__ = ["grouped_gemm_fp4"]

_PAD = 128  # per-group M padding for the variable-K wgrad (BLOCK_SIZE_K).


def _get_fp4_dtype(format: Format):
    if format == Format.E2M1_X2:
        return float4_e2m1fn_x2
    raise ValueError(f"Unsupported FP4 format: {format}")


def _q_row(x: torch.Tensor, rht: bool, sr: bool, two_d: bool):
    """Row-wise (axis=1) MXFP4 quant -> (packed [M, K/2], e8m0 scale [M, K/32])."""
    return quantize_fp4(
        x,
        float4_e2m1fn_x2,
        ScalingGranularity.MX_BLOCKWISE,
        block_size=MXFP4_BLOCK_SIZE,
        axis=1,
        scaling_recipe=ScalingRecipe(use_2d_block=two_d, use_sr=sr, use_rht=rht),
    )


def _q_col(x: torch.Tensor, rht: bool, sr: bool, two_d: bool):
    """Col-wise (axis=0) MXFP4 quant -> (packed [K, M_pad/2], e8m0 scale [K, M_pad/32])."""
    return quantize_fp4(
        x,
        float4_e2m1fn_x2,
        ScalingGranularity.MX_BLOCKWISE,
        block_size=MXFP4_BLOCK_SIZE,
        axis=0,
        scaling_recipe=ScalingRecipe(use_2d_block=two_d, use_sr=sr, use_rht=rht),
    )


def _padded_offs(group_lens_host: List[int], device) -> Tuple[torch.Tensor, int, List[int]]:
    """Per-group offsets after padding each group's M up to a multiple of _PAD."""
    padded = [((gl + _PAD - 1) // _PAD) * _PAD for gl in group_lens_host]
    offs = [0]
    for p in padded:
        offs.append(offs[-1] + p)
    return torch.tensor(offs, dtype=torch.int64, device=device), offs[-1], offs


def _pad_rows(
    x: torch.Tensor, group_offs_host: List[int], padded_offs_host: List[int], total_pad: int
) -> torch.Tensor:
    """Scatter each per-group row-block of ``x`` into a 128-aligned zero buffer."""
    if total_pad == x.shape[0] and group_offs_host == padded_offs_host:
        return x  # already 128-aligned per group (balanced) -> no copy
    out = x.new_zeros((total_pad, x.shape[1]))
    for g in range(len(group_offs_host) - 1):
        s, e = group_offs_host[g], group_offs_host[g + 1]
        ps = padded_offs_host[g]
        out[ps : ps + (e - s)] = x[s:e]
    return out


def _quant_weight_dual(b: torch.Tensor):
    """Per-group dual MXFP4 quant of the 3D weight ``b`` (G, N, K).

    Returns (b_row, b_row_scale, b_col, b_col_scale):
      b_row      (G, N, K/2)  rht=False  -> fwd B operand
      b_col      (G, K, N/2)  rht=True   -> dgrad B operand (col-wise of [N, K])
    """
    G = b.shape[0]
    rows, row_s, cols, col_s = [], [], [], []
    for g in range(G):
        bg = b[g].contiguous()
        br, brs = _q_row(bg, rht=False, sr=False, two_d=True)
        bc, bcs = _q_col(bg, rht=True, sr=False, two_d=True)
        rows.append(br)
        row_s.append(brs)
        cols.append(bc)
        col_s.append(bcs)
    return torch.stack(rows), torch.stack(row_s), torch.stack(cols), torch.stack(col_s)


class FP4GroupedGemmMXFunc(torch.autograd.Function):
    """MXFP4 grouped GEMM autograd (MX_BLOCKWISE, NT-only, Triton backend)."""

    @staticmethod
    def forward(ctx, a, b, group_lens, group_offs, trans_b, out_dtype, config, num_cu):
        assert config.granularity == ScalingGranularity.MX_BLOCKWISE
        assert a.ndim == 2 and b.ndim == 3
        assert out_dtype in (torch.float16, torch.bfloat16)
        assert trans_b, "MXFP4 grouped GEMM only supports trans_b=True (NT layout)."
        assert not config.use_preshuffle, "Triton MXFP4 grouped GEMM does not use preshuffle."
        supported, reason = check_mxfp4_support()
        assert supported, reason

        N, K = int(b.shape[-2]), int(b.shape[-1])
        assert K % _PAD == 0, f"K must be a multiple of {_PAD} (got {K})."
        assert N % _PAD == 0, f"N must be a multiple of {_PAD} (got {N})."
        total_m = int(a.size(0))
        G = int(b.shape[0])

        group_lens_host = group_lens.tolist()
        group_offs_host = group_offs.tolist()
        padded_offs, total_pad, padded_offs_host = _padded_offs(group_lens_host, a.device)

        # --- forward: A_row(rht=F) @ B_row(rht=F)^T, contract K ---
        a_row, a_row_s = _q_row(a, rht=False, sr=False, two_d=False)
        b_row, b_row_s, b_col, b_col_s = _quant_weight_dual(b)

        out = grouped_gemm_fp4_impl(
            a_row, b_row, a_row_s, b_row_s, group_offs, N, K, num_cu, out_dtype, group_offs
        )

        # --- cache the col-wise (rht=T) A transpose for wgrad (per-group 128-padded) ---
        a_pad = _pad_rows(a, group_offs_host, padded_offs_host, total_pad)
        a_col, a_col_s = _q_col(a_pad, rht=True, sr=False, two_d=False)

        ctx.save_for_backward(a_col, a_col_s, b_col, b_col_s, group_offs, padded_offs)
        ctx.N, ctx.K, ctx.G = N, K, G
        ctx.total_m, ctx.total_pad = total_m, total_pad
        ctx.config = config
        ctx.out_dtype = out_dtype
        ctx.num_cu = num_cu
        ctx.group_offs_host = group_offs_host
        ctx.padded_offs_host = padded_offs_host
        return out

    @staticmethod
    def backward(ctx, grad_out):
        a_col, a_col_s, b_col, b_col_s, group_offs, padded_offs = ctx.saved_tensors
        N, K, G = ctx.N, ctx.K, ctx.G
        out_dtype, num_cu = ctx.out_dtype, ctx.num_cu
        sr = ctx.config.use_gradient_sr

        grad_out = grad_out.contiguous().view(ctx.total_m, N)

        # --- dgrad: grad_a = gradO_row(rht=T) @ B_col(rht=T)^T, contract N -> [total_m, K] ---
        go_row, go_row_s = _q_row(grad_out, rht=True, sr=sr, two_d=False)
        grad_a = grouped_gemm_fp4_impl(
            go_row, b_col, go_row_s, b_col_s, group_offs, K, N, num_cu, out_dtype, group_offs
        )

        # --- wgrad: grad_b[g] = gradO_col(rht=T) @ A_col(rht=T)^T, contract M_g -> [G, N, K] ---
        go_pad = _pad_rows(grad_out, ctx.group_offs_host, ctx.padded_offs_host, ctx.total_pad)
        go_col, go_col_s = _q_col(go_pad, rht=True, sr=sr, two_d=False)
        grad_b = grouped_gemm_fp4_variable_k_impl(
            go_col, a_col, go_col_s, a_col_s, padded_offs, N, K, G, num_cu, out_dtype
        )

        # forward args: (a, b, group_lens, group_offs, trans_b, out_dtype, config, num_cu)
        return grad_a, grad_b, None, None, None, None, None, None


@torch._dynamo.disable(
    recursive=True,
    reason=(
        "Grouped MXFP4 GEMM composes the dense quantizer per direction and reads "
        "packed FP4 / E8M0 inner tensors inside its autograd.Function.forward; "
        "Dynamo cannot recover Python sources for those graph-internal tensors."
    ),
)
def grouped_gemm_fp4(
    a: torch.Tensor,
    b: torch.Tensor,
    group_lens: torch.Tensor,
    group_offs: Union[torch.Tensor, None] = None,
    trans_b: bool = True,
    out_dtype: Union[torch.dtype, None] = None,
    config: Union[Float4QuantConfig, None] = None,
    num_cu: int | None = None,
) -> torch.Tensor:
    """Grouped GEMM with MXFP4 (E2M1) quantization, supporting autograd.

    Args:
        a: Activation [total_m, K] (float16 / bfloat16), grouped along M.
        b: Weight [G, N, K] (trans_b=True, NT layout only).
        group_lens: Per-group row counts [G] (int64).
        group_offs: Optional per-group offsets [G+1]; derived from group_lens if None.
        trans_b: Must be True (NT).
        out_dtype: Output dtype (defaults to promote(a, b)).
        config: Float4QuantConfig (MX_BLOCKWISE). use_preshuffle must be False.
        num_cu: Optional cap on compute units for the persistent grid.

    Returns:
        Output [total_m, N].
    """
    if config is None:
        config = Float4QuantConfig()
    if group_offs is None:
        group_offs = group_offs_from_lens(group_lens)
    if out_dtype is None:
        out_dtype = torch.promote_types(a.dtype, b.dtype)

    assert a.ndim == 2, "a must be 2D [total_m, K]"
    assert b.ndim == 3, "b must be 3D [G, N, K]"

    if config.granularity == ScalingGranularity.MX_BLOCKWISE:
        return FP4GroupedGemmMXFunc.apply(a, b, group_lens, group_offs, trans_b, out_dtype, config, num_cu)
    raise ValueError(f"Unsupported FP4 ScalingGranularity: {config.granularity}")
