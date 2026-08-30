###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Autograd-aware MXFP6 (E2M3) GEMM.

Mirrors :mod:`primus_turbo.pytorch.ops.gemm_fp4` closely enough to be a drop-in at the
Megatron call site (``gemm_fp6(x, weight, trans_a=False, trans_b=True, ...)``), but the
MXFP6 packed layout removes most of FP4's knobs:

- No ``preshuffle``. The A6W6 kernels read the packed C0/C1 tile blob directly, so there
  is no un-shuffled layout to opt out of -- the layout is the format.
- No ``ScalingRecipe``. The 32-point Hadamard rotation is mandatory and fused into the
  packer (the GEMM depends on it cancelling between operands), scaling is strictly
  per-1x32 along the contraction axis so a 2D block is meaningless, and stochastic
  rounding is not implemented.
- No ``QuantizedTensor`` fast path yet, so a pre-quantized weight is not reused across
  microbatches the way FP4 allows. Because MXFP6 blobs are opaque and carry no shape,
  that needs the ``PackedQuantizedTensor`` wrapper before it can be wired up.

The three GEMM directions contract over different dimensions, so each operand is needed
packed along two different axes. For ``out = a @ b.T`` with ``a[M, K]``, ``b[N, K]``:

    forward   out    = a @ b.T        contract K  ->  row(a),  row(b)
    backward  grad_a = grad_out @ b   contract N  ->  row(g),  col(b)
    backward  grad_b = grad_out.T @ a contract M  ->  col(g),  col(a)

so forward packs both directions of both operands and saves the column ones, exactly as
the FP4 function saves ``a_col`` / ``b_col``.
"""

from typing import Union

import torch

from primus_turbo.pytorch.core.low_precision import (
    MXFP6_TILE_SIZE,
    Float6QuantConfig,
    ScalingGranularity,
)
from primus_turbo.pytorch.kernels.gemm.gemm_fp6_impl import gemm_fp6_impl
from primus_turbo.pytorch.kernels.quantization.mxfp6_pack import (
    check_mxfp6_support,
    quantize_mxfp6_dual,
)

__all__ = ["FP6GemmMXFunction", "gemm_fp6"]


class FP6GemmMXFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        a: torch.Tensor,
        b: torch.Tensor,
        out_dtype: torch.dtype,
        config: Float6QuantConfig,
    ):
        supported, reason = check_mxfp6_support(a.device)
        if not supported:
            raise RuntimeError(reason)

        m, k = a.shape
        n = b.shape[0]

        a_row, a_row_scale, a_col, a_col_scale = quantize_mxfp6_dual(a, config.block_size)
        b_row, b_row_scale, b_col, b_col_scale = quantize_mxfp6_dual(b, config.block_size)

        out = gemm_fp6_impl(
            a_row,
            a_row_scale,
            b_row,
            b_row_scale,
            m,
            n,
            k,
            out_dtype,
            config.granularity.value,
        )

        ctx.save_for_backward(a_col, a_col_scale, b_col, b_col_scale)
        # The blobs carry no shape, so the logical dims have to travel separately.
        ctx.mnk = (m, n, k)
        ctx.out_dtype = out_dtype
        ctx.config = config
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        a_col, a_col_scale, b_col, b_col_scale = ctx.saved_tensors
        m, n, k = ctx.mnk
        config = ctx.config
        granularity = config.granularity.value

        grad_out = grad_out.view(grad_out.shape[0], -1).contiguous()
        g_row, g_row_scale, g_col, g_col_scale = quantize_mxfp6_dual(grad_out, config.block_size)

        # grad_a[M, K] = grad_out[M, N] @ b[N, K], contracting N.
        # b_col is b packed along N, i.e. logically [K, N] contracting N.
        grad_a = gemm_fp6_impl(g_row, g_row_scale, b_col, b_col_scale, m, k, n, ctx.out_dtype, granularity)

        # grad_b[N, K] = grad_out.T[N, M] @ a[M, K], contracting M.
        grad_b = gemm_fp6_impl(g_col, g_col_scale, a_col, a_col_scale, n, k, m, ctx.out_dtype, granularity)

        return (
            grad_a,  # a
            grad_b,  # b
            None,  # out_dtype
            None,  # config
        )


def gemm_fp6(
    a: torch.Tensor,
    b: torch.Tensor,
    trans_a: bool = False,
    trans_b: bool = True,
    out_dtype: Union[torch.dtype, None] = None,
    config: Union[Float6QuantConfig, None] = None,
    fuse_bgrad_accum_pattern: Union[None, str] = None,
) -> torch.Tensor:
    """GEMM with MXFP6 (E2M3) quantization, supporting autograd.

    Computes ``a @ b.T``. Both operands are quantized to MXFP6 in both contraction
    directions on the way in; backward reuses the saved column-direction blobs.

    Args:
        a: Activation, shape (M, K), 2D.
        b: Weight, shape (N, K), 2D.
        trans_a: Must be False. MXFP6 fixes the contraction axis at pack time, so the
            layout is not selectable here.
        trans_b: Must be True, for the same reason.
        out_dtype: Output dtype. Only bf16 is supported (the A6W6 asm writes bf16), and
            None resolves to bf16.
        config: MXFP6 quantization config.
        fuse_bgrad_accum_pattern: Not supported yet; must be None.

    Returns:
        Output matrix of shape (M, N) in bf16.

    Example::

        >>> a = torch.randn(4096, 3072, device="cuda", dtype=torch.bfloat16)
        >>> b = torch.randn(3072, 3072, device="cuda", dtype=torch.bfloat16)
        >>> out = gemm_fp6(a, b)
    """
    if config is None:
        config = Float6QuantConfig()

    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"Only 2D tensors are supported, got a={a.ndim}D and b={b.ndim}D")
    if trans_a or not trans_b:
        raise ValueError(
            "MXFP6 only supports the NT layout (trans_a=False, trans_b=True): the packed "
            f"layout fixes the contraction axis when quantizing, not at GEMM time. Got "
            f"trans_a={trans_a}, trans_b={trans_b}."
        )
    if a.device != b.device:
        raise ValueError(
            f"MXFP6 GEMM operands must share one device, got a on {a.device} and b on {b.device}."
        )
    if a.dtype != b.dtype:
        raise TypeError(f"MXFP6 GEMM operands must share one dtype, got a={a.dtype} and b={b.dtype}.")
    if a.shape[1] != b.shape[1]:
        raise ValueError(
            f"K mismatch: a is {tuple(a.shape)} and b is {tuple(b.shape)}; with trans_b=True "
            "both must share their last dimension."
        )

    # Every one of M, N and K has to be a multiple of 256, which is stricter than the
    # single-GEMM rule (M/N % 256, K % 128) that GEMMFP6AITERBackend.can_handle applies.
    # The reason is that the backward GEMMs permute the roles: dgrad runs (M, K, N) and
    # wgrad runs (N, K, M), so K becomes an output dimension and inherits the 256
    # constraint. Checking it here rather than in the backend turns what would be a
    # mid-backward failure into an error at the call site.
    m, k = a.shape
    n = b.shape[0]
    misaligned = {name: dim for name, dim in (("M", m), ("N", n), ("K", k)) if dim % MXFP6_TILE_SIZE}
    if misaligned:
        raise ValueError(
            f"MXFP6 training requires M, N and K to be multiples of {MXFP6_TILE_SIZE}, but "
            f"{', '.join(f'{n_}={d}' for n_, d in misaligned.items())} "
            f"{'is' if len(misaligned) == 1 else 'are'} not. K is included because the "
            f"backward GEMMs use it as an output dimension (dgrad computes an M x K result "
            f"and wgrad an N x K one)."
        )
    if fuse_bgrad_accum_pattern is not None:
        raise NotImplementedError(
            "fuse_bgrad_accum_pattern is not supported for MXFP6 yet, got "
            f"{fuse_bgrad_accum_pattern!r}. The A6W6 entry point has no beta=1 accumulate "
            "epilogue, so wgrad cannot write main_grad in place."
        )

    if out_dtype is None:
        out_dtype = torch.bfloat16
    if out_dtype != torch.bfloat16:
        raise ValueError(f"MXFP6 only supports a bf16 output, got {out_dtype}.")

    if config.granularity != ScalingGranularity.MX_BLOCKWISE:
        raise ValueError(f"Unsupported MXFP6 ScalingGranularity: {config.granularity}")

    return FP6GemmMXFunction.apply(a, b, out_dtype, config)
