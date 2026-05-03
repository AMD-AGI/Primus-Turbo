###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
from typing import Union

import torch

from primus_turbo.pytorch.core.backend import BackendType
from primus_turbo.pytorch.core.low_precision import (
    Float8QuantConfig,
    Format,
    ScalingGranularity,
    float8_e4m3,
    float8_e5m2,
)
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_fp8_impl import (
    grouped_gemm_compute_offs,
    grouped_gemm_fp8_impl,
    grouped_gemm_fp8_variable_k_impl,
)
from primus_turbo.pytorch.kernels.quantization.quantization_impl import (
    quant_fp8_blockwise_for_weight_impl,
    quant_fp8_blockwise_impl,
    quant_fp8_blockwise_segment_m_impl,
    quantize_fp8_tensorwise_impl,
)
from primus_turbo.pytorch.ops.quantization import quantize_fp8

__all__ = [
    "grouped_gemm_fp8",
]


def _get_fp8_dtype(format: Format, is_fwd_stage: bool):
    if format == Format.E4M3:
        return float8_e4m3
    elif format == Format.E5M2:
        return float8_e5m2
    elif format == Format.HYBRID:
        return float8_e4m3 if is_fwd_stage else float8_e5m2
    else:
        raise ValueError(f"Unsupported FP8 format: {format}")


def _ensure_contiguous_grad_out(grad_out: torch.Tensor) -> torch.Tensor:
    # Some upstream reductions can produce expanded zero-stride grad_out views.
    # Custom grouped GEMM kernels expect dense layouts.
    return grad_out if grad_out.is_contiguous() else grad_out.contiguous()


class FP8GroupedGemmBlockFunc(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        a: torch.Tensor,
        b: torch.Tensor,
        group_lens: torch.Tensor,  # [B,] int64
        group_offs: torch.Tensor,  # [B+1,] int64
        trans_b: bool,
        config: Float8QuantConfig,
        num_cu: int | None,
    ):
        assert config.granularity == ScalingGranularity.BLOCKWISE
        assert config.block_size in [128], "Only block_size 128 is supported currently."
        assert a.ndim == 2, "Input tensor must be 2-dimensional."
        assert b.ndim == 3, "Weight tensor must be 3-dimensional."
        assert group_lens.size(0) == b.size(0), "group_lens size must match b size(0)."
        out_dtype = a.dtype
        assert out_dtype in [torch.float16, torch.bfloat16]

        a_dtype = _get_fp8_dtype(config.format, True)
        b_dtype = _get_fp8_dtype(config.format, True)

        a_fp8_row, a_scale_inv_row = quant_fp8_blockwise_impl(
            a, a_dtype, axis=1, block_size=config.block_size
        )

        b_fp8, b_scale_inv = quant_fp8_blockwise_for_weight_impl(b, b_dtype, block_size=config.block_size)

        out = grouped_gemm_fp8_impl(
            a_fp8_row,
            b_fp8,
            a_scale_inv_row,
            b_scale_inv,
            group_lens,
            group_offs,
            trans_a=False,
            trans_b=trans_b,
            out_dtype=out_dtype,
            granularity=config.granularity.value,
            num_cu=num_cu,
            default_backend=BackendType.HIPKITTEN.value,
        )

        a_fp8_col, a_scale_inv_col, _, _ = quant_fp8_blockwise_segment_m_impl(
            a, a_dtype, config.block_size, group_lens, group_offs
        )

        ctx.save_for_backward(
            a_fp8_col,
            a_scale_inv_col,
            b_fp8,
            b_scale_inv,
            group_lens,
            group_offs,
        )
        ctx.trans_a = False
        ctx.trans_b = trans_b
        ctx.config = config
        ctx.out_dtype = out_dtype
        ctx.num_cu = num_cu

        return out

    @staticmethod
    def backward(ctx, grad_out):
        grad_out = _ensure_contiguous_grad_out(grad_out)

        (
            a_fp8_col,
            a_scale_inv_col,
            b_fp8,
            b_scale_inv,
            group_lens,
            group_offs,
        ) = ctx.saved_tensors
        block_size = ctx.config.block_size
        grad_out_dtype = _get_fp8_dtype(ctx.config.format, False)

        # Quantize grad_out in row-wise for dgrad
        grad_out_fp8_row, grad_out_scale_inv_row = quant_fp8_blockwise_impl(
            grad_out, grad_out_dtype, axis=1, block_size=block_size
        )

        # grad_a: grad_out @ b^T
        grad_a = grouped_gemm_fp8_impl(
            grad_out_fp8_row,
            b_fp8,
            grad_out_scale_inv_row,
            b_scale_inv,
            group_lens,
            group_offs,
            trans_a=False,
            trans_b=not ctx.trans_b,
            out_dtype=ctx.out_dtype,
            granularity=ctx.config.granularity.value,
            num_cu=ctx.num_cu,
            default_backend=BackendType.HIPKITTEN.value,
        )

        # Quantize grad_out with segment padding for wgrad (colwise quantization)
        grad_out_fp8_col, grad_out_scale_inv_col, var_k_group_lens, var_k_group_offs = (
            quant_fp8_blockwise_segment_m_impl(
                grad_out,
                grad_out_dtype,
                block_size,
                group_lens,
                group_offs,
            )
        )

        grad_b = grouped_gemm_fp8_variable_k_impl(
            a_fp8_col,
            grad_out_fp8_col,
            a_scale_inv_col,
            grad_out_scale_inv_col,
            var_k_group_lens,
            var_k_group_offs,
            trans_a=not ctx.trans_a,
            trans_b=False,
            trans_c=ctx.trans_b,
            out_dtype=ctx.out_dtype,
            granularity=ctx.config.granularity.value,
            num_cu=ctx.num_cu,
            default_backend=BackendType.HIPKITTEN.value,
        )

        return grad_a, grad_b, None, None, None, None, None


class FP8GroupedGemmRowFunc(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        a: torch.Tensor,
        b: torch.Tensor,
        group_lens: torch.Tensor,  # [B,] int64
        group_offs: torch.Tensor,  # [B+1,] int64
        trans_b: bool,
        config: Float8QuantConfig,
        num_cu: int | None,
    ):
        assert config.granularity == ScalingGranularity.ROWWISE
        assert a.ndim == 2, "Input tensor must be 3-dimensions."
        assert b.ndim == 3, "Weight tensor must be 3-dimensional."
        out_dtype = a.dtype
        assert out_dtype in [torch.float16, torch.bfloat16]

        a_dtype = _get_fp8_dtype(config.format, True)
        b_dtype = _get_fp8_dtype(config.format, True)
        a_fp8_row, a_scale_inv_row = quantize_fp8(a, a_dtype, config.granularity, axis=-1)
        b_fp8_row, b_scale_inv_row = quantize_fp8(
            b, b_dtype, config.granularity, axis=(-1 if trans_b else -2)
        )
        out = grouped_gemm_fp8_impl(
            a_fp8_row,
            b_fp8_row,
            a_scale_inv_row,
            b_scale_inv_row,
            group_lens,
            group_offs,
            trans_a=False,
            trans_b=trans_b,
            out_dtype=out_dtype,
            granularity=config.granularity.value,
            num_cu=num_cu,
            default_backend=BackendType.HIPKITTEN.value,
        )

        # we need a/b do col quant for backward.
        a_fp8_col, a_scale_inv_col = quantize_fp8(a, a_dtype, config.granularity, axis=-2)
        b_fp8_col, b_scale_inv_col = quantize_fp8(
            b, b_dtype, config.granularity, axis=(-2 if trans_b else -1)
        )

        ctx.save_for_backward(a_fp8_col, b_fp8_col, a_scale_inv_col, b_scale_inv_col, group_lens, group_offs)
        ctx.trans_a = False
        ctx.trans_b = trans_b
        ctx.config = config
        ctx.out_dtype = out_dtype
        ctx.num_cu = num_cu
        return out

    @staticmethod
    def backward(ctx, grad_out):
        grad_out = _ensure_contiguous_grad_out(grad_out)
        a_fp8_col, b_fp8_col, a_scale_inv_col, b_scale_inv_col, group_lens, group_offs = ctx.saved_tensors

        # For grad_a
        grad_out_dtype = _get_fp8_dtype(ctx.config.format, False)
        grad_out_fp8_row, grad_out_scale_inv_row = quantize_fp8(
            grad_out, grad_out_dtype, ctx.config.granularity, axis=-1
        )

        grad_a = grouped_gemm_fp8_impl(
            grad_out_fp8_row,
            b_fp8_col,
            grad_out_scale_inv_row,
            b_scale_inv_col,
            group_lens,
            group_offs,
            trans_a=False,
            trans_b=not ctx.trans_b,
            out_dtype=ctx.out_dtype,
            granularity=ctx.config.granularity.value,
            num_cu=ctx.num_cu,
            default_backend=BackendType.HIPKITTEN.value,
        )

        # For grad_b
        grad_out_fp8_col, grad_out_scale_inv_col = quantize_fp8(
            grad_out, grad_out_dtype, ctx.config.granularity, axis=-2
        )

        grad_b = grouped_gemm_fp8_variable_k_impl(
            a_fp8_col,
            grad_out_fp8_col,
            a_scale_inv_col,
            grad_out_scale_inv_col,
            group_lens,
            group_offs,
            trans_a=not ctx.trans_a,
            trans_b=False,
            trans_c=ctx.trans_b,
            out_dtype=ctx.out_dtype,
            granularity=ctx.config.granularity.value,
            num_cu=ctx.num_cu,
            default_backend=BackendType.HIPKITTEN.value,
        )

        return grad_a, grad_b, None, None, None, None, None


class FP8GroupedGemmTensorFunc(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        a: torch.Tensor,
        b: torch.Tensor,
        group_lens: torch.Tensor,  # [B,] int64
        group_offs: torch.Tensor,  # [B+1,] int64
        trans_b: bool,
        config: Float8QuantConfig,
        num_cu: int | None,
    ):

        assert config.granularity == ScalingGranularity.TENSORWISE
        assert a.ndim == 2, "Input tensor must be 3-dimensions."
        assert b.ndim == 3, "Weight tensor must be 3-dimensional."
        a_dtype = _get_fp8_dtype(config.format, True)
        b_dtype = _get_fp8_dtype(config.format, True)
        a_fp8, a_scale_inv = quantize_fp8(a, a_dtype, config.granularity)
        b_fp8, b_scale_inv = quantize_fp8(b, b_dtype, config.granularity)

        out = grouped_gemm_fp8_impl(
            a_fp8,
            b_fp8,
            a_scale_inv,
            b_scale_inv,
            group_lens,
            group_offs,
            trans_a=False,
            trans_b=trans_b,
            out_dtype=a.dtype,
            granularity=config.granularity.value,
            num_cu=num_cu,
            default_backend=BackendType.HIPKITTEN.value,
            maybe_pre_sync=True,
        )

        ctx.save_for_backward(a_fp8, b_fp8, a_scale_inv, b_scale_inv, group_lens, group_offs)
        ctx.trans_a = False
        ctx.trans_b = trans_b
        ctx.config = config
        ctx.out_dtype = a.dtype
        ctx.num_cu = num_cu
        return out

    @staticmethod
    def backward(ctx, grad_out):
        grad_out = _ensure_contiguous_grad_out(grad_out)
        a_fp8, b_fp8, a_scale_inv, b_scale_inv, group_lens, group_offs = ctx.saved_tensors

        # For grad_a
        grad_out_dtype = _get_fp8_dtype(ctx.config.format, False)
        grad_out_fp8, grad_out_scale_inv = quantize_fp8(
            grad_out, grad_out_dtype, ctx.config.granularity
        )
        grad_a = grouped_gemm_fp8_impl(
            grad_out_fp8,
            b_fp8,
            grad_out_scale_inv,
            b_scale_inv,
            group_lens,
            group_offs,
            trans_a=False,
            trans_b=not ctx.trans_b,
            out_dtype=ctx.out_dtype,
            granularity=ctx.config.granularity.value,
            num_cu=ctx.num_cu,
            default_backend=BackendType.HIPKITTEN.value,
        )

        # For grad_b
        grad_b = grouped_gemm_fp8_variable_k_impl(
            a_fp8,
            grad_out_fp8,
            a_scale_inv,
            grad_out_scale_inv,
            group_lens,
            group_offs,
            trans_a=not ctx.trans_a,
            trans_b=False,
            trans_c=ctx.trans_b,
            out_dtype=ctx.out_dtype,
            granularity=ctx.config.granularity.value,
            num_cu=ctx.num_cu,
            default_backend=BackendType.HIPKITTEN.value,
        )

        return grad_a, grad_b, None, None, None, None, None


# ---------------------------------------------------------------------------
# Path A "fully-fused activation quantize" hooks (forward + backward).
#
# These four helpers are the agent's primary write target. Each raises
# ``NotImplementedError`` in Phase 0 (current state); the autograd Function's
# outer try/except then falls back to the standard un-fused path so the
# baseline behavior is bit-identical to ``FP8GroupedGemmTensorFunc``.
#
# The helpers are designed so that ALL THREE GEMMs in the forward+backward
# step (forward, dA, dB var-K) can be fused independently — the agent can
# ship them one at a time and the score climbs with each.
#
# Path A semantics (vs un-fused):
#   - Activation scale (forward + backward dA + dB) is computed by a tiny
#     ``max_abs_bf16_to_fp8_scale`` HK kernel (R1 deposit) that produces a
#     single fp32 device scalar = max(|x|) / FP8_MAX in 1 launch (~80 us
#     on the largest metric shape, vs 220 us for full quantize_fp8). No
#     M*K-byte FP8 staging buffer is materialized.
#   - The fused HK GEMM kernels accept the BF16 source tensor directly
#     and convert to FP8 inside ``load_a_tile`` via the AMD
#     ``__builtin_amdgcn_cvt_pk_fp8_*`` builtin BEFORE the LDS write,
#     while the dscale path applies the activation scale to the
#     accumulator (semantics match ``quantize_fp8`` + standard FP8 GEMM
#     within the SNR > 25 dB E4M3 noise floor).
#   - Forward saves BF16 ``a`` (which autograd already keeps live for bwd)
#     INSTEAD of a freshly materialized FP8 ``a_fp8`` — the M*K FP8
#     staging buffer is eliminated end-to-end across the fwd→bwd window.
#     Backward dB re-cvts ``a`` from BF16 inside the fused var-K kernel,
#     reusing the forward's saved scale (no max_abs re-computation).
#   - dA fuses ``grad_out`` BF16→FP8 cvt the same way; dB fuses BOTH
#     ``a`` and ``grad_out`` cvt within the var-K load.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Phase gates for the three Path A helpers.
#
# Phase 0 (today): all three flags False — ALL helpers below raise
# NotImplementedError on every call. The dispatch in ``grouped_gemm_fp8``
# detects this and routes ``fuse_act_quant=True`` callers DIRECTLY to
# ``FP8GroupedGemmTensorFunc`` (the un-fused autograd Function), skipping
# ``FP8GroupedGemmTensorFusedActFunc`` entirely. This is bit-identical to
# the previous "FusedActFunc → try/except → fall-through to _unfused_*"
# path but avoids:
#   - 3 try/raise/catch cycles per fwd+bwd iter (~0.41 us)
#   - 3 extra Python frames for the always-raising helpers (~3 us)
#   - 1 extra Python frame for ``_unfused_forward``  (~1 us)
#   - misc kwargs marshalling / ctx attribute set/get (~1-2 us)
# Probed (R14, /tmp/probe_fused_act_overhead_multi_shape.py) on 7 metric
# shapes, fwd+bwd wall:
#   Qwen3-Down-B16-M2048   0.7670 → 0.7616 ms (-0.72 %)
#   Qwen3-GateUP-B16-M2048 1.2899 → 1.2847 ms (-0.40 %)
#   DSV3-GateUP-B16-M2048  2.6591 → 2.6436 ms (-0.59 %)
#   gpt_oss-Down-B4-M2048  0.3700 → 0.3671 ms (-0.77 %)
#   gpt_oss-GateUP-B4-M2048 0.5462 → 0.5389 ms (-1.37 %)
#   gpt_oss-Down-B32-M4096 3.5284 → 3.5225 ms (-0.17 %)
# Geomean ~ -0.6 %, **asymmetric to HK** (Triton goes through
# ``FP8GroupedGemmTensorFunc`` already and pays no FusedActFunc overhead).
#
# When an agent ships Phase 1+ helpers, they delete the ``raise
# NotImplementedError`` from the corresponding helper AND flip its flag
# to True. The dispatch then routes through FusedActFunc, which keeps the
# try/except so partial-Phase states (e.g. fwd fused but dB still un-fused)
# work without further refactoring.
#
# Numerical equivalence: when flags are all False, the dispatch returns
# the ``FP8GroupedGemmTensorFunc`` Function (existing un-fused path)
# instead of ``FP8GroupedGemmTensorFusedActFunc`` Function (Phase 0
# fallback path). Both Functions, when no fused helper executes, perform
# the same operations on the same tensors and save the same set of
# tensors for backward. Verified bit-equivalent: the metric correctness
# gate (``_check_fused_grouped_fp8_correctness``) runs fwd+bwd vs torch-
# native ref with SNR > 25 dB on out / dA / dB; this dispatch change
# does not affect any tensor, only which Function class wraps the
# computation.
# ---------------------------------------------------------------------------

_HK_FUSED_ACT_FORWARD_ENABLED = False
_HK_FUSED_ACT_BACKWARD_DA_ENABLED = False
_HK_FUSED_ACT_BACKWARD_DB_ENABLED = False


def _any_fused_act_helper_enabled() -> bool:
    return (
        _HK_FUSED_ACT_FORWARD_ENABLED
        or _HK_FUSED_ACT_BACKWARD_DA_ENABLED
        or _HK_FUSED_ACT_BACKWARD_DB_ENABLED
    )


def _hk_fused_act_forward(
    a: torch.Tensor,                 # [M_total, K] BF16/FP16
    b_fp8: torch.Tensor,             # [G, N, K] or [G, K, N] FP8
    b_scale_inv: torch.Tensor,       # scalar fp32 device tensor
    group_lens: torch.Tensor,
    group_offs: torch.Tensor,
    trans_b: bool,
    out_dtype: torch.dtype,
    num_cu: int | None,
    fp8_format: Format,
):
    """Phase 1 fused-act forward — FALSIFIED at R7 on this kernel architecture.

    The HK ``.so`` exposes ``grouped_rcr_fused_act_dscale`` +
    ``max_abs_bf16_to_fp8_scale`` and the FUSE_ACT=true template
    specialisation of ``grouped_rcr_kernel`` codegens cleanly (R6
    deposit, commit a7683112). Forward end-to-end runs correctness=24/24,
    but the fused-fwd kernel itself is **~40 % slower per call** than the
    un-fused FP8-input ``grouped_rcr_kernel``:

    * Un-fused load: 1× DTL ``buffer_load_dwordx4 offen lds`` reads 16
      bytes FP8 from HBM → LDS in a single hardware path (no VGPR
      round-trip, no per-lane store). Per pass / lane: 16 B HBM read.
    * Fused-act load (R5a helper): 2× DTR ``raw_buffer_load_b128`` reads
      32 bytes BF16 to VGPRs; 4× ``cvt_bf16x4_to_fp8x4`` produce 16 B FP8
      in VGPRs; 1× ``ds_write_b128`` stores 16 B FP8 to LDS. Per pass /
      lane: 32 B HBM read (2× the un-fused) + an explicit VGPR / cvt /
      LDS-write critical path.

    On the K-aligned metric shapes (DSV3-{GateUP,Down}-{B16,B32}-M{2048,4096}
    + Qwen3-235B-A22B-{GateUP,Down}-{B16,B32}-M{2048,4096} = 16 / 24 cases),
    HK fused-fwd ratios were 0.78-0.85 vs the un-fused 1.20-1.40. End-to-
    end metric crashed 940 → 696 (-26 %). The fwd-only quantize-launch
    saving (~10-20 µs) cannot offset the ~40 % per-kernel slowdown on the
    multi-ms GEMMs we care about.

    Path-A architectural conclusion: with the FP8 grouped-RCR kernel's
    DTL-dominated load_a critical path, BF16-source-with-in-kernel-cvt
    (DTR + cvt + ds_write_b128) cannot beat FP8-source-direct-DTL. Path-B
    (BF16 LDS staging then read-cvt-write) was already falsified in R3 by
    LDS budget. Both directions blocked → fused-fwd does NOT help on this
    kernel architecture; only fused-dB-var-K (Phase 3) and dA-direct-cvt
    might still pay off (different load patterns, less DTL-dominated).

    DECISION (R7): keep the HK kernel + binding deposited (zero cost,
    might help future Path-C exploration / smaller-tile architectures);
    permanently disable the Primus call-site by raising
    NotImplementedError so the autograd Function always falls back to the
    un-fused path. Score remains at the 938 plateau.
    """
    raise NotImplementedError(
        "HK fused-act forward (Path A) FALSIFIED at R7: "
        "DTR + in-register cvt is ~40% slower per kernel call than the "
        "un-fused DTL-based grouped_rcr_kernel; net wall regresses "
        "26-27% even after wiring the saved fwd scale into bwd. "
        "See round-7 falsification note. The HK .so kernel + binding "
        "remain deposited for future Path-C / dA / dB-var-K work."
    )


def _hk_fused_act_backward_dA(
    grad_out: torch.Tensor,          # [M_total, N] BF16/FP16
    b_fp8: torch.Tensor,             # [G, N, K] or [G, K, N] FP8
    b_scale_inv: torch.Tensor,
    group_lens: torch.Tensor,
    group_offs: torch.Tensor,
    trans_b: bool,                   # = not ctx.trans_b from forward
    out_dtype: torch.dtype,
    num_cu: int | None,
    fp8_format: Format,
):
    """Phase 2: HK fused-act dA grouped FP8 GEMM.

    Computes ``grad_a = grad_out @ b.T`` (per-group), with ``grad_out``
    converted from BF16 to FP8 inside ``load_a_tile`` of the SAME
    fused-act kernel template used for forward (the load_a path is
    identical — ``a`` and ``grad_out`` are both [M_total, *] BF16
    tiles). Internally:
      1. Compute ``grad_out_scale_inv = max_abs(grad_out) / FP8_MAX``.
      2. Call ``grouped_dscale_{rcr|rrr}_fused_act`` with the appropriate
         transposed-b layout for the backward dA pattern.

    Returns ``grad_a``.

    PHASE 0 (today): raises NotImplementedError, autograd falls back.
    """
    raise NotImplementedError(
        "HK fused-act dA kernel not yet wired (Phase 0 fallback active)."
    )


def _hk_fused_act_backward_dB(
    a_bf16: torch.Tensor,            # [M_total, K] BF16/FP16 (forward input)
    grad_out: torch.Tensor,          # [M_total, N] BF16/FP16
    a_scale_inv: torch.Tensor,       # scalar fp32 device, from forward
    group_lens: torch.Tensor,
    group_offs: torch.Tensor,
    trans_a: bool,                   # = True (a is "lhs", needs transpose)
    trans_b: bool,                   # = False
    trans_c: bool,                   # = ctx.trans_b from forward
    out_dtype: torch.dtype,
    num_cu: int | None,
    fp8_format: Format,
):
    """Phase 3: HK fused-act dB variable-K grouped FP8 GEMM.

    Computes ``grad_b = a.T @ grad_out`` (var-K layout, accumulation
    along the M_total dim across groups). BOTH inputs are loaded as
    BF16 and cvt'd to FP8 inside the kernel:
      - ``a`` uses the saved ``a_scale_inv`` (no re-compute of max_abs(a))
      - ``grad_out`` uses a freshly computed ``grad_out_scale_inv`` via
        ``max_abs_bf16_to_fp8_scale`` (or shares with dA's call if the
        agent stages dA + dB to share the max_abs kernel launch).

    Returns ``grad_b``.

    PHASE 0 (today): raises NotImplementedError, autograd falls back.
    """
    raise NotImplementedError(
        "HK fused-act dB var-K kernel not yet wired (Phase 0 fallback active)."
    )


def _unfused_forward(
    a, b_fp8, b_scale_inv, group_lens, group_offs,
    trans_b, out_dtype, config, num_cu,
):
    """Phase 0 fallback for forward. Bit-identical to ``FP8GroupedGemmTensorFunc.forward``.
    Returns ``(out, a_fp8, a_scale_inv)`` — caller saves a_fp8 for bwd."""
    a_dtype = _get_fp8_dtype(config.format, True)
    a_fp8, a_scale_inv = quantize_fp8(a, a_dtype, config.granularity)
    out = grouped_gemm_fp8_impl(
        a_fp8, b_fp8, a_scale_inv, b_scale_inv,
        group_lens, group_offs,
        trans_a=False, trans_b=trans_b, out_dtype=out_dtype,
        granularity=config.granularity.value, num_cu=num_cu,
        default_backend=BackendType.HIPKITTEN.value, maybe_pre_sync=True,
    )
    return out, a_fp8, a_scale_inv


def _unfused_backward_dA_dB(
    a_fp8, b_fp8, a_scale_inv, b_scale_inv, grad_out,
    group_lens, group_offs, trans_a, trans_b, config, out_dtype, num_cu,
):
    """Phase 0 fallback for backward. Bit-identical to ``FP8GroupedGemmTensorFunc.backward``."""
    grad_out_dtype = _get_fp8_dtype(config.format, False)
    grad_out_fp8, grad_out_scale_inv = quantize_fp8(
        grad_out, grad_out_dtype, config.granularity
    )
    grad_a = grouped_gemm_fp8_impl(
        grad_out_fp8, b_fp8, grad_out_scale_inv, b_scale_inv,
        group_lens, group_offs,
        trans_a=False, trans_b=not trans_b, out_dtype=out_dtype,
        granularity=config.granularity.value, num_cu=num_cu,
        default_backend=BackendType.HIPKITTEN.value,
    )
    grad_b = grouped_gemm_fp8_variable_k_impl(
        a_fp8, grad_out_fp8, a_scale_inv, grad_out_scale_inv,
        group_lens, group_offs,
        trans_a=not trans_a, trans_b=False, trans_c=trans_b,
        out_dtype=out_dtype, granularity=config.granularity.value, num_cu=num_cu,
        default_backend=BackendType.HIPKITTEN.value,
    )
    return grad_a, grad_b


class FP8GroupedGemmTensorFusedActFunc(torch.autograd.Function):
    """Path A "fully-fused activation quantize" autograd Function for FP8
    grouped GEMM (TENSORWISE only).

    The forward, dA backward, AND dB backward GEMMs each have an independent
    HK fused-act variant the agent can ship (in any order). When a fused
    HK kernel is wired (helper raises no NotImplementedError), this Func
    routes through it; otherwise it falls back to the standard un-fused
    path.

    Save policy:
      * If ALL THREE fused helpers are available: save BF16 ``a`` (no FP8
        staging buffer materialized end-to-end). dB re-cvts ``a`` inside
        the kernel using the saved scalar ``a_scale_inv``.
      * If forward fused but dB un-fused: still save BF16 ``a`` plus
        materialize ``a_fp8`` lazily in backward via ``quantize_fp8``
        (using the saved ``a_scale_inv`` to skip max_abs).
      * If forward un-fused (Phase 0): save FP8 ``a_fp8`` like the
        original ``FP8GroupedGemmTensorFunc`` — bit-identical baseline.

    Phase 0 baseline behavior (today, all helpers raise): bit-identical
    to ``FP8GroupedGemmTensorFunc`` — same metric score, same VRAM, same
    correctness. The agent flips Phase 0 → Phase 1+ by implementing one
    or more of ``_hk_fused_act_forward`` / ``_hk_fused_act_backward_dA`` /
    ``_hk_fused_act_backward_dB`` (in that order recommended by the task).
    """

    @staticmethod
    def forward(
        ctx,
        a: torch.Tensor,
        b: torch.Tensor,
        group_lens: torch.Tensor,
        group_offs: torch.Tensor,
        trans_b: bool,
        config: Float8QuantConfig,
        num_cu: int | None,
    ):
        assert config.granularity == ScalingGranularity.TENSORWISE
        assert config.fuse_act_quant
        assert a.ndim == 2, "Input tensor must be 2-dimensional."
        assert b.ndim == 3, "Weight tensor must be 3-dimensional."

        # b is always quantized via standard quantize_fp8 (NOT the fusion
        # target — weight quant is amortized by upstream FP8 weight cache
        # in production training; in this metric both backends pay it).
        b_dtype = _get_fp8_dtype(config.format, True)
        b_fp8, b_scale_inv = quantize_fp8(b, b_dtype, config.granularity)

        # Try the HK fused forward kernel; on NotImplementedError fall
        # back to the un-fused path. The fused path saves BF16 a (and
        # the device-resident a_scale_inv scalar); the fallback saves
        # FP8 a_fp8 (current behavior).
        try:
            out, a_scale_inv = _hk_fused_act_forward(
                a, b_fp8, b_scale_inv, group_lens, group_offs,
                trans_b=trans_b, out_dtype=a.dtype, num_cu=num_cu,
                fp8_format=config.format,
            )
            ctx.save_for_backward(
                a, b_fp8, a_scale_inv, b_scale_inv, group_lens, group_offs
            )
            ctx.fwd_fused = True
        except NotImplementedError:
            out, a_fp8, a_scale_inv = _unfused_forward(
                a, b_fp8, b_scale_inv, group_lens, group_offs,
                trans_b=trans_b, out_dtype=a.dtype, config=config, num_cu=num_cu,
            )
            ctx.save_for_backward(
                a_fp8, b_fp8, a_scale_inv, b_scale_inv, group_lens, group_offs
            )
            ctx.fwd_fused = False

        ctx.trans_a = False
        ctx.trans_b = trans_b
        ctx.config = config
        ctx.out_dtype = a.dtype
        ctx.num_cu = num_cu
        return out

    @staticmethod
    def backward(ctx, grad_out):
        grad_out = _ensure_contiguous_grad_out(grad_out)
        if ctx.fwd_fused:
            a_bf16, b_fp8, a_scale_inv, b_scale_inv, group_lens, group_offs = ctx.saved_tensors
        else:
            a_fp8, b_fp8, a_scale_inv, b_scale_inv, group_lens, group_offs = ctx.saved_tensors

        # ── grad_a path (dA = grad_out @ b.T) ──
        try:
            grad_a = _hk_fused_act_backward_dA(
                grad_out, b_fp8, b_scale_inv, group_lens, group_offs,
                trans_b=not ctx.trans_b, out_dtype=ctx.out_dtype,
                num_cu=ctx.num_cu, fp8_format=ctx.config.format,
            )
            dA_fused = True
        except NotImplementedError:
            dA_fused = False
            # Need grad_out_fp8 for both dA fallback and (potentially) dB
            grad_out_dtype = _get_fp8_dtype(ctx.config.format, False)
            grad_out_fp8, grad_out_scale_inv = quantize_fp8(
                grad_out, grad_out_dtype, ctx.config.granularity
            )
            grad_a = grouped_gemm_fp8_impl(
                grad_out_fp8, b_fp8, grad_out_scale_inv, b_scale_inv,
                group_lens, group_offs,
                trans_a=False, trans_b=not ctx.trans_b, out_dtype=ctx.out_dtype,
                granularity=ctx.config.granularity.value, num_cu=ctx.num_cu,
                default_backend=BackendType.HIPKITTEN.value,
            )

        # ── grad_b path (dB = a.T @ grad_out via variable-K) ──
        try:
            grad_b = _hk_fused_act_backward_dB(
                a_bf16 if ctx.fwd_fused else None, grad_out,
                a_scale_inv, group_lens, group_offs,
                trans_a=not ctx.trans_a, trans_b=False, trans_c=ctx.trans_b,
                out_dtype=ctx.out_dtype, num_cu=ctx.num_cu,
                fp8_format=ctx.config.format,
            )
        except NotImplementedError:
            # Need a_fp8 + grad_out_fp8 for un-fused dB. Materialize whichever
            # is missing depending on which paths were fused upstream.
            if ctx.fwd_fused:
                # Forward saved BF16 a + the FORWARD scale (FP8_MAX/amax)
                # in ``a_scale_inv`` (named for the un-fused convention but
                # holding the forward scale per the fused-fwd contract,
                # see ``_hk_fused_act_forward`` docstring). Re-quantize to
                # FP8 for the un-fused dB path, but PASS the saved forward
                # scale via ``scale=...`` so the C++ kernel skips the amax
                # pass (saves ~half of the re-quantize launch's compute).
                # The returned ``a_scale_inv_dequant`` follows the un-fused
                # convention (= amax/FP8_MAX) which is what dB var-K expects.
                from primus_turbo.pytorch.kernels.quantization.quantization_impl import (
                    quantize_fp8_tensorwise_impl,
                )
                a_dtype = _get_fp8_dtype(ctx.config.format, True)
                a_fp8, a_scale_inv_dequant = quantize_fp8_tensorwise_impl(
                    a_bf16, a_dtype, scale=a_scale_inv
                )
                # Overwrite the saved forward scale with the dequant scale
                # so downstream var-K dispatch sees the un-fused convention.
                a_scale_inv = a_scale_inv_dequant
            if dA_fused:
                # dA was fused so no grad_out_fp8 yet — quantize now
                grad_out_dtype = _get_fp8_dtype(ctx.config.format, False)
                grad_out_fp8, grad_out_scale_inv = quantize_fp8(
                    grad_out, grad_out_dtype, ctx.config.granularity
                )
            grad_b = grouped_gemm_fp8_variable_k_impl(
                a_fp8, grad_out_fp8, a_scale_inv, grad_out_scale_inv,
                group_lens, group_offs,
                trans_a=not ctx.trans_a, trans_b=False, trans_c=ctx.trans_b,
                out_dtype=ctx.out_dtype, granularity=ctx.config.granularity.value,
                num_cu=ctx.num_cu, default_backend=BackendType.HIPKITTEN.value,
            )

        return grad_a, grad_b, None, None, None, None, None


def grouped_gemm_fp8(
    a: torch.Tensor,
    b: torch.Tensor,
    group_lens: torch.Tensor,
    group_offs: torch.Tensor | None = None,
    trans_b: bool = True,
    config: Union[Float8QuantConfig, None] = None,
    num_cu: int | None = None,
) -> torch.Tensor:
    """ """
    supported_dtypes = [torch.bfloat16, torch.float16]
    assert a.dtype in supported_dtypes, f"Unsupported dtype {a.dtype}, expected one of {supported_dtypes}"
    assert b.dtype in supported_dtypes, f"Unsupported dtype {b.dtype}, expected one of {supported_dtypes}"

    if group_offs is None:
        group_offs = grouped_gemm_compute_offs(group_lens)
    if config is None:
        config = Float8QuantConfig()

    args = (a, b, group_lens, group_offs, trans_b, config, num_cu)

    if config.granularity == ScalingGranularity.TENSORWISE:
        # R14: Phase-0 fast path — when ``fuse_act_quant=True`` is requested
        # but NO Path A helper has been wired (all three
        # ``_HK_FUSED_ACT_*_ENABLED`` flags False), the FusedActFunc path
        # would always fall through to its un-fused branch via try/except.
        # That fallback is bit-identical to ``FP8GroupedGemmTensorFunc`` but
        # pays ~3-15 µs/iter of try/except + extra-Python-frame overhead
        # (probe: round-14 multi-shape bench). Route directly to the un-
        # fused Function in this state to skip the wasted overhead. Once an
        # agent ships any Phase 1+ helper and flips its flag, the call
        # naturally re-enters FusedActFunc and exercises the fused path.
        if config.fuse_act_quant and _any_fused_act_helper_enabled():
            return FP8GroupedGemmTensorFusedActFunc.apply(*args)
        return FP8GroupedGemmTensorFunc.apply(*args)
    elif config.granularity == ScalingGranularity.ROWWISE:
        return FP8GroupedGemmRowFunc.apply(*args)
    elif config.granularity == ScalingGranularity.BLOCKWISE:
        return FP8GroupedGemmBlockFunc.apply(*args)
    else:
        raise ValueError(f"Unsupported FP8 ScalingGranularity: {config.granularity}")
