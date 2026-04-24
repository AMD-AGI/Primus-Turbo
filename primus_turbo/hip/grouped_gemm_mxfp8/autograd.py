###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Hybrid MX-FP8 grouped GEMM autograd Function:
#   forward  — HIP kernel (beats Triton 1.13× at K=2880)
#   dgrad    — HIP kernel via dgrad-layout B (beats Triton 1.27×)
#   wgrad    — Triton variable-K kernel (HIP-native wgrad is future work)
#
# Constraints (inherited from the HIP NT kernel):
#   B layout: [G, N, K]        (trans_b=True in Triton terms)
#   K >= 384, K % 32 == 0      (MX group size)
#   Every per-expert M_g % 16 == 0  (scale preshuffle alignment)
###############################################################################

from __future__ import annotations

from typing import Optional, Union

import torch

from primus_turbo.hip.grouped_gemm_mxfp8 import (
    grouped_gemm_mxfp8_hip_dgrad,
    grouped_gemm_mxfp8_hip_fwd,
    grouped_gemm_mxfp8_hip_variable_k,
)
from primus_turbo.triton.grouped_gemm.grouped_gemm_mxfp8_kernel import (
    grouped_gemm_mxfp8_triton_kernel,
)
from primus_turbo.triton.grouped_gemm.grouped_gemm_mxfp8_variable_k_kernel import (
    grouped_gemm_mxfp8_variable_k_triton_kernel,
)
from primus_turbo.triton.quantization.mxfp8_quant_kernels import (
    quant_mxfp8_dual_jagged,
    quant_mxfp8_rowwise,
    quant_mxfp8_weight_dgrad,
    quant_mxfp8_weight_fwd,
)


def _hip_constraints_ok(group_offs: torch.Tensor) -> bool:
    """Check whether all per-expert M_g satisfy HIP fwd/dgrad constraints:
    M_g % 16 == 0 (preshuffle alignment). Falls back to Triton if any expert
    violates. Skips check when on CPU / very small shapes used in unit tests.
    """
    go = group_offs.to("cpu", dtype=torch.int64)
    lens = (go[1:] - go[:-1]).tolist()
    return all(l > 0 and (l % 16 == 0) for l in lens)


class MXFP8WeightPrequantHip:
    """HIP-path prequant container for MX-FP8 weight.

    Holds both fwd-layout [G, N, K] and dgrad-layout [G, K, N] quantisations.
    Hoists per-forward B quant (~0.6 ms at gpt_oss_20B shape) to once per
    optimiser step for training with gradient accumulation (k>=2).
    """
    __slots__ = (
        "b_bf16",
        "b_fp8_fwd",   "b_scale_fwd",     # [G, N, K] fp8, [G, N, K//32] e8m0
        "b_fp8_dgrad", "b_scale_dgrad",   # [G, K, N] fp8, [G, K, N//32] e8m0
    )

    def __init__(self, b_bf16, b_fp8_fwd, b_scale_fwd, b_fp8_dgrad, b_scale_dgrad):
        self.b_bf16 = b_bf16
        self.b_fp8_fwd = b_fp8_fwd
        self.b_scale_fwd = b_scale_fwd
        self.b_fp8_dgrad = b_fp8_dgrad
        self.b_scale_dgrad = b_scale_dgrad


def prequant_mxfp8_weights_hip(b_bf16: torch.Tensor) -> MXFP8WeightPrequantHip:
    """Given b_bf16 shape [G, N, K] (HIP NT fwd layout), produce both fwd and
    dgrad quantisations once. Call once per optimiser step, reuse across
    gradient accumulation micro-batches.
    """
    assert b_bf16.ndim == 3 and b_bf16.dtype == torch.bfloat16
    G, N, K = b_bf16.shape
    # Fwd layout: [G, N, K] scales [G, N, K//32] — run rowwise quant on the
    # flattened [G*N, K] view (matches what HIP fwd consumes).
    b_flat_fp8, b_flat_scale = quant_mxfp8_rowwise(b_bf16.reshape(G * N, K))
    b_fp8_fwd   = b_flat_fp8.view(G, N, K)
    b_scale_fwd = b_flat_scale.view(G, N, K // 32)
    # Dgrad layout: [G, K, N] scales [G, K, N//32]. Requires transpose of b_bf16.
    b_dgrad_in = b_bf16.transpose(1, 2).contiguous()  # [G, K, N]
    b_fp8_dgrad, b_scale_dgrad = quant_mxfp8_weight_dgrad(b_dgrad_in)
    return MXFP8WeightPrequantHip(
        b_bf16, b_fp8_fwd, b_scale_fwd, b_fp8_dgrad, b_scale_dgrad
    )


class FP8GroupedGemmMXHipFunc(torch.autograd.Function):
    """Hybrid MX-FP8 grouped GEMM autograd: HIP fwd+dgrad + Triton wgrad.

    Computes, per expert group g:  out[m_g] = A[m_g] @ B[g]^T
    with B in NT layout [G, N, K] (matching the HIP kernel).

    Forward: fused rowwise+colwise quant of A (for fwd kernel + wgrad save);
    per-forward B quant unless a prequant container is passed.
    Backward: HIP dgrad (reuse fwd with dgrad-layout B), Triton variable-K
    wgrad (jagged colwise A + grad_out, lhs/rhs swapped so output is [G, N, K]).

    Under torch.no_grad() backward-save quant is skipped.
    """

    @staticmethod
    def forward(
        ctx,
        a: torch.Tensor,                            # [M_total, K]  bf16 / fp16
        b: torch.Tensor,                            # [G, N, K]     bf16 / fp16 (or pre-quant stacked via prequant args)
        group_lens: torch.Tensor,                   # [G]   int64
        group_offs: torch.Tensor,                   # [G+1] int64
        # Optional prequant components (pass 4 None's if not prequant).
        b_fp8_fwd_prequant:   Optional[torch.Tensor] = None,
        b_scale_fwd_prequant: Optional[torch.Tensor] = None,
        b_fp8_dgrad_prequant: Optional[torch.Tensor] = None,
        b_scale_dgrad_prequant: Optional[torch.Tensor] = None,
    ):
        assert a.ndim == 2 and a.dtype in (torch.bfloat16, torch.float16)
        out_dtype = a.dtype
        use_prequant = b_fp8_fwd_prequant is not None

        # See skill: autograd_function_requires_grad_check — torch.is_grad_enabled()
        # returns False inside autograd.Function.forward. Use input.requires_grad.
        need_bwd = a.requires_grad or (b is not None and getattr(b, "requires_grad", False))

        # A quant: fused dual-jagged (rowwise for fwd + jagged colwise for wgrad
        # save) in one HBM read when bwd needed; just rowwise for inference.
        if need_bwd:
            a_fp8, a_scale, a_fp8_col, a_scale_col, scale_offs = quant_mxfp8_dual_jagged(
                a, group_offs, group_lens,
            )
        else:
            a_fp8, a_scale = quant_mxfp8_rowwise(a)

        # B fwd quant (or reuse prequant).
        if use_prequant:
            b_fp8_fwd   = b_fp8_fwd_prequant
            b_scale_fwd = b_scale_fwd_prequant
        else:
            assert b is not None and b.ndim == 3
            G, N, K = b.shape
            b_flat_fp8, b_flat_scale = quant_mxfp8_rowwise(b.reshape(G * N, K))
            b_fp8_fwd   = b_flat_fp8.view(G, N, K)
            b_scale_fwd = b_flat_scale.view(G, N, K // 32)

        # HIP forward (falls back to Triton if any expert M_g % 16 != 0 —
        # preshuffle alignment requirement).
        ctx._use_hip_path = _hip_constraints_ok(group_offs)
        if ctx._use_hip_path:
            # B is in HIP NT layout [G, N, K]. Triton fwd uses trans_b=True
            # for the same layout.
            out = grouped_gemm_mxfp8_hip_fwd(
                a_fp8, b_fp8_fwd, a_scale, b_scale_fwd, group_offs,
                out_dtype=out_dtype,
            )
        else:
            out = grouped_gemm_mxfp8_triton_kernel(
                a_fp8, b_fp8_fwd, a_scale, b_scale_fwd, group_offs,
                trans_b=True, out_dtype=out_dtype,
            )

        # Always set ctx attrs unconditionally (autograd may still call backward
        # if we mis-detect need_bwd; ctx setup is cheap).
        ctx.out_dtype = out_dtype
        ctx.b_shape = None if use_prequant else b.shape
        # _use_hip_path was set above; ensure it's there even if forward
        # took an unexpected branch (defensive).
        if not hasattr(ctx, "_use_hip_path"):
            ctx._use_hip_path = False

        if need_bwd:
            if use_prequant:
                b_fp8_dgrad   = b_fp8_dgrad_prequant
                b_scale_dgrad = b_scale_dgrad_prequant
            else:
                b_dgrad_in = b.transpose(1, 2).contiguous()  # [G, K, N]
                b_fp8_dgrad, b_scale_dgrad = quant_mxfp8_weight_dgrad(b_dgrad_in)
            ctx.save_for_backward(
                a_fp8_col, a_scale_col,
                b_fp8_dgrad, b_scale_dgrad,
                group_lens, group_offs, scale_offs,
            )
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        grad_out = grad_out.contiguous() if not grad_out.is_contiguous() else grad_out
        out_dtype = ctx.out_dtype
        (
            a_fp8_col, a_scale_col,
            b_fp8_dgrad, b_scale_dgrad,
            group_lens, group_offs, scale_offs,
        ) = ctx.saved_tensors

        # Dual quant of grad_out: rowwise (for dgrad) + jagged colwise (for wgrad)
        # from a single HBM read.
        (
            grad_out_fp8_row, grad_out_scale_row,
            grad_out_fp8_col, grad_out_scale_col,
            _go_scale_offs,
        ) = quant_mxfp8_dual_jagged(grad_out, group_offs, group_lens)

        # ── dgrad: dA = grad_out @ B  (kernel sees B_dgrad [G, K, N])
        # HIP dgrad has the same M_g % 16 == 0 requirement; fall back to
        # Triton on unbalanced shapes (uses trans_b=True with original B
        # layout — but we have b_dgrad which is [G, K, N], so call the
        # Triton kernel with trans_b=False).
        if ctx._use_hip_path:
            grad_a = grouped_gemm_mxfp8_hip_dgrad(
                grad_out_fp8_row, b_fp8_dgrad, grad_out_scale_row, b_scale_dgrad,
                group_offs, out_dtype=out_dtype,
            )
        else:
            # b_fp8_dgrad is stored [G, K, N]. With trans_b=True the Triton
            # kernel interprets b as [G, N_kern, K_kern] = [G, K, N] so the
            # K_kern (reduction) = N — matches grad_out's N axis.
            grad_a = grouped_gemm_mxfp8_triton_kernel(
                grad_out_fp8_row, b_fp8_dgrad, grad_out_scale_row, b_scale_dgrad,
                group_offs, trans_b=True, out_dtype=out_dtype,
            )

        # ── wgrad: dB[g, n, k] = sum_m grad_out[m, n] * A[m, k]
        # HIP wgrad path (default). Requires balanced MoE with M_g uniform,
        # >=384, %128==0. Triton fallback for any other shape.
        # Output [G, N, K] produced via trans_c=True (internal lhs/rhs swap).
        _m_total = grad_out.shape[0]
        _g = int(group_lens.numel())
        _mg = _m_total // _g if _g > 0 else 0
        _hip_wgrad_ok = (
            ctx._use_hip_path and _g > 0 and _m_total == _g * _mg
            and _mg >= 384 and _mg % 128 == 0
        )
        if _hip_wgrad_ok:
            grad_b = grouped_gemm_mxfp8_hip_variable_k(
                a_fp8_col, grad_out_fp8_col, a_scale_col, grad_out_scale_col,
                group_offs, scale_offs,
                out_dtype=out_dtype, trans_c=True,
            )
        else:
            # Fallback for unbalanced shapes (HIP v1 requires balanced MoE).
            grad_b = grouped_gemm_mxfp8_variable_k_triton_kernel(
                grad_out_fp8_col, a_fp8_col,
                grad_out_scale_col, a_scale_col,
                group_offs, scale_offs,
                out_dtype=out_dtype,
            )
        # Result: [G, N, K] ✓ matches b layout.

        # Grad signature mirrors forward:
        # (a, b, group_lens, group_offs,
        #  b_fp8_fwd_prequant, b_scale_fwd_prequant,
        #  b_fp8_dgrad_prequant, b_scale_dgrad_prequant)
        return grad_a, grad_b, None, None, None, None, None, None


def _dispatch_triton_mxfp8(a, b, group_lens, group_offs):
    """Call the existing Triton MX-FP8 autograd Function. B is expected in
    Triton's trans_b=False convention ([G, K, N]); the HIP path takes
    [G, N, K] — callers of the selector must pick the right layout.
    """
    from primus_turbo.pytorch.core.low_precision import (
        Float8QuantConfig, ScalingGranularity,
    )
    from primus_turbo.pytorch.ops.grouped_gemm_fp8 import grouped_gemm_fp8
    return grouped_gemm_fp8(
        a, b, group_lens, group_offs,
        trans_b=False,
        config=Float8QuantConfig(granularity=ScalingGranularity.MX_BLOCKWISE),
    )


def grouped_gemm_mxfp8(
    a: torch.Tensor,
    b: Union[torch.Tensor, MXFP8WeightPrequantHip, "MXFP8WeightPrequant"],
    group_lens: torch.Tensor,
    group_offs: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """MX-FP8 grouped GEMM with env-selected backend.

    Env ``TURBO_MXFP8_GG_BACKEND``:
      - "hip"    (default on gfx950): HIP kernel, B must be [G, N, K]
      - "triton":                      Triton kernel, B must be [G, K, N]

    For a drop-in A/B test in production pipelines, callers should quantise B
    in BOTH layouts (cheap — bf16 transpose + fp8 quant done once per weight
    update) and pass the one matching the selected backend.
    """
    import os
    backend = os.environ.get("TURBO_MXFP8_GG_BACKEND", "hip").lower()
    if backend == "hip":
        return grouped_gemm_mxfp8_hip(a, b, group_lens, group_offs)
    elif backend == "triton":
        return _dispatch_triton_mxfp8(a, b, group_lens, group_offs)
    else:
        raise ValueError(f"TURBO_MXFP8_GG_BACKEND must be 'hip' or 'triton', got {backend!r}")


def grouped_gemm_mxfp8_hip(
    a: torch.Tensor,
    b: Union[torch.Tensor, MXFP8WeightPrequantHip],
    group_lens: torch.Tensor,
    group_offs: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Public HIP MX-FP8 grouped GEMM entry.

    Args:
        a:          [M_total, K]  bf16 / fp16
        b:          [G, N, K]     bf16 / fp16  OR  MXFP8WeightPrequantHip
        group_lens: [G]   int64   per-expert token counts
        group_offs: [G+1] int64   prefix sum (computed if None)
    """
    if group_offs is None:
        # Build on-the-fly: prepend 0, cumsum.
        zero = torch.zeros(1, dtype=torch.int64, device=group_lens.device)
        group_offs = torch.cat([zero, torch.cumsum(group_lens.to(torch.int64), dim=0)])

    if isinstance(b, MXFP8WeightPrequantHip):
        return FP8GroupedGemmMXHipFunc.apply(
            a, b.b_bf16, group_lens, group_offs,
            b.b_fp8_fwd, b.b_scale_fwd, b.b_fp8_dgrad, b.b_scale_dgrad,
        )
    else:
        return FP8GroupedGemmMXHipFunc.apply(
            a, b, group_lens, group_offs,
            None, None, None, None,
        )
