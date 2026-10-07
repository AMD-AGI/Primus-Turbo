###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""A6W4 forward GEMM for FLUX: MXFP6 (E2M3) activations x MXFP4 (E2M1) weights.

The quantizers are primus_turbo.flydsl.quantization.a6w4_quant (H16 rotation, E8M0 per 32,
outputs already in the GEMM's layout). The GEMM is FlyDSL v0.2.4's mxfp4_preshuffle kernel,
installed as primus_turbo.flydsl.gemm.gemm_a6w4_kernel, which picks its own tiles per shape.
Both files live in Turbo's flydsl tree so forge campaigns can edit them.
"""

import os

import torch
import triton

# FlyDSL quantizers (a6w4_quant) or the Triton ones they replaced (a6w4_quant_triton); same bytes.
_QUANT = os.getenv("FLUX_A6W4_QUANT", "flydsl")


def quantizers():
    """(quant_act_a6w4, quant_w_a6w4) per FLUX_A6W4_QUANT."""
    if _QUANT == "triton":
        from primus_turbo.flydsl.quantization import a6w4_quant_triton as q

        def _flat(fn):
            def f(t):
                d, s = fn(t)
                return d.reshape(t.shape[0], -1), s.reshape(-1)

            return f

        return _flat(q.quant_act_a6w4), _flat(q.quant_w_a6w4)
    if _QUANT == "flydsl":
        from primus_turbo.flydsl.quantization import a6w4_quant as q
    else:
        raise ValueError(f"FLUX_A6W4_QUANT must be flydsl or triton, got {_QUANT!r}")
    return q.quant_act_a6w4, q.quant_w_a6w4


def a6w4_mm(a, b, bias=None):
    """a[M, K] bf16 @ b[N, K]^T bf16 (+ bias[N]) -> [M, N] bf16, through the A6W4 GEMM. The
    bias is added in the epilogue and rounds exactly like a separate `out + bias`."""
    quant_w_a6w4 = quantizers()[1]

    wq, sb = quant_w_a6w4(b.to(torch.bfloat16).contiguous())
    return a6w4_mm_prequant_w(a, wq, sb, b.shape[0], bias)


def a6w4_mm_prequant_w(a, wq, sb, n, bias=None):
    """a6w4_mm with the weight already quantized: wq / sb are quant_w_a6w4's bytes for an
    [n, K] weight, in any shape. This is how the MXFP4 all-gather hands the weight over."""
    from primus_turbo.flydsl.gemm.gemm_a6w4_kernel import M_ALIGN, gemm_a6w4

    quant_act_a6w4 = quantizers()[0]

    m, k = a.shape
    if n % 256 or k % 1024:
        raise ValueError(f"a6w4_mm: N={n} must be a multiple of 256 and K={k} of 1024")
    a = a.to(torch.bfloat16).contiguous()
    mp = triton.cdiv(m, M_ALIGN) * M_ALIGN
    if mp != m:
        a = torch.nn.functional.pad(a, (0, 0, 0, mp - m))
    aq, sa = quant_act_a6w4(a)
    c = gemm_a6w4(aq, wq, sa, sb, n, k, bias)
    return c[:m] if mp != m else c


def a6w4_mm_qa(aq, sa, b, bias=None):
    """a6w4_mm with the activation already quantized (quant_act_a6w4's bytes, M % M_ALIGN == 0)."""
    quant_w_a6w4 = quantizers()[1]

    wq, sb = quant_w_a6w4(b.to(torch.bfloat16).contiguous())
    return a6w4_mm_qa_prequant_w(aq, sa, wq, sb, b.shape[0], bias)


def a6w4_mm_qa_prequant_w(aq, sa, wq, sb, n, bias=None):
    """a6w4_mm with both operands already quantized."""
    from primus_turbo.flydsl.gemm.gemm_a6w4_kernel import M_ALIGN, gemm_a6w4

    m, k = aq.shape
    if m % M_ALIGN or n % 256 or k % 1024:
        raise ValueError(f"a6w4_mm_qa: M={m} must be a multiple of {M_ALIGN}, N={n} of 256, K={k} of 1024")
    return gemm_a6w4(aq, wq, sa, sb, n, k, bias)
