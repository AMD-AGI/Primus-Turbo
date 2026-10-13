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

import functools
import os

import torch
import triton

# FlyDSL quantizers (a6w4_quant) or the Triton ones they replaced (a6w4_quant_triton); same bytes.
_QUANT = os.getenv("FLUX_A6W4_QUANT", "flydsl")


def quantizers():
    """(quant_act_a6w4, quant_w_a6w4) per FLUX_A6W4_QUANT, packing A the way the selected
    GEMM reads it (A_BYTES_PER_32: 24 dense for the tuned kernel, 32 padded for v0)."""
    from primus_turbo.flydsl.gemm.gemm_a6w4_kernel import A_BYTES_PER_32

    if _QUANT == "triton":
        from primus_turbo.flydsl.quantization import a6w4_quant_triton as q

        def _flat(fn, **kw):
            def f(t):
                d, s = fn(t, **kw)
                return d.reshape(t.shape[0], -1), s.reshape(-1)

            return f

        return _flat(q.quant_act_a6w4, a32=A_BYTES_PER_32), _flat(q.quant_w_a6w4)
    if _QUANT == "flydsl":
        from primus_turbo.flydsl.quantization import a6w4_quant as q
    else:
        raise ValueError(f"FLUX_A6W4_QUANT must be flydsl or triton, got {_QUANT!r}")
    return functools.partial(q.quant_act_a6w4, a32=A_BYTES_PER_32), q.quant_w_a6w4


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


def _packs_or_build(preact, act_rounded, aq, wq, sa, sb, n, k, bias):
    """The epilogue's operands, or the same bytes the long way when it declined the store.

    Declining is a tile property, not an error: a shape whose LDS the col staging band does
    not fit keeps the three-kernel chain. Returning the packs either way keeps the caller --
    a custom_op, whose return type is fixed -- out of the business of knowing which it got.
    """
    from primus_turbo.flydsl.gemm.gemm_a6w4_kernel import gemm_a6w4
    from primus_turbo.flydsl.quantization.mxfp4_quant_kernel import flydsl_quant_a6w4_act_dual

    packs = getattr(preact, "a6w4_dual", None)
    if packs is not None:
        return packs
    src = preact if act_rounded else gemm_a6w4(aq, wq, sa, sb, n, k, None)
    bias_f = 0.0 if act_rounded else bias.float()
    act = torch.nn.functional.gelu(src.float() + bias_f, approximate="tanh").to(preact.dtype)
    return flydsl_quant_a6w4_act_dual(act)


def a6w4_mm_gelu_dual_prequant_w(a, wq, sb, n, bias=None, act_rounded=False):
    """`a6w4_mm_prequant_w` whose epilogue also builds the activation's A6W4 operands.

    Returns ``(preact, aq, sa, co, cs)``: ``preact`` is the GEMM's own bf16 output, which is
    what the MLP's backward differentiates, and the four packs are
    ``flydsl_quant_a6w4_act_dual(gelu(preact, approximate="tanh"))`` byte for byte -- built
    from the accumulator, so that tensor is never written or read back.

    ``act_rounded`` picks which chain is being replaced: 1 for a consumer that read the
    GEMM's biased output back, 0 for one handed the bias-free output that added the bias in
    f32 ahead of its own GELU. They differ by one rounding and are not interchangeable.

    The fused store is a hint the GEMM can decline, and ``M`` is not padded here because the
    packs would cover the padding; both cases raise rather than return quietly wrong shapes,
    so the caller can keep the materialised path behind its own eligibility check.
    """
    from primus_turbo.flydsl.gemm.gemm_a6w4_kernel import M_ALIGN, gemm_a6w4

    quant_act_a6w4 = quantizers()[0]

    m, k = a.shape
    if m % M_ALIGN or n % 256 or k % 1024:
        raise ValueError(f"a6w4_mm_gelu_dual: M={m} must be a multiple of {M_ALIGN}, N={n} of 256, K={k} of 1024")
    aq_in, sa_in = quant_act_a6w4(a.to(torch.bfloat16).contiguous())
    preact = gemm_a6w4(
        aq_in, wq, sa_in, sb, n, k, bias, gelu=True, keep_preact=True, act_rounded=act_rounded
    )
    return (preact, *_packs_or_build(preact, act_rounded, aq_in, wq, sa_in, sb, n, k, bias))


def a6w4_mm_gelu_dual_qa_prequant_w(aq, sa, wq, sb, n, bias=None, act_rounded=False):
    """`a6w4_mm_gelu_dual_prequant_w` with the activation already quantized."""
    from primus_turbo.flydsl.gemm.gemm_a6w4_kernel import M_ALIGN, gemm_a6w4

    # K from the weight, not from the activation's row length: that row is either K*A/32
    # bytes or FP8-padded to K, and only the weight is one packing.
    m, k = aq.shape[0], wq.shape[1] * 2
    if m % M_ALIGN or n % 256 or k % 1024:
        raise ValueError(f"a6w4_mm_gelu_dual_qa: M={m} must be a multiple of {M_ALIGN}, N={n} of 256, K={k} of 1024")
    preact = gemm_a6w4(
        aq, wq, sa, sb, n, k, bias, gelu=True, keep_preact=True, act_rounded=act_rounded
    )
    return (preact, *_packs_or_build(preact, act_rounded, aq, wq, sa, sb, n, k, bias))


def a6w4_gelu_dual_eligible(m, n, k):
    """Whether `a6w4_mm_gelu_dual_*` would fuse at this shape, without running it.

    False on the v0 kernel, which has no fused store at all."""
    from primus_turbo.flydsl.gemm import gemm_a6w4_kernel as impl

    fits = getattr(impl, "gelu_dual_eligible", None)
    if fits is None:
        return False
    return not (m % impl.M_ALIGN or n % 256 or k % 1024) and fits(m, n, k)


def a6w4_mm_qa(aq, sa, b, bias=None):
    """a6w4_mm with the activation already quantized (quant_act_a6w4's bytes, M % M_ALIGN == 0)."""
    quant_w_a6w4 = quantizers()[1]

    wq, sb = quant_w_a6w4(b.to(torch.bfloat16).contiguous())
    return a6w4_mm_qa_prequant_w(aq, sa, wq, sb, b.shape[0], bias)


def a6w4_mm_qa_prequant_w(aq, sa, wq, sb, n, bias=None):
    """a6w4_mm with both operands already quantized."""
    from primus_turbo.flydsl.gemm.gemm_a6w4_kernel import A_BYTES_PER_32, M_ALIGN, gemm_a6w4

    m = aq.shape[0]
    k = aq.shape[1] * 32 // A_BYTES_PER_32
    if m % M_ALIGN or n % 256 or k % 1024:
        raise ValueError(f"a6w4_mm_qa: M={m} must be a multiple of {M_ALIGN}, N={n} of 256, K={k} of 1024")
    return gemm_a6w4(aq, wq, sa, sb, n, k, bias)
