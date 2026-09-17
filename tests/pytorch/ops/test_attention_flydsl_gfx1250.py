###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Correctness tests for AITER's FlyDSL flash-attention FORWARD on gfx1250.

There is no FlyDSL backward on gfx1250 -- aiter's `flydsl_flash_attn_varlen_bwd`
is gated to gfx942 -- so these tests cover the forward and its LSE only. The
backward is exercised through `tools/gfx1250/tune_attention.py --impl flydsl`,
which pairs this forward with the vendored fused backward and gates all four
tensors.

Two things these tests exist to catch, neither of which raises on its own:

  1. `flydsl_flash_attn_batch_func` RETURNS None when it will not serve a
     configuration. A caller that does not check falls through to a different
     kernel and reports its number as FlyDSL's. `test_gate_*` pin which
     configurations it accepts and which it declines.

  2. Under the wave32 misclassification (flydsl's `is_rdna_arch()` does not
     match "gfx1250", so it builds wave64 on a wave32 dispatch) the phantom
     upper lanes silently drop their work. That shows up as output that is
     wrong, or never written at all -- so every output here is NaN-prefilled
     and checked for full `isfinite` coverage BEFORE its SQNR is looked at.
     "Wrote nothing" and "wrote zero" are different failures.
"""

import pytest
import torch

from tests.pytorch.ref.attention_ref import attention_vanilla_forward_pytorch_ref_impl

fmha = pytest.importorskip(
    "aiter.ops.flydsl.fmha_kernels",
    reason="aiter's FlyDSL kernels are not importable",
)
flydsl_batch = fmha.flydsl_flash_attn_batch_func

pytestmark = [
    pytest.mark.gfx1250,
    pytest.mark.skipif(
        not torch.cuda.is_available()
        or "gfx1250" not in torch.cuda.get_device_properties(0).gcnArchName,
        reason="requires a gfx1250 device",
    ),
]

# bf16 output rounding puts the ceiling near 55.6 dB, and the harness gate for this op is
# 50 dB. Same threshold here so a test failure and a sweep rejection mean the same thing.
MIN_SNR_DB = 50.0

# (batch, seqlen, heads_q, heads_kv, head_dim). The first is a toy shape that exists to be
# launched FIRST in its own process: a bad descriptor or a wave-size mismatch hangs the card,
# and finding that out in 20 s beats finding it out after a full-size run.
SHAPES = [
    (1, 256, 2, 1, 128),        # toy -- launch this one first
    (1, 1024, 8, 2, 128),
    (2, 2048, 8, 2, 128),
    (4, 8192, 32, 8, 128),      # production: Llama-3.1-8B, G=4
]


def snr_db(ref: torch.Tensor, got: torch.Tensor) -> float:
    ref = ref.float()
    got = got.float()
    noise = (ref - got).pow(2).mean().clamp_min(1e-30)
    return float(10.0 * torch.log10(ref.pow(2).mean() / noise))


def _qkv(b, s, hq, hkv, d, dtype=torch.bfloat16):
    torch.manual_seed(0)
    q = torch.randn(b, s, hq, d, device="cuda", dtype=dtype)
    k = torch.randn(b, s, hkv, d, device="cuda", dtype=dtype)
    v = torch.randn(b, s, hkv, d, device="cuda", dtype=dtype)
    return q, k, v


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "b{}_s{}_hq{}_hkv{}_d{}".format(*s))
@pytest.mark.parametrize("causal", [True, False])
def test_forward_matches_reference(shape, causal):
    b, s, hq, hkv, d = shape
    q, k, v = _qkv(b, s, hq, hkv, d)
    scale = d**-0.5

    # NaN prefill: distinguishes "the kernel never wrote here" from "the kernel wrote zero".
    out = torch.full((b, s, hq, d), float("nan"), device="cuda", dtype=q.dtype)
    lse = torch.full((b, hq, s), float("nan"), device="cuda", dtype=torch.float32)

    got = flydsl_batch(q, k, v, softmax_scale=scale, causal=causal, return_lse=True, out=out)
    assert got is not None, (
        f"flydsl_flash_attn_batch_func declined b={b} s={s} hq={hq} hkv={hkv} d={d} "
        f"causal={causal}. It returns None rather than raising, so this is a refusal to "
        "serve the shape. Do not let a caller fall through silently."
    )
    got_out, got_lse = got

    # Coverage before accuracy. A wave-size mismatch drops the upper lanes' work, and the
    # surviving elements can still be close enough to pass an SQNR gate computed over a
    # tensor that is half unwritten.
    assert torch.isfinite(got_out).all(), (
        "output has non-finite elements: "
        f"{int((~torch.isfinite(got_out)).sum())} of {got_out.numel()} "
        "(NaN-prefilled, so these were never written)"
    )
    assert torch.isfinite(got_lse).all(), (
        f"lse has non-finite elements: {int((~torch.isfinite(got_lse)).sum())} of {got_lse.numel()}"
    )

    ref_out, _ = attention_vanilla_forward_pytorch_ref_impl(
        q.float(), k.float(), v.float(), scale, causal, qkv_format="bshd"
    )
    db = snr_db(ref_out, got_out)
    assert db >= MIN_SNR_DB, f"out SQNR {db:.2f} dB < {MIN_SNR_DB} dB"


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "b{}_s{}_hq{}_hkv{}_d{}".format(*s))
def test_lse_shape_and_dtype(shape):
    """The LSE contract the fused backward depends on: [B, Hq, Sq] fp32.

    `dense_fused_backward` accepts exactly this form. If the kernel ever changes it, the
    pairing in `tune_attention.py --impl flydsl` breaks, and it breaks by producing wrong
    gradients rather than by raising.
    """
    b, s, hq, hkv, d = shape
    q, k, v = _qkv(b, s, hq, hkv, d)
    got = flydsl_batch(q, k, v, softmax_scale=d**-0.5, causal=True, return_lse=True)
    assert got is not None
    _, got_lse = got
    assert got_lse.shape == (b, hq, s), f"expected [B, Hq, Sq], got {list(got_lse.shape)}"
    assert got_lse.dtype is torch.float32, f"expected fp32 lse, got {got_lse.dtype}"


@pytest.mark.xfail(
    reason="UNVERIFIED: whether this kernel's LSE is natural log (what dense_fused_backward "
    "expects, and what the ASM forward emits) or log2 -- the kernel carries a LOG2E "
    "constant. Remove the xfail once measured; do not delete the test.",
    strict=False,
)
def test_lse_is_natural_log():
    b, s, hq, hkv, d = 1, 256, 2, 1, 128
    q, k, v = _qkv(b, s, hq, hkv, d)
    scale = d**-0.5
    got = flydsl_batch(q, k, v, softmax_scale=scale, causal=True, return_lse=True)
    assert got is not None
    _, got_lse = got
    _, ref_lse = attention_vanilla_forward_pytorch_ref_impl(
        q.float(), k.float(), v.float(), scale, True, qkv_format="bshd"
    )
    db = snr_db(ref_lse.float(), got_lse.float())
    assert db >= MIN_SNR_DB, (
        f"lse SQNR {db:.2f} dB against a NATURAL-LOG reference. If the ratio against the "
        "reference is ~1.4427 the kernel is emitting log2 and the fused-backward pairing "
        "needs a conversion."
    )


def test_gate_declines_unsupported_head_dim():
    """d=64 is outside this kernel's scope; it must decline rather than serve it wrongly."""
    q, k, v = _qkv(1, 256, 2, 1, 64)
    assert flydsl_batch(q, k, v, softmax_scale=64**-0.5, causal=True) is None


def test_gate_declines_fp32():
    q, k, v = _qkv(1, 256, 2, 1, 128, dtype=torch.float32)
    assert flydsl_batch(q, k, v, softmax_scale=128**-0.5, causal=True) is None


def test_gate_accepts_gqa_ratio_4():
    """G=4 is in scope here.

    Turbo's own _flydsl_common_ok refuses it (_gqa_group_ok wants a power of two in
    [8, 256]), which is why Llama-3.1-8B never reached turbo's FlyDSL path. That is a
    different gate, and this test pins that aiter's is not the same one.
    """
    q, k, v = _qkv(1, 1024, 8, 2, 128)
    assert flydsl_batch(q, k, v, softmax_scale=128**-0.5, causal=True) is not None
