###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fused FP8 grouped MLP, driven through its public op, tensorwise and MXFP8.

``grouped_mlp_fp8`` runs fc1 with the gated activation fused into its epilogue
and fc2 after it, which is only worth doing if it agrees with doing the pieces
separately. So the op is driven end to end -- output and all four gradients --
against an eager per-expert fp32 reference, for every gate it takes.

The input and the fc1 activation are both quantised and the backward adds its
own, which puts the SNR floor well below a single GEMM's; the threshold only has
to catch a wrong answer, not the quantisation. Which backend answers is the
device's choice, not the caller's.

The two granularities differ only in the config they are handed, MX_BLOCKWISE
being NT-only and gfx950-only.
"""

import pytest
import torch
import torch.nn.functional as F

from primus_turbo.pytorch.core.low_precision import (
    MXFP8_BLOCK_SIZE,
    Float8QuantConfig,
    ScaleDtype,
    ScalingGranularity,
    check_mxfp8_support,
)
from primus_turbo.pytorch.core.utils import is_gfx942, is_gfx950
from primus_turbo.pytorch.ops.grouped_mlp_fp8 import grouped_mlp_fp8
from tests.pytorch.test_utils import compute_snr

# fp8 puts a floor of ~55 dB on a single GEMM; stacking two of them and their
# quantisations costs roughly half of that, and the whole op measures ~23.7 dB
# across these shapes, forward and backward alike. MXFP8 lands in the same band.
SNR_THRESHOLD = 20.0

# (M, K, I, G). I=320 leaves a masked column tail (I % 256 != 0), and the uneven
# split exercises a group whose last M-tile is clamped mid-tile. MX additionally
# needs K % 128 == 0 and I % 64 == 0; see ``glu_epi_quant_supported``.
SHAPES = [(2048, 1024, 512, 4), (2048, 1024, 320, 4), (1536, 1152, 384, 3)]

GATES = {"silu": F.silu, "gelu": lambda t: F.gelu(t, approximate="tanh")}

CLAMP_LIMIT = 1.0
GATE_CASES = [("silu", None), ("gelu", None), ("silu", CLAMP_LIMIT)]

CLAMP_SNR_THRESHOLD = 12.0


def _mx_config() -> Float8QuantConfig:
    return Float8QuantConfig(
        granularity=ScalingGranularity.MX_BLOCKWISE,
        scale_dtype=ScaleDtype.E8M0,
        block_size=MXFP8_BLOCK_SIZE,
    )


def _mlp_leaves(M, K, I, G, seed=42):
    """bf16 leaves for the fused MLP, which does its own quantisation.

    One group boundary is pushed off the tile grid. fc2's weight is [G, K, I],
    so the op's output is as wide as its input.
    """
    dev = "cuda"
    gen = torch.Generator(device=dev).manual_seed(seed)
    base = M // G
    lens = [base] * G
    lens[0] += M - base * G
    if G > 1:
        lens[0] -= 17
        lens[1] += 17
    offs = torch.tensor([0] + torch.tensor(lens).cumsum(0).tolist(), device=dev, dtype=torch.int64)
    x = (torch.randn(M, K, device=dev, generator=gen) * 0.1).bfloat16()
    w1 = (torch.randn(G, 2 * I, K, device=dev, generator=gen) * 0.3).bfloat16()
    w2 = (torch.randn(G, K, I, device=dev, generator=gen) * 0.02).bfloat16()
    probs = torch.rand(M, device=dev, dtype=torch.float32, generator=gen) + 0.25
    return offs, offs[1:] - offs[:-1], (x, w1, w2, probs)


def _fused_mlp(x, w1, w2, probs, group_lens, activation, clamp_limit, config):
    return grouped_mlp_fp8(
        x,
        w1,
        w2,
        group_lens,
        probs=probs,
        trans_w1=True,
        trans_w2=True,
        config=config,
        activation=activation,
        clamp_limit=clamp_limit,
    )


def _mlp_ref(x, w1, w2, probs, offs, activation, clamp_limit=None):
    """Per-expert fc1, gated and scaled by probs, then fc2 -- all in fp32."""
    gate_fn = GATES[activation]
    outs = []
    for g in range(w1.shape[0]):
        lo, hi = int(offs[g]), int(offs[g + 1])
        l1 = x[lo:hi].float() @ w1[g].float().t()
        gate, up = torch.chunk(l1, 2, dim=-1)
        if clamp_limit is not None:
            gate = gate.clamp(max=clamp_limit)
            up = up.clamp(min=-clamp_limit, max=clamp_limit)
        act = gate_fn(gate) * up * probs[lo:hi, None].float()
        outs.append(act @ w2[g].float().t())
    return torch.cat(outs, dim=0)


def _run(fn, leaves, cotangent):
    """``fn`` on fresh leaves, returning its output and the gradients it produced.

    The reference runs in fp32 and the op in bf16, so the cotangent is cast to
    whichever the output is rather than being pinned to one of them.
    """
    args = [t.clone().detach().requires_grad_(True) for t in leaves]
    out = fn(*args)
    out.backward(cotangent.to(out.dtype))
    return out.detach(), [t.grad for t in args]


def _check_mlp(shape, activation, clamp_limit, config):
    """The fused op against the same arithmetic done eagerly, expert by expert.

    Output and gradients in one pass: the gradients need the forward anyway, so
    splitting them would only pay for the same kernels twice.
    """
    M, K, I, G = shape
    offs, group_lens, leaves = _mlp_leaves(M, K, I, G)
    gen = torch.Generator(device="cuda").manual_seed(7)
    # Random rather than ones: a cotangent that varies keeps a per-row term
    # like grad_probs from passing on symmetry alone.
    cotangent = torch.randn(M, K, device="cuda", generator=gen)

    out, grads = _run(
        lambda x, w1, w2, p: _fused_mlp(x, w1, w2, p, group_lens, activation, clamp_limit, config),
        leaves,
        cotangent,
    )
    ref, ref_grads = _run(
        lambda x, w1, w2, p: _mlp_ref(x, w1, w2, p, offs, activation, clamp_limit), leaves, cotangent
    )

    grad_threshold = SNR_THRESHOLD if clamp_limit is None else CLAMP_SNR_THRESHOLD

    assert out.shape == (M, K)
    assert compute_snr(ref, out) > SNR_THRESHOLD, "out"
    for name, got, want in zip(("grad_x", "grad_w1", "grad_w2", "grad_probs"), grads, ref_grads):
        assert got is not None, f"{name} was not produced"
        assert got.shape == want.shape, name
        assert compute_snr(want, got) > grad_threshold, name


@pytest.mark.parametrize("activation,clamp_limit", GATE_CASES)
@pytest.mark.parametrize("shape", SHAPES)
def test_grouped_mlp_fp8(shape, activation, clamp_limit):
    if is_gfx942():
        pytest.skip("grouped_mlp_fp8 is not supported on gfx942 currently.")
    _check_mlp(shape, activation, clamp_limit, Float8QuantConfig())


@pytest.mark.parametrize("activation,clamp_limit", GATE_CASES)
@pytest.mark.parametrize("shape", SHAPES)
def test_grouped_mlp_mxfp8(shape, activation, clamp_limit):
    supported, reason = check_mxfp8_support()
    if not supported:
        pytest.skip(reason)
    if not is_gfx950():
        pytest.skip("the fused MXFP8 GLU epilogues are gfx950-only")
    _check_mlp(shape, activation, clamp_limit, _mx_config())
