###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fused MXFP4 grouped MLP, driven through its public op.

The op runs end to end -- output and all four gradients -- against an eager
per-expert fp32 reference. The floor is low because MXFP4 carries ~2 mantissa
bits and this path stacks four quantisations plus the wgrad operands' RHT, so
the bar is here to catch a wrong answer rather than the quantisation.
"""

import pytest
import torch
import torch.nn.functional as F

from primus_turbo.pytorch.core.low_precision import (
    Float4QuantConfig,
    check_mxfp4_support,
)
from primus_turbo.pytorch.ops.grouped_mlp_fp4 import grouped_mlp_fp4
from tests.pytorch.test_utils import compute_snr

# The op measures 10.15-10.88 dB across these shapes and tensors, the same band a
# single MXFP4 grouped GEMM sits in; grad_probs on the last shape is the tightest.
# The threshold sits just under that rather than at the fp4 GEMM suite's 8 dB, so a
# regression in the fused epilogue's rounding shows up instead of being absorbed --
# which leaves it only ~0.15 dB of headroom.
SNR_THRESHOLD = 10.0

# (M, K, I, G). MX needs every GEMM dim to be a 32-multiple; the uneven split
# puts a group boundary off the tile grid. K has to be an odd multiple of 128:
# the fused GLU quant epilogue hides its l1 store in the trailing 128-K block's
# dropped sub-step, which a 256-multiple K does not have. See
# ``glu_epi_quant_supported``.
SHAPES = [(2048, 896, 512, 4), (2048, 1408, 320, 4), (1536, 1152, 384, 3)]


def _mlp_leaves(M, K, I, G, seed=42):
    """bf16 leaves for the fused MLP, which does its own quantisation.

    fc2's weight is [G, K, I], so the output is as wide as the input.
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
    w1 = (torch.randn(G, 2 * I, K, device=dev, generator=gen) * 0.02).bfloat16()
    w2 = (torch.randn(G, K, I, device=dev, generator=gen) * 0.02).bfloat16()
    probs = torch.rand(M, device=dev, dtype=torch.float32, generator=gen) + 0.25
    return offs, offs[1:] - offs[:-1], (x, w1, w2, probs)


def _mlp_ref(x, w1, w2, probs, offs):
    """Per-expert fc1, silu-gated and scaled by probs, then fc2 -- all in fp32."""
    outs = []
    for g in range(w1.shape[0]):
        lo, hi = int(offs[g]), int(offs[g + 1])
        l1 = x[lo:hi].float() @ w1[g].float().t()
        gate, up = torch.chunk(l1, 2, dim=-1)
        act = F.silu(gate) * up * probs[lo:hi, None].float()
        outs.append(act @ w2[g].float().t())
    return torch.cat(outs, dim=0)


def _run(fn, leaves, cotangent):
    """``fn`` on fresh leaves, returning its output and the gradients it produced."""
    args = [t.clone().detach().requires_grad_(True) for t in leaves]
    out = fn(*args)
    out.backward(cotangent.to(out.dtype))
    return out.detach(), [t.grad for t in args]


@pytest.mark.parametrize("shape", SHAPES)
def test_grouped_mlp_fp4(shape):
    """The fused op against the same arithmetic done eagerly, expert by expert."""
    supported, reason = check_mxfp4_support()
    if not supported:
        pytest.skip(reason)

    M, K, I, G = shape
    offs, group_lens, leaves = _mlp_leaves(M, K, I, G)
    gen = torch.Generator(device="cuda").manual_seed(7)
    # Random rather than ones: a cotangent that varies keeps a per-row term like
    # grad_probs from passing on symmetry alone.
    cotangent = torch.randn(M, K, device="cuda", generator=gen)

    out, grads = _run(
        lambda x, w1, w2, p: grouped_mlp_fp4(
            x,
            w1,
            w2,
            group_lens,
            probs=p,
            trans_w1=True,
            trans_w2=True,
            config=Float4QuantConfig(),
            activation="silu",
        ),
        leaves,
        cotangent,
    )
    ref, ref_grads = _run(lambda x, w1, w2, p: _mlp_ref(x, w1, w2, p, offs), leaves, cotangent)

    assert out.shape == (M, K)
    assert compute_snr(ref, out) > SNR_THRESHOLD, "out"
    for name, got, want in zip(("grad_x", "grad_w1", "grad_w2", "grad_probs"), grads, ref_grads):
        assert got is not None, f"{name} was not produced"
        assert got.shape == want.shape, name
        assert compute_snr(want, got) > SNR_THRESHOLD, name
