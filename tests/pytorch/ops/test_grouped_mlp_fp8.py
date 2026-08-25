###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fused FP8 grouped MLP, driven through its public op.

``grouped_mlp_fp8`` runs fc1 with the silu-gated activation fused into its
epilogue and fc2 after it, which is only worth doing if it agrees with doing
the pieces separately. So the op is driven end to end -- output and all four
gradients -- against an eager per-expert reference computed in fp32.

Two fp8 quantisations sit on that path, the input and the fc1 activation, and
the backward adds its own. That is what puts the SNR floor here well below a
single GEMM's; the threshold only has to catch a wrong answer, not the
quantisation. Which backend answers is decided by the device rather than the
caller: FlyDSL takes the fused path where it is supported, Triton covers the
rest.
"""

import pytest
import torch
import torch.nn.functional as F

from primus_turbo.pytorch.core.low_precision import Float8QuantConfig
from primus_turbo.pytorch.core.utils import is_gfx942
from primus_turbo.pytorch.ops.grouped_mlp_fp8 import grouped_mlp_fp8
from tests.pytorch.test_utils import compute_snr

# fp8 puts a floor of ~55 dB on a single GEMM; stacking two of them and their
# quantisations costs roughly half of that, and the whole op measures ~23.7 dB
# across these shapes, forward and backward alike.
SNR_THRESHOLD = 20.0

# (M, K, I, G). I=320 leaves a masked column tail (I % 256 != 0), and the uneven
# split exercises a group whose last M-tile is clamped mid-tile.
SHAPES = [
    (2048, 1024, 512, 4),
    (2048, 1024, 320, 4),
    (1536, 1152, 384, 3),
    (1024, 960, 320, 4),  # both fc1 and fc2 K require a 64-element pad
]


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
    w1 = (torch.randn(G, 2 * I, K, device=dev, generator=gen) * 0.02).bfloat16()
    w2 = (torch.randn(G, K, I, device=dev, generator=gen) * 0.02).bfloat16()
    probs = torch.rand(M, device=dev, dtype=torch.float32, generator=gen) + 0.25
    return offs, offs[1:] - offs[:-1], (x, w1, w2, probs)


def _fused_mlp(x, w1, w2, probs, group_lens):
    return grouped_mlp_fp8(
        x,
        w1,
        w2,
        group_lens,
        probs=probs,
        trans_w1=True,
        trans_w2=True,
        config=Float8QuantConfig(),
        activation="silu",
    )


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
    """``fn`` on fresh leaves, returning its output and the gradients it produced.

    The reference runs in fp32 and the op in bf16, so the cotangent is cast to
    whichever the output is rather than being pinned to one of them.
    """
    args = [t.clone().detach().requires_grad_(True) for t in leaves]
    out = fn(*args)
    out.backward(cotangent.to(out.dtype))
    return out.detach(), [t.grad for t in args]


@pytest.mark.parametrize("shape", SHAPES)
def test_grouped_mlp_fp8(shape):
    """The fused op against the same arithmetic done eagerly, expert by expert.

    Output and gradients in one pass: the gradients need the forward anyway, so
    splitting them would only pay for the same kernels twice.
    """
    if is_gfx942():
        pytest.skip("grouped_mlp_fp8 is not supported on gfx942 currently.")

    M, K, I, G = shape
    offs, group_lens, leaves = _mlp_leaves(M, K, I, G)
    gen = torch.Generator(device="cuda").manual_seed(7)
    # Random rather than ones: a cotangent that varies keeps a per-row term
    # like grad_probs from passing on symmetry alone.
    cotangent = torch.randn(M, K, device="cuda", generator=gen)

    out, grads = _run(lambda x, w1, w2, p: _fused_mlp(x, w1, w2, p, group_lens), leaves, cotangent)
    ref, ref_grads = _run(lambda x, w1, w2, p: _mlp_ref(x, w1, w2, p, offs), leaves, cotangent)

    assert out.shape == (M, K)
    assert compute_snr(ref, out) > SNR_THRESHOLD, "out"
    for name, got, want in zip(("grad_x", "grad_w1", "grad_w2", "grad_probs"), grads, ref_grads):
        assert got is not None, f"{name} was not produced"
        assert got.shape == want.shape, name
        assert compute_snr(want, got) > SNR_THRESHOLD, name
