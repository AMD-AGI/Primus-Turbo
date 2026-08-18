###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""HipKittens attention, gfx950 only.

Two shape families, both taken from the tables the kernels were tuned against:

* META -- ten (Hq, Hkv, Sq, Skv, window) configs, each run full-causal and windowed, which
  is where the rectangular and sliding-window coverage lives.
* MODEL -- the head structures real pretrains run attention at, head dim 128, square and
  full causal, swept over batch 1/2/4. gpt-oss is excluded: it is head dim 64, which the
  META set already covers.

The reference is fp32 with the GQA expansion done explicitly and a bottom-right causal mask,
because `is_causal` is top-left aligned and would only agree with these kernels while
Sq == Skv.
"""

import pytest
import torch

from primus_turbo.pytorch.core.utils import is_gfx950

# The kernels and their Python layer are only built on gfx950, so importing them at all has to
# be conditional -- otherwise collection fails on every other card.
gfx950_only = pytest.mark.skipif(
    not (torch.cuda.is_available() and is_gfx950()),
    reason="HipKittens attention is gfx950-only",
)

pytestmark = gfx950_only


# The MI355X meta configs: (Hq, Hkv, Sq, Skv, window_left). Each runs twice, full-causal and
# windowed. Batch is 4, matching the table these were measured on.
_META = [
    (128, 16, 2048, 16384, 2048),
    (128, 16, 4096, 16384, 2048),
    (128, 16, 8192, 16384, 2048),
    (128, 16, 16384, 16384, 2048),
    (48, 6, 4096, 4096, 2047),
    (48, 6, 4096, 8192, 2047),
    (48, 6, 4096, 12288, 2047),
    (48, 6, 4096, 16384, 2047),
    (64, 8, 1024, 1024, 2047),
    (64, 8, 1024, 16384, 2047),
]

# Run at a reduced sequence length: the fp32 reference materialises a whole [B, Hq, Sq, Skv]
# score matrix, which at the table's own lengths peaks well past what can share a device with
# anything else. The head structure, the rectangular ratios and the window are what this set
# is covering, and they all survive the scale-down.
_META_SCALE = 8
_META_BATCH = 1

META_CASES = []
for hq, hkv, sq, skv, w in _META:
    sq_s, skv_s = sq // _META_SCALE, skv // _META_SCALE
    META_CASES.append((hq, hkv, sq_s, skv_s, -1))
    META_CASES.append((hq, hkv, sq_s, skv_s, max(1, w // _META_SCALE)))

# (Hq, Hkv) of eleven real pretrain configs, deduplicated to eight. All head dim 128.
_MODEL_HEADS = [
    (40, 8),   # llama4_17B128E, llama4_17B16E
    (48, 8),   # minimax_m2.5
    (64, 4),   # qwen3_235B_A22B
    (32, 4),   # qwen3_30B_A3B
    (32, 8),   # lfm2_8B_A1B, mixtral_8x7B_v0.1
    (16, 16),  # deepseek_v2_lite (MHA)
    (64, 8),   # grok2
    (48, 8),   # grok1, mixtral_8x22B_v0.1
]
_MODEL_SEQLEN = 1024
_MODEL_BATCHES = (1, 2, 4)


def _ref(q, k, v, window_left, scale):
    """fp32 reference. q is [Sq, B, Hq, D]; k/v are [Skv, B, Hkv, D]."""
    Sq, B, Hq, D = q.shape
    Skv, _, Hkv, _ = k.shape
    g = Hq // Hkv
    qf = q.float().permute(1, 2, 0, 3)
    kf = k.float().permute(1, 2, 0, 3).repeat_interleave(g, 1)
    vf = v.float().permute(1, 2, 0, 3).repeat_interleave(g, 1)
    s = (qf @ kf.transpose(-1, -2)) * scale
    off = Skv - Sq  # bottom-right alignment
    qi = torch.arange(Sq, device=q.device)[:, None]
    ki = torch.arange(Skv, device=q.device)[None, :]
    keep = ki <= qi + off
    if window_left >= 0:
        keep &= ki >= qi + off - window_left
    p = torch.softmax(s.masked_fill(~keep, float("-inf")), dim=-1)
    return (p @ vf).permute(2, 0, 1, 3)


def _snr_db(ref, got):
    ref, got = ref.float(), got.float()
    noise = (ref - got).pow(2).sum().item()
    if noise == 0.0:
        return 99.0
    return 10.0 * torch.log10(ref.pow(2).sum() / noise).item()


def _run_case(Sq, Skv, B, Hq, Hkv, D, window_left, check_bwd=True, bar=40.0):
    from primus_turbo.hipkittens.attention.gfx950 import (
        hipkittens_attn_backward,
        hipkittens_attn_forward,
    )

    torch.manual_seed(0)
    scale = D ** -0.5

    def mk(s, h):
        return torch.randn(s, B, h, D, device="cuda", dtype=torch.bfloat16) * 0.5

    q, k, v = mk(Sq, Hq), mk(Skv, Hkv), mk(Skv, Hkv)
    out, lse = hipkittens_attn_forward(q, k, v, scale, True, (window_left, 0))
    assert out.shape == q.shape
    assert lse.shape == (B, Hq, 1, Sq)

    ref_out = _ref(q, k, v, window_left, scale)
    snr_o = _snr_db(ref_out, out)
    assert snr_o > bar, f"forward SNR {snr_o:.2f} dB below {bar}"

    if not check_bwd:
        return

    dout = torch.randn_like(out)
    dq, dk, dv = hipkittens_attn_backward(dout, q, k, v, out, lse, scale, True, (window_left, 0))
    assert (dq.shape, dk.shape, dv.shape) == (q.shape, k.shape, v.shape)

    qd, kd, vd = (t.detach().clone().float().requires_grad_(True) for t in (q, k, v))
    _ref(qd, kd, vd, window_left, scale).backward(dout.float())
    for name, ref_g, got in (("dq", qd.grad, dq), ("dk", kd.grad, dk), ("dv", vd.grad, dv)):
        snr = _snr_db(ref_g, got)
        assert snr > bar, f"{name} SNR {snr:.2f} dB below {bar}"


@pytest.mark.parametrize(
    "Hq, Hkv, Sq, Skv, window_left", META_CASES,
    ids=lambda x: str(x),
)
def test_meta_shapes(Hq, Hkv, Sq, Skv, window_left):
    """The meta table: rectangular Sq < Skv, GQA, full-causal and sliding-window, head dim 64."""
    _run_case(Sq, Skv, _META_BATCH, Hq, Hkv, 64, window_left)


@pytest.mark.parametrize("B", _MODEL_BATCHES)
@pytest.mark.parametrize("heads", _MODEL_HEADS, ids=lambda h: f"H{h[0]}x{h[1]}")
def test_model_attn_shapes(heads, B):
    """Head dim 128, square, full causal -- the head structures real pretrains run, at the
    batches the benchmark sweeps."""
    Hq, Hkv = heads
    _run_case(_MODEL_SEQLEN, _MODEL_SEQLEN, B, Hq, Hkv, 128, -1)


def test_deterministic():
    """Same inputs must give bit-identical results across launches, forward and backward."""
    from primus_turbo.hipkittens.attention.gfx950 import (
        hipkittens_attn_backward,
        hipkittens_attn_forward,
    )

    D, Sq, Skv, B, Hq, Hkv = 128, 1024, 1024, 2, 32, 4
    torch.manual_seed(0)
    scale = D ** -0.5
    q = torch.randn(Sq, B, Hq, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(Skv, B, Hkv, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(Skv, B, Hkv, D, device="cuda", dtype=torch.bfloat16)

    o1, l1 = hipkittens_attn_forward(q, k, v, scale, True, (-1, 0))
    o2, l2 = hipkittens_attn_forward(q, k, v, scale, True, (-1, 0))
    assert torch.equal(o1, o2) and torch.equal(l1, l2), "forward is not run-to-run deterministic"

    do = torch.randn_like(o1)
    g1 = hipkittens_attn_backward(do, q, k, v, o1, l1, scale, True, (-1, 0))
    g2 = hipkittens_attn_backward(do, q, k, v, o1, l1, scale, True, (-1, 0))
    for name, a, b in zip(("dq", "dk", "dv"), g1, g2):
        assert torch.equal(a, b), f"{name} is not run-to-run deterministic"


# --------------------------------------------------------------------------------------
# The envelope. Everything outside it must be refused rather than computed wrongly: these
# kernels read out of bounds or leave output unwritten instead of failing.
# --------------------------------------------------------------------------------------


def _qkv(Sq=128, Skv=128, B=1, Hq=8, Hkv=8, D=64, dtype=torch.bfloat16):
    q = torch.randn(Sq, B, Hq, D, device="cuda", dtype=dtype)
    k = torch.randn(Skv, B, Hkv, D, device="cuda", dtype=dtype)
    v = torch.randn(Skv, B, Hkv, D, device="cuda", dtype=dtype)
    return q, k, v


@pytest.mark.parametrize(
    "kwargs, needle",
    [
        (dict(causal=False), "causal"),
        (dict(dropout_p=0.1), "dropout"),
        (dict(bias=object()), "bias"),
        (dict(window_size=(64, 64)), "left window"),
    ],
)
def test_unsupported_options_refused(kwargs, needle):
    from primus_turbo.hipkittens.attention.gfx950 import hipkittens_attn_supported

    q, k, v = _qkv()
    call = dict(causal=True, window_size=(-1, -1))
    call.update(kwargs)
    ok, why = hipkittens_attn_supported(q, k, v, **call)
    assert not ok and needle in why, why


def test_sink_refused():
    """A learned sink changes the softmax denominator; these kernels have no term for it."""
    from primus_turbo.hipkittens.attention.gfx950 import hipkittens_attn_supported

    q, k, v = _qkv()
    ok, why = hipkittens_attn_supported(
        q, k, v, causal=True, sink=torch.zeros(8, device="cuda", dtype=torch.float32)
    )
    assert not ok and "sink" in why, why


def test_fp16_refused():
    """bf16 only: the kernels declare gl<bf16, ...>, so any other 16-bit dtype is
    reinterpreted bit-for-bit rather than converted, returning garbage with no error."""
    from primus_turbo.hipkittens.attention.gfx950 import hipkittens_attn_supported

    q, k, v = _qkv(dtype=torch.float16)
    ok, why = hipkittens_attn_supported(q, k, v, causal=True)
    assert not ok and "bf16" in why, why


@pytest.mark.parametrize("D", [32, 96, 192, 256])
def test_head_dim_refused(D):
    from primus_turbo.hipkittens.attention.gfx950 import hipkittens_attn_supported

    q, k, v = _qkv(D=D)
    ok, why = hipkittens_attn_supported(q, k, v, causal=True)
    assert not ok and "head dim" in why, why


@pytest.mark.parametrize("Sq, Skv", [(256, 128), (512, 256)])
def test_sq_gt_skv_refused(Sq, Skv):
    """Sq > Skv leaves the leading Sq - Skv query rows attending to no key at all; the
    forward kernel returns before its store for exactly those waves, so their output comes
    back holding whatever was in the buffer."""
    from primus_turbo.hipkittens.attention.gfx950 import hipkittens_attn_supported

    q, k, v = _qkv(Sq=Sq, Skv=Skv)
    ok, why = hipkittens_attn_supported(q, k, v, causal=True)
    assert not ok and "Sq > Skv" in why, why


def test_varlen_refused():
    """THD packing arrives as 3-D; these kernels only take dense 4-D SBHD."""
    from primus_turbo.hipkittens.attention.gfx950 import hipkittens_attn_supported

    t = torch.randn(1024, 8, 64, device="cuda", dtype=torch.bfloat16)
    ok, why = hipkittens_attn_supported(t, t, t, causal=True)
    assert not ok and "varlen" in why, why


def test_non_contiguous_refused():
    from primus_turbo.hipkittens.attention.gfx950 import hipkittens_attn_supported

    q, k, v = _qkv(B=2)
    ok, why = hipkittens_attn_supported(q.transpose(0, 1), k, v, causal=True)
    assert not ok and "contiguous" in why, why
