###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""RL attention on the FlyDSL backend (gfx950): accuracy, and the same bitwise guarantees as the
Triton backend (see test_rl_attention.py), within the FlyDSL limits (bf16, D 64 / 128, pages of 64
or 128). The FlyDSL backend has its own bits, so every comparison stays within it."""

import pytest
import torch

from primus_turbo.common import envs
from primus_turbo.pytorch.core.utils import is_gfx950
from primus_turbo.pytorch.ops.attention import flash_attn_varlen_func
from primus_turbo.pytorch.ops.attention.rl_attention import (
    create_block_mask,
    rl_flash_attn_varlen_func,
    rl_paged_attention_func,
)
from tests.pytorch.ops.test_rl_attention import (
    DEVICE,
    _bits_equal,
    _cu,
    _mixed_mask,
    _paged,
    _sinks,
    _varlen,
)
from tests.pytorch.ref.rl_attention_ref import rl_attention_ref

pytestmark = pytest.mark.skipif(not is_gfx950(), reason="the FlyDSL RL forward is gfx950 only")

FLY = {"BACKEND": "FLYDSL"}
HEADS = [(32, 8, 128), (64, 8, 64), (8, 8, 128)]
LENS = [1, 77, 300, 129, 1024]
BF16 = torch.bfloat16


def _train(q, k, v, cu, max_len, cu_k=None, kernel_options=None, **kw):
    cu_k = cu if cu_k is None else cu_k
    return rl_flash_attn_varlen_func(
        q, k, v, cu, cu_k, max_len, max_len, kernel_options=kernel_options or FLY, **kw
    )


@pytest.mark.parametrize("heads", HEADS)
@pytest.mark.parametrize("window", [-1, 128])
@pytest.mark.parametrize("with_sinks", [False, True])
def test_matches_reference(heads, window, with_sinks):
    hq, hkv, d = heads
    q, k, v, cu = _varlen(LENS, hq, hkv, d, BF16, seed=1)
    sinks = _sinks(hq, 1) if with_sinks else None
    out = _train(q, k, v, cu, 1024, window_left=window, sinks=sinks)
    if sinks is None:
        ref = rl_attention_ref(q, k, v, cu, cu, d**-0.5, window_left=window)
    else:
        pos = torch.cat([torch.arange(n) for n in LENS]).to(DEVICE)
        base = torch.repeat_interleave(cu[:-1], cu[1:] - cu[:-1])
        lo = base + (pos - window).clamp(min=0) if window >= 0 else base
        ref = rl_attention_ref(q, k, v, cu, cu, d**-0.5, lo, base + pos, sinks)
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("heads", HEADS[:2])
def test_masks_match_reference(heads):
    hq, hkv, d = heads
    q, k, v, cu = _varlen(LENS, hq, hkv, d, BF16, seed=2)
    lo, hi, mask = _mixed_mask(LENS, seed=2)
    sinks = _sinks(hq, 2)
    out, lse = _train(q, k, v, cu, 1024, block_mask=mask, sinks=sinks, return_lse=True)
    ref = rl_attention_ref(q, k, v, cu, cu, d**-0.5, lo, hi, sinks)
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)
    assert torch.all(out[lo > hi] == 0)
    nolse = _train(q, k, v, cu, 1024, block_mask=mask, return_lse=True)[1]
    assert torch.all(torch.isinf(nolse[lo > hi]) & (nolse[lo > hi] > 0))


@pytest.mark.parametrize("heads", HEADS[:2])
def test_large_logits_accuracy(heads):
    """Scores are scaled in fp32 after the MFMA, so logits in the hundreds stay accurate."""
    hq, hkv, d = heads
    n = 2048
    q, k, v, cu = _varlen([n], hq, hkv, d, BF16, seed=3)
    growth = (1 + 8 * torch.linspace(0, 1, n, device=DEVICE)).view(n, 1, 1)
    k = (k.float() * growth).to(BF16)
    out = _train(q, k, v, cu, n)
    ref = rl_attention_ref(q, k, v, cu, cu, d**-0.5)
    torch.testing.assert_close(out.float(), ref, atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize("heads", HEADS[:2])
@pytest.mark.parametrize("page", [64, 128])
@pytest.mark.parametrize("masked", [False, True])
def test_paged_and_builtin_equal_contiguous(heads, page, masked):
    """Paged, intervals and the built-in causal path give the same bits. Unused cache slots hold
    NaN, so reading past a sequence would show."""
    hq, hkv, d = heads
    q, k, v, cu = _varlen(LENS, hq, hkv, d, BF16, seed=4)
    mask = _mixed_mask(LENS, seed=4)[2] if masked else None
    contiguous = _train(q, k, v, cu, 1024, block_mask=mask)
    if not masked:
        # a different cu_seqlens_k object routes plain causal through the interval path
        assert _bits_equal(_train(q, k, v, cu, 1024, cu_k=cu.clone()), contiguous)
    k_cache, v_cache, block_table, seq_lens = _paged(k, v, cu, page)
    unused = torch.ones_like(k_cache, dtype=torch.bool)
    for s, n in enumerate(seq_lens.tolist()):
        for b in range((n + page - 1) // page):
            unused[block_table[s, b], : min(page, n - b * page)] = False
    k_cache[unused], v_cache[unused] = float("nan"), float("nan")
    paged = rl_paged_attention_func(
        q, k_cache, v_cache, cu, seq_lens, block_table, 1024, 1024, block_mask=mask, kernel_options=FLY
    )
    assert _bits_equal(paged, contiguous)


@pytest.mark.parametrize("split", [False, True])
def test_paged_out_buffer(split):
    """out= on FlyDSL: a caller's padded buffer gets the returned bits; rows past the tokens are kept."""
    lens = [1, 1, 1, 1] if split else [1, 77, 300]
    kv_lens = [3000, 2100, 1500, 1025] if split else lens
    q = _varlen(lens, 32, 8, 128, BF16, seed=17)[0]
    _, k, v, cu_k = _varlen(kv_lens, 32, 8, 128, BF16, seed=18)
    k_cache, v_cache, block_table, seq_lens = _paged(k, v, cu_k, 64)
    args = (q, k_cache, v_cache, _cu(lens), seq_lens, block_table, max(lens), max(kv_lens))
    opts = {**FLY, "MODE": "split"} if split else FLY
    want = rl_paged_attention_func(*args, kernel_options=opts)
    buf = torch.full((q.shape[0] + 3, 32 * 128), 7.0, dtype=BF16, device=DEVICE)
    assert rl_paged_attention_func(*args, kernel_options=opts, out=buf) is buf
    assert _bits_equal(buf[: q.shape[0]].view_as(want), want)
    assert bool((buf[q.shape[0] :] == 7.0).all())


@pytest.mark.parametrize("heads", HEADS[:2])
def test_batch_and_launch_invariance(heads):
    hq, hkv, d = heads
    q, k, v, cu = _varlen([500], hq, hkv, d, BF16, seed=5)
    alone = _train(q, k, v, cu, 500)
    oq, ok, ov, _ = _varlen([700, 33], hq, hkv, d, BF16, seed=6)
    bq, bk, bv = (torch.cat([o[:700], x, o[700:]]) for o, x in ((oq, q), (ok, k), (ov, v)))
    batched = _train(bq, bk, bv, _cu([700, 500, 33]), 700)
    assert _bits_equal(batched[700:1200], alone)
    for block_m in (64, 128):
        assert _bits_equal(_train(q, k, v, cu, 500, kernel_options={**FLY, "BLOCK_M": block_m}), alone)


@pytest.mark.parametrize("heads", HEADS)
@pytest.mark.parametrize("window", [-1, 128])
@pytest.mark.parametrize("page", [64, 128])
def test_causal_rollout_matches_training(heads, window, page):
    """Paged prefill, then one token per step next to another decoding sequence (packed decode
    rows), against one training forward."""
    hq, hkv, d = heads
    prompt, total = 400, 440
    q, k, v, cu = _varlen([total], hq, hkv, d, BF16, seed=7)
    training = _train(q, k, v, cu, total, window_left=window)
    oq, ok, ov, _ = _varlen([700], hq, hkv, d, BF16, seed=8)
    k_cache, v_cache, block_table, _ = _paged(torch.cat([k, ok]), torch.cat([v, ov]), _cu([total, 700]), page)
    for t0, t1 in [(0, prompt)] + [(t, t + 1) for t in range(prompt, total)]:
        other = 200 + t0  # the other sequence (700 tokens) stays in range
        step_q = torch.cat([q[t0:t1], oq[other : other + 1]])
        seq_lens = torch.tensor([t1, other + 1], dtype=torch.int32, device=DEVICE)
        out = rl_paged_attention_func(
            step_q, k_cache, v_cache, _cu([t1 - t0, 1]), seq_lens, block_table, t1 - t0, max(t1, other + 1),
            window_left=window, kernel_options=FLY,
        )  # fmt: skip
        assert _bits_equal(out[: t1 - t0], training[t0:t1]), f"tokens [{t0}, {t1})"


@pytest.mark.parametrize("heads", HEADS[:2])
@pytest.mark.parametrize("window", [-1, 128])
@pytest.mark.parametrize("with_sinks", [False, True])
def test_long_rollout_matches_training(heads, window, with_sinks):
    """Sequences longer than SEGMENT_SIZE: the training forward merges fixed 1024-key segments
    in-kernel; the rollout prefills, then decodes through split-KV programs (few sequences), alone
    and next to another long sequence. Every token must match."""
    hq, hkv, d = heads
    prompt, total = 2990, 3000
    q, k, v, cu = _varlen([total], hq, hkv, d, BF16, seed=13)
    sinks = _sinks(hq, 13) if with_sinks else None
    training = _train(q, k, v, cu, total, window_left=window, sinks=sinks)
    oq, ok, ov, _ = _varlen([4000], hq, hkv, d, BF16, seed=14)
    k_cache, v_cache, block_table, _ = _paged(torch.cat([k, ok]), torch.cat([v, ov]), _cu([total, 4000]), 64)
    for t0, t1 in [(0, prompt)] + [(t, t + 1) for t in range(prompt, total)]:
        for with_other in (False, True):
            other = 3000 + t0 - prompt
            step_q = torch.cat([q[t0:t1], oq[other : other + 1]]) if with_other else q[t0:t1]
            lens = [t1, other + 1] if with_other else [t1]
            q_lens = [t1 - t0, 1] if with_other else [t1 - t0]
            out = rl_paged_attention_func(
                step_q, k_cache, v_cache, _cu(q_lens), torch.tensor(lens, dtype=torch.int32, device=DEVICE),
                block_table[: len(lens)], t1 - t0, max(lens), window_left=window, sinks=sinks, kernel_options=FLY,
            )  # fmt: skip
            assert _bits_equal(out[: t1 - t0], training[t0:t1]), f"tokens [{t0}, {t1}) other={with_other}"


@pytest.mark.parametrize("heads", HEADS[:2])
def test_segments_leave_short_sequences_unchanged(heads):
    """A batch with a long sequence runs the segmented build; a short sequence in it must keep the
    bits it has alone (one segment, no merge)."""
    hq, hkv, d = heads
    q, k, v, cu = _varlen([700], hq, hkv, d, BF16, seed=15)
    alone = _train(q, k, v, cu, 700)
    oq, ok, ov, _ = _varlen([3000], hq, hkv, d, BF16, seed=16)
    batched = _train(torch.cat([oq, q]), torch.cat([ok, k]), torch.cat([ov, v]), _cu([3000, 700]), 3000)
    assert _bits_equal(batched[3000:], alone)


@pytest.mark.parametrize("heads", HEADS[:2])
@pytest.mark.parametrize("chunk", [None, 37])
def test_masked_rollout_matches_training(heads, chunk):
    """A prompt with a bidirectional image span, prefilled whole or in 37-token chunks, then decode,
    with sinks, through flash_attn_varlen_func on both sides."""
    hq, hkv, d = heads
    prompt, total = 300, 330
    q, k, v, cu = _varlen([total], hq, hkv, d, BF16, seed=9)
    sinks = _sinks(hq, 9)
    lo = torch.zeros(total, dtype=torch.int32, device=DEVICE)
    hi = torch.arange(total, dtype=torch.int32, device=DEVICE)
    hi[40:200] = 199
    training = flash_attn_varlen_func(
        q, k, v, cu, cu, total, total, sink=sinks, block_mask=create_block_mask(lo, hi), kernel_options=FLY
    )
    k_cache, v_cache, block_table, _ = _paged(k, v, cu, 64)
    steps = [(0, prompt)] if chunk is None else [(s, min(s + chunk, prompt)) for s in range(0, prompt, chunk)]
    steps += [(t, t + 1) for t in range(prompt, total)]
    for t0, t1 in steps:
        keys = max(t1, int(hi[t0:t1].max()) + 1)
        out = flash_attn_varlen_func(
            q[t0:t1], k_cache, v_cache, _cu([t1 - t0]), None, t1 - t0, keys, sink=sinks,
            block_mask=create_block_mask(lo[t0:t1], hi[t0:t1]), kernel_options=FLY, block_table=block_table,
            seqused_k=torch.tensor([keys], dtype=torch.int32, device=DEVICE),
        )  # fmt: skip
        assert _bits_equal(out, training[t0:t1]), f"tokens [{t0}, {t1})"


@pytest.mark.parametrize("heads", HEADS[:2])
@pytest.mark.parametrize("masked", [False, True])
def test_backward(heads, masked):
    hq, hkv, d = heads
    lens = [33, 200, 129]
    q, k, v, cu = _varlen(lens, hq, hkv, d, BF16, seed=10)
    lo, hi, mask = _mixed_mask(lens, seed=10)
    sinks = _sinks(hq, 10)
    dout = torch.randn(q.shape[0], hq, d, generator=torch.Generator().manual_seed(11)).to(DEVICE, BF16)
    grads = []
    for _ in range(2):
        qq, kk, vv, ss = (t.clone().requires_grad_() for t in (q, k, v, sinks))
        _train(qq, kk, vv, cu, 200, block_mask=mask if masked else None, sinks=ss).backward(dout)
        grads.append((qq.grad, kk.grad, vv.grad, ss.grad))
    for a, b in zip(*grads):
        assert torch.equal(a, b), "backward is not deterministic"
    qr, kr, vr, sr = (t.float().clone().requires_grad_() for t in (q, k, v, sinks))
    if masked:
        ref = rl_attention_ref(qr, kr, vr, cu, cu, d**-0.5, lo, hi, sr)
    else:
        pos = torch.cat([torch.arange(n) for n in lens]).to(DEVICE)
        base = torch.repeat_interleave(cu[:-1], cu[1:] - cu[:-1])
        ref = rl_attention_ref(qr, kr, vr, cu, cu, d**-0.5, base, base + pos, sr)
    ref.backward(dout.float())
    for name, got, want in zip(("dq", "dk", "dv", "dsink"), grads[0], (qr.grad, kr.grad, vr.grad, sr.grad)):
        torch.testing.assert_close(got.float(), want, atol=5e-2, rtol=5e-2, msg=name)


def test_env_selects_flydsl():
    q, k, v, cu = _varlen([300], 32, 8, 128, BF16, seed=12)
    want = _train(q, k, v, cu, 300)
    saved = envs.PRIMUS_TURBO_RL_ATTN_KERNEL_OPTIONS
    try:
        envs.set_env("PRIMUS_TURBO_RL_ATTN_KERNEL_OPTIONS", "BACKEND=FLYDSL")
        got = rl_flash_attn_varlen_func(q, k, v, cu, cu, 300, 300)
    finally:
        envs.set_env("PRIMUS_TURBO_RL_ATTN_KERNEL_OPTIONS", saved)
    assert _bits_equal(got, want)


def test_rejects_unsupported():
    q, k, v, cu = _varlen([64], 8, 2, 128, torch.float16)
    with pytest.raises(ValueError, match="cannot handle"):
        _train(q, k, v, cu, 64)
    q, k, v, cu = _varlen([64], 8, 2, 256, BF16)
    with pytest.raises(ValueError, match="cannot handle"):
        _train(q, k, v, cu, 64)
    q, k, v, cu = _varlen([64], 8, 2, 128, BF16)
    k_cache, v_cache, block_table, seq_lens = _paged(k, v, cu, 16)
    with pytest.raises(ValueError, match="cannot handle"):
        rl_paged_attention_func(q, k_cache, v_cache, cu, seq_lens, block_table, 64, 64, kernel_options=FLY)


# ----------------------------------------------------------------------------- schedule knobs


def _knob_sets(decode):
    """Schedule-only kernel options; none may change the bits."""
    sets = [{"BLOCK_M": bm, "waves_per_eu": w} for bm in (64, 128) for w in (1, 2)]
    # The 32-row layout (32x32x16 MFMAs) against the 16-row default, and 16-row blocks of 32.
    sets += [{"WAVE_ROWS": 32}, {"WAVE_ROWS": 32, "BLOCK_M": 128}, {"WAVE_ROWS": 16, "BLOCK_M": 32}]
    if decode:
        sets += [{"MODE": "fused"}, {"MODE": "fused", "PACK": "off"}]
        sets += [{"MODE": "split", "SPLIT_SEGS": k, "BLOCK_M": bm} for k in (1, 2, 4) for bm in (64, 128)]
        sets += [{"MODE": "split", "SPLIT_SEGS": 2, "waves_per_eu": 2}]
    else:
        sets += [{"PACK": "on"}, {"PACK": "on", "BLOCK_M": 128}]
    return sets


@pytest.mark.parametrize("heads", HEADS[:2])
def test_schedule_knobs_do_not_change_bits(heads):
    """Every schedule knob combination gives the training forward's bits for decode (3 segments,
    next to another sequence), chunked prefill, and the training forward itself."""
    hq, hkv, d = heads
    total = 3000
    q, k, v, cu = _varlen([total], hq, hkv, d, BF16, seed=17)
    sinks = _sinks(hq, 17)
    training = _train(q, k, v, cu, total, sinks=sinks)
    for opts in _knob_sets(decode=False):
        got = _train(q, k, v, cu, total, sinks=sinks, kernel_options={**FLY, **opts})
        assert _bits_equal(got, training), f"training with {opts}"
    oq, ok, ov, _ = _varlen([2600], hq, hkv, d, BF16, seed=18)
    k_cache, v_cache, block_table, _ = _paged(torch.cat([k, ok]), torch.cat([v, ov]), _cu([total, 2600]), 64)
    for t0, t1, decode in [(2999, 3000, True), (2963, 3000, False)]:
        step_q = torch.cat([q[t0:t1], oq[2599:2600]])
        seq_lens = torch.tensor([t1, 2600], dtype=torch.int32, device=DEVICE)
        for opts in [{}] + _knob_sets(decode):
            out = rl_paged_attention_func(
                step_q, k_cache, v_cache, _cu([t1 - t0, 1]), seq_lens, block_table, t1 - t0, total,
                sinks=sinks, kernel_options={**FLY, **opts},
            )  # fmt: skip
            assert _bits_equal(out[: t1 - t0], training[t0:t1]), f"tokens [{t0}, {t1}) with {opts}"
