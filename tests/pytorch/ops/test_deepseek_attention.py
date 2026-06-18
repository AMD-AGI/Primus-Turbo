###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the ported DeepSeek-V4 attention Triton kernels.

Covers the three per-layer attention types against their eager
references:

* dense / SWA          (``compress_ratio == 0``)   -> ``hca_attention``
* HCA                  (``compress_ratio == 128``)  -> ``hca_attention`` (split-mask)
* CSA (pre-gathered)   (``compress_ratio == 4``)    -> ``csa_attention``
* CSA (from compressed pool, in-kernel gather)      -> ``csa_attention_from_pool``

Forward is checked by SNR vs the eager reference; backward gradients are
checked by SNR against autograd through the eager reference.

Both production model envelopes are exercised (HuggingFace
``deepseek-ai/DeepSeek-V4-{Flash,Pro}`` config.json). They share
head_dim=512, MQA (K_H=1) and sliding_window=128; at the attention-op
level they differ only in head count and indexer top-k:

* V4-Flash: num_attention_heads = 64,  index_topk = 512
* V4-Pro:   num_attention_heads = 128, index_topk = 1024

The shapes below use test-sized B/H/S so they stay fast, but keep the
two head counts distinct (Flash<=Pro) and add a larger top-k case so the
Pro indexer regime is covered. Real per-model values are recorded here:
"""

import pytest
import torch

from primus_turbo.pytorch.core.backend import BackendType
from primus_turbo.pytorch.core.utils import get_device_compute_capability
from primus_turbo.pytorch.ops.attention import (
    eager_hca_attention,
    eager_csa_attention,
    sliding_window_causal_mask,
    hca_attention,
    csa_attention,
    csa_attention_from_pool,
)
from tests.pytorch.test_utils import compute_snr

# bf16 SNR threshold (matches the existing flash-attn test bar).
SNR_FWD = 40.0
SNR_BWD = 35.0

# Attention backends exercised in the same process (Triton vs FlyDSL A/B,
# design §3.2 / §7.1). ``None`` keeps the default (Triton). The FlyDSL forward
# is gfx950 + D=512 only; it falls back to Triton elsewhere and for the CSA
# paths (the CSA FlyDSL sparse branch is a later round, design §4.7), so these
# parametrisations also cover the dispatch fallback contract.
_BACKENDS = [None, BackendType.FLYDSL]
_BACKEND_IDS = ["triton", "flydsl"]


def _skip_if_flydsl_unsupported(backend, D):
    """FlyDSL attention forward needs gfx950 (ds_read_*_tr / 16 B G2S) and the
    D = 512 specialisation (design §3.3). Off-support, ``backend=FLYDSL`` falls
    back to Triton in the dispatcher, so the run is still valid — we only skip
    when the FlyDSL kernel cannot be reached at all to avoid a misleading pass."""
    if backend is BackendType.FLYDSL:
        if get_device_compute_capability() < (9, 5):
            pytest.skip("FlyDSL attention forward requires gfx950 (CDNA4).")

# Real per-model attention envelope (HuggingFace config.json), for reference.
# Only num_heads and index_topk differ at the attention-op level.
V4_FLASH = {"num_heads": 64, "head_dim": 512, "sliding_window": 128, "index_topk": 512}
V4_PRO = {"num_heads": 128, "head_dim": 512, "sliding_window": 128, "index_topk": 1024}


def _clone_leaf(t):
    out = t.detach().clone().requires_grad_(t.requires_grad)
    return out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "B,H,S,D",
    [
        (1, 8, 256, 512),    # V4-Flash-like (toy heads)
        (2, 4, 512, 512),    # V4-Flash-like (toy heads)
        (1, 16, 256, 512),   # V4-Pro-like: wider head count
    ],
)
@pytest.mark.parametrize("swa_window", [128, 0])
@pytest.mark.parametrize("mqa", [True, False])
@pytest.mark.parametrize("enable_sink", [True, False])
@pytest.mark.parametrize("backend", _BACKENDS, ids=_BACKEND_IDS)
def test_v4_dense_attention(dtype, B, H, S, D, swa_window, mqa, enable_sink, backend):
    _skip_if_flydsl_unsupported(backend, D)
    torch.manual_seed(0)
    dev = "cuda"
    scale = D**-0.5
    KH = 1 if mqa else H

    q = torch.randn(B, H, S, D, device=dev, dtype=dtype, requires_grad=True)
    k = torch.randn(B, KH, S, D, device=dev, dtype=dtype, requires_grad=True)
    v = torch.randn(B, KH, S, D, device=dev, dtype=dtype, requires_grad=True)
    sink = (
        torch.randn(H, device=dev, dtype=dtype, requires_grad=True) if enable_sink else None
    )

    # Reference uses broadcast K/V to full H.
    qr, kr, vr = _clone_leaf(q), _clone_leaf(k), _clone_leaf(v)
    sinkr = _clone_leaf(sink) if enable_sink else None

    out = hca_attention(
        q, k, v, sink=sink, swa_window=swa_window, additive_mask=None,
        attn_dropout=0.0, training=True, scale=scale, backend=backend,
    )
    out_ref = eager_hca_attention(
        qr, kr.expand(B, H, S, D), vr.expand(B, H, S, D), sink=sinkr,
        swa_window=swa_window, additive_mask=None, attn_dropout=0.0,
        training=True, scale=scale,
    )
    assert compute_snr(out_ref, out) > SNR_FWD

    g = torch.randn_like(out)
    out.backward(g)
    out_ref.backward(g)
    assert compute_snr(qr.grad, q.grad) > SNR_BWD
    assert compute_snr(kr.grad, k.grad) > SNR_BWD
    assert compute_snr(vr.grad, v.grad) > SNR_BWD
    if enable_sink:
        assert compute_snr(sinkr.grad, sink.grad) > SNR_BWD


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("B,H,S,D", [(1, 8, 256, 512)])
@pytest.mark.parametrize("swa_window", [128])
@pytest.mark.parametrize("enable_sink", [True, False])
@pytest.mark.parametrize("backend", _BACKENDS, ids=_BACKEND_IDS)
def test_v4_hca_attention(dtype, B, H, S, D, swa_window, enable_sink, backend):
    _skip_if_flydsl_unsupported(backend, D)
    torch.manual_seed(0)
    dev = "cuda"
    scale = D**-0.5
    P = S // 128  # HCA pool size

    q = torch.randn(B, H, S, D, device=dev, dtype=dtype, requires_grad=True)
    k_local = torch.randn(B, 1, S, D, device=dev, dtype=dtype, requires_grad=True)
    v_local = torch.randn(B, 1, S, D, device=dev, dtype=dtype, requires_grad=True)
    pool_k = torch.randn(B, 1, P, D, device=dev, dtype=dtype, requires_grad=True)
    pool_v = torch.randn(B, 1, P, D, device=dev, dtype=dtype, requires_grad=True)
    sink = (
        torch.randn(H, device=dev, dtype=dtype, requires_grad=True) if enable_sink else None
    )

    # Caller concatenates [local | pool] and passes pool-only mask.
    k_cat = torch.cat([k_local, pool_k], dim=2)
    v_cat = torch.cat([v_local, pool_v], dim=2)
    pool_mask = torch.zeros(S, P, device=dev, dtype=dtype)  # fully visible pool

    out = hca_attention(
        q, k_cat, v_cat, sink=sink, swa_window=swa_window, additive_mask=pool_mask,
        attn_dropout=0.0, training=True, scale=scale, hca_local_seqlen=S, backend=backend,
    )

    # Reference: joint mask [S, S+P] = cat([local_swa_mask, pool_mask]).
    local_mask = sliding_window_causal_mask(S, swa_window, device=dev, dtype=dtype)
    joint_mask = torch.cat([local_mask, pool_mask], dim=1)
    qr = _clone_leaf(q)
    sinkr = _clone_leaf(sink) if enable_sink else None
    out_ref = eager_hca_attention(
        qr, k_cat.detach().expand(B, H, S + P, D), v_cat.detach().expand(B, H, S + P, D),
        sink=sinkr, swa_window=swa_window, additive_mask=joint_mask,
        attn_dropout=0.0, training=True, scale=scale,
    )
    assert compute_snr(out_ref, out) > SNR_FWD


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "B,H,S,D",
    [
        (1, 8, 256, 512),    # V4-Flash-like (toy heads)
        (2, 4, 512, 512),    # V4-Flash-like (toy heads)
        (1, 16, 256, 512),   # V4-Pro-like: wider head count
    ],
)
@pytest.mark.parametrize("K_topk", [16, 64, 128])  # 128 covers the larger V4-Pro top-k regime
@pytest.mark.parametrize("enable_sink", [True, False])
@pytest.mark.parametrize("backend", _BACKENDS, ids=_BACKEND_IDS)
def test_v4_csa_gathered(dtype, B, H, S, D, K_topk, enable_sink, backend):
    _skip_if_flydsl_unsupported(backend, D)
    torch.manual_seed(0)
    dev = "cuda"
    scale = D**-0.5
    swa_window = 128

    q = torch.randn(B, H, S, D, device=dev, dtype=dtype, requires_grad=True)
    k_local = torch.randn(B, H, S, D, device=dev, dtype=dtype, requires_grad=True)
    v_local = torch.randn(B, H, S, D, device=dev, dtype=dtype, requires_grad=True)
    gathered = torch.randn(B, S, K_topk, D, device=dev, dtype=dtype, requires_grad=True)
    sparse_mask = torch.zeros(B, S, K_topk, device=dev, dtype=dtype)
    sink = (
        torch.randn(H, device=dev, dtype=dtype, requires_grad=True) if enable_sink else None
    )

    qr, kr, vr, gr = (_clone_leaf(t) for t in (q, k_local, v_local, gathered))
    sinkr = _clone_leaf(sink) if enable_sink else None

    out = csa_attention(
        q, k_local, v_local, gathered, sink=sink, swa_window=swa_window,
        sparse_mask=sparse_mask, attn_dropout=0.0, training=True, scale=scale, backend=backend,
    )
    out_ref = eager_csa_attention(
        qr, kr, vr, gr, sink=sinkr, swa_window=swa_window,
        sparse_mask=sparse_mask, attn_dropout=0.0, training=True, scale=scale,
    )
    assert compute_snr(out_ref, out) > SNR_FWD

    g = torch.randn_like(out)
    out.backward(g)
    out_ref.backward(g)
    assert compute_snr(qr.grad, q.grad) > SNR_BWD
    assert compute_snr(kr.grad, k_local.grad) > SNR_BWD
    assert compute_snr(vr.grad, v_local.grad) > SNR_BWD
    assert compute_snr(gr.grad, gathered.grad) > SNR_BWD
    if enable_sink:
        assert compute_snr(sinkr.grad, sink.grad) > SNR_BWD


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize(
    "B,H,S,D",
    [
        (1, 8, 256, 512),    # V4-Flash-like (toy heads)
        (1, 16, 256, 512),   # V4-Pro-like: wider head count
    ],
)
@pytest.mark.parametrize(
    "K_topk,P",
    [(16, 64), (32, 128), (64, 256)],  # (64, 256) covers the larger V4-Pro top-k regime
)
@pytest.mark.parametrize("enable_sink", [True, False])
@pytest.mark.parametrize("backend", _BACKENDS, ids=_BACKEND_IDS)
def test_v4_csa_from_pool(dtype, B, H, S, D, K_topk, P, enable_sink, backend):
    _skip_if_flydsl_unsupported(backend, D)
    torch.manual_seed(0)
    dev = "cuda"
    scale = D**-0.5
    swa_window = 128

    q = torch.randn(B, H, S, D, device=dev, dtype=dtype, requires_grad=True)
    k_local = torch.randn(B, H, S, D, device=dev, dtype=dtype, requires_grad=True)
    v_local = torch.randn(B, H, S, D, device=dev, dtype=dtype, requires_grad=True)
    pool = torch.randn(B, P, D, device=dev, dtype=dtype, requires_grad=True)
    topk = torch.randint(0, P, (B, S, K_topk), device=dev, dtype=torch.int64)
    sink = (
        torch.randn(H, device=dev, dtype=dtype, requires_grad=True) if enable_sink else None
    )

    qr, kr, vr, pr = (_clone_leaf(t) for t in (q, k_local, v_local, pool))
    sinkr = _clone_leaf(sink) if enable_sink else None

    out = csa_attention_from_pool(
        q, k_local, v_local, pool, topk_idxs=topk, sink=sink, swa_window=swa_window,
        attn_dropout=0.0, training=True, scale=scale, backend=backend,
    )

    # Reference: gather pool by topk into [B, S, K, D].
    bidx = torch.arange(B, device=dev).view(B, 1, 1)
    gathered_ref = pr[bidx, topk]
    sparse_mask = torch.zeros(B, S, K_topk, device=dev, dtype=dtype)
    out_ref = eager_csa_attention(
        qr, kr, vr, gathered_ref, sink=sinkr, swa_window=swa_window,
        sparse_mask=sparse_mask, attn_dropout=0.0, training=True, scale=scale,
    )
    assert compute_snr(out_ref, out) > SNR_FWD

    g = torch.randn_like(out)
    out.backward(g)
    out_ref.backward(g)
    assert compute_snr(qr.grad, q.grad) > SNR_BWD
    assert compute_snr(kr.grad, k_local.grad) > SNR_BWD
    assert compute_snr(vr.grad, v_local.grad) > SNR_BWD
    assert compute_snr(pr.grad, pool.grad) > SNR_BWD
    if enable_sink:
        assert compute_snr(sinkr.grad, sink.grad) > SNR_BWD


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_v4_csa_ktopk_zero_falls_back_to_dense():
    """K_topk == 0 short-circuits to the dense kernel (CSA local-only limit)."""
    torch.manual_seed(0)
    dev = "cuda"
    dtype = torch.bfloat16
    B, H, S, D = 1, 8, 256, 512
    scale = D**-0.5
    swa_window = 128

    q = torch.randn(B, H, S, D, device=dev, dtype=dtype, requires_grad=True)
    k_local = torch.randn(B, H, S, D, device=dev, dtype=dtype, requires_grad=True)
    v_local = torch.randn(B, H, S, D, device=dev, dtype=dtype, requires_grad=True)
    gathered = torch.empty(B, S, 0, D, device=dev, dtype=dtype)
    sparse_mask = torch.empty(B, S, 0, device=dev, dtype=dtype)

    out = csa_attention(
        q, k_local, v_local, gathered, sink=None, swa_window=swa_window,
        sparse_mask=sparse_mask, attn_dropout=0.0, training=True, scale=scale,
    )
    out_dense = hca_attention(
        q, k_local, v_local, sink=None, swa_window=swa_window, additive_mask=None,
        attn_dropout=0.0, training=True, scale=scale,
    )
    torch.testing.assert_close(out, out_dense, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_v4_dense_attention_flydsl_all_swa_masked():
    """Degenerate FlyDSL forward edge (design §7.1): the first query rows under
    SWA see an empty window for some K-tiles; the NEG_INF / 0-guards must keep
    the output finite and match the Triton backend. Skips unless the FlyDSL
    forward is reachable (gfx950)."""
    _skip_if_flydsl_unsupported(BackendType.FLYDSL, 512)
    torch.manual_seed(0)
    dev = "cuda"
    dtype = torch.bfloat16
    B, H, S, D = 1, 8, 256, 512
    scale = D**-0.5
    swa_window = 128

    q = torch.randn(B, H, S, D, device=dev, dtype=dtype)
    k = torch.randn(B, 1, S, D, device=dev, dtype=dtype)
    v = torch.randn(B, 1, S, D, device=dev, dtype=dtype)

    out_fly = hca_attention(
        q, k, v, sink=None, swa_window=swa_window, additive_mask=None,
        attn_dropout=0.0, training=False, scale=scale, backend=BackendType.FLYDSL,
    )
    out_tri = hca_attention(
        q, k, v, sink=None, swa_window=swa_window, additive_mask=None,
        attn_dropout=0.0, training=False, scale=scale, backend=BackendType.TRITON,
    )
    assert torch.isfinite(out_fly).all()
    assert compute_snr(out_tri, out_fly) > SNR_FWD


# ---------------------------------------------------------------------------
# Real production envelopes (V4-Flash / V4-Pro). These use the real head count
# (64 / 128), head_dim 512, MQA, SWA 128 and index_topk (512 / 1024). Sequence
# length is kept small (S=256) so the eager [B, H, S, Sk] reference fits in
# memory; the head count / head dim / sliding window / top-k match the HF
# config exactly. Marked ``slow`` -- deselect with ``-m 'not slow'``.
# ---------------------------------------------------------------------------

_REAL_S = 256  # eager reference materializes [B, H, S, Sk]; keep S small.


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("model", [V4_FLASH, V4_PRO], ids=["flash", "pro"])
@pytest.mark.parametrize("enable_sink", [True, False])
def test_v4_dense_attention_real_shape(model, enable_sink):
    torch.manual_seed(0)
    dev = "cuda"
    dtype = torch.bfloat16
    B, H, S, D = 1, model["num_heads"], _REAL_S, model["head_dim"]
    swa_window = model["sliding_window"]
    scale = D**-0.5

    q = torch.randn(B, H, S, D, device=dev, dtype=dtype, requires_grad=True)
    k = torch.randn(B, 1, S, D, device=dev, dtype=dtype, requires_grad=True)  # MQA
    v = torch.randn(B, 1, S, D, device=dev, dtype=dtype, requires_grad=True)
    sink = (
        torch.randn(H, device=dev, dtype=dtype, requires_grad=True) if enable_sink else None
    )

    qr, kr, vr = _clone_leaf(q), _clone_leaf(k), _clone_leaf(v)
    sinkr = _clone_leaf(sink) if enable_sink else None

    out = hca_attention(
        q, k, v, sink=sink, swa_window=swa_window, additive_mask=None,
        attn_dropout=0.0, training=True, scale=scale,
    )
    out_ref = eager_hca_attention(
        qr, kr.expand(B, H, S, D), vr.expand(B, H, S, D), sink=sinkr,
        swa_window=swa_window, additive_mask=None, attn_dropout=0.0,
        training=True, scale=scale,
    )
    assert compute_snr(out_ref, out) > SNR_FWD

    g = torch.randn_like(out)
    out.backward(g)
    out_ref.backward(g)
    assert compute_snr(qr.grad, q.grad) > SNR_BWD
    assert compute_snr(kr.grad, k.grad) > SNR_BWD
    assert compute_snr(vr.grad, v.grad) > SNR_BWD
    if enable_sink:
        assert compute_snr(sinkr.grad, sink.grad) > SNR_BWD


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("model", [V4_FLASH, V4_PRO], ids=["flash", "pro"])
@pytest.mark.parametrize("enable_sink", [True, False])
def test_v4_csa_from_pool_real_shape(model, enable_sink):
    torch.manual_seed(0)
    dev = "cuda"
    dtype = torch.bfloat16
    B, H, S, D = 1, model["num_heads"], _REAL_S, model["head_dim"]
    swa_window = model["sliding_window"]
    K_topk = model["index_topk"]
    P = K_topk  # pool sized so the real top-k is exercised (not P-capped)
    scale = D**-0.5

    q = torch.randn(B, H, S, D, device=dev, dtype=dtype, requires_grad=True)
    k_local = torch.randn(B, H, S, D, device=dev, dtype=dtype, requires_grad=True)
    v_local = torch.randn(B, H, S, D, device=dev, dtype=dtype, requires_grad=True)
    pool = torch.randn(B, P, D, device=dev, dtype=dtype, requires_grad=True)
    topk = torch.randint(0, P, (B, S, K_topk), device=dev, dtype=torch.int64)
    sink = (
        torch.randn(H, device=dev, dtype=dtype, requires_grad=True) if enable_sink else None
    )

    qr, kr, vr, pr = (_clone_leaf(t) for t in (q, k_local, v_local, pool))
    sinkr = _clone_leaf(sink) if enable_sink else None

    out = csa_attention_from_pool(
        q, k_local, v_local, pool, topk_idxs=topk, sink=sink, swa_window=swa_window,
        attn_dropout=0.0, training=True, scale=scale,
    )

    bidx = torch.arange(B, device=dev).view(B, 1, 1)
    gathered_ref = pr[bidx, topk]
    sparse_mask = torch.zeros(B, S, K_topk, device=dev, dtype=dtype)
    out_ref = eager_csa_attention(
        qr, kr, vr, gathered_ref, sink=sinkr, swa_window=swa_window,
        sparse_mask=sparse_mask, attn_dropout=0.0, training=True, scale=scale,
    )
    assert compute_snr(out_ref, out) > SNR_FWD

    g = torch.randn_like(out)
    out.backward(g)
    out_ref.backward(g)
    assert compute_snr(qr.grad, q.grad) > SNR_BWD
    assert compute_snr(kr.grad, k_local.grad) > SNR_BWD
    assert compute_snr(vr.grad, v_local.grad) > SNR_BWD
    assert compute_snr(pr.grad, pool.grad) > SNR_BWD
    if enable_sink:
        assert compute_snr(sinkr.grad, sink.grad) > SNR_BWD
