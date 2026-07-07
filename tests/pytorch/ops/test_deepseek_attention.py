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
* CSA (from compressed pool, in-kernel gather)  (``compress_ratio == 4``)
  -> ``csa_attention_from_pool``

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
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("B,H,S,D", [(1, 8, 256, 512)])
@pytest.mark.parametrize("swa_window", [128])
@pytest.mark.parametrize("enable_sink", [True, False])
@pytest.mark.parametrize("backend", _BACKENDS, ids=_BACKEND_IDS)
def test_v4_hca_attention_bwd(dtype, B, H, S, D, swa_window, enable_sink, backend):
    """HCA split-mask backward (dq/dk/dv/dsink) vs an fp32 autograd reference.

    The FlyDSL HCA backward composes the local SWA dq/dkv kernels with the
    ported HCA pool dq/dkv kernels (MQA, B=1, pool_size<=32,
    hca_local_seqlen % 64 == 0). The forward stays on Triton (the FlyDSL HCA
    forward is not wired), so this exercises a Triton-fwd / FlyDSL-bwd pairing
    through the same joint-domain LSE convention.
    """
    _skip_if_flydsl_unsupported(backend, D)
    torch.manual_seed(0)
    dev = "cuda"
    scale = D**-0.5
    P = S // 128  # HCA pool size (2 for S=256)

    q = torch.randn(B, H, S, D, device=dev, dtype=dtype, requires_grad=True)
    k_local = torch.randn(B, 1, S, D, device=dev, dtype=dtype)
    v_local = torch.randn(B, 1, S, D, device=dev, dtype=dtype)
    pool_k = torch.randn(B, 1, P, D, device=dev, dtype=dtype)
    pool_v = torch.randn(B, 1, P, D, device=dev, dtype=dtype)
    # The op takes the concatenated [local | pool] K/V; make those the leaves.
    k_cat = torch.cat([k_local, pool_k], dim=2).detach().clone().requires_grad_(True)
    v_cat = torch.cat([v_local, pool_v], dim=2).detach().clone().requires_grad_(True)
    sink = (
        torch.randn(H, device=dev, dtype=dtype, requires_grad=True) if enable_sink else None
    )
    pool_mask = torch.zeros(S, P, device=dev, dtype=dtype)  # fully visible pool

    out = hca_attention(
        q, k_cat, v_cat, sink=sink, swa_window=swa_window, additive_mask=pool_mask,
        attn_dropout=0.0, training=True, scale=scale, hca_local_seqlen=S, backend=backend,
    )

    # fp32 autograd reference over the joint [local_swa | pool] mask.
    qf = q.detach().float().requires_grad_(True)
    kf = k_cat.detach().float().requires_grad_(True)
    vf = v_cat.detach().float().requires_grad_(True)
    sinkf = sink.detach().float().requires_grad_(True) if enable_sink else None
    local_mask = sliding_window_causal_mask(S, swa_window, device=dev, dtype=torch.float32)
    joint_mask = torch.cat([local_mask, pool_mask.float()], dim=1)
    out_ref = eager_hca_attention(
        qf, kf.expand(B, H, S + P, D), vf.expand(B, H, S + P, D),
        sink=sinkf, swa_window=swa_window, additive_mask=joint_mask,
        attn_dropout=0.0, training=True, scale=scale,
    )
    assert compute_snr(out_ref, out) > SNR_FWD

    g = torch.randn_like(out)
    out.backward(g)
    out_ref.backward(g.float())
    assert compute_snr(qf.grad, q.grad) > SNR_BWD
    assert compute_snr(kf.grad, k_cat.grad) > SNR_BWD
    assert compute_snr(vf.grad, v_cat.grad) > SNR_BWD
    if enable_sink:
        assert compute_snr(sinkf.grad, sink.grad) > SNR_BWD


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
    P = 64
    scale = D**-0.5
    swa_window = 128

    q = torch.randn(B, H, S, D, device=dev, dtype=dtype, requires_grad=True)
    k_local = torch.randn(B, H, S, D, device=dev, dtype=dtype, requires_grad=True)
    v_local = torch.randn(B, H, S, D, device=dev, dtype=dtype, requires_grad=True)
    pool = torch.randn(B, P, D, device=dev, dtype=dtype, requires_grad=True)
    topk = torch.empty(B, S, 0, device=dev, dtype=torch.int64)

    out = csa_attention_from_pool(
        q, k_local, v_local, pool, topk_idxs=topk, sink=None, swa_window=swa_window,
        attn_dropout=0.0, training=True, scale=scale,
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
    if enable_sink:
        assert compute_snr(sinkr.grad, sink.grad) > SNR_BWD

    # Pool gradient: validate against an fp32 reference instead of the bf16
    # one above. At the real index_topk (512 / 1024) each pool row is reused
    # by hundreds of queries, so a bf16 reference accumulates dpool in bf16 and
    # only reaches ~31 dB SNR vs fp32 truth — below SNR_BWD even for a correct
    # kernel. The kernel accumulates dpool in fp32, so the fair comparison is
    # against an fp32 reference (verify-accuracy: compare to higher precision).
    pf = pool.detach().float().requires_grad_(True)
    qf, kf, vf = q.detach().float(), k_local.detach().float(), v_local.detach().float()
    sinkf = sink.detach().float() if enable_sink else None
    gathered_f = pf[bidx, topk]
    out_f = eager_csa_attention(
        qf, kf, vf, gathered_f, sink=sinkf, swa_window=swa_window,
        sparse_mask=torch.zeros(B, S, K_topk, device=dev, dtype=torch.float32),
        attn_dropout=0.0, training=True, scale=scale,
    )
    out_f.backward(g.float())
    assert compute_snr(pf.grad, pool.grad) > SNR_BWD


# ---------------------------------------------------------------------------
# Fused single-latent sparse-MLA v2 path (the `flydsl_turbo` benchmark backend).
#
# The tests above exercise the SPLIT CSA path (csa_attention_from_pool with
# separate k_local/v_local/pool). The Primus benchmark bench_v4_attention.py
# instead drives the FUSED single-latent (K==V) sparse-MLA kernels directly:
#   sparse_mla_{fwd,bwd}_v4_flydsl(q, kv, topk, attn_sink, kv_lora_rank, scale)
# with kv = [local latent ++ pool] and topk = [SWA window ++ pool], zero rope pad.
# That entry point (tr16 fwd + M=16 dq bwd + the LDS-pad / interm-tiling perf
# work) had NO direct unit test — this guards it against regression across the
# real V4 shapes, INCLUDING the large-topk V4-Pro CSA case (topk=1152) where the
# dKV-intermediate tensor exceeds 2^31 elements. Reference = the same-repo Triton
# v2 fused kernel (independently validated); we compare fwd out + dq + dkv by SNR.
# NOTE gluon_v2 is intentionally NOT the reference: its dKV uses a different
# accumulation convention at large topk (norm differs ~1.7x), so turbo and Triton
# agree with each other but not with gluon — that is a gluon-convention artifact,
# not a turbo bug.
# ---------------------------------------------------------------------------
_ROPE_V2 = 64


def _build_v4form_fused(*, cr, H, S, D, seed=0):
    """Mirror bench_v4_attention._build_gluon_v4form: V4-form fused inputs."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    dev, dt = "cuda", torch.bfloat16
    W = 128
    latent = torch.randn(S, D, generator=g, device=dev, dtype=dt)
    q512 = torch.randn(S, H, D, generator=g, device=dev, dtype=dt)
    z_q = torch.zeros(S, H, _ROPE_V2, device=dev, dtype=dt)
    q_g = torch.cat([q512, z_q], dim=-1).contiguous()
    sink = torch.randn(H, generator=g, device=dev, dtype=torch.float32) * 0.1
    do = torch.randn(S, H, D, generator=g, device=dev, dtype=dt)

    ti = torch.arange(S, device=dev).view(S, 1)
    win = ti - W + 1 + torch.arange(W, device=dev).view(1, W)
    win = torch.where(win >= 0, win, torch.full_like(win, -1))

    if cr == 0:
        kv512 = latent.unsqueeze(1)
        topk = win
    else:
        P = max(S // cr, 1)
        pool = torch.randn(P, D, generator=g, device=dev, dtype=dt)
        kv512 = torch.cat([latent, pool], dim=0).unsqueeze(1)
        if cr == 4:
            K = min(1024 if H == 128 else 512, P)
            sp = torch.randint(0, P, (S, K), generator=g, device=dev)
            pool_topk = S + sp
        else:  # cr == 128 HCA full causal pool
            ps = torch.arange(P, device=dev).view(1, P)
            vis = ((ps + 1) * cr - 1) <= ti
            pool_topk = torch.where(vis, S + ps, torch.full_like(ps.expand(S, P), -1))
        topk = torch.cat([win, pool_topk], dim=1)

    tk = topk.shape[1]
    pad = ((tk + 63) // 64) * 64 - tk
    if pad > 0:
        topk = torch.cat([topk, torch.full((S, pad), -1, device=dev, dtype=topk.dtype)], dim=1)
    topk_g = topk.to(torch.int32).contiguous()
    z_kv = torch.zeros(kv512.shape[0], 1, _ROPE_V2, device=dev, dtype=dt)
    kv_g = torch.cat([kv512, z_kv], dim=-1).contiguous()
    return q_g, kv_g, topk_g, sink, do


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("H", [64, 128], ids=["flash", "pro"])
@pytest.mark.parametrize("cr", [0, 4, 128], ids=["swa", "csa", "hca"])
def test_v4_sparse_mla_v2_fused_flydsl_vs_triton(H, cr):
    """flydsl (tr16 fwd + M=16 dq bwd) vs same-repo Triton v2 fused reference,
    on the real V4-form shapes the benchmark uses (S=4096). Guards the fused
    entry point incl. V4-Pro CSA topk=1152 (interm > 2^31 elems)."""
    if get_device_compute_capability() < (9, 5):
        pytest.skip("FlyDSL sparse-MLA requires gfx950 (CDNA4).")
    from primus_turbo.flydsl.attention.kernels.sparse_mla_v2.dsa_fwd import sparse_mla_fwd_v4_flydsl
    from primus_turbo.flydsl.attention.kernels.sparse_mla_v2.dsa_bwd import sparse_mla_bwd_v4_flydsl
    from primus_turbo.triton.attention.deepseek.sparse_mla_v2.dsa_fwd import sparse_mla_fwd_v4_triton
    from primus_turbo.triton.attention.deepseek.sparse_mla_v2.dsa_bwd import sparse_mla_bwd_v4_triton

    import math
    D, S = 512, 4096
    q, kv, topk, sink, do = _build_v4form_fused(cr=cr, H=H, S=S, D=D)
    scale = 1.0 / math.sqrt(D)

    of, lf = sparse_mla_fwd_v4_flydsl(q, kv, topk, attn_sink=sink, kv_lora_rank=D, scale=scale)
    ot, lt = sparse_mla_fwd_v4_triton(q, kv, topk, attn_sink=sink, kv_lora_rank=D, scale=scale)
    assert compute_snr(ot, of) > SNR_FWD, f"fwd out SNR too low (H{H} cr{cr})"

    dqf, dkvf, dsf = sparse_mla_bwd_v4_flydsl(q, kv, of, do, topk, lf, attn_sink=sink, kv_lora_rank=D, scale=scale)
    dqt, dkvt, dst = sparse_mla_bwd_v4_triton(q, kv, ot, do, topk, lt, attn_sink=sink, kv_lora_rank=D, scale=scale)
    assert compute_snr(dqt[:, :, :D], dqf[:, :, :D]) > SNR_BWD, f"dq SNR too low (H{H} cr{cr})"
    assert compute_snr(dkvt[..., :D], dkvf[..., :D]) > SNR_BWD, f"dkv SNR too low (H{H} cr{cr})"
    if dsf is not None and dst is not None:
        assert compute_snr(dst, dsf) > SNR_BWD, f"dsink SNR too low (H{H} cr{cr})"
