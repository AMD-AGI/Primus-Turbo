###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""gfx1250 FlyDSL sparse-MLA kernels called directly: forward, backward, dsink and the dKV band gather.

Inputs follow the single-latent DeepSeek-V4 layout (512 latent + 64 rope columns, sliding window of 128 plus an
optional pool): pure window (cr 0), random pool (cr 4) and the deterministic causal pool (cr 128). Outputs are
checked by SNR against the Triton kernels, the backward tiling variants (staged loads, warp-specialized CSA
backward, runtime-loop window / pool backward) must give the bits of the default backward, and a backward whose
dKV work runs on a caller stream must give the same bits as the in-stream one. H = 64 is the tuned head count;
H = 32 and H = 128 take the generic kernels.
"""

import math

import pytest
import torch

from primus_turbo.pytorch.kernels.attention.sparse_mla_impl import SparseMlaFwdDispatcher  # noqa: F401
from primus_turbo.triton.attention.sparse_mla import sparse_mla_bwd_triton, sparse_mla_fwd_triton
from tests.pytorch.ops.gfx1250_helper import bits_equal, rel_l2, requires_flydsl, requires_gfx1250
from tests.pytorch.test_utils import compute_snr

pytestmark = [requires_gfx1250, requires_flydsl]

D, ROPE, WINDOW = 512, 64, 128
SNR_FLOOR = 40.0


def _topk_pool(cr, seqlen):
    if cr == 0:
        return 0, 0
    pool = max(seqlen // cr, 1)
    return pool, (min(64, pool) if cr == 4 else 0)


def _build(cr, heads, seqlen, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    dev, dt = "cuda", torch.bfloat16
    pool, topk_pool = _topk_pool(cr, seqlen)
    latent = torch.randn(seqlen, D, generator=gen, device=dev, dtype=dt)
    q = torch.randn(seqlen, heads, D, generator=gen, device=dev, dtype=dt)
    q = torch.cat([q, torch.zeros(seqlen, heads, ROPE, device=dev, dtype=dt)], -1)
    sink = torch.randn(heads, generator=gen, device=dev, dtype=torch.float32) * 0.1
    grad_out = torch.randn(seqlen, heads, D, generator=gen, device=dev, dtype=dt)
    tok = torch.arange(seqlen, device=dev).view(seqlen, 1)
    win = tok - WINDOW + 1 + torch.arange(WINDOW, device=dev).view(1, WINDOW)
    win = torch.where(win >= 0, win, torch.full_like(win, -1))
    if cr == 0:
        kv, topk = latent.unsqueeze(1), win
    else:
        p = torch.randn(pool, D, generator=gen, device=dev, dtype=dt)
        kv = torch.cat([latent, p], 0).unsqueeze(1)
        if cr == 4:
            # each pool row at most once per token (the window + pool band contract)
            pool_topk = (
                seqlen + torch.rand(seqlen, pool, generator=gen, device=dev).topk(topk_pool, dim=1).indices
            )
        else:
            ps = torch.arange(pool, device=dev).view(1, pool)
            pool_topk = torch.where(
                ((ps + 1) * cr - 1) <= tok, seqlen + ps, torch.full_like(ps.expand(seqlen, pool), -1)
            )
        topk = torch.cat([win, pool_topk], 1)
    pad = ((topk.shape[1] + 63) // 64) * 64 - topk.shape[1]
    if pad > 0:
        topk = torch.cat([topk, torch.full((seqlen, pad), -1, device=dev, dtype=topk.dtype)], 1)
    kv = torch.cat([kv, torch.zeros(kv.shape[0], 1, ROPE, device=dev, dtype=dt)], -1)
    return q.contiguous(), kv.contiguous(), topk.to(torch.int32).contiguous(), sink, grad_out, pool


CASES = [(cr, h) for cr in (0, 4, 128) for h in (64, 32, 128)]
CASE_IDS = [f"cr{cr}-H{h}" for cr, h in CASES]
SEQLEN = 512
SCALE = 1.0 / math.sqrt(D)


@pytest.mark.parametrize("cr, heads", CASES, ids=CASE_IDS)
def test_forward_matches_triton(cr, heads):
    from primus_turbo.flydsl.attention.gfx1250_sparse_mla.fwd_kernel import sparse_mla_fwd_gfx1250

    q, kv, topk, sink, _, _ = _build(cr, heads, SEQLEN)
    o_ref, lse_ref = sparse_mla_fwd_triton(q, kv, topk, attn_sink=sink, kv_lora_rank=D, scale=SCALE)
    o, lse = sparse_mla_fwd_gfx1250(q, kv, topk, attn_sink=sink, kv_lora_rank=D, scale=SCALE)
    assert torch.isfinite(o).all()
    assert compute_snr(o_ref, o) > SNR_FLOOR
    assert compute_snr(lse_ref, lse) > SNR_FLOOR
    o2, lse2 = sparse_mla_fwd_gfx1250(q, kv, topk, attn_sink=sink, kv_lora_rank=D, scale=SCALE)
    assert bits_equal(o, o2) and bits_equal(lse, lse2)


def _bwd(q, kv, o, g, topk, lse, sink, cr, pool, **kw):
    from primus_turbo.flydsl.attention.gfx1250_sparse_mla.bwd_kernel import sparse_mla_bwd_gfx1250

    band = (WINDOW, q.shape[0], pool) if cr == 4 else None
    out = sparse_mla_bwd_gfx1250(
        q, kv, o, g, topk, lse, attn_sink=sink, kv_lora_rank=D, scale=SCALE, band_pool=band, **kw
    )
    torch.cuda.synchronize()
    return out


@pytest.mark.parametrize("cr, heads", CASES, ids=CASE_IDS)
def test_backward_matches_triton_and_is_repeatable(cr, heads):
    from primus_turbo.flydsl.attention.gfx1250_sparse_mla.fwd_kernel import sparse_mla_fwd_gfx1250

    q, kv, topk, sink, g, pool = _build(cr, heads, SEQLEN)
    o, lse = sparse_mla_fwd_gfx1250(q, kv, topk, attn_sink=sink, kv_lora_rank=D, scale=SCALE)
    dq_ref, dkv_ref, dsink_ref = sparse_mla_bwd_triton(
        q, kv, o, g, topk, lse, attn_sink=sink, kv_lora_rank=D, scale=SCALE
    )
    dq, dkv, dsink = _bwd(q, kv, o, g, topk, lse, sink, cr, pool)
    assert compute_snr(dq_ref, dq) > SNR_FLOOR
    assert compute_snr(dkv_ref, dkv) > SNR_FLOOR
    assert compute_snr(dsink_ref, dsink) > SNR_FLOOR
    dq2, dkv2, dsink2 = _bwd(q, kv, o, g, topk, lse, sink, cr, pool)
    assert bits_equal(dq, dq2) and bits_equal(dkv, dkv2) and bits_equal(dsink, dsink2)


@pytest.mark.parametrize(
    "env",
    [
        {"PRIMUS_SMLA_BWD_V2": "1"},
        {"PRIMUS_SMLA_BWD_V2": "1", "PRIMUS_CSABWD_PP": "1"},
        {"PRIMUS_SWAHCA_LOOP": "1"},
        {"PRIMUS_SMLA_BWD_V2": "1", "PRIMUS_CSABWD_PP": "1", "PRIMUS_SWAHCA_LOOP": "1"},
    ],
    ids=["v2", "v2_pp", "swahca_loop", "all"],
)
@pytest.mark.parametrize(
    "cr, heads",
    [(0, 64), (4, 64), (128, 64), (4, 32), (4, 128)],
    ids=["cr0-H64", "cr4-H64", "cr128-H64", "cr4-H32", "cr4-H128"],
)
def test_backward_variants_bits_equal_default(monkeypatch, env, cr, heads):
    """Backward tiling variants write the bits of the default backward (dq, dKV and dsink)."""
    from primus_turbo.flydsl.attention.gfx1250_sparse_mla.fwd_kernel import sparse_mla_fwd_gfx1250

    for name in ("PRIMUS_SMLA_BWD_V2", "PRIMUS_CSABWD_PP", "PRIMUS_SWAHCA_LOOP"):
        monkeypatch.delenv(name, raising=False)
    q, kv, topk, sink, g, pool = _build(cr, heads, SEQLEN, seed=3)
    o, lse = sparse_mla_fwd_gfx1250(q, kv, topk, attn_sink=sink, kv_lora_rank=D, scale=SCALE)
    base = _bwd(q, kv, o, g, topk, lse, sink, cr, pool)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    got = _bwd(q, kv, o, g, topk, lse, sink, cr, pool)
    for a, b in zip(base, got):
        assert bits_equal(a, b)


def test_forward_pipeline_variant_bits_equal_default(monkeypatch):
    from primus_turbo.flydsl.attention.gfx1250_sparse_mla.fwd_kernel import sparse_mla_fwd_gfx1250

    q, kv, topk, sink, _, _ = _build(4, 64, SEQLEN, seed=5)
    res = []
    for pipe in ("0", "1"):
        monkeypatch.setenv("PRIMUS_SMLA_FWD_PIPE", pipe)
        res.append(
            sparse_mla_fwd_gfx1250(q, kv, topk, attn_sink=sink, kv_lora_rank=D, scale=SCALE, use_swa=False)
        )
    assert bits_equal(res[0][0], res[1][0]) and bits_equal(res[0][1], res[1][1])


def test_backward_dkv_on_caller_stream_bits_equal():
    """With ``dkv_stream`` the dKV gather is issued on the caller's stream; after joining it equals the in-stream dKV."""
    from primus_turbo.flydsl.attention.gfx1250_sparse_mla.fwd_kernel import sparse_mla_fwd_gfx1250

    cr, heads = 4, 64
    q, kv, topk, sink, g, pool = _build(cr, heads, SEQLEN, seed=7)
    o, lse = sparse_mla_fwd_gfx1250(q, kv, topk, attn_sink=sink, kv_lora_rank=D, scale=SCALE)
    base = _bwd(q, kv, o, g, topk, lse, sink, cr, pool)
    stream, keep = torch.cuda.Stream(), []
    dq, dkv, dsink = _bwd(q, kv, o, g, topk, lse, sink, cr, pool, dkv_stream=stream, dkv_keep=keep)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    assert bits_equal(dq, base[0]) and bits_equal(dkv, base[1]) and bits_equal(dsink, base[2])


def test_backward_dpool_out_is_the_pool_rows_of_dkv():
    """On the single-chunk band path ``dpool_out`` receives the contiguous bf16 pool rows of dKV (so the framework
    adapter can skip its strided copy); dKV itself is unchanged by asking for it."""
    from primus_turbo.flydsl.attention.gfx1250_sparse_mla.fwd_kernel import sparse_mla_fwd_gfx1250

    cr, heads = 4, 64
    q, kv, topk, sink, g, pool = _build(cr, heads, SEQLEN, seed=9)
    o, lse = sparse_mla_fwd_gfx1250(q, kv, topk, attn_sink=sink, kv_lora_rank=D, scale=SCALE)
    base = _bwd(q, kv, o, g, topk, lse, sink, cr, pool)
    dpool = []
    dq, dkv, dsink = _bwd(q, kv, o, g, topk, lse, sink, cr, pool, dpool_out=dpool)
    assert len(dpool) == 1 and dpool[0].is_contiguous() and tuple(dpool[0].shape) == (pool, D)
    assert dpool[0].dtype == torch.bfloat16
    assert bits_equal(dpool[0], dkv.view(-1, dkv.shape[-1])[SEQLEN:, :D])
    assert bits_equal(dq, base[0]) and bits_equal(dkv, base[1]) and bits_equal(dsink, base[2])


@pytest.mark.parametrize("accumulate", [False, True], ids=["overwrite", "accumulate"])
@pytest.mark.parametrize("cr", [0, 4], ids=["cr0", "cr4"])
def test_backward_dsink_into_main_grad(cr, accumulate):
    """``dsink_main_grad=(mg, accumulate)`` writes the sink gradient into a bf16 ``main_grad`` (overwrite or bf16 add)
    and returns None for it; dQ and dKV keep the bits of the plain backward."""
    from primus_turbo.flydsl.attention.gfx1250_sparse_mla.fwd_kernel import sparse_mla_fwd_gfx1250

    heads = 64
    q, kv, topk, sink, g, pool = _build(cr, heads, SEQLEN, seed=11)
    o, lse = sparse_mla_fwd_gfx1250(q, kv, topk, attn_sink=sink, kv_lora_rank=D, scale=SCALE)
    base = _bwd(q, kv, o, g, topk, lse, sink, cr, pool)
    mg0 = (torch.randn(heads, device="cuda") * 0.1).bfloat16()
    mg = mg0.clone()
    dq, dkv, dsink = _bwd(q, kv, o, g, topk, lse, sink, cr, pool, dsink_main_grad=(mg, accumulate))
    assert dsink is None
    assert bits_equal(dq, base[0]) and bits_equal(dkv, base[1])
    want = (mg0.float() + base[2].float()) if accumulate else base[2].float()
    assert rel_l2(mg.float(), want) < 2e-2
