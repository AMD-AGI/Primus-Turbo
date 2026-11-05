###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Paged KV in the FlyDSL flash attention forward (FA4's page_table + seqused_k).

k / v caches are ``[num_pages, page_size, Hk, D]``; ``page_table[b, i]`` is the physical
page holding keys ``i * page_size ..`` of sequence ``b``, and ``seqused_k[b]`` its
length. Page size, lengths and the table are runtime values, so each test sweeps them
inside one compiled kernel.
"""

import pytest
import torch

pytest.importorskip("flydsl")
from primus_turbo.flydsl.flex_attention import FlashAttentionForward  # noqa: E402
from primus_turbo.flydsl.flex_attention.mods import make_softcap_mods  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a ROCm GPU")

DTYPE = torch.bfloat16


def _paged_cache(seqlens, page_size, Hk, D, *, seed=0, extra_pages=7, shared_prefix_pages=0, dtype=DTYPE):
    """A shuffled page pool and its table. Unused table entries point past the pool, so
    a read of one would be out of range. With ``shared_prefix_pages``, every sequence's
    first pages are the same physical pages (prefix caching)."""
    gen = torch.Generator().manual_seed(seed)
    B = len(seqlens)
    pages_per_seq = [-(-n // page_size) for n in seqlens]
    max_pages = max(max(pages_per_seq), 1)
    shared = shared_prefix_pages
    num_pages = shared + sum(max(p - shared, 0) for p in pages_per_seq) + extra_pages
    perm = torch.randperm(num_pages, generator=gen)
    table = torch.full((B, max_pages), num_pages + 1000, dtype=torch.int32)
    nxt = shared
    for b, p in enumerate(pages_per_seq):
        for i in range(p):
            if i < shared:
                table[b, i] = perm[i]
            else:
                table[b, i] = perm[nxt]
                nxt += 1
    k_cache = torch.randn(num_pages, page_size, Hk, D, generator=gen).to("cuda", dtype)
    v_cache = torch.randn(num_pages, page_size, Hk, D, generator=gen).to("cuda", dtype)
    return k_cache, v_cache, table.cuda(), torch.tensor(seqlens, dtype=torch.int32, device="cuda")


def _gather(cache, table, b, n):
    page_size = cache.shape[1]
    pages = table[b, : -(-n // page_size)].long()
    return cache[pages].reshape(-1, *cache.shape[2:])[:n]


def _reference(q, k, v, scale, causal=False, window=None, sink=None, softcap=None):
    """fp32 attention for one sequence: q [Sq, Hq, D], k / v [Sk, Hk, D]; causal and
    window bottom-right aligned (FA4). Returns (out [Sq, Hq, Dv], lse [Hq, Sq])."""
    Sq, Hq, _ = q.shape
    Sk, Hk, _ = k.shape
    g = Hq // Hk
    qf = q.float().transpose(0, 1)
    kf = k.float().transpose(0, 1).repeat_interleave(g, 0)
    vf = v.float().transpose(0, 1).repeat_interleave(g, 0)
    s = qf @ kf.transpose(-1, -2) * scale
    if softcap is not None:
        s = softcap * torch.tanh(s / softcap)
    rel = torch.arange(Sk, device=q.device)[None, :] - torch.arange(Sq, device=q.device)[:, None] - (Sk - Sq)
    keep = torch.ones_like(rel, dtype=torch.bool)
    if causal:
        keep &= rel <= 0
    if window is not None:
        left, right = window
        if left >= 0:
            keep &= rel >= -left
        if right >= 0:
            keep &= rel <= right
    s = s.masked_fill(~keep, float("-inf"))
    if sink is not None:
        s = torch.cat([s, sink.float().view(Hq, 1, 1).expand(Hq, Sq, 1)], dim=-1)
    lse = torch.logsumexp(s, -1)
    p = torch.softmax(s, -1).nan_to_num(0.0)
    if sink is not None:
        p = p[..., :-1]
    return (p @ vf).transpose(0, 1), lse


def _check(out, lse, q, k_cache, v_cache, table, seqlens, scale, o_tol=2e-2, lse_tol=1e-3, **ref_kw):
    for b, n in enumerate(seqlens):
        k, v = _gather(k_cache, table, b, n), _gather(v_cache, table, b, n)
        ref_o, ref_lse = _reference(q[b], k, v, scale, **ref_kw)
        torch.testing.assert_close(out[b].float(), ref_o, atol=o_tol, rtol=o_tol, msg=f"out, sequence {b}")
        finite = torch.isfinite(ref_lse)
        torch.testing.assert_close(lse[b][finite], ref_lse[finite], atol=lse_tol, rtol=lse_tol,
                                   msg=f"lse, sequence {b}")


SEQLENS = (1, 37, 64, 500, 1024, 129)  # length 1, shorter than a page, page-aligned, ragged


@pytest.mark.parametrize("page_size", [16, 64, 256])
def test_decode_gqa(page_size):
    Hq, Hk, D = 8, 2, 128
    k_cache, v_cache, table, used = _paged_cache(SEQLENS, page_size, Hk, D, seed=page_size)
    q = torch.randn(len(SEQLENS), 1, Hq, D, device="cuda").to(DTYPE)
    fwd = FlashAttentionForward(head_dim=D, qhead_per_kvhead=Hq // Hk, has_seqused_k=True, paged_kv=True)
    out, lse = fwd(q, k_cache, v_cache, D**-0.5, seqused_k=used, page_table=table)
    _check(out, lse, q, k_cache, v_cache, table, SEQLENS, D**-0.5)


@pytest.mark.parametrize("page_size", [16, 128])
def test_chunked_prefill_causal(page_size):
    # 200 new queries against each sequence's cached keys (bottom-right causal).
    seqlens = (200, 700, 1300)
    Hq, Hk, D = 8, 2, 128
    k_cache, v_cache, table, used = _paged_cache(seqlens, page_size, Hk, D, seed=1)
    q = torch.randn(len(seqlens), 200, Hq, D, device="cuda").to(DTYPE)
    fwd = FlashAttentionForward(head_dim=D, qhead_per_kvhead=Hq // Hk, is_causal=True, has_seqused_k=True,
                                paged_kv=True)
    out, lse = fwd(q, k_cache, v_cache, D**-0.5, seqused_k=used, page_table=table)
    _check(out, lse, q, k_cache, v_cache, table, seqlens, D**-0.5, causal=True)


def test_shared_prefix_pages():
    seqlens = (300, 512, 260)
    Hq, Hk, D = 8, 2, 128
    k_cache, v_cache, table, used = _paged_cache(seqlens, 64, Hk, D, seed=2, shared_prefix_pages=4)
    assert torch.equal(table[0, :4], table[1, :4]) and torch.equal(table[0, :4], table[2, :4])
    q = torch.randn(len(seqlens), 1, Hq, D, device="cuda").to(DTYPE)
    fwd = FlashAttentionForward(head_dim=D, qhead_per_kvhead=Hq // Hk, has_seqused_k=True, paged_kv=True)
    out, lse = fwd(q, k_cache, v_cache, D**-0.5, seqused_k=used, page_table=table)
    _check(out, lse, q, k_cache, v_cache, table, seqlens, D**-0.5)


@pytest.mark.parametrize("num_splits", [0, 4])  # 0: FA4's heuristic
def test_decode_split_kv(num_splits):
    seqlens = (8192, 3000, 77, 5000)
    Hq, Hk, D = 8, 1, 128
    k_cache, v_cache, table, used = _paged_cache(seqlens, 64, Hk, D, seed=3)
    q = torch.randn(len(seqlens), 1, Hq, D, device="cuda").to(DTYPE)
    fwd = FlashAttentionForward(head_dim=D, qhead_per_kvhead=Hq // Hk, has_seqused_k=True, paged_kv=True,
                                split_kv=True)
    out, lse = fwd(q, k_cache, v_cache, D**-0.5, seqused_k=used, page_table=table, num_splits=num_splits)
    _check(out, lse, q, k_cache, v_cache, table, seqlens, D**-0.5)


def test_head_dim_64_mha():
    seqlens = (333, 64, 1)
    k_cache, v_cache, table, used = _paged_cache(seqlens, 32, 4, 64, seed=4)
    q = torch.randn(len(seqlens), 3, 4, 64, device="cuda").to(DTYPE)
    fwd = FlashAttentionForward(head_dim=64, is_causal=True, has_seqused_k=True, paged_kv=True)
    out, lse = fwd(q, k_cache, v_cache, 64**-0.5, seqused_k=used, page_table=table)
    _check(out, lse, q, k_cache, v_cache, table, seqlens, 64**-0.5, causal=True)


def test_sliding_window_with_sink():
    # gpt-oss style decode: a 128-key window plus a learnable sink per head.
    seqlens = (1000, 90, 513)
    Hq, Hk, D = 8, 1, 64
    k_cache, v_cache, table, used = _paged_cache(seqlens, 16, Hk, D, seed=5)
    q = torch.randn(len(seqlens), 1, Hq, D, device="cuda").to(DTYPE)
    sink = torch.randn(Hq, device="cuda")
    fwd = FlashAttentionForward(head_dim=D, qhead_per_kvhead=Hq // Hk, is_local=True, has_seqused_k=True,
                                paged_kv=True, has_learnable_sink=True)
    out, lse = fwd(q, k_cache, v_cache, D**-0.5, window_left=127, window_right=0, seqused_k=used,
                   page_table=table, learnable_sink=sink)
    _check(out, lse, q, k_cache, v_cache, table, seqlens, D**-0.5, window=(127, 0), sink=sink)


def test_softcap():
    seqlens = (700, 40)
    Hq, Hk, D = 4, 2, 128
    k_cache, v_cache, table, used = _paged_cache(seqlens, 64, Hk, D, seed=6)
    q = (torch.randn(len(seqlens), 1, Hq, D, device="cuda") * 3).to(DTYPE)
    softcap, _ = make_softcap_mods(10.0)
    fwd = FlashAttentionForward(head_dim=D, qhead_per_kvhead=Hq // Hk, has_seqused_k=True, paged_kv=True,
                                score_mod=softcap)
    out, lse = fwd(q, k_cache, v_cache, D**-0.5, seqused_k=used, page_table=table)
    _check(out, lse, q, k_cache, v_cache, table, seqlens, D**-0.5, softcap=10.0)


def _prefix_or_recent(b, h, q_idx, kv_idx, seqlen_info, aux):
    """Keep a per-sequence prefix (aux[0][b]) and the last 64 keys."""
    import flydsl.expr as fx

    return (kv_idx < fx.Int32(aux[0][b])) | (kv_idx >= seqlen_info.seqlen_k - fx.Int32(64))


def _alibi(score, b, h, q_idx, kv_idx, seqlen_info, aux):
    """score + slope[h] * (kv_idx - query position); slopes in aux[1]."""
    import flydsl.expr as fx

    return score + fx.Float32(aux[1][h]) * fx.Float32(kv_idx - (seqlen_info.seqlen_k - fx.Int32(1)))


def test_decode_mask_mod_score_mod_aux():
    # Decode-only shapes: the mask empties whole 256-key partitions in the middle of the
    # long sequences, and the bias depends on each sequence's length.
    seqlens = (1, 90, 257, 1000, 5000)
    Hq, Hk, D = 12, 2, 128
    k_cache, v_cache, table, used = _paged_cache(seqlens, 16, Hk, D, seed=8)
    q = torch.randn(len(seqlens), 1, Hq, D, device="cuda").to(DTYPE)
    prefix = torch.tensor([0, 10, 300, 40, 600], dtype=torch.int32, device="cuda")
    slopes = -torch.rand(Hq, device="cuda") * 0.05
    sink = torch.randn(Hq, device="cuda")
    fwd = FlashAttentionForward(head_dim=D, qhead_per_kvhead=Hq // Hk, is_local=True, has_seqused_k=True,
                                paged_kv=True, mask_mod=_prefix_or_recent, score_mod=_alibi, num_aux_tensors=2,
                                has_learnable_sink=True)
    out, lse = fwd(q, k_cache, v_cache, D**-0.5, window_left=900, window_right=0, seqused_k=used,
                   page_table=table, aux_tensors=(prefix, slopes), learnable_sink=sink)
    for b, n in enumerate(seqlens):
        k, v = _gather(k_cache, table, b, n), _gather(v_cache, table, b, n)
        kv = torch.arange(n, device="cuda")
        s = torch.einsum("hd,nhd->hn", q[b, 0].float(), k.float().repeat_interleave(Hq // Hk, 1)) * D**-0.5
        s = s + slopes[:, None] * (kv - (n - 1))
        keep = ((kv < prefix[b]) | (kv >= n - 64)) & (kv >= n - 1 - 900)
        s = torch.cat([s.masked_fill(~keep, float("-inf")), sink[:, None]], 1)
        ref_o = torch.einsum("hn,nhd->hd", torch.softmax(s, -1)[:, :n], v.float().repeat_interleave(Hq // Hk, 1))
        torch.testing.assert_close(out[b, 0].float(), ref_o, atol=2e-2, rtol=2e-2, msg=f"out, sequence {b}")
        torch.testing.assert_close(lse[b, :, 0], torch.logsumexp(s, -1), atol=1e-3, rtol=1e-3)


def _head_or_sparse_tail(b, h, q_idx, kv_idx, seqlen_info, aux):
    """The first 256 keys, then every third key of the last 1000."""
    import flydsl.expr as fx

    tail = kv_idx >= seqlen_info.seqlen_k - fx.Int32(1000)
    return (kv_idx < fx.Int32(256)) | (((kv_idx % fx.Int32(3)) == fx.Int32(0)) & tail)


@pytest.mark.parametrize("shared_heads", [False, True])
def test_decode_block_sparse(shared_heads):
    # FA4 block lists from the dense mask: full head blocks, partial tail blocks, and
    # absent middle blocks whose partitions the decode kernel skips.
    from primus_turbo.flydsl.flex_attention.torch_utils import dense_to_block_sparse

    seqlens = (1, 700, 5000, 2100)
    Hq, Hk, D = 8, 2, 128
    k_cache, v_cache, table, used = _paged_cache(seqlens, 16, Hk, D, seed=13)
    q = torch.randn(len(seqlens), 1, Hq, D, device="cuda").to(DTYPE)
    fwd = FlashAttentionForward(head_dim=D, qhead_per_kvhead=Hq // Hk, has_seqused_k=True, paged_kv=True,
                                mask_mod=_head_or_sparse_tail, block_sparse=True, has_full_blocks=True)
    sk = table.shape[1] * 16
    kv = torch.arange(sk, device="cuda")
    dense = torch.zeros(len(seqlens), 1 if shared_heads else Hq, 1, sk, dtype=torch.bool, device="cuda")
    for b, n in enumerate(seqlens):
        dense[b, ..., :n] = ((kv < 256) | ((kv % 3 == 0) & (kv >= n - 1000)))[:n]
    bst = dense_to_block_sparse(dense, (fwd.tile_m, fwd.tile_n))
    out, lse = fwd(q, k_cache, v_cache, D**-0.5, seqused_k=used, page_table=table, block_sparse_tensors=bst)
    for b, n in enumerate(seqlens):
        k, v = _gather(k_cache, table, b, n), _gather(v_cache, table, b, n)
        s = torch.einsum("hd,nhd->hn", q[b, 0].float(), k.float().repeat_interleave(Hq // Hk, 1)) * D**-0.5
        s = s.masked_fill(~dense[b, :, 0, :n].expand(Hq, n), float("-inf"))
        ref_o = torch.einsum("hn,nhd->hd", torch.softmax(s, -1), v.float().repeat_interleave(Hq // Hk, 1))
        torch.testing.assert_close(out[b, 0].float(), ref_o, atol=2e-2, rtol=2e-2, msg=f"out, sequence {b}")
        torch.testing.assert_close(lse[b, :, 0], torch.logsumexp(s, -1), atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("group", [1, 16])
def test_decode_group_sizes_and_empty_sequence(group):
    seqlens = (300, 0, 4097)
    Hk, D = 2, 64
    k_cache, v_cache, table, used = _paged_cache(seqlens, 64, Hk, D, seed=9)
    q = torch.randn(len(seqlens), 1, Hk * group, D, device="cuda").to(DTYPE)
    fwd = FlashAttentionForward(head_dim=D, qhead_per_kvhead=group, has_seqused_k=True, paged_kv=True)
    out, lse = fwd(q, k_cache, v_cache, D**-0.5, seqused_k=used, page_table=table)
    assert torch.all(out[1] == 0) and torch.all(torch.isneginf(lse[1]))
    _check(out[[0, 2]], lse[[0, 2]], q[[0, 2]], k_cache, v_cache, table[[0, 2]], (300, 4097), D**-0.5)


def test_fp8_kv_cache():
    # e4m3 in the GPU's native format, per-(batch, kv head) descales; bf16 output.
    from primus_turbo.flydsl.flex_attention.flash_fwd import _arch_has_ocp_fp8

    fmt = torch.float8_e4m3fn if _arch_has_ocp_fp8() else torch.float8_e4m3fnuz
    seqlens = (900, 64, 333)
    B, Hq, Hk, D = len(seqlens), 8, 2, 128
    k_cache, v_cache, table, used = _paged_cache(seqlens, 64, Hk, D, seed=7, dtype=torch.float32)
    q = torch.randn(B, 1, Hq, D, device="cuda")
    fmax = torch.finfo(fmt).max
    ones = torch.ones(B, Hk, device="cuda")
    # Per-tensor scales from the amax: a value past the format's range would cast to NaN.
    q_d, k_d, v_d = (ones * (t.abs().max() / fmax) for t in (q, k_cache, v_cache))
    q8, k8, v8 = ((t / d[0, 0]).to(fmt) for t, d in ((q, q_d), (k_cache, k_d), (v_cache, v_d)))
    fwd = FlashAttentionForward(dtype="fp8_e4m3", head_dim=D, qhead_per_kvhead=Hq // Hk, has_seqused_k=True,
                                paged_kv=True)
    out, lse = fwd(q8, k8, v8, D**-0.5, seqused_k=used, page_table=table,
                   q_descale=q_d, k_descale=k_d, v_descale=v_d)
    assert out.dtype == torch.bfloat16
    # Reference on the dequantized values; P's rounding to fp8 is not shared.
    deq = lambda t, d: t.float() * d[0, 0]  # noqa: E731
    _check(out, lse, deq(q8, q_d), deq(k8, k_d), deq(v8, v_d), table, seqlens, D**-0.5, o_tol=6e-2, lse_tol=1e-2)


@pytest.mark.parametrize("per_token", [False, True])
def test_decode_fp8_kv_cache_bf16_query(per_token):
    # aiter / SGLang decode: bf16 q, native-fp8 cache, k / v scales per tensor or per token.
    from primus_turbo.flydsl.flex_attention.flash_decode import FlashAttentionDecode
    from primus_turbo.flydsl.flex_attention.flash_fwd import _arch_has_ocp_fp8

    fmt = torch.float8_e4m3fn if _arch_has_ocp_fp8() else torch.float8_e4m3fnuz
    fmax = torch.finfo(fmt).max
    seqlens = (1, 900, 64, 4097)
    Hq, Hk, D = 12, 2, 128
    k_cache, v_cache, table, used = _paged_cache(seqlens, 16, Hk, D, seed=11, dtype=torch.float32)
    if per_token:  # token magnitudes that differ, so a misindexed scale shows
        gen = torch.Generator().manual_seed(12)
        k_cache = k_cache * torch.rand(*k_cache.shape[:3], 1, generator=gen).cuda() * 4
        v_cache = v_cache * torch.rand(*v_cache.shape[:3], 1, generator=gen).cuda() * 4
    amax = lambda t: t.abs().amax(-1) if per_token else t.abs().amax().reshape(1)  # noqa: E731
    k_s, v_s = amax(k_cache) / fmax, amax(v_cache) / fmax
    expand = lambda s: s[..., None] if per_token else s  # noqa: E731
    k8, v8 = (k_cache / expand(k_s)).to(fmt), (v_cache / expand(v_s)).to(fmt)
    q = torch.randn(len(seqlens), Hq, D, device="cuda").to(DTYPE)
    dec = FlashAttentionDecode(head_dim=D, qhead_per_kvhead=Hq // Hk, kv_dtype="fp8",
                               kv_scale="token" if per_token else "tensor")
    out, lse = dec(q, k8, v8, table, used, D**-0.5, k_scale=k_s, v_scale=v_s)
    deq_k, deq_v = k8.float() * expand(k_s), v8.float() * expand(v_s)
    _check(out[:, None], lse[:, :, None], q[:, None], deq_k, deq_v, table, seqlens, D**-0.5, o_tol=3e-2,
           lse_tol=2e-3)
    with pytest.raises(ValueError, match="needs k_scale and v_scale"):
        dec(q, k8, v8, table, used, D**-0.5)


def test_validation():
    k_cache, v_cache, table, used = _paged_cache((100,), 16, 1, 64)
    q = torch.randn(1, 1, 1, 64, device="cuda").to(DTYPE)
    fwd = FlashAttentionForward(head_dim=64, has_seqused_k=True, paged_kv=True)
    with pytest.raises(ValueError, match="page_table must be int32"):
        fwd(q, k_cache, v_cache, 0.125, seqused_k=used, page_table=table.long())
    with pytest.raises(ValueError, match="page_table does not match"):
        fwd(q, k_cache, v_cache, 0.125, seqused_k=used)
    with pytest.raises(ValueError, match="not supported with cu_seqlens_k"):
        FlashAttentionForward(head_dim=64, varlen_k=True, paged_kv=True)
