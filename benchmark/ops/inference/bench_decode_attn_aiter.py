###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Inference decode attention micro-benchmark for aiter (per-rank, single GPU).

Each family maps to the exact kernel SGLang's aiter backend uses for that model
on ROCm (verified against sglang/srt/layers/attention/aiter_backend.py), with a
fallback only where the fast path has no variant for the shape:
  - gqa : paged_attention_ragged   (full GQA, NHD paged KV cache; e.g. MiniMax-M2)
  - mla : mla_decode_fwd (asm)      (DeepSeek/Kimi absorbed MLA, latent KV).
          fp8 has asm variants only for per-rank nhead in {16, 128}; other nhead
          (e.g. Kimi-K2 nhead=64) fall back to the triton MLA decode kernel.
  - swa : unified_attention (triton) (sliding-window layers; SGLang forces this
          path whenever a sliding-window KV pool is active), with attention sinks
          (gpt-oss). For asymmetric qk/v head dims (mimo qk=192/v=128) no native
          aiter kernel exists, so v is padded to qk_head_dim and the output is
          sliced back -- the working fallback (see bench_swa for details).
  - dsa : fp8 paged-MQA-logits indexer + sparse mla_decode_fwd (DeepSeek sparse
          attention; GLM-5.1). The indexer scores the whole context and keeps the
          top-`index_topk` tokens, then absorbed-MLA decode runs over just those
          (KV bounded by topk). fp8 nhead the asm MLA lacks -> triton MLA fallback.

Decode is memory-bound, so we sweep batch x context_len and report KV bandwidth.
KV cache runs in bf16 and fp8. TP shards heads on this rank via
attn_tp = tp_size // dp_size (SGLang DP-attention semantics).

    python bench_decode_attn_aiter.py [--model M] [--ctx-spread S] [--tp-size N] [--dp-size N] [--check]
"""

import argparse
import contextlib
import os
from datetime import datetime

import torch
import aiter.mla
from aiter.ops.attention import paged_attention_ragged

# family "gqa": num_q_heads, num_kv_heads, head_dim.
# family "mla": num_heads, kv_lora_rank, qk_rope_head_dim (absorbed decode:
#               qk_head_dim = kv_lora_rank + qk_rope, v_head_dim = kv_lora_rank).
# family "swa": num_q_heads, num_kv_heads, qk_head_dim, v_head_dim, window, sinks
#               (sliding-window full attention layers; unified_attention kernel).
# family "dsa": num_heads, kv_lora_rank, qk_rope_head_dim (absorbed MLA, as "mla")
#               + index_n_heads, index_head_dim, index_topk (the fp8 indexer).
MODELS = {
    # https://huggingface.co/MiniMaxAI/MiniMax-M2.7  (229B; full GQA)
    "minimax-m2.7": {"family": "gqa", "num_q_heads": 48, "num_kv_heads": 8, "head_dim": 128},
    # GLM-5.1 (GlmMoeDsaForCausalLM; MLA + DeepSeek sparse attention). Absorbed
    # MLA decode: qk = kv_lora_rank + qk_rope = 576, v = kv_lora_rank = 512
    # (qk_nope=192/v_head_dim=256 are absorbed away at decode). Indexer scores
    # the full ctx with index_n_heads x index_head_dim, keeps index_topk tokens.
    "glm-5.1": {"family": "dsa", "num_heads": 64, "kv_lora_rank": 512, "qk_rope_head_dim": 64,
                "index_n_heads": 32, "index_head_dim": 128, "index_topk": 2048},
    # https://huggingface.co/deepseek-ai/DeepSeek-R1  (671B; MLA)
    "deepseek-r1": {"family": "mla", "num_heads": 128, "kv_lora_rank": 512, "qk_rope_head_dim": 64},
    # https://huggingface.co/moonshotai/Kimi-K2.6  (1T; MLA, DeepSeek-V3 arch)
    "kimi-k2.6": {"family": "mla", "num_heads": 64, "kv_lora_rank": 512, "qk_rope_head_dim": 64},
    # https://huggingface.co/openai/gpt-oss-120b  (117B; SWA layers, window 128, attention sinks)
    "gpt-oss-120b": {"family": "swa", "num_q_heads": 64, "num_kv_heads": 8,
                     "qk_head_dim": 64, "v_head_dim": 64, "window": 128, "sinks": True},
    # https://huggingface.co/XiaomiMiMo/MiMo-V2.5-Pro  (1.02T; SWA layers, window 128)
    "mimo-v2.5-pro": {"family": "swa", "num_q_heads": 128, "num_kv_heads": 8,
                      "qk_head_dim": 192, "v_head_dim": 128, "window": 128, "sinks": False},
}

DEFAULT_BATCHES = [1, 8, 16, 32, 64, 128, 256]
DEFAULT_CTX = [1024, 4096, 8192, 16384, 32768, 65536]

_PARTITION_SIZE = 256  # aiter paged_attention_ragged partition size.
_MLA_PAGE_SIZE = 1  # MLA latent cache addressed at token granularity.
# aiter's fp8 MLA decode *asm* kernel only has variants for these per-rank head
# counts; others (e.g. 64) abort hard (SIGABRT). For unsupported nhead we fall
# back to the triton MLA decode (newer aiter only); if unavailable we mark the
# cell unsupported rather than let the asm kernel crash the process.
_MLA_FP8_NHEADS = {16, 128}
_MLA_TRITON_BLOCK = 64  # triton MLA decode block_size (>= 16).


@contextlib.contextmanager
def suppress_output():
    """Silence aiter's chatty per-call stdout/stderr at the fd level."""
    with open(os.devnull, "w") as devnull:
        old_out, old_err = os.dup(1), os.dup(2)
        os.dup2(devnull.fileno(), 1)
        os.dup2(devnull.fileno(), 2)
        try:
            yield
        finally:
            os.dup2(old_out, 1)
            os.dup2(old_err, 2)
            os.close(old_out)
            os.close(old_err)


def compute_snr(ref: torch.Tensor, actual: torch.Tensor) -> float:
    """Signal-to-Noise Ratio (dB); higher is better."""
    ref_f = ref.to(torch.float64)
    act_f = actual.to(torch.float64)
    signal = ref_f.norm().pow(2)
    noise = (ref_f - act_f).norm().pow(2)
    return 10 * torch.log10(signal / (noise + 1e-12)).item()


def fp8_dtype():
    """OCP fp8 (e4m3fn) on gfx950 (MI355X); e4m3fnuz on older CDNA (MI300)."""
    props = torch.cuda.get_device_properties(0)
    if (props.major, props.minor) == (9, 5):
        return torch.float8_e4m3fn
    return torch.float8_e4m3fnuz


def _time_ms(fn, warmup=10, iters=50):
    """GPU time per call via CUDA events (excludes host/Python launch overhead)."""
    with suppress_output():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def gen_ctx_lens(batch, ctx, spread, seed):
    """Per-request context lengths; deterministic per cell so bf16/fp8 align.
    spread=0 -> all equal; else uniform in [ctx*(1-s), ctx*(1+s)] (mean ~ctx)."""
    if spread <= 0:
        return torch.full((batch,), ctx, dtype=torch.int64)
    lo = max(1, round(ctx * (1 - spread)))
    hi = max(lo, round(ctx * (1 + spread)))
    g = torch.Generator().manual_seed(seed)
    return torch.randint(lo, hi + 1, (batch,), generator=g, dtype=torch.int64)


def shuffled_pages(num_pages, seed):
    """Random (reproducible) physical page placement. Real serving allocates KV
    pages from a shared pool, so a sequence's pages are scattered across HBM
    rather than contiguous; the resulting gather is less cache-friendly. Returns
    a CPU permutation `perm` where logical page slot i lives at physical page
    perm[i] -- used for both the kernel's page table and the reference gather."""
    g = torch.Generator().manual_seed(int(seed))
    return torch.randperm(num_pages, generator=g)


# --------------------------------------------------------------------------- #
# GQA family: paged_attention_ragged
# --------------------------------------------------------------------------- #


def gqa_ref(query, key_cache, value_cache, ctx_lens, page_offsets, page_indices, scale, num_q_heads, num_kv_heads):
    """Naive paged GQA decode reference (fp32). query [B, Hq, D];
    caches [num_pages, page_size, Hkv, D]; seq i uses physical pages
    page_indices[off[i]:off[i+1]] (same scattered layout the kernel sees)."""
    B, Hq, D = query.shape
    rep = num_q_heads // num_kv_heads
    out = torch.empty(B, Hq, D, dtype=torch.float32, device=query.device)
    kf, vf = key_cache.float(), value_cache.float()
    for b in range(B):
        c = int(ctx_lens[b])
        phys = page_indices[int(page_offsets[b]) : int(page_offsets[b + 1])].to(kf.device)
        flat_k = kf[phys].reshape(-1, num_kv_heads, D)[:c]
        flat_v = vf[phys].reshape(-1, num_kv_heads, D)[:c]
        k = flat_k.repeat_interleave(rep, dim=1)
        v = flat_v.repeat_interleave(rep, dim=1)
        q = query[b].float()
        probs = torch.softmax(torch.einsum("hd,chd->hc", q, k) * scale, dim=-1)
        out[b] = torch.einsum("hc,chd->hd", probs, v)
    return out


def bench_gqa(batch, ctx, cfg, dtype, kv_dtype, page_size, attn_tp, ctx_lens_cpu, seed, device, check):
    Hq = cfg["num_q_heads"] // attn_tp
    Hkv = max(1, cfg["num_kv_heads"] // attn_tp)  # KV replicated when attn_tp > kv_heads
    D = cfg["head_dim"]
    scale = 1.0 / (D**0.5)

    pages_per_seq = (ctx_lens_cpu + page_size - 1) // page_size
    page_offsets = torch.cat([torch.zeros(1, dtype=torch.int64), pages_per_seq.cumsum(0)])
    num_pages = int(page_offsets[-1])
    total_tokens = int(ctx_lens_cpu.sum())
    max_ctx = int(ctx_lens_cpu.max())
    perm = shuffled_pages(num_pages, seed)  # scattered physical page layout

    query = torch.randn(batch, Hq, D, device=device, dtype=dtype)
    k_ref = torch.randn(num_pages, page_size, Hkv, D, device=device, dtype=dtype)
    v_ref = torch.randn(num_pages, page_size, Hkv, D, device=device, dtype=dtype)
    if kv_dtype == "fp8":
        cdt = fp8_dtype()
        key_cache, value_cache, kv_cache_dtype = k_ref.to(cdt), v_ref.to(cdt), "fp8_e4m3"
    else:
        key_cache, value_cache, kv_cache_dtype = k_ref, v_ref, "auto"

    kv_indptr = page_offsets.to(device=device, dtype=torch.int32)
    kv_page_indices = perm.to(device=device, dtype=torch.int32)
    kv_last_page_lens = (((ctx_lens_cpu - 1) % page_size) + 1).to(device=device, dtype=torch.int32)
    ctx_lens = ctx_lens_cpu.to(device=device, dtype=torch.int32)

    max_num_partitions = (max_ctx + _PARTITION_SIZE - 1) // _PARTITION_SIZE
    ws_bytes = (batch * Hq * max_num_partitions * D) * 4 + 2 * (batch * Hq * max_num_partitions) * 4
    workspace = torch.empty(ws_bytes, dtype=torch.uint8, device=device)
    k_scale = torch.ones(1, dtype=torch.float32, device=device)
    v_scale = torch.ones(1, dtype=torch.float32, device=device)
    out = torch.empty(batch, Hq, D, device=device, dtype=dtype)

    def fn():
        return paged_attention_ragged(
            out, workspace, query, key_cache, value_cache, scale,
            kv_indptr, kv_page_indices, kv_last_page_lens, page_size, max_num_partitions,
            None, kv_cache_dtype, "NHD", 0.0, k_scale, v_scale, None, _PARTITION_SIZE,
        )

    snr = None
    if check:
        with suppress_output():
            fn()
        ref = gqa_ref(query, k_ref, v_ref, ctx_lens, page_offsets, perm, scale, Hq, Hkv)
        snr = compute_snr(ref, out.float())

    time_ms = _time_ms(fn)
    kv_bytes = 2 * total_tokens * Hkv * D * key_cache.element_size()
    bw_gbps = kv_bytes / (time_ms * 1e-3) / 1e9
    tflops = (2 * 2 * Hq * D * total_tokens) / (time_ms * 1e-3) / 1e12
    return time_ms, bw_gbps, tflops, snr


# --------------------------------------------------------------------------- #
# MLA family: mla_decode_fwd (absorbed)
# --------------------------------------------------------------------------- #


def mla_ref(q, kv_buffer, kv_indptr, kv_indices, kv_lora_rank, scale):
    """Naive absorbed-MLA decode reference (fp32). q [bs, nhead, qk];
    kv_buffer [num_tokens, 1, qk]; K = latent, V = latent[:kv_lora_rank]."""
    bs, nhead, qk = q.shape
    out = torch.empty(bs, nhead, kv_lora_rank, dtype=torch.float32, device=q.device)
    kvf = kv_buffer.float()
    for b in range(bs):
        idx = kv_indices[kv_indptr[b] : kv_indptr[b + 1]]
        kvc = kvf[idx, 0]
        probs = torch.softmax(torch.einsum("hd,cd->hc", q[b].float(), kvc) * scale, dim=-1)
        out[b] = torch.einsum("hc,cd->hd", probs, kvc[:, :kv_lora_rank])
    return out


def bench_mla_triton(batch, nhead, lora, qk, scale, ctx_lens_cpu, dtype, seed, device, check, pool_tokens=None):
    """fp8 MLA decode via the triton paged kernel (newer aiter), used when the
    asm kernel has no variant for this nhead. Lazy import: unavailable -> error
    caught by the caller and marked unsupported. pool_tokens (DSA) scatters each
    seq's read blocks across a full-ctx-sized block pool."""
    try:
        from aiter.ops.triton.attention.mla import mla_decode_fwd as triton_mla
    except ImportError as e:
        raise RuntimeError(f"fp8 MLA unsupported for nhead={nhead} (no asm variant; triton mla unavailable)") from e

    rope = qk - lora
    blk = _MLA_TRITON_BLOCK
    pages_per_seq = (ctx_lens_cpu + blk - 1) // blk
    page_off = torch.cat([torch.zeros(1, dtype=torch.int64), pages_per_seq.cumsum(0)])
    num_blocks = int(page_off[-1])
    max_blocks = int(pages_per_seq.max())
    pool_blocks = (int(pool_tokens) + blk - 1) // blk if pool_tokens is not None else num_blocks
    perm = shuffled_pages(pool_blocks, seed)[:num_blocks]  # read blocks scattered across the pool

    cu_seqlens_q = torch.arange(0, batch + 1, device=device, dtype=torch.int32)  # 1 q/seq
    seqused_k = ctx_lens_cpu.to(device=device, dtype=torch.int32)
    block_tables = torch.zeros(batch, max_blocks, device=device, dtype=torch.int32)
    for i in range(batch):
        n = int(pages_per_seq[i])
        block_tables[i, :n] = perm[int(page_off[i]) : int(page_off[i]) + n].to(device)

    cdt = fp8_dtype()
    q = torch.randn(batch, nhead, qk, device=device, dtype=dtype)
    kv_ref = torch.randn(pool_blocks, blk, 1, qk, device=device, dtype=dtype)
    q_in, kv_buffer = q.to(cdt), kv_ref.to(cdt)
    q_descale = torch.ones(1, dtype=torch.float32, device=device)
    kv_descale = torch.ones(1, dtype=torch.float32, device=device)
    out = torch.empty(batch, nhead, lora, device=device, dtype=dtype)

    def fn():
        return triton_mla(
            q=q_in, kv_buffer=kv_buffer, out=out, cu_seqlens_q=cu_seqlens_q,
            seqused_k=seqused_k, max_seqlen_kv=int(ctx_lens_cpu.max()), block_tables=block_tables,
            softmax_scale=scale, kv_lora_rank=lora, qk_rope_head_dim=rope, causal=True,
            q_descale=q_descale, kv_descale=kv_descale,
        )

    snr = None
    if check:
        with suppress_output():
            fn()
        ref = torch.empty(batch, nhead, lora, dtype=torch.float32, device=device)
        kvf = kv_ref.float()  # [pool_blocks, blk, 1, qk]
        for i in range(batch):
            c = int(ctx_lens_cpu[i])
            n = int(pages_per_seq[i])
            phys = perm[int(page_off[i]) : int(page_off[i]) + n].to(device)
            kvc = kvf[phys].reshape(-1, qk)[:c]  # logical tokens 0..c-1, scattered blocks
            probs = torch.softmax(torch.einsum("hd,cd->hc", q[i].float(), kvc) * scale, dim=-1)
            ref[i] = torch.einsum("hc,cd->hd", probs, kvc[:, :lora])
        snr = compute_snr(ref, out.float())
    return fn, kv_buffer.element_size(), snr


def bench_mla(batch, ctx, cfg, dtype, kv_dtype, attn_tp, ctx_lens_cpu, seed, device, check):
    nhead = cfg["num_heads"] // attn_tp  # TP shards q heads; latent KV is replicated
    lora = cfg["kv_lora_rank"]
    qk = lora + cfg["qk_rope_head_dim"]
    scale = 1.0 / (qk**0.5)
    total_tokens = int(ctx_lens_cpu.sum())

    # fp8 nhead the asm kernel lacks -> triton fallback (avoids asm SIGABRT).
    if kv_dtype == "fp8" and nhead not in _MLA_FP8_NHEADS:
        fn, elem, snr = bench_mla_triton(batch, nhead, lora, qk, scale, ctx_lens_cpu, dtype, seed, device, check)
    else:
        fn, elem, snr = _bench_mla_asm(batch, nhead, lora, qk, scale, ctx_lens_cpu, dtype, kv_dtype, seed, device, check)

    time_ms = _time_ms(fn)
    kv_bytes = total_tokens * qk * elem  # latent read once (K/V shared)
    bw_gbps = kv_bytes / (time_ms * 1e-3) / 1e9
    tflops = (2 * nhead * total_tokens * (qk + lora)) / (time_ms * 1e-3) / 1e12
    return time_ms, bw_gbps, tflops, snr


def _bench_mla_asm(batch, nhead, lora, qk, scale, ctx_lens_cpu, dtype, kv_dtype, seed, device, check, pool_tokens=None):
    """fp8/bf16 MLA decode via the asm kernel (fast path). Returns (fn, elem, snr).
    Reads ctx_lens_cpu[b] latent tokens/seq scattered across a pool of `pool_tokens`
    (defaults to the read count). DSA passes pool_tokens=full ctx so the selected
    topk tokens gather across the whole KV cache, not just a topk-sized buffer."""
    total_read = int(ctx_lens_cpu.sum())
    total_pool = int(pool_tokens) if pool_tokens is not None else total_read
    qo_indptr = torch.arange(0, batch + 1, device=device, dtype=torch.int32)  # 1 q token/seq
    kv_indptr = torch.zeros(batch + 1, device=device, dtype=torch.int32)
    kv_indptr[1:] = torch.cumsum(ctx_lens_cpu, dim=0).to(device)
    # Latent KV is token-paged (page_size=1); scatter the read slots across the pool.
    kv_indices = shuffled_pages(total_pool, seed)[:total_read].to(device=device, dtype=torch.int32)
    kv_last_page_lens = torch.ones(batch, device=device, dtype=torch.int32)

    q = torch.randn(batch, nhead, qk, device=device, dtype=dtype)
    kv_ref = torch.randn(total_pool, 1, qk, device=device, dtype=dtype)
    q_scale = kv_scale = None
    if kv_dtype == "fp8":
        cdt = fp8_dtype()
        q_in, kv_buffer = q.to(cdt), kv_ref.to(cdt)
        q_scale = torch.ones(1, dtype=torch.float32, device=device)
        kv_scale = torch.ones(1, dtype=torch.float32, device=device)
    else:
        q_in, kv_buffer = q, kv_ref
    out = torch.empty(batch, nhead, lora, device=device, dtype=dtype)
    kv_view = kv_buffer.view(total_pool, _MLA_PAGE_SIZE, 1, qk)

    def fn():
        return aiter.mla.mla_decode_fwd(
            q_in, kv_view, out, qo_indptr, kv_indptr, kv_indices, kv_last_page_lens,
            1, _MLA_PAGE_SIZE, 1, scale, q_scale=q_scale, kv_scale=kv_scale,
        )

    snr = None
    if check:
        with suppress_output():
            fn()
        ref = mla_ref(q, kv_ref, kv_indptr, kv_indices, lora, scale)
        snr = compute_snr(ref, out.float())
    return fn, kv_buffer.element_size(), snr


# --------------------------------------------------------------------------- #
# DSA family: fp8 paged-MQA-logits indexer + sparse mla_decode_fwd
# (DeepSeek sparse attention; GLM-5.1 / DeepSeek-V3.2). Decode = score all ctx
# with a lightweight fp8 indexer, pick top-`index_topk` tokens, then run the
# absorbed-MLA decode over only those tokens (KV traffic bounded by topk).
# --------------------------------------------------------------------------- #

_DSA_INDEX_PAGE = 64  # HIP preshuffle paged-MQA blocksize (page_size % 16 == 0).


def _build_dsa_indexer(batch, ctx_lens_cpu, n_heads, head_dim, topk, seed, device):
    """DSA indexer fast-path: aiter fp8 paged-MQA logits over the full context +
    top-k selection. Always fp8 by design. Random index-K cache -> the logit
    values are meaningless, but the kernel runs at representative speed (we time
    it, we don't validate the data-dependent top-k numerically). Returns
    (fn, bytes_per_token) where bytes_per_token includes the per-block fp8 scale."""
    from aiter.ops.triton.pa_mqa_logits import deepgemm_fp8_paged_mqa_logits

    page = _DSA_INDEX_PAGE
    hdsf = head_dim + head_dim // 128 * 4  # fp8 key bytes + per-128 fp32 scale bytes
    pages_per_seq = (ctx_lens_cpu + page - 1) // page
    page_off = torch.cat([torch.zeros(1, dtype=torch.int64), pages_per_seq.cumsum(0)])
    num_pages = int(page_off[-1])
    max_blocks = int(pages_per_seq.max())
    max_ctx = int(ctx_lens_cpu.max())
    perm = shuffled_pages(num_pages, seed)

    block_tables = torch.zeros(batch, max_blocks, device=device, dtype=torch.int32)
    for i in range(batch):
        n = int(pages_per_seq[i])
        block_tables[i, :n] = perm[int(page_off[i]) : int(page_off[i]) + n].to(device)
    ctx32 = ctx_lens_cpu.to(device=device, dtype=torch.int32)
    max_seq = max_blocks * page

    q = torch.randn(batch, n_heads, head_dim, device=device).to(fp8_dtype()).unsqueeze(1)
    weights = torch.randn(batch, n_heads, device=device, dtype=torch.float32)
    kv = torch.randint(0, 255, (num_pages, page * hdsf), dtype=torch.uint8, device=device)
    kv = kv.view(num_pages, page, 1, hdsf)
    logits = torch.empty(batch, max_seq, device=device, dtype=torch.float32)
    tk = min(topk, max_ctx)

    def fn():
        deepgemm_fp8_paged_mqa_logits(
            q, kv, weights, logits, ctx32, block_tables, max_seq,
            Preshuffle=True, KVBlockSize=page,
        )
        torch.topk(logits[:, :max_ctx], tk, dim=-1)  # top-k token selection

    return fn, hdsf


def bench_dsa(batch, ctx, cfg, dtype, kv_dtype, attn_tp, ctx_lens_cpu, seed, device, check):
    lora = cfg["kv_lora_rank"]
    qk = lora + cfg["qk_rope_head_dim"]
    nhead = cfg["num_heads"] // attn_tp  # TP shards q heads; latent KV replicated
    scale = 1.0 / (qk**0.5)
    topk = cfg["index_topk"]
    # Sparse decode attends to <= topk tokens (all of them when ctx <= topk),
    # gathered from the full-ctx latent pool (pool_tokens) -- the selected tokens
    # are scattered across the whole KV cache, not a topk-sized buffer.
    sel_lens = torch.minimum(ctx_lens_cpu, torch.tensor(topk, dtype=torch.int64))
    pool_tokens = int(ctx_lens_cpu.sum())

    # Stage 2 -- sparse absorbed-MLA decode over the selected tokens. Reuses the
    # MLA fast path (asm) with the fp8 nhead triton fallback. SNR validates this
    # stage; the data-dependent indexer top-k is timed but not numerically checked.
    if kv_dtype == "fp8" and nhead not in _MLA_FP8_NHEADS:
        mla_fn, elem, snr = bench_mla_triton(batch, nhead, lora, qk, scale, sel_lens, dtype, seed, device, check, pool_tokens)
    else:
        mla_fn, elem, snr = _bench_mla_asm(batch, nhead, lora, qk, scale, sel_lens, dtype, kv_dtype, seed, device, check, pool_tokens)

    # Stage 1 -- fp8 indexer (scores the full context, picks top-k).
    idx_fn, idx_bytes = _build_dsa_indexer(
        batch, ctx_lens_cpu, cfg["index_n_heads"], cfg["index_head_dim"], topk, seed + 7, device
    )

    def fn():
        idx_fn()
        return mla_fn()

    total_ctx = pool_tokens
    total_sel = int(sel_lens.sum())
    time_ms = _time_ms(fn)
    # Indexer reads the fp8 index-K (idx_bytes/token incl scale) over the full
    # context; sparse MLA reads the latent (qk wide) over only selected tokens.
    kv_bytes = total_ctx * idx_bytes + total_sel * qk * elem
    bw_gbps = kv_bytes / (time_ms * 1e-3) / 1e9
    tflops = (
        2 * cfg["index_n_heads"] * cfg["index_head_dim"] * total_ctx
        + 2 * nhead * total_sel * (qk + lora)
    ) / (time_ms * 1e-3) / 1e12
    return time_ms, bw_gbps, tflops, snr


# --------------------------------------------------------------------------- #
# SWA family: unified_attention (sliding window, optional attention sinks)
# --------------------------------------------------------------------------- #


def swa_ref(query, k_ref, v_ref, ctx_lens, page_offsets, page_indices, page_size, scale, Hq, Hkv, window, sinks):
    """Naive sliding-window decode reference (fp32). Each query attends to the
    last `window` keys. Optional per-head attention sinks add a no-value logit.
    Seq b's tokens live in physical pages page_indices[off[b]:off[b+1]]."""
    D = query.shape[-1]
    Dv = v_ref.shape[-1]
    rep = Hq // Hkv
    out = torch.empty(query.shape[0], Hq, Dv, dtype=torch.float32, device=query.device)
    kf, vf = k_ref.float(), v_ref.float()
    for b in range(query.shape[0]):
        c = int(ctx_lens[b])
        w = min(c, window)
        phys = page_indices[int(page_offsets[b]) : int(page_offsets[b + 1])].to(kf.device)
        ktok = kf[phys].reshape(-1, Hkv, D)[:c]  # logical tokens 0..c-1, scattered pages
        vtok = vf[phys].reshape(-1, Hkv, Dv)[:c]
        k = ktok[c - w : c].repeat_interleave(rep, dim=1)  # [w, Hq, D]
        v = vtok[c - w : c].repeat_interleave(rep, dim=1)
        scores = torch.einsum("hd,whd->hw", query[b].float(), k) * scale  # [Hq, w]
        if sinks is not None:
            aug = torch.cat([sinks.float().unsqueeze(-1), scores], dim=-1)  # [Hq, 1+w]
            probs = torch.softmax(aug, dim=-1)[:, 1:]  # drop sink column (no value)
        else:
            probs = torch.softmax(scores, dim=-1)
        out[b] = torch.einsum("hw,whd->hd", probs, v)
    return out


def bench_swa(batch, ctx, cfg, dtype, kv_dtype, page_size, attn_tp, ctx_lens_cpu, seed, device, check):
    from aiter.ops.triton.attention.unified_attention import unified_attention

    Hq = cfg["num_q_heads"] // attn_tp
    Hkv = max(1, cfg["num_kv_heads"] // attn_tp)
    D = cfg["qk_head_dim"]
    Dv = cfg["v_head_dim"]
    window = cfg["window"]
    scale = 1.0 / (D**0.5)

    # Symmetric qk/v (gpt-oss 64/64): unified_attention runs directly -- the
    # SGLang aiter-backend SWA fast-path (forced on with a sliding-window KV pool).
    # Asymmetric qk/v (mimo 192/128) has no native aiter kernel: unified_attention
    # uses one HEAD_SIZE for q/k/v (v read out-of-bounds -> memory fault) and
    # flash_attn_varlen_func's paged path is broken (unbound `filter_fwd`). vLLM
    # uses FlashAttention's head_size_v, unavailable here for paged decode.
    # Fallback: pad v to qk_head_dim, run symmetric, slice output back to Dv.
    asym = D != Dv

    pages_per_seq = (ctx_lens_cpu + page_size - 1) // page_size
    page_offsets = torch.cat([torch.zeros(1, dtype=torch.int64), pages_per_seq.cumsum(0)])
    num_pages = int(page_offsets[-1])
    max_blocks = int(pages_per_seq.max())
    perm = shuffled_pages(num_pages, seed)  # scattered physical page layout
    # Effective KV traffic is bounded by the window (SWA reads only recent tokens).
    eff_tokens = int(torch.minimum(ctx_lens_cpu, torch.tensor(window)).sum())

    q = torch.randn(batch, Hq, D, device=device, dtype=dtype)
    k_ref = torch.randn(num_pages, page_size, Hkv, D, device=device, dtype=dtype)
    v_ref = torch.randn(num_pages, page_size, Hkv, Dv, device=device, dtype=dtype)
    k_descale = v_descale = None
    if kv_dtype == "fp8":
        cdt = fp8_dtype()
        k_in, v_in = k_ref.to(cdt), v_ref.to(cdt)
        k_descale = torch.ones(1, dtype=torch.float32, device=device)
        v_descale = torch.ones(1, dtype=torch.float32, device=device)
    else:
        k_in, v_in = k_ref, v_ref

    # Kernel sees v (and out) at qk_head_dim; pad with zeros when asymmetric.
    if asym:
        v_kernel = torch.zeros(num_pages, page_size, Hkv, D, device=device, dtype=v_in.dtype)
        v_kernel[..., :Dv] = v_in
    else:
        v_kernel = v_in
    out_kernel = torch.empty(batch, Hq, D, device=device, dtype=dtype)

    cu_seqlens_q = torch.arange(0, batch + 1, device=device, dtype=torch.int32)  # 1 q/seq
    seqused_k = ctx_lens_cpu.to(device=device, dtype=torch.int32)
    block_tables = torch.zeros(batch, max_blocks, device=device, dtype=torch.int32)
    for i in range(batch):
        n = int(pages_per_seq[i])
        block_tables[i, :n] = perm[int(page_offsets[i]) : int(page_offsets[i]) + n].to(device)
    sinks = torch.randn(Hq, dtype=torch.float32, device=device) if cfg["sinks"] else None

    def fn():
        return unified_attention(
            q=q, k=k_in, v=v_kernel, out=out_kernel, cu_seqlens_q=cu_seqlens_q, max_seqlen_q=1,
            seqused_k=seqused_k, max_seqlen_k=int(ctx_lens_cpu.max()), softmax_scale=scale,
            causal=True, window_size=(window - 1, 0), block_table=block_tables, softcap=0,
            q_descale=None, k_descale=k_descale, v_descale=v_descale, sinks=sinks,
        )

    out = out_kernel[..., :Dv]  # slice padded v back to v_head_dim
    snr = None
    if check:
        with suppress_output():
            fn()
        ref = swa_ref(q, k_ref, v_ref, seqused_k, page_offsets, perm, page_size, scale, Hq, Hkv, window, sinks)
        snr = compute_snr(ref, out.float())

    # Metric accounting for the padded (asymmetric) case: time_ms is the real
    # measured latency (which includes the extra reads of the zero-padded v
    # columns), but KV bytes and FLOPs use the *theoretical* model dims (qk + real
    # v_head_dim Dv, never the padded width D). So KV-BW / TFLOPS are deliberately
    # penalized by the padding overhead -- that lower number is the true cost of
    # having no native asymmetric kernel, which is exactly what we want to report.
    time_ms = _time_ms(fn)
    kv_bytes = (eff_tokens * Hkv * D + eff_tokens * Hkv * Dv) * k_in.element_size()
    bw_gbps = kv_bytes / (time_ms * 1e-3) / 1e9
    tflops = (2 * Hq * eff_tokens * (D + Dv)) / (time_ms * 1e-3) / 1e12
    return time_ms, bw_gbps, tflops, snr


def benchmark_one(batch, ctx, cfg, dtype, kv_dtype, page_size, attn_tp, ctx_spread, base_seed, device, check):
    # Per-request context lengths (varied when ctx_spread > 0); deterministic per cell.
    cell_seed = base_seed * 1_000_003 + batch * 9973 + ctx
    ctx_lens_cpu = gen_ctx_lens(batch, ctx, ctx_spread, cell_seed)
    # Page-layout seed: distinct from the ctx-len stream but deterministic per cell
    # (so bf16 and fp8 share the same scattered layout for a fair comparison).
    page_seed = cell_seed + 1
    if cfg["family"] == "gqa":
        return bench_gqa(batch, ctx, cfg, dtype, kv_dtype, page_size, attn_tp, ctx_lens_cpu, page_seed, device, check)
    if cfg["family"] == "swa":
        return bench_swa(batch, ctx, cfg, dtype, kv_dtype, page_size, attn_tp, ctx_lens_cpu, page_seed, device, check)
    if cfg["family"] == "dsa":
        return bench_dsa(batch, ctx, cfg, dtype, kv_dtype, attn_tp, ctx_lens_cpu, page_seed, device, check)
    return bench_mla(batch, ctx, cfg, dtype, kv_dtype, attn_tp, ctx_lens_cpu, page_seed, device, check)


def head_count(cfg):
    return cfg["num_heads"] if cfg["family"] in ("mla", "dsa") else cfg["num_q_heads"]


_OP_BY_FAMILY = {
    "gqa": "paged_attention_ragged",
    "mla": "mla_decode_fwd",
    "swa": "unified_attention",
    "dsa": "fp8_paged_mqa_logits + mla_decode_fwd",
}


def save_excel(path, args, cfg, results):
    import pandas as pd

    op = _OP_BY_FAMILY[cfg["family"]]
    meta = {
        "backend": "aiter",
        "op": f"{op} (decode)",
        "model": args.model,
        "family": cfg["family"],
        **{k: v for k, v in cfg.items() if k != "family"},
        "ctx_spread": args.ctx_spread,
        "tp_size": args.tp_size,
        "dp_size": args.dp_size,
        "attn_tp": args.tp_size // args.dp_size,
        "seed": args.seed,
        "gpu": torch.cuda.get_device_name(0),
    }
    if cfg["family"] in ("gqa", "swa"):
        meta["page_size"] = args.page_size
    if not path.endswith(".xlsx"):
        path += ".xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame(meta.items(), columns=["field", "value"]).to_excel(
            writer, sheet_name="config", index=False
        )
        for kv_dtype, rows in results.items():
            pd.DataFrame(rows).to_excel(writer, sheet_name=kv_dtype, index=False)
    return path


def main():
    parser = argparse.ArgumentParser(description="Decode attention benchmark (aiter; GQA + MLA + SWA + DSA)")
    parser.add_argument("--model", choices=list(MODELS), default="minimax-m2.7")
    parser.add_argument("--batches", type=int, nargs="+", default=DEFAULT_BATCHES)
    parser.add_argument("--ctx", type=int, nargs="+", default=DEFAULT_CTX)
    parser.add_argument("--page-size", type=int, default=16, help="GQA paged KV block size (MLA fixed at 1).")
    parser.add_argument(
        "--ctx-spread",
        type=float,
        default=0.0,
        help="Per-request context-length variation: lengths ~ uniform[ctx*(1-s), ctx*(1+s)].",
    )
    parser.add_argument("--tp-size", type=int, default=1, help="Global tensor-parallel size.")
    parser.add_argument(
        "--dp-size",
        type=int,
        default=1,
        help="DP-attention size. Per-rank heads split by attn_tp = tp_size // dp_size.",
    )
    parser.add_argument("--check", action="store_true", help="Run the SNR correctness check.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", "-o", type=str, default=None, help="Save to .xlsx (auto-named if omitted).")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "This benchmark requires a GPU."
    torch.manual_seed(args.seed)
    tp_size, dp_size = args.tp_size, args.dp_size
    assert tp_size >= 1 and dp_size >= 1, "tp/dp size must be >= 1"
    assert tp_size % dp_size == 0, "tp_size not divisible by dp_size"
    attn_tp = tp_size // dp_size

    device = "cuda"
    dtype = torch.bfloat16
    cfg = MODELS[args.model]
    heads = head_count(cfg)
    assert heads % attn_tp == 0, "heads not divisible by attn_tp (tp//dp)"

    op = _OP_BY_FAMILY[cfg["family"]]
    print()
    print(f"  Model     : {args.model}  family={cfg['family']}")
    print(f"  Op        : aiter {op} (decode)  ctx_spread={args.ctx_spread}"
          + (f"  page_size={args.page_size}" if cfg["family"] in ("gqa", "swa") else ""))
    print(f"  Parallel  : tp={tp_size} dp={dp_size} -> attn_tp={attn_tp}  (per-rank heads={heads // attn_tp})")
    print(f"  GPU       : {torch.cuda.get_device_name(0)}")

    results = {}
    for kv_dtype in ("bf16", "fp8"):
        print(f"\n=== KV cache: {kv_dtype} ===")
        header = f"{'Batch':>6} | {'Ctx':>7} | {'Time (ms)':>10} | {'KV-BW (GB/s)':>13} | {'TFLOPS':>8} | {'SNR (dB)':>8}"
        print(header)
        print("-" * len(header))
        rows = []
        for batch in args.batches:
            for ctx in args.ctx:
                try:
                    time_ms, bw, tflops, snr = benchmark_one(
                        batch, ctx, cfg, dtype, kv_dtype, args.page_size, attn_tp,
                        args.ctx_spread, args.seed, device, args.check
                    )
                    snr_str = "-" if snr is None else f"{snr:.1f}"
                    print(f"{batch:>6} | {ctx:>7} | {time_ms:>10.3f} | {bw:>13.0f} | {tflops:>8.1f} | {snr_str:>8}")
                    rows.append(
                        {"Batch": batch, "Ctx": ctx, "Time (ms)": round(time_ms, 3),
                         "KV-BW (GB/s)": round(bw), "TFLOPS": round(tflops, 1), "SNR (dB)": snr_str}
                    )
                except Exception as e:
                    msg = str(e).splitlines()[0][:60]
                    print(f"{batch:>6} | {ctx:>7} | ERROR: {msg}")
                    rows.append({"Batch": batch, "Ctx": ctx, "Time (ms)": f"ERROR: {msg}"})
        results[kv_dtype] = rows

    output = args.output
    if output is None:
        date = datetime.now().strftime("%Y%m%d")
        output = f"bench_decode_attn_{args.model}_tp{args.tp_size}_dp{args.dp_size}_{date}.xlsx"
    print(f"\nSaved results to {save_excel(output, args, cfg, results)}")


if __name__ == "__main__":
    main()
