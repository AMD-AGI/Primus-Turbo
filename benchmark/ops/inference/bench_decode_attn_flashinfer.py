###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Inference decode attention micro-benchmark for FlashInfer (per-rank, single GPU).

NVIDIA counterpart of ``bench_decode_attn_aiter.py``; mirrors its structure
one-to-one (same MODELS, helpers, metrics, CLI, Excel layout) so AMD/aiter and
NVIDIA/FlashInfer numbers compare directly. Each family maps to the TensorRT-LLM
-gen kernel FlashInfer dispatches on Blackwell (B200 / sm100):

  - gqa : trtllm_batch_decode_with_kv_cache (paged GQA, HND KV cache).
  - mla : trtllm_batch_decode_with_kv_cache_mla (DeepSeek/Kimi absorbed MLA,
          latent KV: qk = kv_lora_rank + qk_rope, v = kv_lora_rank).
  - swa : trtllm_batch_decode_with_kv_cache with window_left = window-1 and
          per-head attention sinks (gpt-oss). Sliding-window decode.
  - dsa : deep_gemm.fp8_paged_mqa_logits indexer (full-ctx fp8 scoring + top-k)
          + trtllm_batch_decode_with_kv_cache_mla with sparse_mla_top_k (DeepSeek
          sparse attention; GLM-5.1). Both stages are timed (matching aiter's
          two-stage bench_dsa); the indexer uses DeepGEMM's fp8 paged-MQA logits,
          the exact NV counterpart of aiter's deepgemm_fp8_paged_mqa_logits.

Decode is memory-bound, so we sweep batch x context_len and report KV bandwidth.
KV cache runs in bf16 and fp8. TP shards heads on this rank via
attn_tp = tp_size // dp_size (SGLang DP-attention semantics).

NOTE: targets B200/sm100 and is written against FlashInfer's documented trtllm
API; validate kernel availability/shapes on real Blackwell hardware.

    python bench_decode_attn_flashinfer.py [--model M] [--ctx-spread S] [--tp-size N] [--dp-size N] [--check]
"""

import argparse
import contextlib
import os
from datetime import datetime

import torch
import flashinfer

# family "gqa": num_q_heads, num_kv_heads, head_dim.
# family "mla": num_heads, kv_lora_rank, qk_rope_head_dim, qk_nope_head_dim
#               (absorbed decode: q/k width = kv_lora_rank + qk_rope, v = kv_lora_rank;
#               qk_nope_head_dim is passed to the kernel for the bmm1 scale).
# family "swa": num_q_heads, num_kv_heads, qk_head_dim, v_head_dim, window, sinks.
# family "dsa": as "mla" + index_n_heads, index_head_dim, index_topk (the fp8
#               indexer that scores the full ctx and keeps the top-`index_topk`).
MODELS = {
    # https://huggingface.co/MiniMaxAI/MiniMax-M2.7  (229B; full GQA)
    "minimax-m2.7": {"family": "gqa", "num_q_heads": 48, "num_kv_heads": 8, "head_dim": 128},
    # GLM-5.1 (MLA + DeepSeek sparse attention). Absorbed MLA decode: qk = 512+64.
    # Indexer scores full ctx with index_n_heads x index_head_dim, keeps index_topk.
    "glm-5.1": {"family": "dsa", "num_heads": 64, "kv_lora_rank": 512, "qk_rope_head_dim": 64,
                "qk_nope_head_dim": 128, "index_n_heads": 32, "index_head_dim": 128, "index_topk": 2048},
    # https://huggingface.co/deepseek-ai/DeepSeek-R1  (671B; MLA)
    "deepseek-r1": {"family": "mla", "num_heads": 128, "kv_lora_rank": 512, "qk_rope_head_dim": 64,
                    "qk_nope_head_dim": 128},
    # https://huggingface.co/moonshotai/Kimi-K2.6  (1T; MLA, DeepSeek-V3 arch)
    "kimi-k2.6": {"family": "mla", "num_heads": 64, "kv_lora_rank": 512, "qk_rope_head_dim": 64,
                  "qk_nope_head_dim": 128},
    # https://huggingface.co/openai/gpt-oss-120b  (117B; SWA layers, window 128, attention sinks)
    "gpt-oss-120b": {"family": "swa", "num_q_heads": 64, "num_kv_heads": 8,
                     "qk_head_dim": 64, "v_head_dim": 64, "window": 128, "sinks": True},
    # https://huggingface.co/XiaomiMiMo/MiMo-V2.5-Pro  (1.02T; SWA layers, window 128)
    "mimo-v2.5-pro": {"family": "swa", "num_q_heads": 128, "num_kv_heads": 8,
                      "qk_head_dim": 192, "v_head_dim": 128, "window": 128, "sinks": False},
}

DEFAULT_BATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 256]
DEFAULT_CTX = [1024, 4096, 8192, 16384, 32768, 65536, 131072]

_GQA_PAGE_SIZE = 16  # paged KV block size for GQA/SWA.
_MLA_PAGE_SIZE = 32  # trtllm-gen MLA latent cache page size (see test_trtllm_gen_mla).
_DSA_INDEX_PAGE = 64  # DeepGEMM paged-MQA logits block size (page_size % 16 == 0).
_WORKSPACE_BYTES = 256 * 1024 * 1024  # trtllm-gen fmha workspace.
_USE_CUDAGRAPH = True  # time via CUDA-graph replay (matches SGLang/vLLM decode); set by main().


@contextlib.contextmanager
def suppress_output():
    """Silence chatty per-call stdout/stderr at the fd level."""
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
    """OCP fp8 (e4m3fn) -- the format the trtllm-gen kernels consume on sm100."""
    return torch.float8_e4m3fn


def _time_ms(fn, warmup=10, iters=50, use_graph=None):
    """GPU time per call via CUDA events (excludes host/Python launch overhead).

    With use_graph (default, controlled by --cudagraph) fn is captured into a
    CUDA graph and timed via replay, removing per-launch dispatch overhead and
    inter-kernel gaps that dominate low-concurrency cells, matching how
    SGLang/vLLM run decode. Falls back to eager launches if the kernel cannot be
    captured. Mirrors bench_decode_attn_aiter._time_ms one-to-one."""
    if use_graph is None:
        use_graph = _USE_CUDAGRAPH

    def event_loop(call):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            call()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters

    with suppress_output():
        if use_graph:
            try:
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(warmup):
                        fn()
                torch.cuda.current_stream().wait_stream(s)
                torch.cuda.synchronize()

                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    fn()
                for _ in range(3):  # warm the replay
                    graph.replay()
                return event_loop(graph.replay)
            except Exception:
                pass  # capture unsupported for this kernel -> eager timing below

        for _ in range(warmup):
            fn()
        return event_loop(fn)


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
    """Random (reproducible) physical page placement: a serving KV pool scatters
    a sequence's pages across HBM, so the gather is cache-unfriendly. Returns a
    CPU permutation `perm` (logical slot i -> physical page perm[i])."""
    g = torch.Generator().manual_seed(int(seed))
    return torch.randperm(num_pages, generator=g)


def _block_tables(ctx_lens_cpu, page_size, perm, device):
    """Per-seq page table [batch, max_blocks] (int32), pages drawn from `perm`."""
    pages_per_seq = (ctx_lens_cpu + page_size - 1) // page_size
    page_off = torch.cat([torch.zeros(1, dtype=torch.int64), pages_per_seq.cumsum(0)])
    max_blocks = int(pages_per_seq.max())
    batch = ctx_lens_cpu.numel()
    bt = torch.zeros(batch, max_blocks, device=device, dtype=torch.int32)
    for i in range(batch):
        n = int(pages_per_seq[i])
        bt[i, :n] = perm[int(page_off[i]) : int(page_off[i]) + n].to(device)
    return bt, pages_per_seq, page_off


# --------------------------------------------------------------------------- #
# GQA family: trtllm_batch_decode_with_kv_cache (paged, HND)
# --------------------------------------------------------------------------- #


def gqa_ref(query, k_cache, v_cache, ctx_lens, page_off, perm, scale, Hq, Hkv, page_size):
    """Naive paged GQA decode reference (fp32). query [B, Hq, D]; caches HND
    [num_pages, Hkv, page_size, D]; seq b uses physical pages perm[off[b]:off[b+1]]."""
    B, _, D = query.shape
    rep = Hq // Hkv
    out = torch.empty(B, Hq, D, dtype=torch.float32, device=query.device)
    kf, vf = k_cache.float(), v_cache.float()
    for b in range(B):
        c = int(ctx_lens[b])
        phys = perm[int(page_off[b]) : int(page_off[b + 1])].to(kf.device)
        # HND page: [Hkv, page_size, D] -> token-major [page_size*npages, Hkv, D]
        flat_k = kf[phys].permute(0, 2, 1, 3).reshape(-1, Hkv, D)[:c]
        flat_v = vf[phys].permute(0, 2, 1, 3).reshape(-1, Hkv, D)[:c]
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

    perm_pages = (ctx_lens_cpu + page_size - 1) // page_size
    num_pages = int(perm_pages.sum())
    total_tokens = int(ctx_lens_cpu.sum())
    max_ctx = int(ctx_lens_cpu.max())
    perm = shuffled_pages(num_pages, seed)
    block_tables, _, page_off = _block_tables(ctx_lens_cpu, page_size, perm, device)
    seq_lens = ctx_lens_cpu.to(device=device, dtype=torch.int32)

    query = torch.randn(batch, Hq, D, device=device, dtype=dtype)
    # HND single-tensor KV cache: [num_pages, 2, Hkv, page_size, D].
    k_ref = torch.randn(num_pages, Hkv, page_size, D, device=device, dtype=dtype)
    v_ref = torch.randn(num_pages, Hkv, page_size, D, device=device, dtype=dtype)
    if kv_dtype == "fp8":
        cdt = fp8_dtype()
        kv_cache = torch.stack([k_ref.to(cdt), v_ref.to(cdt)], dim=1).contiguous()
    else:
        kv_cache = torch.stack([k_ref, v_ref], dim=1).contiguous()

    workspace = torch.zeros(_WORKSPACE_BYTES, dtype=torch.uint8, device=device)
    out = torch.empty(batch, Hq, D, device=device, dtype=dtype)

    def fn():
        return flashinfer.decode.trtllm_batch_decode_with_kv_cache(
            query=query, kv_cache=kv_cache, workspace_buffer=workspace,
            block_tables=block_tables, seq_lens=seq_lens, max_seq_len=max_ctx,
            bmm1_scale=scale, bmm2_scale=1.0, kv_layout="HND", out=out,
        )

    snr = None
    if check:
        with suppress_output():
            fn()
        ref = gqa_ref(query, k_ref, v_ref, ctx_lens_cpu, page_off, perm, scale, Hq, Hkv, page_size)
        snr = compute_snr(ref, out.float())

    time_ms = _time_ms(fn)
    elem = kv_cache.element_size()
    kv_bytes = 2 * total_tokens * Hkv * D * elem
    bw_gbps = kv_bytes / (time_ms * 1e-3) / 1e9
    tflops = (2 * 2 * Hq * D * total_tokens) / (time_ms * 1e-3) / 1e12
    return time_ms, bw_gbps, tflops, snr


# --------------------------------------------------------------------------- #
# SWA family: trtllm_batch_decode_with_kv_cache (window_left + attention sinks)
# --------------------------------------------------------------------------- #


def swa_ref(query, k_ref, v_ref, ctx_lens, page_off, perm, page_size, scale, Hq, Hkv, window, sinks):
    """Naive sliding-window decode reference (fp32): each query attends to the
    last `window` keys; optional per-head sink adds a value-less logit."""
    D = query.shape[-1]
    Dv = v_ref.shape[-1]
    rep = Hq // Hkv
    out = torch.empty(query.shape[0], Hq, Dv, dtype=torch.float32, device=query.device)
    kf, vf = k_ref.float(), v_ref.float()
    for b in range(query.shape[0]):
        c = int(ctx_lens[b])
        w = min(c, window)
        phys = perm[int(page_off[b]) : int(page_off[b + 1])].to(kf.device)
        ktok = kf[phys].permute(0, 2, 1, 3).reshape(-1, Hkv, D)[:c]
        vtok = vf[phys].permute(0, 2, 1, 3).reshape(-1, Hkv, Dv)[:c]
        k = ktok[c - w : c].repeat_interleave(rep, dim=1)
        v = vtok[c - w : c].repeat_interleave(rep, dim=1)
        scores = torch.einsum("hd,whd->hw", query[b].float(), k) * scale
        if sinks is not None:
            aug = torch.cat([sinks.float().unsqueeze(-1), scores], dim=-1)
            probs = torch.softmax(aug, dim=-1)[:, 1:]
        else:
            probs = torch.softmax(scores, dim=-1)
        out[b] = torch.einsum("hw,whd->hd", probs, v)
    return out


def bench_swa(batch, ctx, cfg, dtype, kv_dtype, page_size, attn_tp, ctx_lens_cpu, seed, device, check):
    Hq = cfg["num_q_heads"] // attn_tp
    Hkv = max(1, cfg["num_kv_heads"] // attn_tp)
    D = cfg["qk_head_dim"]
    Dv = cfg["v_head_dim"]
    window = cfg["window"]
    scale = 1.0 / (D**0.5)
    # trtllm-gen decode expects symmetric qk/v; pad v then slice. NOTE: trtllm-gen
    # only ships decode kernels for head_dim in {32,64,128,256} -- there is no
    # D=192 variant, so mimo-v2.5-pro (qk=192) raises "Missing TRTLLM-GEN kernel"
    # (no NVIDIA fallback; aiter uses the triton unified_attention path instead).
    asym = D != Dv

    perm_pages = (ctx_lens_cpu + page_size - 1) // page_size
    num_pages = int(perm_pages.sum())
    max_ctx = int(ctx_lens_cpu.max())
    perm = shuffled_pages(num_pages, seed)
    block_tables, _, page_off = _block_tables(ctx_lens_cpu, page_size, perm, device)
    seq_lens = ctx_lens_cpu.to(device=device, dtype=torch.int32)
    eff_tokens = int(torch.minimum(ctx_lens_cpu, torch.tensor(window)).sum())

    Dk = D
    q = torch.randn(batch, Hq, Dk, device=device, dtype=dtype)
    k_ref = torch.randn(num_pages, Hkv, page_size, Dk, device=device, dtype=dtype)
    v_ref = torch.randn(num_pages, Hkv, page_size, Dv, device=device, dtype=dtype)
    # Kernel sees v at qk_head_dim; pad with zeros when asymmetric, slice out back.
    if asym:
        v_kernel = torch.zeros(num_pages, Hkv, page_size, Dk, device=device, dtype=dtype)
        v_kernel[..., :Dv] = v_ref
    else:
        v_kernel = v_ref
    if kv_dtype == "fp8":
        cdt = fp8_dtype()
        kv_cache = torch.stack([k_ref.to(cdt), v_kernel.to(cdt)], dim=1).contiguous()
    else:
        kv_cache = torch.stack([k_ref, v_kernel], dim=1).contiguous()

    workspace = torch.zeros(_WORKSPACE_BYTES, dtype=torch.uint8, device=device)
    out_kernel = torch.empty(batch, Hq, Dk, device=device, dtype=dtype)
    # trtllm decode expects sinks as a single [Hq] tensor (not a list).
    sinks = torch.randn(Hq, dtype=torch.float32, device=device) if cfg["sinks"] else None

    def fn():
        return flashinfer.decode.trtllm_batch_decode_with_kv_cache(
            query=q, kv_cache=kv_cache, workspace_buffer=workspace,
            block_tables=block_tables, seq_lens=seq_lens, max_seq_len=max_ctx,
            bmm1_scale=scale, bmm2_scale=1.0, window_left=window - 1,
            sinks=sinks, kv_layout="HND", out=out_kernel,
        )

    out = out_kernel[..., :Dv]
    snr = None
    if check:
        with suppress_output():
            fn()
        ref = swa_ref(q, k_ref, v_ref, ctx_lens_cpu, page_off, perm, page_size, scale, Hq, Hkv, window, sinks)
        snr = compute_snr(ref, out.float())

    time_ms = _time_ms(fn)
    elem = kv_cache.element_size()
    kv_bytes = (eff_tokens * Hkv * D + eff_tokens * Hkv * Dv) * elem
    bw_gbps = kv_bytes / (time_ms * 1e-3) / 1e9
    tflops = (2 * Hq * eff_tokens * (D + Dv)) / (time_ms * 1e-3) / 1e12
    return time_ms, bw_gbps, tflops, snr


# --------------------------------------------------------------------------- #
# MLA / DSA families: trtllm_batch_decode_with_kv_cache_mla (absorbed; sparse)
# --------------------------------------------------------------------------- #


def mla_ref(q, kv_cache, ctx_lens, page_off, perm, page_size, kv_lora_rank, scale):
    """Naive absorbed-MLA decode reference (fp32). q [B, nhead, qk];
    kv_cache [num_blocks, page_size, qk]; K = latent, V = latent[:kv_lora_rank].
    seq b uses physical blocks perm[off[b]:off[b+1]]."""
    B, nhead, qk = q.shape
    out = torch.empty(B, nhead, kv_lora_rank, dtype=torch.float32, device=q.device)
    kvf = kv_cache.float()
    for b in range(B):
        c = int(ctx_lens[b])
        phys = perm[int(page_off[b]) : int(page_off[b + 1])].to(kvf.device)
        kvc = kvf[phys].reshape(-1, qk)[:c]
        probs = torch.softmax(torch.einsum("hd,cd->hc", q[b].float(), kvc) * scale, dim=-1)
        out[b] = torch.einsum("hc,cd->hd", probs, kvc[:, :kv_lora_rank])
    return out


def bench_mla(batch, ctx, cfg, dtype, kv_dtype, attn_tp, ctx_lens_cpu, seed, device, check):
    nhead = cfg["num_heads"] // attn_tp  # TP shards q heads; latent KV is replicated
    lora = cfg["kv_lora_rank"]
    rope = cfg["qk_rope_head_dim"]
    nope = cfg["qk_nope_head_dim"]
    qk = lora + rope
    scale = 1.0 / ((nope + rope) ** 0.5)
    page_size = _MLA_PAGE_SIZE
    total_read = int(ctx_lens_cpu.sum())  # dense MLA reads the full ctx

    perm_pages = (ctx_lens_cpu + page_size - 1) // page_size
    num_blocks = int(perm_pages.sum())
    max_ctx = int(ctx_lens_cpu.max())
    perm = shuffled_pages(num_blocks, seed)
    block_tables, _, page_off = _block_tables(ctx_lens_cpu, page_size, perm, device)
    seq_lens = ctx_lens_cpu.to(device=device, dtype=torch.int32)

    q = torch.randn(batch, nhead, qk, device=device, dtype=dtype)
    kv_ref = torch.randn(num_blocks, page_size, qk, device=device, dtype=dtype)
    if kv_dtype == "fp8":
        cdt = fp8_dtype()
        q_in, kv_cache = q.to(cdt), kv_ref.to(cdt)
    else:
        q_in, kv_cache = q, kv_ref
    workspace = torch.zeros(_WORKSPACE_BYTES, dtype=torch.uint8, device=device)

    def fn():
        return flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
            query=q_in.unsqueeze(1),  # [B, q_len=1, nhead, qk]
            kv_cache=kv_cache.unsqueeze(1),  # [num_blocks, 1, page_size, qk]
            workspace_buffer=workspace,
            qk_nope_head_dim=nope, kv_lora_rank=lora, qk_rope_head_dim=rope,
            block_tables=block_tables, seq_lens=seq_lens, max_seq_len=max_ctx,
            bmm1_scale=scale, bmm2_scale=1.0,
        )

    snr = None
    if check:
        with suppress_output():
            out = fn()
        out_t = out[0] if isinstance(out, (tuple, list)) else out
        ref = mla_ref(q, kv_ref, ctx_lens_cpu, page_off, perm, page_size, lora, scale)
        snr = compute_snr(ref, out_t.float().reshape(batch, nhead, lora))

    time_ms = _time_ms(fn)
    elem = kv_cache.element_size()
    kv_bytes = total_read * qk * elem  # latent read once (K/V shared)
    bw_gbps = kv_bytes / (time_ms * 1e-3) / 1e9
    tflops = (2 * nhead * total_read * (qk + lora)) / (time_ms * 1e-3) / 1e12
    return time_ms, bw_gbps, tflops, snr


def _build_dsa_indexer(batch, ctx_lens_cpu, n_heads, head_dim, topk, seed, device):
    """DSA indexer (NV): DeepGEMM fp8 paged-MQA logits over the full context +
    top-k selection. The exact NV counterpart of aiter's deepgemm_fp8_paged_mqa_
    logits (both wrap DeepSeek's DeepGEMM kernel). Random index-K cache -> the
    logit values are meaningless, but the kernel runs at representative speed (we
    time it; the data-dependent top-k is not validated numerically). Returns
    (fn, bytes_per_token) where bytes_per_token includes the per-block fp8 scale."""
    import deep_gemm

    page = _DSA_INDEX_PAGE
    # DeepGEMM kv layout: [num_blocks, page, 1, head_dim + 4] (fp8 key + fp32 scale).
    hdsf = head_dim + 4  # 128 fp8 bytes + 4-byte per-128 fp32 scale (head_dim=128)
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
    seqlens_2d = ctx_lens_cpu.to(device=device, dtype=torch.int32).unsqueeze(-1)
    sm_count = deep_gemm.get_num_sms()
    schedule_meta = deep_gemm.get_paged_mqa_logits_metadata(seqlens_2d, page, sm_count)

    # q [B, next_n=1, heads, head_dim] fp8; weights [B*1, heads] fp32.
    # kv_cache stays uint8 (kByte): fp8 key bytes + per-128 fp32 scale, packed,
    # exactly as sglang passes it to deep_gemm (the kernel asserts kByte).
    q = torch.randn(batch, 1, n_heads, head_dim, device=device, dtype=torch.bfloat16).to(fp8_dtype())
    weights = torch.randn(batch * 1, n_heads, device=device, dtype=torch.float32)
    kv = torch.randint(0, 255, (num_pages, page, 1, hdsf), dtype=torch.uint8, device=device)
    tk = min(topk, max_ctx)

    def fn():
        logits = deep_gemm.fp8_paged_mqa_logits(
            q, kv, weights, seqlens_2d, block_tables, schedule_meta, max_ctx, clean_logits=False,
        )
        torch.topk(logits[:, :max_ctx], tk, dim=-1)  # top-k token selection

    return fn, hdsf


def bench_dsa(batch, ctx, cfg, dtype, kv_dtype, attn_tp, ctx_lens_cpu, seed, device, check):
    """Two-stage DeepSeek sparse attention decode, mirroring aiter bench_dsa:
      Stage 1 (indexer): DeepGEMM fp8 paged-MQA logits over the full ctx + top-k.
      Stage 2 (sparse MLA): trtllm_batch_decode_with_kv_cache_mla with
        sparse_mla_top_k -- the page_table is (num_seqs, 1, top_k) token indices
        into a page_size=1 latent cache (flashinfer mla/_core _check_trtllm_gen_
        mla_shape). The selected tokens scatter across the full-ctx latent pool.
    Both stages are timed in fn() and bytes/FLOPs include the indexer, so the
    headline Total numbers are apples-to-apples with aiter; per-stage Idx/MLA
    breakdowns are also reported (the MLA columns are the cross-backend-cleanest)."""
    nhead = cfg["num_heads"] // attn_tp  # TP shards q heads; latent KV replicated
    lora = cfg["kv_lora_rank"]
    rope = cfg["qk_rope_head_dim"]
    nope = cfg["qk_nope_head_dim"]
    qk = lora + rope
    scale = 1.0 / ((nope + rope) ** 0.5)
    topk = cfg["index_topk"]

    # Per-seq selected-token count (all of ctx when ctx <= topk); top_k is the
    # fixed page-table width, seq_lens carries the per-seq valid count.
    sel_lens = torch.minimum(ctx_lens_cpu, torch.tensor(topk, dtype=torch.int64))
    top_k = int(sel_lens.max())
    total_sel = int(sel_lens.sum())
    total_ctx = int(ctx_lens_cpu.sum())

    # Latent KV is paged at MLA page_size (sglang real_page_size: 32/64); the
    # sparse page_table holds TOKEN-level indices into this paged pool (the kernel
    # maps each to its page internally, matching sglang _forward_trtllm). Each seq
    # owns a contiguous token region [off[b], off[b]+ctx_len[b]); its selected
    # tokens scatter across that region.
    pg = _MLA_PAGE_SIZE  # 32: a supported sparse MLA block_size
    off = torch.cat([torch.zeros(1, dtype=torch.int64), ctx_lens_cpu.cumsum(0)])
    pool_tokens = int(off[-1])
    num_pages = (pool_tokens + pg - 1) // pg
    g = torch.Generator().manual_seed(int(seed))
    # Sparse top-k must be a multiple of (128/page_size); pad the table width.
    pad = int(128 // pg)
    top_k_padded = ((top_k + pad - 1) // pad) * pad
    page_table = torch.zeros(batch, 1, top_k_padded, device=device, dtype=torch.int32)
    for b in range(batch):
        c = int(ctx_lens_cpu[b])
        k = int(sel_lens[b])
        chosen = torch.randperm(c, generator=g)[:k] + int(off[b])
        page_table[b, 0, :k] = chosen.to(device=device, dtype=torch.int32)

    seq_lens = sel_lens.to(device=device, dtype=torch.int32)
    q = torch.randn(batch, nhead, qk, device=device, dtype=dtype)
    kv_ref = torch.randn(num_pages, pg, qk, device=device, dtype=dtype)  # [num_pages, page, qk]
    if kv_dtype == "fp8":
        cdt = fp8_dtype()
        q_in, kv_cache = q.to(cdt), kv_ref.to(cdt)
    else:
        q_in, kv_cache = q, kv_ref
    workspace = torch.zeros(_WORKSPACE_BYTES, dtype=torch.uint8, device=device)

    # Stage 2 -- sparse absorbed-MLA decode over the selected tokens.
    def mla_fn():
        return flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
            query=q_in.unsqueeze(1),  # [B, q_len=1, nhead, qk]
            kv_cache=kv_cache.unsqueeze(1),  # [num_pages, 1, page, qk]
            workspace_buffer=workspace,
            qk_nope_head_dim=nope, kv_lora_rank=lora, qk_rope_head_dim=rope,
            block_tables=page_table, seq_lens=seq_lens, max_seq_len=pool_tokens,
            bmm1_scale=scale, bmm2_scale=1.0, sparse_mla_top_k=top_k_padded,
        )

    # Stage 1 -- fp8 indexer (scores the full context, picks top-k).
    idx_fn, idx_bytes = _build_dsa_indexer(
        batch, ctx_lens_cpu, cfg["index_n_heads"], cfg["index_head_dim"], topk, seed + 7, device
    )

    def fn():
        idx_fn()
        return mla_fn()

    elem = kv_cache.element_size()
    # Indexer reads index-K over the full ctx; sparse MLA reads the latent over
    # only the <=topk selected tokens.
    idx_bytes_total = total_ctx * idx_bytes
    mla_bytes_total = total_sel * qk * elem
    idx_flops = 2 * cfg["index_n_heads"] * cfg["index_head_dim"] * total_ctx
    mla_flops = 2 * nhead * total_sel * (qk + lora)

    # Time the full decode and each stage in isolation (the MLA columns are the
    # apples-to-apples ones if another backend times only the sparse stage).
    time_ms = _time_ms(fn)
    idx_ms = _time_ms(idx_fn)
    mla_ms = _time_ms(mla_fn)

    bw_gbps = (idx_bytes_total + mla_bytes_total) / (time_ms * 1e-3) / 1e9
    tflops = (idx_flops + mla_flops) / (time_ms * 1e-3) / 1e12
    extra = {
        "idx_ms": idx_ms,
        "idx_bw": idx_bytes_total / (idx_ms * 1e-3) / 1e9,
        "idx_tflops": idx_flops / (idx_ms * 1e-3) / 1e12,
        "mla_ms": mla_ms,
        "mla_bw": mla_bytes_total / (mla_ms * 1e-3) / 1e9,
        "mla_tflops": mla_flops / (mla_ms * 1e-3) / 1e12,
    }
    return time_ms, bw_gbps, tflops, None, extra  # data-dependent top-k: SNR not checked


def benchmark_one(batch, ctx, cfg, dtype, kv_dtype, page_size, attn_tp, ctx_spread, base_seed, device, check):
    cell_seed = base_seed * 1_000_003 + batch * 9973 + ctx
    ctx_lens_cpu = gen_ctx_lens(batch, ctx, ctx_spread, cell_seed)
    page_seed = cell_seed + 1
    if cfg["family"] == "gqa":
        return (*bench_gqa(batch, ctx, cfg, dtype, kv_dtype, page_size, attn_tp, ctx_lens_cpu, page_seed, device, check), None)
    if cfg["family"] == "swa":
        return (*bench_swa(batch, ctx, cfg, dtype, kv_dtype, page_size, attn_tp, ctx_lens_cpu, page_seed, device, check), None)
    if cfg["family"] == "dsa":
        return bench_dsa(batch, ctx, cfg, dtype, kv_dtype, attn_tp, ctx_lens_cpu, page_seed, device, check)
    return (*bench_mla(batch, ctx, cfg, dtype, kv_dtype, attn_tp, ctx_lens_cpu, page_seed, device, check), None)


def head_count(cfg):
    return cfg["num_heads"] if cfg["family"] in ("mla", "dsa") else cfg["num_q_heads"]


_OP_BY_FAMILY = {
    "gqa": "trtllm_batch_decode_with_kv_cache",
    "mla": "trtllm_batch_decode_with_kv_cache_mla",
    "swa": "trtllm_batch_decode_with_kv_cache (window+sinks)",
    "dsa": "trtllm_batch_decode_with_kv_cache_mla (sparse)",
}


def save_excel(path, args, cfg, results):
    import pandas as pd

    op = _OP_BY_FAMILY[cfg["family"]]
    meta = {
        "backend": "flashinfer",
        "op": f"{op} (decode)",
        "model": args.model,
        "family": cfg["family"],
        **{k: v for k, v in cfg.items() if k != "family"},
        "ctx_spread": args.ctx_spread,
        "tp_size": args.tp_size,
        "dp_size": args.dp_size,
        "attn_tp": args.tp_size // args.dp_size,
        "cudagraph": args.cudagraph,
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
    parser = argparse.ArgumentParser(description="Decode attention benchmark (FlashInfer trtllm; GQA + MLA + SWA + DSA)")
    parser.add_argument("--model", choices=list(MODELS), default="minimax-m2.7")
    parser.add_argument("--batches", type=int, nargs="+", default=DEFAULT_BATCHES)
    parser.add_argument("--ctx", type=int, nargs="+", default=DEFAULT_CTX)
    parser.add_argument("--page-size", type=int, default=_GQA_PAGE_SIZE, help="GQA/SWA paged KV block size.")
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
    parser.add_argument(
        "--no-cudagraph",
        dest="cudagraph",
        action="store_false",
        help="Time with eager launches instead of CUDA-graph replay (default: cudagraph on).",
    )
    parser.set_defaults(cudagraph=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", "-o", type=str, default=None, help="Save to .xlsx (auto-named if omitted).")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "This benchmark requires a GPU."
    torch.manual_seed(args.seed)
    global _USE_CUDAGRAPH
    _USE_CUDAGRAPH = args.cudagraph
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
    print(f"  Op        : flashinfer {op} (decode)  ctx_spread={args.ctx_spread}"
          + (f"  page_size={args.page_size}" if cfg["family"] in ("gqa", "swa") else ""))
    print(f"  Parallel  : tp={tp_size} dp={dp_size} -> attn_tp={attn_tp}  (per-rank heads={heads // attn_tp})")
    print(f"  Timing    : {'cudagraph' if args.cudagraph else 'eager'}")
    print(f"  GPU       : {torch.cuda.get_device_name(0)}")

    is_dsa = cfg["family"] == "dsa"
    results = {}
    for kv_dtype in ("bf16", "fp8"):
        print(f"\n=== KV cache: {kv_dtype} ===")
        if is_dsa:
            header = (f"{'Batch':>6} | {'Ctx':>7} | {'Tot ms':>8} | {'Tot BW':>8} | {'Tot TF':>7} | "
                      f"{'Idx ms':>8} | {'Idx BW':>8} | {'MLA ms':>8} | {'MLA BW':>8} | {'SNR':>6}")
        else:
            header = f"{'Batch':>6} | {'Ctx':>7} | {'Time (ms)':>10} | {'KV-BW (GB/s)':>13} | {'TFLOPS':>8} | {'SNR (dB)':>8}"
        print(header)
        print("-" * len(header))
        rows = []
        for batch in args.batches:
            for ctx in args.ctx:
                try:
                    time_ms, bw, tflops, snr, extra = benchmark_one(
                        batch, ctx, cfg, dtype, kv_dtype, args.page_size, attn_tp,
                        args.ctx_spread, args.seed, device, args.check
                    )
                    snr_str = "-" if snr is None else f"{snr:.1f}"
                    if is_dsa:
                        print(f"{batch:>6} | {ctx:>7} | {time_ms:>8.3f} | {bw:>8.0f} | {tflops:>7.1f} | "
                              f"{extra['idx_ms']:>8.3f} | {extra['idx_bw']:>8.0f} | "
                              f"{extra['mla_ms']:>8.3f} | {extra['mla_bw']:>8.0f} | {snr_str:>6}")
                        rows.append(
                            {"Batch": batch, "Ctx": ctx,
                             "Total ms": round(time_ms, 3), "Total BW (GB/s)": round(bw), "Total TFLOPS": round(tflops, 1),
                             "Idx ms": round(extra["idx_ms"], 3), "Idx BW (GB/s)": round(extra["idx_bw"]),
                             "Idx TFLOPS": round(extra["idx_tflops"], 1),
                             "MLA ms": round(extra["mla_ms"], 3), "MLA BW (GB/s)": round(extra["mla_bw"]),
                             "MLA TFLOPS": round(extra["mla_tflops"], 1),
                             "SNR (dB)": snr_str}
                        )
                    else:
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
        output = f"bench_decode_attn_fi_{args.model}_tp{args.tp_size}_dp{args.dp_size}_{date}.xlsx"
    print(f"\nSaved results to {save_excel(output, args, cfg, results)}")


if __name__ == "__main__":
    main()
