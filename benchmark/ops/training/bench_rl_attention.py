###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""RL attention benchmark (single GPU): the batch-invariant kernel on rollout and training shapes.

Phases, all with bf16 / fp16 and real model head shapes (heads sharded by --tp-size):
  - decode  : paged KV cache, one query token per sequence, batch x context sweep. Reports time and
              KV bandwidth. Baseline: aiter unified_attention (the Triton kernel SGLang / vLLM use).
  - decode-graph : decode replayed from one HIP graph captured with max_seqlen_k = --graph-max-ctx
              (a serving engine's max model length), against a graph at max_seqlen_k = ctx; checks
              the bits match. Not in the default phases.
  - prefill : paged KV cache, a chunk of query tokens on top of existing context. Same baseline.
  - train   : packed varlen sequences, forward and forward + backward. Reports time and TFLOPS.
              Baseline: Turbo's default flash_attn_varlen_func backend (not batch invariant).

The RL kernel runs on each backend in --backend (TRITON, FLYDSL; columns prefixed triton / flydsl)
three ways: built-in causal, the same causal mask written as visibility intervals ("+mask", the
overhead of create_block_mask), and an image-span mask ("+image", a bidirectional span inside a
causal sequence; train only, models without a sliding window). The paged cache uses 64-key pages,
which both backends take. The two backends give different bits; every bitwise check stays within one. Sliding-window models pass their window.

With --check, each case also verifies accuracy against the baseline (SNR) and, for decode and
prefill, that the rollout result equals a training forward of the same tokens bitwise.

    python benchmark/ops/training/bench_rl_attention.py [--phase decode prefill train]
        [--model llama3-8b gpt-oss-120b] [--backend TRITON FLYDSL] [--tp-size N] [--check] [-o out.csv]
        [--decode-batch 1 8 32 --decode-ctx 131072 262144 524288]
    python benchmark/ops/training/bench_rl_attention.py --phase decode-graph [--graph-max-ctx 131072]
"""

import argparse
from datetime import datetime

import pandas as pd
import torch
import triton
from tabulate import tabulate

from primus_turbo.pytorch.ops.attention import (
    flash_attn_varlen_func,
    rl_flash_attn_varlen_func,
    rl_paged_attention_func,
)
from primus_turbo.pytorch.ops.attention.rl_attention import create_block_mask

MODELS = {
    # https://huggingface.co/meta-llama/Llama-3.1-8B
    "llama3-8b": {"num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "window": -1, "sinks": False},
    # https://huggingface.co/Qwen/Qwen3-32B
    "qwen3-32b": {"num_q_heads": 64, "num_kv_heads": 8, "head_dim": 128, "window": -1, "sinks": False},
    # https://huggingface.co/openai/gpt-oss-120b  (sliding-window layers: window 128, attention sinks)
    "gpt-oss-120b": {"num_q_heads": 64, "num_kv_heads": 8, "head_dim": 64, "window": 128, "sinks": True},
    # gpt-oss-120b's full-attention layers (alternating with the window-128 ones): no window, sinks
    "gpt-oss-120b-full": {"num_q_heads": 64, "num_kv_heads": 8, "head_dim": 64, "window": -1, "sinks": True},
}

DECODE_CASES = [(1, 4096), (8, 4096), (64, 4096), (256, 2048), (1, 32768), (8, 32768), (32, 32768)]
# Real rollout decode shapes: batch 64-256 per GPU, context 4k-32k.
DECODE_CASES += [(b, c) for c in (4096, 16384, 32768) for b in (64, 128, 256) if (b, c) not in DECODE_CASES]
PREFILL_CASES = [(1, 4096, 512), (4, 4096, 512), (1, 16384, 2048), (8, 2048, 2048)]  # (batch, ctx, q_len)
TRAIN_CASES = [(8, 4096), (4, 8192), (2, 16384)]  # (num_seqs, seqlen)
BLOCK_SIZE = 64

_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


def bench_ms(fn):
    return triton.testing.do_bench(fn, warmup=25, rep=100)


def bench_graph_ms(fn, iters=200):
    """Milliseconds per call of ``fn`` captured once in a HIP graph and replayed."""
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    for _ in range(10):
        graph.replay()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def compute_snr(ref, actual):
    ref, actual = ref.double(), actual.double()
    return (10 * torch.log10(ref.norm().pow(2) / ((ref - actual).norm().pow(2) + 1e-12))).item()


def bits_equal(a, b):
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


def heads(model, tp_size):
    m = MODELS[model]
    hq, hkv = m["num_q_heads"] // tp_size, max(m["num_kv_heads"] // tp_size, 1)
    return hq, hkv, m["head_dim"], m["window"], m["sinks"]


def causal_bounds(q_lens, k_lens, window_left=-1):
    """Global-index (lo, hi) of bottom-right causal attention, narrowed by a window."""
    lo, hi, base = [], [], 0
    for ql, kl in zip(q_lens, k_lens):
        pos = torch.arange(kl - ql, kl) + base
        start = torch.full_like(pos, base)
        lo.append(torch.maximum(start, pos - window_left) if window_left >= 0 else start)
        hi.append(pos)
        base += kl
    return torch.cat(lo).int().cuda(), torch.cat(hi).int().cuda()


def image_bounds(num_seqs, seqlen, span=(256, 1280)):
    """Causal sequences with a bidirectional span [span) (e.g. an image) in each."""
    pos = torch.arange(seqlen)
    lo, hi = torch.zeros(seqlen, dtype=torch.long), pos.clone()
    hi[span[0] : span[1]] = span[1] - 1
    base = (torch.arange(num_seqs) * seqlen).repeat_interleave(seqlen)
    return (lo.repeat(num_seqs) + base).int().cuda(), (hi.repeat(num_seqs) + base).int().cuda()


def paged_inputs(batch, ctx, q_len, hq, hkv, d, dtype):
    num_blocks = batch * triton.cdiv(ctx, BLOCK_SIZE)
    k_cache = torch.randn(num_blocks, BLOCK_SIZE, hkv, d, dtype=dtype, device="cuda")
    v_cache = torch.randn_like(k_cache)
    block_table = torch.randperm(num_blocks, device="cuda", dtype=torch.int32).view(batch, -1)
    q = torch.randn(batch * q_len, hq, d, dtype=dtype, device="cuda")
    seq_lens = torch.full((batch,), ctx, dtype=torch.int32, device="cuda")
    cu_q = torch.arange(0, (batch + 1) * q_len, q_len, dtype=torch.int32, device="cuda")
    return q, k_cache, v_cache, block_table, seq_lens, cu_q


def gather_contiguous(k_cache, block_table, ctx):
    """The paged cache as contiguous [batch * ctx, Hkv, D], in sequence order."""
    return k_cache[block_table.long()].flatten(1, 2)[:, :ctx].flatten(0, 1).contiguous()


def _aiter_unified_attention():
    try:
        from aiter.ops.triton.attention.unified_attention import unified_attention

        return unified_attention
    except Exception:  # aiter is optional; the baseline column is left empty without it
        return None


def _rollout_case(phase, model, batch, ctx, q_len, shape, dtype, check, unified_attention, backends):
    hq, hkv, d, window, has_sinks = shape
    scale = d**-0.5
    sinks = torch.randn(hq, device="cuda") if has_sinks else None
    q, k_cache, v_cache, block_table, seq_lens, cu_q = paged_inputs(batch, ctx, q_len, hq, hkv, d, dtype)
    mask = create_block_mask(*causal_bounds([q_len] * batch, [ctx] * batch, window))
    common = (q, k_cache, v_cache, cu_q, seq_lens, block_table, q_len, ctx, scale)
    keys_read = ctx if window < 0 else min(ctx, window + q_len)
    kv_bytes = 2 * batch * keys_read * hkv * d * q.element_size()
    row = {"phase": phase, "model": model, "batch": batch, "ctx": ctx, "q_len": q_len}

    t_aiter, aiter_out = float("nan"), None
    if unified_attention is not None and not has_sinks:
        aiter_out = torch.empty_like(q)

        def aiter_fn():
            unified_attention(
                q, k_cache, v_cache, aiter_out, cu_q, q_len, seq_lens, ctx, scale, True, (window, -1),
                block_table, 0, None, None, None,
            )  # fmt: skip

        t_aiter = bench_ms(aiter_fn) * 1e3
        aiter_fn()
    row["aiter_us"] = round(t_aiter, 1)

    for backend in backends:
        opts = {"BACKEND": backend}
        name = backend.lower()

        def rl(block_mask=None, opts=opts):
            w = -1 if block_mask is not None else window
            return rl_paged_attention_func(
                *common, window_left=w, block_mask=block_mask, sinks=sinks, kernel_options=opts
            )

        t_rl = bench_ms(rl) * 1e3
        t_mask = bench_ms(lambda: rl(mask)) * 1e3
        row[f"{name}_us"] = round(t_rl, 1)
        row[f"{name}+mask_us"] = round(t_mask, 1)
        row[f"{name}_vs_aiter"] = round(t_aiter / t_rl, 2)
        row[f"{name}_kv_GB/s"] = round(kv_bytes / (t_rl * 1e-6) / 1e9)
        if check:
            if aiter_out is not None:
                row[f"{name}_snr_db"] = round(compute_snr(aiter_out, rl()), 1)
            k = gather_contiguous(k_cache, block_table, ctx)
            v = gather_contiguous(v_cache, block_table, ctx)
            cu_k = torch.arange(0, (batch + 1) * ctx, ctx, dtype=torch.int32, device="cuda")
            train = rl_flash_attn_varlen_func(
                q, k, v, cu_q, cu_k, q_len, ctx, scale, window, sinks=sinks, kernel_options=opts
            )
            row[f"{name}==train"] = bits_equal(rl(), train) and bits_equal(rl(mask), train)
    return row


def bench_rollout(phase, model, tp_size, dtype, check, backends, decode_cases=DECODE_CASES):
    shape = heads(model, tp_size)
    unified_attention = _aiter_unified_attention()
    cases = [(b, c, 1) for b, c in decode_cases] if phase == "decode" else PREFILL_CASES
    return [
        _rollout_case(phase, model, batch, ctx, q_len, shape, dtype, check, unified_attention, backends)
        for batch, ctx, q_len in cases
    ]


def _graph_case(model, batch, ctx, max_ctx, shape, dtype, backends):
    """Decode as a serving engine captures it: one HIP graph with max_seqlen_k = max_ctx (the
    model's max length) replayed at the real context, against a graph captured at max_seqlen_k =
    ctx. The block table has max_ctx // BLOCK_SIZE columns, as an engine allocates it."""
    hq, hkv, d, window, has_sinks = shape
    scale = d**-0.5
    sinks = torch.randn(hq, device="cuda") if has_sinks else None
    q, k_cache, v_cache, block_table, seq_lens, cu_q = paged_inputs(batch, ctx, 1, hq, hkv, d, dtype)
    wide = torch.zeros(batch, triton.cdiv(max_ctx, BLOCK_SIZE), dtype=torch.int32, device="cuda")
    wide[:, : block_table.shape[1]] = block_table
    row = {"phase": "decode-graph", "model": model, "batch": batch, "ctx": ctx, "max_ctx": max_ctx}
    for backend in backends:
        opts = {"BACKEND": backend}
        name = backend.lower()

        def rl(max_k, opts=opts):
            return rl_paged_attention_func(
                q, k_cache, v_cache, cu_q, seq_lens, wide, 1, max_k, scale, window_left=window, sinks=sinks,
                kernel_options=opts,
            )  # fmt: skip

        t_ctx = bench_graph_ms(lambda: rl(ctx)) * 1e3
        t_max = bench_graph_ms(lambda: rl(max_ctx)) * 1e3
        row[f"{name}_graph_us"] = round(t_ctx, 1)
        row[f"{name}_graph_max_us"] = round(t_max, 1)
        row[f"{name}_max/ctx"] = round(t_max / t_ctx, 2)
        row[f"{name}_bits_equal"] = bits_equal(rl(ctx), rl(max_ctx))
    return row


def bench_decode_graph(model, tp_size, dtype, backends, decode_cases, max_ctx):
    shape = heads(model, tp_size)
    return [_graph_case(model, b, c, max_ctx, shape, dtype, backends) for b, c in decode_cases if c <= max_ctx]


def _time_fwd_and_fwdbwd(fn, dout):
    with torch.no_grad():
        fwd = bench_ms(fn)
    return fwd, bench_ms(lambda: fn().backward(dout))


def _train_case(model, num_seqs, seqlen, shape, dtype, check, backends):
    hq, hkv, d, window, has_sinks = shape
    scale = d**-0.5
    total = num_seqs * seqlen
    q = torch.randn(total, hq, d, dtype=dtype, device="cuda", requires_grad=True)
    k = torch.randn(total, hkv, d, dtype=dtype, device="cuda", requires_grad=True)
    v = torch.randn(total, hkv, d, dtype=dtype, device="cuda", requires_grad=True)
    sinks = torch.randn(hq, device="cuda", requires_grad=True) if has_sinks else None
    dout = torch.randn(total, hq, d, dtype=dtype, device="cuda")
    cu = torch.arange(0, (num_seqs + 1) * seqlen, seqlen, dtype=torch.int32, device="cuda")
    causal = create_block_mask(*causal_bounds([seqlen] * num_seqs, [seqlen] * num_seqs, window))
    image = create_block_mask(*image_bounds(num_seqs, seqlen))
    args = (q, k, v, cu, cu, seqlen, seqlen, scale)
    variants = {
        "turbo": lambda: flash_attn_varlen_func(
            *args[:7], softmax_scale=scale, causal=True, window_size=(window, 0), sink=sinks
        )
    }
    for backend in backends:
        opts = {"BACKEND": backend}
        name = backend.lower()
        variants[name] = lambda o=opts: rl_flash_attn_varlen_func(
            *args, window, sinks=sinks, kernel_options=o
        )
        variants[f"{name}+mask"] = lambda o=opts: rl_flash_attn_varlen_func(
            *args, block_mask=causal, sinks=sinks, kernel_options=o
        )
        if window < 0:  # the span ignores sliding windows, so it is only comparable without one
            variants[f"{name}+image"] = lambda o=opts: rl_flash_attn_varlen_func(
                *args, block_mask=image, sinks=sinks, kernel_options=o
            )
    row = {"phase": "train", "model": model, "num_seqs": num_seqs, "seqlen": seqlen}
    for name, fn in variants.items():
        try:
            fwd, fwdbwd = _time_fwd_and_fwdbwd(fn, dout)
        except Exception as e:  # e.g. the default backend lacks a feature (sinks, windows)
            print(f"  {name} skipped for {model} {num_seqs}x{seqlen}: {type(e).__name__}: {e}")
            fwd = fwdbwd = float("nan")
        row[f"{name}_fwd_ms"], row[f"{name}_fwdbwd_ms"] = round(fwd, 3), round(fwdbwd, 3)
    # causal FLOPs: 4 * Hq * D per (query, key) pair kept; the backward is about 2.5 forwards
    kept = seqlen * (seqlen + 1) / 2 if window < 0 else seqlen * min(window + 1, seqlen)
    flops = 4 * num_seqs * kept * hq * d
    for backend in backends:
        name = backend.lower()
        row[f"{name}_fwd_tflops"] = round(flops / (row[f"{name}_fwd_ms"] * 1e-3) / 1e12, 1)
        if check:
            with torch.no_grad():
                row[f"{name}_snr_db"] = round(compute_snr(variants["turbo"](), variants[name]()), 1)
                row[f"{name}_mask==builtin"] = bits_equal(variants[name](), variants[f"{name}+mask"]())
    return row


def bench_train(model, tp_size, dtype, check, backends):
    shape = heads(model, tp_size)
    return [_train_case(model, n, s, shape, dtype, check, backends) for n, s in TRAIN_CASES]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Benchmark batch-invariant RL attention")
    parser.add_argument(
        "--phase",
        nargs="+",
        default=["decode", "prefill", "train"],
        choices=["decode", "decode-graph", "prefill", "train"],
    )
    parser.add_argument("--model", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--backend", nargs="+", default=["TRITON", "FLYDSL"], choices=["TRITON", "FLYDSL"])
    parser.add_argument("--tp-size", type=int, default=1, help="shard attention heads over this many ranks")
    parser.add_argument("--dtype", default="bf16", choices=list(_DTYPES))
    parser.add_argument("--check", action="store_true", help="also check accuracy and rollout == train bits")
    parser.add_argument("--decode-batch", nargs="+", type=int, help="decode batches (with --decode-ctx)")
    parser.add_argument("--decode-ctx", nargs="+", type=int, help="decode contexts, e.g. 131072 262144 524288")
    parser.add_argument(
        "--graph-max-ctx", type=int, default=131072, help="decode-graph: max_seqlen_k a captured graph passes"
    )
    parser.add_argument("-o", "--output", help="CSV path (default: rl_attention_<date>_<gpu>.csv)")
    args = parser.parse_args(argv)

    dtype = _DTYPES[args.dtype]
    decode_cases = DECODE_CASES
    if args.decode_batch or args.decode_ctx:
        decode_cases = [
            (b, c) for c in args.decode_ctx or sorted({c for _, c in DECODE_CASES})
            for b in args.decode_batch or sorted({b for b, _ in DECODE_CASES})
        ]  # fmt: skip
    gpu = torch.cuda.get_device_name(0)
    print(f"{gpu}, triton {triton.__version__}, {args.dtype}, tp {args.tp_size}")
    results = []
    for phase in args.phase:
        rows = []
        for model in args.model:
            if phase == "train":
                rows += bench_train(model, args.tp_size, dtype, args.check, args.backend)
            elif phase == "decode-graph":
                rows += bench_decode_graph(
                    model, args.tp_size, dtype, args.backend, decode_cases, args.graph_max_ctx
                )
            else:
                rows += bench_rollout(phase, model, args.tp_size, dtype, args.check, args.backend, decode_cases)
        print(f"\n{phase}")
        print(tabulate(rows, headers="keys", tablefmt="github"))
        results += rows
    name = args.output or f"rl_attention_{datetime.now():%Y%m%d}_{gpu.replace(' ', '_')}.csv"
    pd.DataFrame(results).to_csv(name, index=False)
    print(f"\nwrote {name}")


if __name__ == "__main__":
    main()
