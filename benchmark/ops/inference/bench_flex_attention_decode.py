###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Paged-KV GQA decode: the FlyDSL flash attention forward vs aiter (per rank, one GPU).

FlyDSL runs ``FlashAttentionForward(paged_kv=True)`` (FA4's page_table + seqused_k),
which sends one-token queries to the ``FlashAttentionDecode`` kernel. With
``--kv-dtype fp8`` both sides take a bf16 query and the same native-fp8 cache with
per-tensor k / v scales (FlyDSL calls ``FlashAttentionDecode(kv_dtype="fp8")``
directly). aiter runs ``paged_attention_ragged``, the
kernel SGLang's aiter backend uses for full-GQA decode (skipped when aiter is not
installed). Both read the same shuffled page pool. Decode is memory-bound, so cells
report KV bandwidth; ``--check`` adds the SNR against an fp32 reference.

    python bench_flex_attention_decode.py [--model M] [--batches ...] [--ctx ...]
        [--page-size 16] [--ctx-cv 0.0] [--tp-size 8] [--kv-dtype bf16|fp8] [--check]
"""

import argparse
import contextlib
import math
import os

import torch
from tabulate import tabulate

from primus_turbo.flydsl.flex_attention import FlashAttentionForward
from primus_turbo.flydsl.flex_attention.flash_decode import FlashAttentionDecode

try:
    from aiter.ops.attention import paged_attention_ragged
except ImportError:  # aiter is optional: the FlyDSL column still runs
    paged_attention_ragged = None

# Global (unsharded) head counts; heads are split over --tp-size ranks.
MODELS = {
    "minimax-m2.7": {"num_q_heads": 48, "num_kv_heads": 8, "head_dim": 128},
    "llama-3-70b": {"num_q_heads": 64, "num_kv_heads": 8, "head_dim": 128},
    # sliding-window layers: 128-token window plus a learnable sink per head
    "gpt-oss-120b-swa": {"num_q_heads": 64, "num_kv_heads": 8, "head_dim": 64, "window": 128, "sinks": True},
}
DEFAULT_BATCHES = [1, 8, 32, 128]
DEFAULT_CTX = [1024, 8192, 32768]
PARTITION_SIZE = 256  # aiter paged_attention_ragged partition size
PEAK_HBM_GBPS = {"gfx942": 5300, "gfx950": 8000}


@contextlib.contextmanager
def quiet():
    """Silence aiter's per-call prints at the fd level."""
    with open(os.devnull, "w") as devnull:
        saved = os.dup(1), os.dup(2)
        os.dup2(devnull.fileno(), 1)
        os.dup2(devnull.fileno(), 2)
        try:
            yield
        finally:
            for fd, old in zip((1, 2), saved):
                os.dup2(old, fd)
                os.close(old)


def time_ms(fn, warmup=10, iters=50):
    """Per-call GPU time, replaying a CUDA graph (as serving decode does); eager when
    the call cannot be captured."""

    def loop(call):
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            call()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    try:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        graph.replay()
        return loop(graph.replay), "graph"
    except Exception:  # noqa: BLE001 - not capturable: time eager launches
        torch.cuda.synchronize()
        return loop(fn), "eager"


def ctx_lengths(batch, ctx, cv, seed):
    """Lognormal context lengths with mean ``ctx`` and coefficient of variation ``cv``
    (as bench_decode_attn_aiter.py), clamped to [1, 8 * ctx]."""
    if cv <= 0:
        return torch.full((batch,), ctx, dtype=torch.int64)
    sigma = math.sqrt(math.log(1.0 + cv * cv))
    g = torch.Generator().manual_seed(seed)
    lens = torch.exp(torch.randn(batch, generator=g) * sigma + math.log(ctx) - 0.5 * sigma * sigma)
    return lens.round().clamp_(1, 8 * ctx).to(torch.int64)


def snr_db(ref, got):
    ref, got = ref.double(), got.double()
    return (10 * torch.log10(ref.norm() ** 2 / ((ref - got).norm() ** 2 + 1e-12))).item()


def reference(q, k_cache, v_cache, table, lens, scale, window=None, sink=None):
    """fp32 decode reference: q [B, Hq, D] -> [B, Hq, D]."""
    B, Hq, D = q.shape
    Hk, page_size = k_cache.shape[2], k_cache.shape[1]
    out = torch.empty(B, Hq, D, device=q.device)
    for b in range(B):
        n = int(lens[b])
        pages = table[b, : -(-n // page_size)].long()
        k = k_cache[pages].reshape(-1, Hk, D)[:n].float().repeat_interleave(Hq // Hk, 1)
        v = v_cache[pages].reshape(-1, Hk, D)[:n].float().repeat_interleave(Hq // Hk, 1)
        s = torch.einsum("hd,nhd->hn", q[b].float(), k) * scale
        if window is not None:
            s[:, : max(n - 1 - window, 0)] = float("-inf")
        if sink is not None:
            s = torch.cat([s, sink.float()[:, None]], 1)
        p = torch.softmax(s, -1)
        out[b] = torch.einsum("hn,nhd->hd", p[:, :n], v)
    return out


def fp8_format():
    return torch.float8_e4m3fn if "gfx95" in torch.cuda.get_device_properties(0).gcnArchName else torch.float8_e4m3fnuz


def bench_cell(cfg, batch, ctx, args, seed):
    Hq, Hk, D = cfg["num_q_heads"] // args.tp_size, max(cfg["num_kv_heads"] // args.tp_size, 1), cfg["head_dim"]
    window, use_sink = cfg.get("window"), cfg.get("sinks", False)
    ps, dev, scale = args.page_size, "cuda", D**-0.5
    lens = ctx_lengths(batch, ctx, args.ctx_cv, seed)
    pages_per_seq = (lens + ps - 1) // ps
    offsets = torch.cat([torch.zeros(1, dtype=torch.int64), pages_per_seq.cumsum(0)])
    num_pages = int(offsets[-1])
    perm = torch.randperm(num_pages, generator=torch.Generator().manual_seed(seed))
    max_pages = int(pages_per_seq.max())
    table = torch.zeros(batch, max_pages, dtype=torch.int32)
    for b in range(batch):
        table[b, : pages_per_seq[b]] = perm[offsets[b] : offsets[b + 1]]
    table, used = table.to(dev), lens.to(dev, torch.int32)

    q = torch.randn(batch, 1, Hq, D, device=dev, dtype=torch.bfloat16)
    k_cache = torch.randn(num_pages, ps, Hk, D, device=dev, dtype=torch.bfloat16)
    v_cache = torch.randn(num_pages, ps, Hk, D, device=dev, dtype=torch.bfloat16)
    sink = torch.randn(Hq, device=dev) if use_sink else None
    if args.kv_dtype == "fp8":  # one quantized cache for both kernels; the reference reads it dequantized
        fmt = fp8_format()
        k_scale, v_scale = (t.abs().amax().float().reshape(1) / torch.finfo(fmt).max for t in (k_cache, v_cache))
        k_cache, v_cache = (k_cache / k_scale).to(fmt), (v_cache / v_scale).to(fmt)
        k_ref, v_ref = k_cache.float() * k_scale, v_cache.float() * v_scale
    else:
        k_scale = v_scale = torch.ones(1, device=dev)
        k_ref, v_ref = k_cache, v_cache
    # Bytes a decode step must read: a sliding window only touches its last window + 1 keys.
    read_lens = lens if window is None else lens.clamp(max=window + 1)
    kv_bytes = 2 * int(read_lens.sum()) * Hk * D * (1 if args.kv_dtype == "fp8" else 2)
    peak = PEAK_HBM_GBPS.get(torch.cuda.get_device_properties(0).gcnArchName.split(":")[0])
    ref = reference(q[:, 0], k_ref, v_ref, table, lens, scale, window, sink) if args.check else None

    def result(ms, mode, out):
        gbps = kv_bytes / (ms * 1e-3) / 1e9
        snr = snr_db(ref, out.float()) if ref is not None else None
        return {"ms": ms, "gbps": gbps, "pct": 100 * gbps / peak if peak else None, "snr": snr, "mode": mode}

    # ---- FlyDSL ----
    flags = dict(head_dim=D, qhead_per_kvhead=Hq // Hk, has_seqused_k=True, paged_kv=True)
    call = dict(seqused_k=used, page_table=table)
    if window is not None:
        flags.update(is_local=True)
        call.update(window_left=window, window_right=0)
    if use_sink:  # FlyDSL's split-KV does not combine sinks yet
        flags.update(has_learnable_sink=True)
        call.update(learnable_sink=sink)
    else:
        flags.update(split_kv=True)
        call.update(num_splits=0)
    holder = {}
    if args.kv_dtype == "fp8":
        dec = FlashAttentionDecode(head_dim=D, qhead_per_kvhead=Hq // Hk, is_local=window is not None,
                                   has_learnable_sink=use_sink, kv_dtype="fp8")
        parts = -(-int(lens.max()) // 128)
        partials = (torch.empty(batch, Hk, parts, Hq // Hk, D, device=dev),
                    torch.empty(batch, Hk, parts, Hq // Hk, device=dev))
        out = torch.empty(batch, Hq, D, device=dev, dtype=torch.bfloat16)
        lse = torch.empty(batch, Hq, device=dev)

        def run_flydsl():
            holder["out"], _ = dec(q[:, 0], k_cache, v_cache, table, used, scale, out=out, lse=lse,
                                   window_left=-1 if window is None else window, learnable_sink=sink,
                                   partials=partials, k_scale=k_scale, v_scale=v_scale)
    else:
        fwd = FlashAttentionForward(**flags)

        def run_flydsl():
            holder["out"], _ = fwd(q, k_cache, v_cache, scale, **call)
            holder["out"] = holder["out"][:, 0]

    ms, mode = time_ms(run_flydsl)
    row = {"flydsl": result(ms, mode, holder["out"])}

    # ---- aiter (full-attention GQA only) ----
    if paged_attention_ragged is not None and window is None and not use_sink:
        max_parts = -(-int(lens.max()) // PARTITION_SIZE)
        workspace = torch.empty((batch * Hq * max_parts * D) * 4 + 2 * (batch * Hq * max_parts) * 4,
                                dtype=torch.uint8, device=dev)
        kv_dtype = "fp8_e4m3" if args.kv_dtype == "fp8" else "auto"
        a_out = torch.empty(batch, Hq, D, device=dev, dtype=torch.bfloat16)
        indptr = offsets.to(dev, torch.int32)
        indices = perm.to(dev, torch.int32)
        last = (((lens - 1) % ps) + 1).to(dev, torch.int32)

        def run_aiter():
            paged_attention_ragged(a_out, workspace, q[:, 0], k_cache, v_cache, scale, indptr, indices, last, ps,
                                   max_parts, None, kv_dtype, "NHD", 0.0, k_scale, v_scale, None, PARTITION_SIZE)

        with quiet():
            ms, mode = time_ms(run_aiter)
        row["aiter"] = result(ms, mode, a_out)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS), default="minimax-m2.7")
    parser.add_argument("--batches", type=int, nargs="+", default=DEFAULT_BATCHES)
    parser.add_argument("--ctx", type=int, nargs="+", default=DEFAULT_CTX)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--ctx-cv", type=float, default=0.0, help="context-length spread (lognormal cv)")
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--kv-dtype", choices=("bf16", "fp8"), default="bf16")
    parser.add_argument("--check", action="store_true", help="SNR against an fp32 reference")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    cfg = MODELS[args.model]
    props = torch.cuda.get_device_properties(0)
    print(f"{props.name} ({props.gcnArchName.split(':')[0]}), model={args.model} TP={args.tp_size} "
          f"(Hq={cfg['num_q_heads'] // args.tp_size}, Hk={max(cfg['num_kv_heads'] // args.tp_size, 1)}, "
          f"D={cfg['head_dim']}), page_size={args.page_size}, kv={args.kv_dtype}, ctx_cv={args.ctx_cv}"
          + ("" if paged_attention_ragged is not None else "  [aiter not installed]"), flush=True)

    rows = []
    for batch in args.batches:
        for ctx in args.ctx:
            cell = bench_cell(cfg, batch, ctx, args, args.seed + 1000 * batch + ctx)
            fl, ai = cell["flydsl"], cell.get("aiter")
            fmt = lambda r, k, f: "-" if r is None or r[k] is None else format(r[k], f)  # noqa: E731
            rows.append([
                batch, ctx, fmt(fl, "ms", ".3f"), fmt(fl, "gbps", ".0f"), fmt(fl, "pct", ".0f"),
                fmt(ai, "ms", ".3f"), fmt(ai, "gbps", ".0f"), fmt(ai, "pct", ".0f"),
                "-" if ai is None else f"{ai['ms'] / fl['ms']:.2f}x",
                fmt(fl, "snr", ".1f"), fmt(ai, "snr", ".1f"), fl["mode"],
            ])
            print(rows[-1], flush=True)
            torch.cuda.empty_cache()
    print(tabulate(rows, headers=["batch", "ctx", "flydsl ms", "GB/s", "% peak", "aiter ms", "GB/s", "% peak",
                                  "flydsl speedup", "flydsl SNR dB", "aiter SNR dB", "timing"], tablefmt="github"))


if __name__ == "__main__":
    main()
