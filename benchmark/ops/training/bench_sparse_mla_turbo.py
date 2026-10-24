###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""DeepSeek-V4 sparse-MLA attention training benchmark (``turbo.ops.sparse_mla_func``).

Shapes follow the DSV4 attention variants: flash (64 heads, index top-k 512) and pro (128
heads, top-k 1024), each at three compression ratios that are different workloads, so the
results are reported per row and never averaged across them:

* cr=0   : sliding window only (128 keys per token).
* cr=4   : window + top-k distinct random picks from a seqlen/4 compressed pool, causally
  masked like DSV4 CSA (pool row p is visible from token 4 * (p + 1) - 1), ``-1``-padded.
* cr=128 : window + every row of a seqlen/128 pool, causally masked (HCA).

TFLOPS count only valid (non ``-1``) top-k slots: forward ``2 * H * valid * (512 + 512)``,
backward 2.5x that. Every backend is checked against the Triton kernels called directly.
"""

import argparse
import math
from datetime import datetime

import pandas as pd
import torch
import torch.utils.benchmark as benchmark
from config import compute_snr, get_platform_info
from tabulate import tabulate

import primus_turbo.pytorch as turbo
from primus_turbo.pytorch.core.backend import (
    BackendType,
    GlobalBackendManager,
    PrecisionType,
)
from primus_turbo.triton.attention.sparse_mla import (
    sparse_mla_bwd_triton,
    sparse_mla_fwd_triton,
)

ROPE_DIM = 64
LATENT_DIM = 512
SWA_WINDOW = 128
VARIANTS = {"flash": (64, 512), "pro": (128, 1024)}  # num_heads, index top-k cap
CRS = (0, 4, 128)
SEQLENS = (4096, 8192, 16384)
BACKENDS = ("FLYDSL", "TRITON")
_PRECISION = PrecisionType.BF16_FP16_FP32


def build_inputs(variant, cr, seqlen, seed=0):
    """q, kv, top-k indices (SWA band + optional pool, ``-1`` padded to 64), sink, grad_out."""
    num_heads, topk_cap = VARIANTS[variant]
    gen = torch.Generator(device="cuda").manual_seed(seed)
    dev, dt, d, w = "cuda", torch.bfloat16, LATENT_DIM, SWA_WINDOW
    latent = torch.randn(seqlen, d, generator=gen, device=dev, dtype=dt)
    q = torch.randn(seqlen, num_heads, d, generator=gen, device=dev, dtype=dt)
    q = torch.cat([q, torch.zeros(seqlen, num_heads, ROPE_DIM, device=dev, dtype=dt)], -1)
    sink = torch.randn(num_heads, generator=gen, device=dev, dtype=torch.float32) * 0.1
    grad_out = torch.randn(seqlen, num_heads, d, generator=gen, device=dev, dtype=dt)

    tok = torch.arange(seqlen, device=dev).view(seqlen, 1)
    win = tok - w + 1 + torch.arange(w, device=dev).view(1, w)
    win = torch.where(win >= 0, win, torch.full_like(win, -1))
    if cr == 0:
        kv = latent
        topk = win
    else:
        pool = max(seqlen // cr, 1)
        kv = torch.cat([latent, torch.randn(pool, d, generator=gen, device=dev, dtype=dt)], 0)
        ps = torch.arange(pool, device=dev).view(1, pool)
        visible = ((ps + 1) * cr - 1) <= tok
        if cr == 4:
            # Distinct, causally visible picks in random order, as the indexer emits them; tokens
            # that see fewer than topk_cap pool entries get trailing -1.
            n = min(topk_cap, pool)
            score = torch.rand(seqlen, pool, generator=gen, device=dev).masked_fill(~visible, float("-inf"))
            val, pick = score.topk(n, dim=1)
            pool_topk = torch.where(torch.isneginf(val), torch.full_like(pick, -1), seqlen + pick)
        else:
            pool_topk = torch.where(visible, seqlen + ps, torch.full_like(ps.expand(seqlen, pool), -1))
        topk = torch.cat([win, pool_topk], 1)
    pad = -topk.shape[1] % 64
    if pad:
        topk = torch.cat([topk, torch.full((seqlen, pad), -1, device=dev, dtype=topk.dtype)], 1)
    kv = torch.cat([kv, torch.zeros(kv.shape[0], ROPE_DIM, device=dev, dtype=dt)], -1).unsqueeze(1)
    return q.contiguous(), kv.contiguous(), topk.to(torch.int32).contiguous(), sink, grad_out


def _timeit(fn, iters):
    return benchmark.Timer(stmt="fn()", globals={"fn": fn}).timeit(iters).mean * 1e3


def profile_backend(backend, q, kv, topk, sink, grad_out, ref, scale, iters):
    """(min SNR over o/dq/dkv/dsink, fwd ms, bwd ms, bwd peak GB) for one pinned backend."""
    GlobalBackendManager.set_sparse_attn_backend(BackendType[backend], _PRECISION)
    try:
        qg, kvg, sg = (t.detach().clone().requires_grad_() for t in (q, kv, sink))

        def fwd():
            return turbo.ops.sparse_mla_func(
                qg, kvg, topk, attn_sink=sg, kv_lora_rank=LATENT_DIM, scale=scale
            )

        out = fwd()
        out.backward(grad_out)
        snr = min(compute_snr(r, x) for r, x in zip(ref, (out, qg.grad, kvg.grad.view_as(ref[2]), sg.grad)))

        # autograd.grad returns fresh gradients: accumulating into .grad would add a q-sized
        # elementwise pass per call that a training step does not pay for activations.
        out = fwd()
        bwd = lambda: torch.autograd.grad(out, (qg, kvg, sg), grad_out, retain_graph=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        bwd()
        torch.cuda.synchronize()
        peak_gb = (torch.cuda.max_memory_allocated() - base) / 1e9
        for _ in range(10):
            fwd()
            bwd()
        torch.cuda.synchronize()
        fwd_ms = _timeit(fwd, iters)
        bwd_ms = _timeit(bwd, iters)
        return snr, fwd_ms, bwd_ms, peak_gb
    finally:
        GlobalBackendManager.reset()


def benchmark_sparse_mla(variants, crs, seqlens, backends, iters, output_csv):
    platform, gpu_name = get_platform_info()
    rows = []
    for variant in variants:
        num_heads = VARIANTS[variant][0]
        for cr in crs:
            for seqlen in seqlens:
                q, kv, topk, sink, grad_out = build_inputs(variant, cr, seqlen)
                scale = 1.0 / math.sqrt(LATENT_DIM)
                valid = int((topk >= 0).sum())
                fwd_flops = 2 * num_heads * valid * (LATENT_DIM + LATENT_DIM)
                bwd_flops = 2.5 * fwd_flops
                o_r, lse_r = sparse_mla_fwd_triton(
                    q, kv, topk, attn_sink=sink, kv_lora_rank=LATENT_DIM, scale=scale
                )
                dq_r, dkv_r, ds_r = sparse_mla_bwd_triton(
                    q, kv, o_r, grad_out, topk, lse_r, attn_sink=sink, kv_lora_rank=LATENT_DIM, scale=scale
                )
                ref = (o_r, dq_r, dkv_r, ds_r)
                print(f"\n{'=' * 60}\n{variant} H={num_heads} cr={cr} seqlen={seqlen} topk={topk.shape[1]}")
                for backend in backends:
                    row = {
                        "Platform": platform,
                        "GPU": gpu_name,
                        "Variant": variant,
                        "num_heads": num_heads,
                        "CR": cr,
                        "SeqLen": seqlen,
                        "TopK": topk.shape[1],
                        "Valid (%)": f"{100 * valid / topk.numel():.1f}",
                        "Backend": backend,
                    }
                    try:
                        snr, fwd_ms, bwd_ms, peak_gb = profile_backend(
                            backend, q, kv, topk, sink, grad_out, ref, scale, iters
                        )
                        floor = 60.0 if backend == "TRITON" else 40.0
                        step_ms = fwd_ms + bwd_ms
                        row.update(
                            {
                                "Check": f"{'PASS' if snr > floor else 'FAIL'} ({snr:.1f} dB)",
                                "Forward Time (ms)": f"{fwd_ms:.3f}",
                                "Forward TFLOPS": f"{fwd_flops / fwd_ms / 1e9:.1f}",
                                "Backward Time (ms)": f"{bwd_ms:.3f}",
                                "Backward TFLOPS": f"{bwd_flops / bwd_ms / 1e9:.1f}",
                                "Step Time (ms)": f"{step_ms:.3f}",
                                "Step TFLOPS": f"{(fwd_flops + bwd_flops) / step_ms / 1e9:.1f}",
                                "Backward Peak (GB)": f"{peak_gb:.2f}",
                            }
                        )
                    except Exception as e:  # noqa: BLE001
                        print(f"Failed {backend}: {e}")
                        row.update({"Check": "ERROR"})
                    print(row, flush=True)
                    rows.append(row)
                del q, kv, topk, sink, grad_out, ref, o_r, dq_r, dkv_r
                torch.cuda.empty_cache()

    results = pd.DataFrame(rows)
    keys = ["Variant", "CR", "SeqLen"]
    passes = ("Forward", "Backward", "Step")
    speedup = {}
    if "TRITON" in backends and "Step Time (ms)" in results.columns:
        for p in passes:
            t = pd.to_numeric(results[f"{p} Time (ms)"], errors="coerce")
            triton = results.assign(_t=t)[results["Backend"] == "TRITON"].set_index(keys)["_t"]
            base = results[keys].apply(lambda r, tr=triton: tr.get(tuple(r), float("nan")), axis=1)
            speedup[p] = base / t
            results[f"{p} Speedup vs Triton"] = speedup[p].map(lambda x: f"{x:.2f}x" if x == x else "-")

    print("\nFinal Results:")
    print(tabulate(results, headers="keys", tablefmt="grid", showindex=False))

    # Forward, backward and step separately, one line per (variant, seqlen) with compression
    # ratios side by side: no averages across passes or compression ratios.
    for backend in backends:
        part = results[results["Backend"] == backend]
        if "Step TFLOPS" not in part.columns or part.empty:
            continue
        for p in passes:
            tf = pd.to_numeric(part[f"{p} TFLOPS"], errors="coerce")
            cell = tf.map(lambda x: f"{x:.1f}")
            if p in speedup and backend != "TRITON":
                cell = cell + speedup[p][part.index].map(lambda x: f" ({x:.2f}x)" if x == x else "")
            pivot = part.assign(_c=cell).pivot_table(
                index=["Variant", "SeqLen"], columns="CR", values="_c", aggfunc="first"
            )
            pivot.columns = [f"cr={c}" for c in pivot.columns]
            vs = " (speedup vs Triton)" if p in speedup and backend != "TRITON" else ""
            print(f"\n{backend} {p.lower()} TFLOPS{vs}, valid slots, by compression ratio:")
            print(tabulate(pivot.reset_index(), headers="keys", tablefmt="grid", showindex=False))

    filename = output_csv or f"sparse_mla_benchmark_result_{datetime.now():%Y%m%d}_{gpu_name}.csv"
    results.to_csv(filename, index=False)
    print(f"Results saved to {filename}")


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Benchmark DeepSeek-V4 sparse-MLA attention (fwd + bwd)")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output CSV filename.")
    parser.add_argument(
        "--backend",
        choices=("all", "flydsl", "triton"),
        default="all",
        help="Backends to pin and time (default: all; Triton is the reference).",
    )
    parser.add_argument("--variants", nargs="+", choices=tuple(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--cr", nargs="+", type=int, choices=CRS, default=list(CRS))
    parser.add_argument("--seqlens", nargs="+", type=int, default=list(SEQLENS))
    parser.add_argument("--iters", type=int, default=50, help="Timed iterations per pass (default: 50).")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    backends = BACKENDS if args.backend == "all" else (args.backend.upper(),)
    benchmark_sparse_mla(args.variants, args.cr, args.seqlens, backends, args.iters, args.output)
