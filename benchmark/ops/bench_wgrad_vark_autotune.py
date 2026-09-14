#!/usr/bin/env python3
"""Sweep variable-K wgrad autotune candidates for dW1/dW2 shapes (single GPU).

Run:
  PYTHONPATH=<repo> python benchmark/ops/bench_wgrad_vark_autotune.py [--shape dw1|dw2|both]
"""

import argparse
import math

import numpy as np
import torch

import primus_turbo.pytorch  # noqa: F401
import primus_turbo.flydsl.grouped_gemm.mxfp8_grouped_kernel as mgk
from primus_turbo.flydsl.grouped_gemm.mxfp8_grouped_kernel import (
    _PRESHUF_KT,
    _get_grouped_wgrad_workspace,
    _get_wgrad_launch,
    _gwg_wgrad_candidates,
    _robust_time,
    ceildiv,
    grouped_gemm_mxfp8_variable_k_flydsl_kernel,
)
from primus_turbo.pytorch.core.low_precision import float8_e5m2


def _setup_dw2(M, H, I, G, meta):
    q = torch.randn(M, H, device="cuda").to(torch.float8_e4m3fn)
    sc = torch.randint(120, 140, (M, H // 32), device="cuda", dtype=torch.uint8)
    act = torch.randn(M, I, device="cuda", dtype=torch.bfloat16)
    from primus_turbo.flydsl.mega.fp8 import colwise_requant_fp8in_and_quant_bf16_grouped_flydsl

    a_t, a_ts, b_t, b_ts, _, _ = colwise_requant_fp8in_and_quant_bf16_grouped_flydsl(
        q, sc, act, float8_e5m2, meta=meta,
    )
    return a_t, a_ts, b_t, b_ts, H, I


def _setup_dw1(M, H, I, G, meta):
    from primus_turbo.flydsl.mega.fp8 import (
        colwise_requant_fp8in_and_quant_bf16_grouped_flydsl,
    )

    gl1 = torch.randn(M, 2 * I, device="cuda", dtype=torch.bfloat16)
    pool = torch.randn(M, H, device="cuda").to(torch.float8_e4m3fn)
    psc = torch.randint(120, 140, (M, H // 32), device="cuda", dtype=torch.uint8)
    # dW1 takes the operand roles the other way round from dW2 -- `a` is the bf16 grad_l1 quant and
    # `b` the fp8 pool requant -- so the dual's two outputs land swapped here.
    b_t, b_ts, a_t, a_ts, _, _ = colwise_requant_fp8in_and_quant_bf16_grouped_flydsl(
        pool, psc, gl1, float8_e5m2, meta=meta,
    )
    return a_t, a_ts, b_t, b_ts, 2 * I, H


def _wgrad_args(a_t, a_ts, b_t, b_ts, offs_pc, OUT_M, OUT_N, G):
    cbsz = 1 if a_t.dtype == torch.float8_e5m2 else 0
    blgp = 1 if b_t.dtype == torch.float8_e5m2 else 0
    out_fp16 = False
    M_total = a_t.shape[1]
    K128 = M_total // 128
    a_raw = a_ts.contiguous().view(torch.int32).reshape(-1)
    b_raw = b_ts.contiguous().view(torch.int32).reshape(-1)
    a8 = a_t.contiguous().view(torch.int8)
    b8 = b_t.contiguous().view(torch.int8)
    out = torch.empty((G, OUT_M, OUT_N), dtype=torch.bfloat16, device=a_t.device)
    go = offs_pc.to(torch.int64).view(torch.int32)
    stream = torch.cuda.current_stream()
    a_sp, b_sp = _get_grouped_wgrad_workspace(OUT_M, OUT_N, K128, a_t.device, stream)
    a_ngrp = ceildiv(OUT_M, 64)
    b_ngrp = ((OUT_N + 255) // 256) * 4
    n_kt = ceildiv(K128, _PRESHUF_KT)
    a_blocks = a_ngrp * n_kt
    pre_grid = a_blocks + b_ngrp * n_kt
    args = (
        a8, b8, out, a_raw, b_raw, a_sp, b_sp, go, M_total, K128, n_kt, a_blocks, pre_grid, stream,
    )
    at_key = (OUT_M, OUT_N, M_total, G, cbsz, blgp, out_fp16)
    flops = 2.0 * M_total * OUT_M * OUT_N
    return args, out, at_key, cbsz, blgp, out_fp16, flops


def sweep_shape(label, OUT_M, OUT_N, G, setup_fn, M, H, I, meta, *, warmup, iters):
    mgk._GWG_CFG_CACHE.clear()
    mgk._GWG_AT_CACHE.clear()

    a_t, a_ts, b_t, b_ts, om, on = setup_fn(M, H, I, G, meta)
    assert om == OUT_M and on == OUT_N, f"{label}: expected OUT_M/N={OUT_M}/{OUT_N}, got {om}/{on}"
    args, out, at_key, cbsz, blgp, out_fp16, flops = _wgrad_args(
        a_t, a_ts, b_t, b_ts, meta["offs_pc"], OUT_M, OUT_N, G,
    )

    cands = [c for c in _gwg_wgrad_candidates() if OUT_M % c[0] == 0 and OUT_N % c[1] == 0]
    print(f"\n{'=' * 72}")
    print(f"[{label}] OUT_M={OUT_M} OUT_N={OUT_N} M={M} G={G}  candidates={len(cands)}")
    print(f"{'=' * 72}")
    print(f"  {'cfg (bm,bn,gm,xcd,gn)':<28} {'ms':>8} {'TFLOPS':>10}  finite")

    base_cfg = cands[0]
    base_launch = _get_wgrad_launch(OUT_M, OUT_N, G, *base_cfg, cbsz, blgp, out_fp16)
    base_launch(*args)
    torch.cuda.synchronize()
    ref = out.detach().clone().float()
    ref_n = float((ref * ref).sum().item()) or 1.0

    results = []
    for cfg in cands:
        launch = _get_wgrad_launch(OUT_M, OUT_N, G, *cfg, cbsz, blgp, out_fp16)
        try:
            launch(*args)
            torch.cuda.synchronize()
            o = out.detach().float()
            err = float(((o - ref) * (o - ref)).sum().item())
            ok = (err / ref_n) < (2e-2**2) and torch.isfinite(o.reshape(-1)[:1024]).all().item()
            if not ok:
                print(f"  {str(cfg):<28} {'SKIP':>8}  (output drift)")
                continue
            t = _robust_time(launch, args, warmup=warmup, iters=iters)
            tf = flops / (t * 1e-3) / 1e12
            results.append((t, cfg, tf))
            mark = " *" if cfg == base_cfg else ""
            print(f"  {str(cfg):<28} {t:8.3f} {tf:10.1f}  {ok}{mark}")
        except Exception as ex:
            print(f"  {str(cfg):<28} {'FAIL':>8}  ({ex})")

    if not results:
        print("  no valid candidates")
        return None

    best_t, best_cfg, best_tf = min(results, key=lambda x: x[0])
    base_t = next(t for t, c, _ in results if c == base_cfg)
    print(f"\n  best: {best_cfg}  {best_t:.3f} ms ({best_tf:.1f} TFLOPS)")
    print(f"  base: {base_cfg}  {base_t:.3f} ms  delta vs base: {(best_t/base_t - 1)*100:+.1f}%")

    # run built-in autotune selector
    mgk._GWG_CFG_CACHE.clear()
    mgk._GWG_AT_CACHE.clear()
    mgk._GWG_CFG_CACHE[at_key] = None  # force miss
    t_auto = _bench_kernel(a_t, a_ts, b_t, b_ts, meta["offs_pc"], OUT_M, OUT_N, G)
    picked = mgk._GWG_CFG_CACHE.get(at_key, mgk._GWG_WGRAD_DEFAULT_CFG)
    print(f"  autotune picked: {picked}  kernel={t_auto:.3f} ms")
    return best_cfg, best_t, picked


def _bench_kernel(a_t, a_ts, b_t, b_ts, offs_pc, OUT_M, OUT_N, G, warmup=3, iters=15):
    for _ in range(warmup):
        grouped_gemm_mxfp8_variable_k_flydsl_kernel(
            a_t, a_ts, b_t, b_ts, offs_pc.to(torch.int64), OUT_M, OUT_N, G,
        )
    torch.cuda.synchronize()
    s = [torch.cuda.Event(True) for _ in range(iters)]
    e = [torch.cuda.Event(True) for _ in range(iters)]
    for i in range(iters):
        s[i].record()
        grouped_gemm_mxfp8_variable_k_flydsl_kernel(
            a_t, a_ts, b_t, b_ts, offs_pc.to(torch.int64), OUT_M, OUT_N, G,
        )
        e[i].record()
    torch.cuda.synchronize()
    return float(np.median([s[i].elapsed_time(e[i]) for i in range(iters)]))


def _make_meta(M: int, G: int):
    """Balanced G groups summing to M (like load_balanced EP8 local pool)."""
    base, rem = divmod(M, G)
    lens = torch.tensor([base + (1 if i < rem else 0) for i in range(G)], device="cuda", dtype=torch.int32)
    offs = torch.cat([torch.zeros(1, dtype=torch.int32, device="cuda"), torch.cumsum(lens, 0)])
    from primus_turbo.flydsl.mega.fp8 import colwise_grouped_meta
    return colwise_grouped_meta(lens, offs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", choices=["dw1", "dw2", "both"], default="both")
    ap.add_argument("--M", type=int, default=8192, help="total unpadded M on this rank")
    ap.add_argument("--H", type=int, default=7168)
    ap.add_argument("--I", type=int, default=2048)
    ap.add_argument("--G", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=12)
    args = ap.parse_args()

    meta = _make_meta(args.M, args.G)
    print(f"meta: M={args.M} G={args.G} M_pad={meta['total_M_pad']}")

    if args.shape in ("dw2", "both"):
        sweep_shape(
            "dW2", args.H, args.I, args.G, _setup_dw2,
            args.M, args.H, args.I, meta, warmup=args.warmup, iters=args.iters,
        )
    if args.shape in ("dw1", "both"):
        sweep_shape(
            "dW1", 2 * args.I, args.H, args.G, _setup_dw1,
            args.M, args.H, args.I, meta, warmup=args.warmup, iters=args.iters,
        )


if __name__ == "__main__":
    main()
