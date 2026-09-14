#!/usr/bin/env python3
"""Compare MoE backward grouped GEMM vs CK / hipBLASLt / Triton / FlyDSL.

Uses DeepSeek-V3 shapes: T=8192, EP8, topk=8, H=7168, I=2048.
Per rank: G=32 experts, M=8192 pool rows (256 rows/expert, BM-padded, uniform).

Two grouped-GEMM APIs (same math, different metadata):
  * turbo / CK / hipBLASLt / Triton: ``group_lens`` + expert-major concatenated rows
  * FlyDSL dispatch-style: ``tile_to_expert`` + ``num_tile_blocks``

MoE cases (M x K_out x N_contract, matching bench_mega_moe ``dense_dims``):
  L2 dgrad : pool[M,H]   @ W2[G,H,I]   -> [M,I]    8192 x 2048 x 7168
  L1 dgrad : pool[M,2I]  @ W1[G,2I,H]  -> [M,H]    8192 x 7168 x 4096
  dW1 wgrad: x^T @ grad_l1  (variable-K TN, trans_c) 8192 x 7168 x 4096
  dW2 wgrad: dy^T @ act     (variable-K TN)            8192 x 2048 x 7168

Run (must use workspace tree — pip install has no flydsl):
  PYTHONPATH=/perf_apps/xiaoming/MegaMoE python benchmark/ops/bench_bwd_gg_vs_baselines.py
"""

from __future__ import annotations

import argparse
import importlib.util
import os
from dataclasses import dataclass

import torch

from primus_turbo.pytorch.core.backend import BackendType
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_impl import (
    grouped_gemm_impl,
    grouped_gemm_variable_k_impl,
)
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_utils import group_offs_from_lens
from primus_turbo.pytorch.ops.grouped_gemm import grouped_gemm as turbo_grouped_gemm
from primus_turbo.flydsl.grouped_gemm.grouped_gemm_bf16_dgrad_kernel import (
    grouped_gemm_bf16_dgrad_flydsl_kernel,
)
from primus_turbo.flydsl.grouped_gemm.grouped_gemm_bf16_kernel import (
    grouped_gemm_bf16_variable_k_flydsl_kernel,
)

DSV3_H = 7168
DSV3_I = 2048
DSV3_T = 8192
DSV3_TOPK = 8
DSV3_EP = 8
DSV3_EPR = 256 // DSV3_EP
DSV3_M = DSV3_T * DSV3_TOPK // DSV3_EP

BACKENDS = (
    ("CK", BackendType.CK.value),
    ("hipBLASLt", BackendType.HIPBLASLT.value),
    ("Triton", BackendType.TRITON.value),
)


@dataclass(frozen=True)
class Case:
    name: str
    M: int
    k_out: int
    n_contract: int
    op: str  # "dgrad" | "wgrad"
    trans_c: bool = True


def dsv3_cases() -> tuple[Case, ...]:
    M = DSV3_M
    return (
        Case("L2 dgrad (dy@W2)", M, DSV3_I, DSV3_H, "dgrad"),
        Case("L1 dgrad (dgrad@W1)", M, DSV3_H, 2 * DSV3_I, "dgrad"),
        Case("dW1 wgrad (x^T@grad_l1)", M, DSV3_H, 2 * DSV3_I, "wgrad", trans_c=True),
        Case("dW2 wgrad (dy^T@act)", M, DSV3_I, DSV3_H, "wgrad", trans_c=False),
    )


def _load_bench_mega_moe():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    training = os.path.join(root, "benchmark", "ops", "training")
    path = os.path.join(training, "bench_mega_moe.py")
    spec = importlib.util.spec_from_file_location("bench_mega_moe", path)
    mod = importlib.util.module_from_spec(spec)
    # bench_mega_moe expects training/ on sys.path for `config`
    import sys

    sys.path.insert(0, training)
    spec.loader.exec_module(mod)
    return mod


def _bench(fn, *, warmup: int, iters: int) -> float:
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return float(s.elapsed_time(e) / iters)


def _snr(ref: torch.Tensor, out: torch.Tensor) -> float:
    ref, out = ref.float(), out.float()
    return float(10.0 * torch.log10(ref.pow(2).sum() / ((ref - out).pow(2).sum() + 1e-12)))


def _setup_dgrad(M: int, k_out: int, n_contract: int, G: int, bm: int):
    """NN dgrad: pool[M, n_contract] @ W[G, n_contract, k_out]."""
    m_pg = M // G
    group_lens = torch.full((G,), m_pg, dtype=torch.int64, device="cuda")
    group_offs = group_offs_from_lens(group_lens)
    n_mblk = M // bm
    tile_to_expert = torch.arange(n_mblk, device="cuda", dtype=torch.int32) // (m_pg // bm)
    num_tile_blocks = torch.tensor([n_mblk], device="cuda", dtype=torch.int32)

    pool = torch.randn(M, n_contract, device="cuda", dtype=torch.bfloat16)
    # NN layout for FlyDSL: [G, K_contract, N_out]; turbo API uses same layout for trans_b=False
    weight = torch.randn(G, n_contract, k_out, device="cuda", dtype=torch.bfloat16)
    out_turbo = torch.empty(M, k_out, device="cuda", dtype=torch.bfloat16)
    out_fly = torch.empty(M, k_out, device="cuda", dtype=torch.bfloat16)
    return dict(
        M=M,
        k_out=k_out,
        n_contract=n_contract,
        G=G,
        flops=2.0 * M * k_out * n_contract,
        group_lens=group_lens,
        group_offs=group_offs,
        tile_to_expert=tile_to_expert,
        num_tile_blocks=num_tile_blocks,
        pool=pool,
        weight=weight,
        out_turbo=out_turbo,
        out_fly=out_fly,
    )


def _setup_wgrad(M: int, k_out: int, n_contract: int, G: int):
    """TN variable-K wgrad: lhs[M, k_out]^T @ rhs[M, n_contract] -> dW per group."""
    m_pg = M // G
    group_lens = torch.full((G,), m_pg, dtype=torch.int64, device="cuda")
    group_offs = group_offs_from_lens(group_lens)
    group_offs_i64 = group_offs.to(torch.int64)

    lhs = torch.randn(M, k_out, device="cuda", dtype=torch.bfloat16)
    rhs = torch.randn(M, n_contract, device="cuda", dtype=torch.bfloat16)
    return dict(
        M=M,
        k_out=k_out,
        n_contract=n_contract,
        G=G,
        flops=2.0 * M * k_out * n_contract,
        group_lens=group_lens,
        group_offs=group_offs,
        group_offs_i64=group_offs_i64,
        lhs=lhs,
        rhs=rhs,
    )


def _run_dgrad_turbo(ctx, backend_id: int | None, *, warmup: int, iters: int):
    pool, weight = ctx["pool"], ctx["weight"]
    gl, go = ctx["group_lens"], ctx["group_offs"]

    def fn():
        if backend_id is None:
            return turbo_grouped_gemm(pool, weight, gl, trans_b=False)
        return grouped_gemm_impl(
            pool,
            weight,
            gl,
            go,
            trans_a=False,
            trans_b=False,
            num_cu=None,
            default_backend=backend_id,
            schedule="static",
        )

    fn()
    ms = _bench(fn, warmup=warmup, iters=iters)
    return ms, ctx["flops"] / (ms * 1e-3) / 1e12


def _run_dgrad_flydsl(ctx, bench_mod, *, bm: int, bn: int, group_m: int, warmup: int, iters: int):
    pool, weight = ctx["pool"], ctx["weight"]
    out = ctx["out_fly"]

    def fn_kernel():
        return grouped_gemm_bf16_dgrad_flydsl_kernel(
            pool,
            weight,
            ctx["tile_to_expert"],
            ctx["num_tile_blocks"],
            BLOCK_M=bm,
            BLOCK_N=bn,
            GROUP_M=group_m,
        )

    def fn_bench_only():
        bench_mod.grouped_gemm_bf16_only(
            pool,
            weight,
            out,
            ctx["tile_to_expert"],
            ctx["num_tile_blocks"],
            layout="nn",
            BLOCK_M=bm,
            BLOCK_N=bn,
            GROUP_M=group_m,
        )
        return out

    fn_kernel()
    ms_k = _bench(fn_kernel, warmup=warmup, iters=iters)
    ms_b = _bench(fn_bench_only, warmup=warmup, iters=iters)
    tf_k = ctx["flops"] / (ms_k * 1e-3) / 1e12
    tf_b = ctx["flops"] / (ms_b * 1e-3) / 1e12
    return (ms_k, tf_k), (ms_b, tf_b)


def _run_wgrad_turbo(ctx, backend_id: int | None, *, trans_c: bool, warmup: int, iters: int):
    lhs, rhs = ctx["lhs"], ctx["rhs"]
    gl, go = ctx["group_lens"], ctx["group_offs"]

    def fn():
        return grouped_gemm_variable_k_impl(
            lhs,
            rhs,
            gl,
            go,
            trans_a=True,
            trans_b=False,
            trans_c=trans_c,
            num_cu=None,
            default_backend=backend_id,
            schedule="static",
        )

    fn()
    ms = _bench(fn, warmup=warmup, iters=iters)
    return ms, ctx["flops"] / (ms * 1e-3) / 1e12


def _run_wgrad_flydsl(ctx, bench_mod, *, bm: int, bn: int, trans_c: bool, warmup: int, iters: int):
    lhs, rhs = ctx["lhs"], ctx["rhs"]
    go = ctx["group_offs_i64"]
    gl = ctx["group_lens"]
    out_shape = (ctx["G"], ctx["n_contract"], ctx["k_out"]) if trans_c else (ctx["G"], ctx["k_out"], ctx["n_contract"])
    out_bench = torch.empty(out_shape, device="cuda", dtype=torch.bfloat16)

    def fn_kernel():
        return grouped_gemm_bf16_variable_k_flydsl_kernel(
            lhs, rhs, go, masked_k=gl, BLOCK_M=bm, BLOCK_N=bn, trans_c=trans_c
        )

    def fn_bench_only():
        bench_mod.grouped_gemm_variable_k_only(lhs, rhs, go, out_bench, BLOCK_M=bm, BLOCK_N=bn, trans_c=trans_c)
        return out_bench

    fn_kernel()
    ms_k = _bench(fn_kernel, warmup=warmup, iters=iters)
    ms_b = _bench(fn_bench_only, warmup=warmup, iters=iters)
    tf_k = ctx["flops"] / (ms_k * 1e-3) / 1e12
    tf_b = ctx["flops"] / (ms_b * 1e-3) / 1e12
    return (ms_k, tf_k), (ms_b, tf_b)


def _check_dgrad(ctx):
    ref = turbo_grouped_gemm(ctx["pool"], ctx["weight"], ctx["group_lens"], trans_b=False)
    ck = grouped_gemm_impl(
        ctx["pool"],
        ctx["weight"],
        ctx["group_lens"],
        ctx["group_offs"],
        trans_a=False,
        trans_b=False,
        num_cu=None,
        default_backend=BackendType.CK.value,
    )
    fly = grouped_gemm_bf16_dgrad_flydsl_kernel(
        ctx["pool"],
        ctx["weight"],
        ctx["tile_to_expert"],
        ctx["num_tile_blocks"],
        GROUP_M=8,
    )
    print(f"  correctness SNR vs turbo_grouped_gemm: CK={_snr(ref, ck):.0f} dB  FlyDSL={_snr(ref, fly):.0f} dB")


def _check_wgrad(ctx, trans_c: bool):
    ref = grouped_gemm_variable_k_impl(
        ctx["lhs"],
        ctx["rhs"],
        ctx["group_lens"],
        ctx["group_offs"],
        trans_a=True,
        trans_b=False,
        trans_c=trans_c,
        num_cu=None,
        default_backend=BackendType.TRITON.value,
    )
    fly = grouped_gemm_bf16_variable_k_flydsl_kernel(
        ctx["lhs"], ctx["rhs"], ctx["group_offs_i64"], masked_k=ctx["group_lens"], trans_c=trans_c
    )
    print(f"  correctness SNR vs Triton var_k: FlyDSL={_snr(ref, fly):.0f} dB")


def _profile_dgrad(case: Case, ctx, bench_mod, args):
    print("=" * 78)
    print(f"{case.name}: M×K×N = {case.M}×{case.k_out}×{case.n_contract}  (G={ctx['G']}, grouped via group_lens / tile_to_expert)")
    _check_dgrad(ctx)
    print(f"{'backend':<28} {'ms':>8} {'TFLOPS':>10}")
    rows = []
    ms, tf = _run_dgrad_turbo(ctx, None, warmup=args.warmup, iters=args.iters)
    rows.append(("turbo_grouped_gemm (Triton)", ms, tf))
    for name, bid in BACKENDS:
        try:
            ms, tf = _run_dgrad_turbo(ctx, bid, warmup=args.warmup, iters=args.iters)
            rows.append((name, ms, tf))
        except Exception as e:
            print(f"  {name:<28} FAILED: {e}")
    (ms_k, tf_k), (ms_b, tf_b) = _run_dgrad_flydsl(
        ctx, bench_mod, bm=args.bm, bn=args.bn, group_m=args.group_m, warmup=args.warmup, iters=args.iters
    )
    rows.append(("FlyDSL dgrad_kernel", ms_k, tf_k))
    rows.append(("bench gg_bf16_only(nn)", ms_b, tf_b))
    best = max(tf for _, _, tf in rows)
    for name, ms, tf in rows:
        print(f"  {name:<28} {ms:8.3f} {tf:10.1f}  ({100 * tf / best:5.1f}% of best)")
    print()


def _profile_wgrad(case: Case, ctx, bench_mod, args):
    print("=" * 78)
    tc = case.trans_c
    print(f"{case.name}: M×K×N = {case.M}×{case.k_out}×{case.n_contract}  (G={ctx['G']}, variable-K TN, trans_c={tc})")
    _check_wgrad(ctx, tc)
    print(f"{'backend':<28} {'ms':>8} {'TFLOPS':>10}")
    rows = []
    for name, bid in BACKENDS:
        try:
            ms, tf = _run_wgrad_turbo(ctx, bid, trans_c=tc, warmup=args.warmup, iters=args.iters)
            rows.append((name, ms, tf))
        except Exception as e:
            print(f"  {name:<28} FAILED: {e}")
    (ms_k, tf_k), (ms_b, tf_b) = _run_wgrad_flydsl(
        ctx, bench_mod, bm=args.bm, bn=args.bn, trans_c=tc, warmup=args.warmup, iters=args.iters
    )
    rows.append(("FlyDSL var_k", ms_k, tf_k))
    if abs(tf_k - tf_b) / max(tf_b, 1) > 0.05:
        rows.append(("bench var_k_only (ref)", ms_b, tf_b))
    best = max(tf for _, _, tf in rows)
    for name, ms, tf in rows:
        print(f"  {name:<28} {ms:8.3f} {tf:10.1f}  ({100 * tf / best:5.1f}% of best)")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bm", type=int, default=256)
    ap.add_argument("--bn", type=int, default=256)
    ap.add_argument("--group-m", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    bench_mod = _load_bench_mega_moe()
    print(f"device={torch.cuda.get_device_name()}  BM={args.bm} BN={args.bn} GROUP_M={args.group_m}")
    print(
        f"DeepSeek-V3 grouped GEMM: T={DSV3_T} EP={DSV3_EP} topk={DSV3_TOPK} "
        f"H={DSV3_H} I={DSV3_I} -> per-rank M={DSV3_M} G={DSV3_EPR}\n"
    )
    print(
        "API note: CK/Triton/hipBLASLt use turbo grouped_gemm (group_lens, expert-major rows).\n"
        "FlyDSL uses tile_to_expert (dgrad) or variable_k wgrad kernel (same launch as bench_mega_moe).\n"
        "Requires PYTHONPATH=/perf_apps/xiaoming/MegaMoE (pip primus_turbo 0.2.0 has no flydsl).\n"
    )

    for case in dsv3_cases():
        if case.op == "dgrad":
            ctx = _setup_dgrad(case.M, case.k_out, case.n_contract, DSV3_EPR, args.bm)
            _profile_dgrad(case, ctx, bench_mod, args)
        else:
            ctx = _setup_wgrad(case.M, case.k_out, case.n_contract, DSV3_EPR)
            _profile_wgrad(case, ctx, bench_mod, args)


if __name__ == "__main__":
    main()
