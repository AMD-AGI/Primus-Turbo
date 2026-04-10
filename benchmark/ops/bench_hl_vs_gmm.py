###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Head-to-head: hipBLASLt grouped GEMM vs grouped_gemm (GMM) library.

Tests all model configs (DeepSeek-V3, LFM2-8B-A1B, gpt_oss_20B) with both
balanced and unbalanced routing, for fwd / dgrad / wgrad separately.
"""

import sys
import time

import grouped_gemm.backend as gmm_backend
import grouped_gemm.ops as gmm_ops
import torch
from config import gen_grouped_gemm_group_lens, gen_grouped_gemm_test_cases

sys.path.insert(0, "/shared_nfs/kyle/Primus-Turbo")
from primus_turbo.pytorch.core.backend import BackendType
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_impl import (
    grouped_gemm_impl,
    grouped_gemm_variable_k_impl,
)

DEVICE = "cuda"
DTYPE  = torch.bfloat16
HL     = BackendType.HIPBLASLT.value
WARMUP = 20
ITERS  = 60


def timeit_us(fn):
    """Measure per-call latency with sync after every iteration.

    Sync-per-call is necessary because hipBLASLt uses multiple streams and
    events per invocation — queuing hundreds of calls without synchronization
    saturates the ROCm command buffer and inflates latency artificially.
    """
    for _ in range(WARMUP):
        fn()
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        fn()
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / ITERS * 1e6


def tflops(M_total, N, K, us):
    return 2 * M_total * N * K / us / 1e6


def bench_case(case):
    B, M, N, K = case["B"], case["M"], case["N"], case["K"]
    balance   = case["balance"]
    num_topk  = case.get("num_topk")

    group_lens     = gen_grouped_gemm_group_lens(B, M, balance=balance, num_topk=num_topk).to(DEVICE)
    group_lens_cpu = group_lens.cpu()
    group_offs     = torch.cat([torch.zeros(1, dtype=torch.int64, device=DEVICE),
                                group_lens.cumsum(0)])
    M_total = int(group_lens.sum().item())

    results = {}

    # ── fwd: a=[M,K]  b=[B,K,N]  → out=[M,N] ────────────────────────────────
    a_fwd = torch.randn(M_total, K, dtype=DTYPE, device=DEVICE)
    b_fwd = torch.randn(B, K, N, dtype=DTYPE, device=DEVICE)

    hl_us  = timeit_us(lambda: grouped_gemm_impl(
        a_fwd, b_fwd, group_lens, group_offs, False, False, None, HL))
    gmm_us = timeit_us(lambda: gmm_ops.gmm(a_fwd, b_fwd, group_lens_cpu, trans_b=False))

    results["fwd"] = (tflops(M_total, N, K, hl_us), tflops(M_total, N, K, gmm_us))

    # ── dgrad: a=[M,K]  b=[B,N,K] transB=True  → out=[M,N] ──────────────────
    a_dgrad = torch.randn(M_total, K, dtype=DTYPE, device=DEVICE)
    b_dgrad = torch.randn(B, N, K, dtype=DTYPE, device=DEVICE)

    hl_us  = timeit_us(lambda: grouped_gemm_impl(
        a_dgrad, b_dgrad, group_lens, group_offs, False, True, None, HL))
    gmm_us = timeit_us(lambda: gmm_ops.gmm(a_dgrad, b_dgrad, group_lens_cpu, trans_b=True))

    results["dgrad"] = (tflops(M_total, N, K, hl_us), tflops(M_total, N, K, gmm_us))

    # ── wgrad: lhs=[M,K]  rhs=[M,N]  transA=True  → out=[B,K,N] ─────────────
    lhs = torch.randn(M_total, K, dtype=DTYPE, device=DEVICE)
    rhs = torch.randn(M_total, N, dtype=DTYPE, device=DEVICE)

    hl_us  = timeit_us(lambda: grouped_gemm_variable_k_impl(
        lhs, rhs, group_lens, group_offs, True, False, False, None, HL))
    gmm_us = timeit_us(lambda: gmm_backend.gmm(lhs, rhs, group_lens_cpu, True, False))

    results["wgrad"] = (tflops(M_total, K, N, hl_us), tflops(M_total, K, N, gmm_us))

    return results


def model_tag(case_name):
    if "DeepSeek" in case_name:
        return "DSV3"
    if "LFM2" in case_name:
        return "LFM2"
    if "gpt_oss" in case_name:
        return "20B"
    return "?"


def main():
    test_cases = gen_grouped_gemm_test_cases()

    print(f"\n{'hipBLASLt vs GMM — all models, balanced & unbalanced':^80}")
    print(f"{'BF16, MI355X':^80}")
    print("=" * 80)
    hdr = f"{'Model':>5} {'B':>3} {'M':>5} {'N':>5} {'K':>5} {'bal':>3}  "
    hdr += f"{'op':>5}  {'HL(T)':>7} {'GMM(T)':>7} {'ratio':>7}  result"
    print(hdr)
    print("-" * 80)

    all_rows = []
    for case in test_cases:
        tag     = model_tag(case["Case"])
        bal_str = "Y" if case["balance"] else "N"
        B, M, N, K = case["B"], case["M"], case["N"], case["K"]

        res = bench_case(case)
        for op in ("fwd", "dgrad", "wgrad"):
            hl_T, gmm_T = res[op]
            ratio = hl_T / gmm_T
            beat  = ratio >= 1.0
            row = dict(model=tag, B=B, M=M, N=N, K=K, bal=bal_str, op=op,
                       hl_T=hl_T, gmm_T=gmm_T, ratio=ratio, beat=beat)
            all_rows.append(row)
            status = "BEAT" if beat else "BEHIND"
            print(f"{tag:>5} {B:>3} {M:>5} {N:>5} {K:>5} {bal_str:>3}  "
                  f"{op:>5}  {hl_T:>7.0f} {gmm_T:>7.0f} {ratio:>7.3f}x  {status}")

    # Summary
    print("=" * 80)
    n_beat  = sum(r["beat"] for r in all_rows)
    n_total = len(all_rows)
    print(f"\nOVERALL: {n_beat}/{n_total} BEAT GMM\n")

    # Per-model per-balance summary
    for model in ("DSV3", "20B", "LFM2"):
        for bal in ("Y", "N"):
            sub = [r for r in all_rows if r["model"] == model and r["bal"] == bal]
            if not sub:
                continue
            n_b = sum(r["beat"] for r in sub)
            avg_hl  = sum(r["hl_T"]  for r in sub) / len(sub)
            avg_gmm = sum(r["gmm_T"] for r in sub) / len(sub)
            bal_name = "balanced" if bal == "Y" else "unbalanced"
            print(f"  {model:>5} {bal_name:>10}: {n_b:>2}/{len(sub):>2} BEAT  "
                  f"avg HL={avg_hl:.0f}T  avg GMM={avg_gmm:.0f}T  "
                  f"ratio={avg_hl/avg_gmm:.3f}x")

    # Per-op summary
    print()
    for op in ("fwd", "dgrad", "wgrad"):
        sub = [r for r in all_rows if r["op"] == op]
        n_b = sum(r["beat"] for r in sub)
        avg_ratio = sum(r["ratio"] for r in sub) / len(sub)
        print(f"  {op:>5}: {n_b}/{len(sub)} BEAT  avg ratio={avg_ratio:.3f}x")


if __name__ == "__main__":
    main()
