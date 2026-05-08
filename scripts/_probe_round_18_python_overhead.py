#!/usr/bin/env python3
"""R18 — line-by-line Python overhead profile of GroupedGEMMFP8HipKittenBackend.execute.

R17 closed the dispatcher track; remaining ~3 µs gap on Down-B4-M2048
fwd lives in the public-op path (probe 1472 T direct call vs metric 1425
T full grouped_gemm_fp8_impl). This probe times stage-incremental
versions of the execute body to localise where the 3 µs lives.

The smallest gpt_oss kernel is Down-B4-M2048 fwd (~92 µs); 3 µs there
is ~3% — measurable in the metric.

Method: build N "stage-incremental" callables, each adding one more
stage of the execute body. Time each in a tight loop. Diffs between
adjacent stages give per-stage cost.
"""
import os
import sys
import statistics
import time

os.environ.setdefault("PRIMUS_TURBO_HIPKITTEN_PATH", "/workspace/code/HipKittens")
sys.path.insert(0, "/workspace/code/Primus-Turbo")

import torch  # noqa: E402

import primus_turbo.pytorch as turbo  # noqa: F401  E402
from primus_turbo.pytorch.core.low_precision import ScalingGranularity  # noqa: E402
from primus_turbo.pytorch.kernels import hipkitten as hipkitten  # noqa: E402
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_fp8_impl import (  # noqa: E402
    grouped_gemm_compute_offs,
    grouped_gemm_fp8_impl,
    GroupedGEMMFP8HipKittenBackend,
)
from primus_turbo.pytorch.ops.quantization import quantize_fp8  # noqa: E402

_FP8_DTYPE = torch.float8_e4m3fnuz
_GRAN = ScalingGranularity.TENSORWISE


def _bench(fn, warmup=50, iters=2000):
    """Time fn() in a tight loop, return p20 wall ms per call."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    se = torch.cuda.Event(enable_timing=True)
    ee = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        torch.cuda.synchronize()
        se.record()
        fn()
        ee.record()
        torch.cuda.synchronize()
        times.append(se.elapsed_time(ee))
    times.sort()
    return times[len(times) // 5]


def _bench_python_only(fn, iters=20000):
    """Pure-Python timing (no GPU events) — for stage components that
    don't launch a kernel.

    Returns p20 of perf_counter-ns deltas.
    """
    fn()  # warmup
    torch.cuda.synchronize()
    deltas = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        fn()
        deltas.append(time.perf_counter_ns() - t0)
    deltas.sort()
    return deltas[len(deltas) // 5] / 1000.0  # ns → µs


def setup(B, M, N, K):
    g_lens = torch.full((B,), M, dtype=torch.int64, device="cuda")
    g_offs = grouped_gemm_compute_offs(g_lens)
    a = torch.randn((B * M, K), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((B, N, K), dtype=torch.bfloat16, device="cuda")
    a_fp8, a_s = quantize_fp8(a, _FP8_DTYPE, _GRAN)
    b_fp8, b_s = quantize_fp8(b, _FP8_DTYPE, _GRAN)
    out_pre = torch.empty((B * M, N), dtype=torch.bfloat16, device="cuda")
    return dict(
        a=a_fp8, b=b_fp8, a_scales=a_s, b_scales=b_s,
        group_lens=g_lens, group_offs=g_offs,
        trans_a=False, trans_b=True,
        out_dtype=torch.bfloat16,
        out_pre=out_pre, B=B, M=M, N=N, K=K,
    )


def main():
    print(f"[probe] R18 line-by-line Python overhead profile")
    print(f"[probe] device: {torch.cuda.get_device_name(0)}")
    print()

    # Anchor on Down-B4-M2048 fwd — smallest gpt_oss kernel.
    ctx = setup(B=4, M=2048, N=2880, K=2880)
    print(f"[probe] anchor: Down-B4-M2048 fwd (B={ctx['B']} M={ctx['M']} N={ctx['N']} K={ctx['K']})")
    print()

    hk = hipkitten.load_fp8()
    a = ctx["a"]
    b = ctx["b"]
    a_scales = ctx["a_scales"]
    b_scales = ctx["b_scales"]
    group_offs = ctx["group_offs"]
    out_pre = ctx["out_pre"]
    trans_b = ctx["trans_b"]
    out_dtype = ctx["out_dtype"]

    flops = 2.0 * ctx["B"] * ctx["M"] * ctx["N"] * ctx["K"]
    print(f"  FLOPs/call = {flops/1e9:.0f} GF")
    print()

    # ---- Stage 0: bare HK kernel call with pre-allocated `out`. -----------
    def stage0_bare_kernel():
        hk.grouped_rcr_dscale(
            a, b, out_pre, a_scales, b_scales, group_offs, 16,
            m_per_group=2048, num_xcds=2,
        )
    t0 = _bench(stage0_bare_kernel)
    tflops0 = flops / (t0 * 1e9)
    print(f"  [stage 0] bare hk.grouped_rcr_dscale (out pre-alloc):  {t0*1e3:.2f} µs  {tflops0:.1f} TF")

    # ---- Stage 1: + torch.empty inside the timed region. -------------------
    def stage1_with_empty():
        out = torch.empty((ctx["B"] * ctx["M"], ctx["N"]), dtype=out_dtype, device=a.device)
        hk.grouped_rcr_dscale(
            a, b, out, a_scales, b_scales, group_offs, 16,
            m_per_group=2048, num_xcds=2,
        )
    t1 = _bench(stage1_with_empty)
    tflops1 = flops / (t1 * 1e9)
    print(f"  [stage 1] + torch.empty(out):                          {t1*1e3:.2f} µs  {tflops1:.1f} TF  Δ={(t1-t0)*1e6:+.2f} µs")

    # ---- Stage 2: + select_default_config call. ---------------------------
    def stage2_with_cfg():
        out = torch.empty((ctx["B"] * ctx["M"], ctx["N"]), dtype=out_dtype, device=a.device)
        cfg = hipkitten.select_default_config(
            ctx["M"], ctx["N"], ctx["K"], "rcr", "fp8", m_total=ctx["B"] * ctx["M"],
        )
        gm = cfg.group_m
        xc = cfg.num_xcds if cfg.num_xcds is not None else 0
        hk.grouped_rcr_dscale(
            a, b, out, a_scales, b_scales, group_offs, gm,
            m_per_group=2048, num_xcds=xc,
        )
    t2 = _bench(stage2_with_cfg)
    tflops2 = flops / (t2 * 1e9)
    print(f"  [stage 2] + select_default_config + cfg arg unpack:    {t2*1e3:.2f} µs  {tflops2:.1f} TF  Δ={(t2-t1)*1e6:+.2f} µs")

    # ---- Stage 3: + shape lookups + avg_m + ternaries. ---------------------
    def stage3_with_shapes():
        layout = "rcr" if trans_b else "rrr"
        bs = b.shape[0]
        m_total = a.shape[0]
        n = b.shape[-2] if trans_b else b.shape[-1]
        k = a.shape[1]
        avg_m = max(m_total // bs, 1) if bs > 0 else max(m_total, 1)
        grouped_dscale_fn = hk.grouped_rcr_dscale if trans_b else hk.grouped_rrr_dscale
        use_dscale = grouped_dscale_fn is not None and a_scales.is_cuda  # noqa: F841
        cfg = hipkitten.select_default_config(avg_m, n, k, layout, "fp8", m_total=m_total)
        out = torch.empty((m_total, n), dtype=out_dtype, device=a.device)
        a_in = a if a.is_contiguous() else a.contiguous()
        b_in = b if b.is_contiguous() else b.contiguous()
        xcds_arg = cfg.num_xcds if cfg.num_xcds is not None else 0
        grouped_dscale_fn(
            a_in, b_in, out, a_scales, b_scales, group_offs, cfg.group_m,
            m_per_group=avg_m, num_xcds=xcds_arg,
        )
    t3 = _bench(stage3_with_shapes)
    tflops3 = flops / (t3 * 1e9)
    print(f"  [stage 3] + shapes/avg_m/ternaries/contig (full body): {t3*1e3:.2f} µs  {tflops3:.1f} TF  Δ={(t3-t2)*1e6:+.2f} µs")

    # ---- Stage 4: full GroupedGEMMFP8HipKittenBackend.execute(...). -------
    def stage4_via_execute():
        GroupedGEMMFP8HipKittenBackend.execute(
            a=a, b=b,
            a_scales=a_scales, b_scales=b_scales,
            group_lens=ctx["group_lens"], group_offs=group_offs,
            trans_a=False, trans_b=True,
            out_dtype=out_dtype,
            granularity=_GRAN,
            num_cu=None,
        )
    t4 = _bench(stage4_via_execute)
    tflops4 = flops / (t4 * 1e9)
    print(f"  [stage 4] via Backend.execute(...):                     {t4*1e3:.2f} µs  {tflops4:.1f} TF  Δ={(t4-t3)*1e6:+.2f} µs")

    # ---- Stage 5: full grouped_gemm_fp8_impl public op. -------------------
    from primus_turbo.pytorch.core.backend import BackendType  # noqa: E402

    def stage5_public_op():
        grouped_gemm_fp8_impl(
            a, b, a_scales, b_scales, ctx["group_lens"], group_offs,
            trans_a=False, trans_b=True, out_dtype=out_dtype,
            granularity=_GRAN.value, num_cu=None,
            default_backend=BackendType.HIPKITTEN.value,
        )
    t5 = _bench(stage5_public_op)
    tflops5 = flops / (t5 * 1e9)
    print(f"  [stage 5] via grouped_gemm_fp8_impl public op:          {t5*1e3:.2f} µs  {tflops5:.1f} TF  Δ={(t5-t4)*1e6:+.2f} µs")

    print()
    print(f"  Total Python overhead (stage 5 vs stage 0): {(t5-t0)*1e6:.2f} µs")
    print()

    # ---- Pure-Python micro-timings of the suspect ops. --------------------
    print("[probe] Pure-Python micro-times of suspect ops (perf_counter, no GPU):")

    def py_empty():
        torch.empty((4 * 2048, 2880), dtype=out_dtype, device=a.device)
    print(f"  torch.empty((m_total, n), bf16, cuda):  {_bench_python_only(py_empty):.3f} µs")

    def py_cfg():
        hipkitten.select_default_config(2048, 2880, 2880, "rcr", "fp8", m_total=8192)
    print(f"  select_default_config (lru hit):        {_bench_python_only(py_cfg):.3f} µs")

    def py_is_contig():
        a.is_contiguous()
        b.is_contiguous()
    print(f"  a.is_contiguous() + b.is_contiguous():  {_bench_python_only(py_is_contig):.3f} µs")

    def py_is_cuda():
        a_scales.is_cuda
    print(f"  a_scales.is_cuda (property):            {_bench_python_only(py_is_cuda):.3f} µs")

    def py_shapes():
        bs = b.shape[0]  # noqa: F841
        mt = a.shape[0]  # noqa: F841
        n = b.shape[-2] if trans_b else b.shape[-1]  # noqa: F841
        k = a.shape[1]  # noqa: F841
    print(f"  4 shape lookups + 1 ternary:            {_bench_python_only(py_shapes):.3f} µs")


if __name__ == "__main__":
    main()
