#!/usr/bin/env python3
"""Round 14 (gpt_oss FP8 kernel-only ceiling) — RCR forward chunk_size sweep.

R13 shipped the chunk_size lever for var-K wgrad with +1.49% / +1.35% on
Down-B4 wgrad. This round extends the same lever to **forward RCR**:
HK grouped_rcr_kernel calls ``chiplet_transform_chunked(blockIdx.x,
slots_eff, xcds_eff, 64)`` at line ~2734 (file-scope) and ~3544
(kernel_b128:: BLOCK_SIZE=128 variant).

Critical observation from the swizzle math:

  block = num_xcds * chunk_size
  limit = (slots / block) * block
  if (workgroup_id > limit) return workgroup_id   # NO swizzle

For the prevailing default RCR cell (xcds=8 + slots=NUM_CUS=256):
  cs=64 → block=512, limit=0    → ALL workgroup_id > 0 fall through
                                  to round-robin (swizzle is NO-OP)
  cs=32 → block=256, limit=256  → ALL 256 workgroups participate in
                                  one clean partition: 32 PIDs per XCD
                                  (8 chiplets × 32 = 256)
  cs=16 → block=128, limit=256  → 16 PIDs per XCD per chunk × 2 chunks
                                  = 256 (also clean)
  cs=24 → block=192, limit=192  → 192 chunked + 64 round-robin (mixed)
  cs=48 → block=384, limit=0    → swizzle NO-OP (same as cs=64)

So at the default cell, only chunk_size in {16, 32, 64} produce
non-mixed schedules (and 64 = NO-OP). chunk_size=32 is the most
promising candidate: enables the swizzle in clean form.

This probe times the IMPL path (the same path the metric uses) with
each chunk_size value monkey-patched in. Bit-equivalence verified by
comparing outputs across chunk_size values for the same input.

Methodology: 1500-iter × 5-trial p20 × 3 seeds, mirror R13.
"""
import os
import sys
import statistics
import time

os.environ.setdefault("PRIMUS_TURBO_HIPKITTEN_PATH", "/workspace/code/HipKittens")
sys.path.insert(0, "/workspace/code/Primus-Turbo")
sys.path.insert(0, "/workspace/code/Primus-Turbo/scripts")

import torch
import primus_turbo.pytorch as turbo  # noqa: F401
from primus_turbo.pytorch.core.backend import BackendType, PrecisionType
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_fp8_impl import (
    grouped_gemm_compute_offs,
    grouped_gemm_fp8_impl,
    GroupedGEMMFP8HipKittenBackend,
)
from primus_turbo.pytorch.ops.quantization import quantize_fp8
from primus_turbo.pytorch.core.low_precision import ScalingGranularity
from primus_turbo.pytorch.kernels import hipkitten as hipkit_module
import _metric_hk_ratio as hk_ratio

_FP8_DTYPE = torch.float8_e4m3fn
_GRAN = ScalingGranularity.TENSORWISE


def _bench_p20(fn, warmup=20, iters=1500):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    se = torch.cuda.Event(enable_timing=True)
    ee = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        se.record()
        fn()
        ee.record()
        torch.cuda.synchronize()
        times.append(se.elapsed_time(ee))
    times.sort()
    return times[len(times) // 5]


# Capture the original execute method so we can wrap it
_ORIG_EXECUTE = GroupedGEMMFP8HipKittenBackend.execute


def _patch_chunk_size(value):
    """Wrap execute() to inject chunk_size=value into grouped_rcr_dscale call."""
    @staticmethod
    def patched_execute(*args, **kwargs):
        # Easier: copy the execute body here is too heavy. Instead patch
        # hk.grouped_rcr_dscale to default chunk_size=value via wrapper.
        return _ORIG_EXECUTE(*args, **kwargs)
    return patched_execute


def _patch_hk_dscale(value):
    """Replace hk.grouped_rcr_dscale with a wrapper that injects chunk_size."""
    hk = hipkit_module.load_fp8()
    orig = hk.grouped_rcr_dscale
    if orig is None:
        return None

    def wrapped(*args, **kwargs):
        kwargs.setdefault("chunk_size", value)
        return orig(*args, **kwargs)

    # Monkey-patch the loaded singleton directly. Note the loader caches it
    # as a frozen dataclass attr — we mutate via object.__setattr__.
    object.__setattr__(hk, "grouped_rcr_dscale", wrapped)
    return orig


def _restore_hk_dscale(orig):
    if orig is None:
        return
    hk = hipkit_module.load_fp8()
    object.__setattr__(hk, "grouped_rcr_dscale", orig)


def time_fwd_via_impl(B, M, N, K):
    """Time the metric's fwd path with current hk.grouped_rcr_dscale wrapper."""
    g_lens = torch.full((B,), M, dtype=torch.int64, device="cuda")
    g_offs = grouped_gemm_compute_offs(g_lens)
    a = torch.randn((B * M, K), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((B, N, K), dtype=torch.bfloat16, device="cuda")
    a_fp8, a_s = quantize_fp8(a, _FP8_DTYPE, _GRAN)
    b_fp8, b_s = quantize_fp8(b, _FP8_DTYPE, _GRAN)

    def _call():
        return grouped_gemm_fp8_impl(
            a_fp8, b_fp8, a_s, b_s, g_lens, g_offs,
            trans_a=False, trans_b=True, out_dtype=torch.bfloat16,
            granularity=_GRAN.value, num_cu=None,
            default_backend=BackendType.HIPKITTEN.value,
        )

    flops = 2.0 * (B * M) * N * K
    with hk_ratio.force_grouped_gemm_backend(BackendType.HIPKITTEN, PrecisionType.FP8):
        ms = _bench_p20(_call, warmup=20, iters=1500)
    return flops / (ms * 1e9), ms


def time_dgrad_via_impl(B, M, N, K):
    """Same as fwd but trans_b=False → triggers H4 reroute → still hits RCR kernel."""
    g_lens = torch.full((B,), M, dtype=torch.int64, device="cuda")
    g_offs = grouped_gemm_compute_offs(g_lens)
    grad_out = torch.randn((B * M, N), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((B, N, K), dtype=torch.bfloat16, device="cuda")
    g_out_fp8, g_out_s = quantize_fp8(grad_out, _FP8_DTYPE, _GRAN)
    b_fp8, b_s = quantize_fp8(b, _FP8_DTYPE, _GRAN)

    def _call():
        return grouped_gemm_fp8_impl(
            g_out_fp8, b_fp8, g_out_s, b_s, g_lens, g_offs,
            trans_a=False, trans_b=False, out_dtype=torch.bfloat16,
            granularity=_GRAN.value, num_cu=None,
            default_backend=BackendType.HIPKITTEN.value,
        )

    flops = 2.0 * (B * M) * N * K
    with hk_ratio.force_grouped_gemm_backend(BackendType.HIPKITTEN, PrecisionType.FP8):
        ms = _bench_p20(_call, warmup=20, iters=1500)
    return flops / (ms * 1e9), ms


def verify_bit_eq_fwd(B, M, N, K, chunk_sizes):
    """Verify bit-equivalence of fwd output across chunk_size values."""
    g_lens = torch.full((B,), M, dtype=torch.int64, device="cuda")
    g_offs = grouped_gemm_compute_offs(g_lens)
    torch.manual_seed(7)
    a = torch.randn((B * M, K), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((B, N, K), dtype=torch.bfloat16, device="cuda")
    a_fp8, a_s = quantize_fp8(a, _FP8_DTYPE, _GRAN)
    b_fp8, b_s = quantize_fp8(b, _FP8_DTYPE, _GRAN)

    outs = {}
    for cs in chunk_sizes:
        orig = _patch_hk_dscale(cs)
        with hk_ratio.force_grouped_gemm_backend(BackendType.HIPKITTEN, PrecisionType.FP8):
            outs[cs] = grouped_gemm_fp8_impl(
                a_fp8, b_fp8, a_s, b_s, g_lens, g_offs,
                trans_a=False, trans_b=True, out_dtype=torch.bfloat16,
                granularity=_GRAN.value, num_cu=None,
                default_backend=BackendType.HIPKITTEN.value,
            )
        _restore_hk_dscale(orig)

    base = outs[chunk_sizes[0]]
    print(f"  bit-eq vs chunk_size={chunk_sizes[0]} (base):")
    all_eq = True
    for cs, out in outs.items():
        eq = torch.equal(base, out)
        if not eq:
            max_abs = (base - out).abs().max().item()
        else:
            max_abs = 0.0
        print(f"    chunk_size={cs:>3}: bit_eq={eq}, max_abs={max_abs:.6f}")
        if not eq:
            all_eq = False
    return all_eq


def run_fwd(label, B, M, N, K, chunk_sizes, seeds=(42, 137, 2024)):
    print(f"\n=== {label} fwd (B={B}, M={M}, N={N}, K={K}) ===")
    bit_ok = verify_bit_eq_fwd(B, M, N, K, chunk_sizes)
    print(f"  bit-eq across all chunk_sizes: {bit_ok}")
    assert bit_ok

    print(f"\n  {'cs':>4}  {'med ms':>8}  {'min ms':>8}  {'max ms':>8}  {'TFLOPS':>8}  {'Δ%':>6}")
    results = {}
    for cs in chunk_sizes:
        orig = _patch_hk_dscale(cs)
        seed_meds = []
        for seed in seeds:
            torch.manual_seed(seed)
            t, ms = time_fwd_via_impl(B, M, N, K)
            seed_meds.append(ms)
        med = statistics.median(seed_meds)
        results[cs] = (med, min(seed_meds), max(seed_meds))
        _restore_hk_dscale(orig)

    flops = 2.0 * (B * M) * N * K
    base_ms = results[64][0]
    for cs in chunk_sizes:
        med, lo, hi = results[cs]
        tflops = flops / (med * 1e9)
        delta_pp = (base_ms - med) / base_ms * 100
        marker = " *base" if cs == 64 else ""
        print(f"  {cs:>4}  {med:>8.4f}  {lo:>8.4f}  {hi:>8.4f}  {tflops:>8.1f}  {delta_pp:+6.2f}{marker}")
    best_cs = min(chunk_sizes, key=lambda cs: results[cs][0])
    best_med = results[best_cs][0]
    lift = (base_ms - best_med) / base_ms * 100
    print(f"\n  BEST: cs={best_cs}  ({lift:+.2f}% over default 64)")
    return results, best_cs, lift


def run_dgrad(label, B, M, N, K, chunk_sizes, seeds=(42, 137, 2024)):
    print(f"\n=== {label} dgrad-via-H4 (B={B}, M={M}, N={N}, K={K}) ===")
    print(f"  {'cs':>4}  {'med ms':>8}  {'TFLOPS':>8}  {'Δ%':>6}")
    flops = 2.0 * (B * M) * N * K
    results = {}
    for cs in chunk_sizes:
        orig = _patch_hk_dscale(cs)
        seed_meds = []
        for seed in seeds:
            torch.manual_seed(seed)
            t, ms = time_dgrad_via_impl(B, M, N, K)
            seed_meds.append(ms)
        med = statistics.median(seed_meds)
        results[cs] = (med, min(seed_meds), max(seed_meds))
        _restore_hk_dscale(orig)

    base_ms = results[64][0]
    for cs in chunk_sizes:
        med, lo, hi = results[cs]
        tflops = flops / (med * 1e9)
        delta_pp = (base_ms - med) / base_ms * 100
        marker = " *base" if cs == 64 else ""
        print(f"  {cs:>4}  {med:>8.4f}  {tflops:>8.1f}  {delta_pp:+6.2f}{marker}")
    best_cs = min(chunk_sizes, key=lambda cs: results[cs][0])
    best_med = results[best_cs][0]
    lift = (base_ms - best_med) / base_ms * 100
    print(f"  BEST: cs={best_cs}  ({lift:+.2f}% over default 64)")
    return results, best_cs, lift


if __name__ == "__main__":
    print(f"[probe] R14 RCR chunk_size sweep on metric fwd shapes")
    print(f"[probe] HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES', '<unset>')}")
    print(f"[probe] device: {torch.cuda.get_device_name(0)}")

    chunk_sizes = [16, 24, 32, 48, 64]

    t0 = time.monotonic()

    # Anchor: lowest-ratio shape (Down-B32-M4096 fwd at 1.06)
    run_fwd("Down-B32-M4096", 32, 4096, 2880, 2880, chunk_sizes)
    print(f"[probe] wall {time.monotonic()-t0:.1f}s")

    # Sibling: GateUP-B32-M4096 (1.09)
    t1 = time.monotonic()
    run_fwd("GateUP-B32-M4096", 32, 4096, 5760, 2880, chunk_sizes)
    print(f"[probe] +{time.monotonic()-t1:.1f}s")

    # Sibling: Down-B32-M2048 (1.10)
    t1 = time.monotonic()
    run_fwd("Down-B32-M2048", 32, 2048, 2880, 2880, chunk_sizes)
    print(f"[probe] +{time.monotonic()-t1:.1f}s")

    # Sibling: GateUP-B4-M4096 (1.07-1.10 drift)
    t1 = time.monotonic()
    run_fwd("GateUP-B4-M4096", 4, 4096, 5760, 2880, chunk_sizes)
    print(f"[probe] +{time.monotonic()-t1:.1f}s")

    # Counter: Down-B4-M2048 (R3 cell already at non-default num_slots)
    t1 = time.monotonic()
    run_fwd("Down-B4-M2048", 4, 2048, 2880, 2880, chunk_sizes)
    print(f"[probe] +{time.monotonic()-t1:.1f}s")

    print(f"\n[probe] total wall {time.monotonic()-t0:.1f}s")
