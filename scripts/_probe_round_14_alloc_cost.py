"""Round-14 probe — Stream-K K-split alloc-cost validation.

R11 cost decomp (analysis/_notes/round-11-A1prime-variant-2-K-split-refined-
cost-decomp-GREEN-LIGHT-R12-scaffold.md) GREEN-LIT the multi-round Stream-K
arc on the assumption that per-call hipMallocAsync overhead is ~3 us per
call (~3% of the 105 us Down-B4-M2048 main kernel wall) — leaving the
projected +30% per-cell B=4 lift envelope net-positive.

R13a (HK 43f37f8b) chose per-call hipMallocAsync as the fallback after
the host-cached pattern was found inapplicable (done_counter is caller-
allocated). The actual overhead has NEVER been measured on gfx950 for
the 44-88 MiB buffer sizes the cost decomp predicts.

R14 (this probe) validates that premise BEFORE R15 commits to the
700-line kernel K-split branch. If alloc overhead is <<5 us, GREEN-LIGHT
R15. If 5-15 us, the per-cell envelope shrinks but stays positive on
B=4. If >>15 us, the entire arc EV vanishes and we save 2-3 rounds of
doomed work by pivoting now (e.g., pre-allocated pool from PT side, or
rotate to Direction E barrier scheme).

Mechanism:
  * R13a alloc/free machinery is in `dispatch_grouped_rcr` already, gated
    on `g.sk_split_n > 0`. The kernel itself does NOT yet read
    `g.sk_partial_buf` (R15 will), so accumulation math is bit-identical
    regardless of `sk_split_n`. Only the host-side alloc/free fires.
  * R14 adds `sk_split_n` as a trailing pybind kwarg on `grouped_rcr` /
    `grouped_rcr_dscale` (default 0; production calls bit-identical).
    Setting >0 from this probe triggers the alloc branch.
  * Measurement: per-call wall delta between sk_split_n=2 and sk_split_n=0
    on the same shape. The diff isolates alloc/free + memset overhead
    (the kernel itself is bit-identical between the two — same launch
    parameters, same threads, same memory access pattern; only g.sk_split_n
    and g.sk_partial_buf differ in the launched struct, neither is read by
    the kernel).
  * Anchor shape: Down-B4-M2048 fwd (T_max = 8*22 = 176 tiles → 11 MiB
    per call). This is the smallest B=4 cell — others up to 4x larger
    buffer (Down-B4-M4096 is 22 MiB; GateUP-B4-M4096 is 88 MiB). A
    secondary measurement on GateUP-B4-M4096 bounds the buffer-size
    scaling.

Falsification gate:
  GREEN: median delta < 5 us per call → R15 GREEN-LIT, kernel branch next.
  YELLOW: 5 us <= delta < 15 us → R15 still net-positive on B=4 cells but
          per-cell margin tighter; document the new envelope and proceed.
  RED: delta >= 15 us → R11 cost decomp falsified at the alloc primitive
       level. Pivot: either (a) alloc once at PT-side and pass through
       pybind (mirror done_counter pattern), or (b) rotate to Direction E.
"""
from __future__ import annotations

import os
import statistics
import sys

os.environ.setdefault("PRIMUS_TURBO_HIPKITTEN_PATH", "/workspace/code/HipKittens")
sys.path.insert(0, "/workspace/code/Primus-Turbo")

import torch
from primus_turbo.pytorch.kernels.hipkitten import loader as _hk_loader

hk = _hk_loader.load_fp8()
fn = hk.grouped_rcr_dscale


def _quantize(t):
    amax = t.abs().max()
    s = (amax / 240.0).clamp(min=1e-6).to(torch.float32)
    q = (t.to(torch.float32) / s).clamp(-240.0, 240.0).to(torch.float8_e4m3fn)
    return q, s


def _bench_per_call(B, M, N, K, gm, xcd, sk_split_n,
                    warmup=80, iters=2000, trials=5):
    """Median p20 per-call wall ms, with sk_split_n flag applied to every call."""
    torch.manual_seed(42)
    g_lens = torch.full((B,), M, dtype=torch.int64, device="cuda")
    g_offs = torch.zeros(B + 1, dtype=torch.int64, device="cuda")
    g_offs[1:] = torch.cumsum(g_lens, dim=0)
    a_bf = torch.randn((B * M, K), dtype=torch.bfloat16, device="cuda")
    b_bf = torch.randn((B, N, K), dtype=torch.bfloat16, device="cuda")
    a, sa = _quantize(a_bf)
    b, sb = _quantize(b_bf)
    out = torch.empty((B * M, N), dtype=torch.bfloat16, device="cuda")

    def _call():
        fn(a, b, out, sa, sb, g_offs, gm,
           m_per_group=M, num_xcds=xcd,
           num_slots=0, chunk_size=0,
           fuse_ktail_off=0, sk_split_n=sk_split_n)

    for _ in range(warmup):
        _call()
    torch.cuda.synchronize()

    p20s = []
    for _ in range(trials):
        se = torch.cuda.Event(enable_timing=True)
        ee = torch.cuda.Event(enable_timing=True)
        times = []
        for _ in range(iters):
            torch.cuda.synchronize()
            se.record()
            _call()
            ee.record()
            torch.cuda.synchronize()
            times.append(se.elapsed_time(ee))
        times.sort()
        p20s.append(times[len(times) // 5])
    return statistics.median(p20s), min(p20s), max(p20s)


# Anchor: Down_B4_M2048 fwd (smallest B=4 cell — 11 MiB partial buffer).
# Secondary: GateUP_B4_M4096 fwd (largest B=4 cell — 88 MiB partial buffer).
# (gm, xcd) values are placeholder; alloc cost is independent of (gm, xcd).
SHAPES = [
    # name                     B   M     N     K     gm   xcd
    ("Down_B4_M2048_fwd",      4, 2048, 2880, 2880, 16,  2),
    ("Down_B4_M4096_fwd",      4, 4096, 2880, 2880, 1,   4),
    ("GateUP_B4_M2048_fwd",    4, 2048, 5760, 2880, 4,   8),
    ("GateUP_B4_M4096_fwd",    4, 4096, 5760, 2880, 4,   8),
]

print(f"\n{'=' * 78}")
print(f"  R14 alloc-cost probe (Stream-K K-split per-call hipMallocAsync overhead)")
print(f"{'=' * 78}")
print(f"  Premise (R11): ~3 us per call. GREEN < 5; YELLOW 5-15; RED >= 15.")

for name, B, M, N, K, gm, xcd in SHAPES:
    bpc = N // 256
    tiles_m_total = (B * M) // 256
    T_max = tiles_m_total * bpc
    buf_MiB = T_max * 256 * 256 * 4 / (1024 * 1024)
    print(f"\n  {name}  (B={B}, M={M}, N={N}, K={K}; T_max={T_max} tiles, "
          f"partial_buf={buf_MiB:.1f} MiB)")

    base_med, base_lo, base_hi = _bench_per_call(B, M, N, K, gm, xcd,
                                                 sk_split_n=0)
    sk_med, sk_lo, sk_hi = _bench_per_call(B, M, N, K, gm, xcd,
                                           sk_split_n=2)
    delta_us = (sk_med - base_med) * 1000.0  # ms -> us
    base_pct = (sk_med - base_med) / base_med * 100.0
    print(f"    sk_split_n=0  med={base_med:.4f} ms  range[{base_lo:.4f},{base_hi:.4f}]")
    print(f"    sk_split_n=2  med={sk_med:.4f} ms  range[{sk_lo:.4f},{sk_hi:.4f}]")
    print(f"    >> alloc+free+memset overhead = {delta_us:+.2f} us per call "
          f"({base_pct:+.2f}% of base wall)")
    if delta_us < 5.0:
        verdict = "GREEN"
    elif delta_us < 15.0:
        verdict = "YELLOW"
    else:
        verdict = "RED"
    print(f"    >> verdict: {verdict}")
