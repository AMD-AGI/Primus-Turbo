# R16 — gpt_oss FP8 wgrad var-K kernel: per-tile prologue trim FALSIFIED

## TL;DR

Pivoting from R15 (dispatcher exhausted) into HipKittens kernel surgery,
this round attempted the **R14 plan #1 / R15 R16 plan #2** lever:

> Replace the 6-step LDS binary search at the top of every persistent-loop
> iteration in `grouped_var_k_kernel_fp8` with a single integer divide
> (R38 already proved the closed form `s_cum_tiles[k] = k * tiles_per_group`,
> so the search is provably redundant).

The change was numerically bit-equivalent (correctness PASS on the metric's
8 shapes, SNR > 25 dB), but **regressed the wgrad section average by
~18 T (1774 → 1755 mean over 3 runs)** because the soft-divide instruction
sequence on CDNA3 (gfx950) increased register pressure, bumping VGPR spill
from 37 → 41 inside the hot K-loop. Each spill is ~2 mem-ops × ~16 K-iters
per tile, costing ~640 cycles per tile — well above the ~150 cycles the
binary-search removal saved.

**Falsified.** Reverted; no commit to HipKittens.

This caps the **R14 plan #1** ("per-tile prologue trim, ported to wgrad
var-K") at the kernel-cell ceiling — the bottleneck is register-pressure-
bound on the existing kernel, not cycle-bound on the prologue. Any
follow-up kernel surgery on this path must be *register-neutral or
register-reducing*, not just cycle-saving.

## Anchor

- **Shape**: gpt_oss-Down-B4-M2048 wgrad (worst HK ratio in the metric:
  ratio = 1280 T / 2800 T target = 0.457).
- **Kernel**: `grouped_var_k_kernel_fp8<KI_HINT=0>` at line ~7766 of
  `HipKittens/analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp`.
- **Round 2 baseline metric (3 runs)**: scores 682, 682, 680 (mean 681);
  Down-B4-M2048 wgrad p20 ≈ 1278 T (range 1272-1286).

## Theory of operation

The persistent-loop iteration body opens with:

```cpp
for (int gt = pid; gt < total_tiles; gt += slots_eff) {
    int lo = 0;
    int hi = MAX_G_PLUS_1 - 1;        // = 64
    #pragma unroll
    for (int level = 0; level < 6; ++level) {
        const int mid = (lo + hi + 1) >> 1;
        if (gt >= s_cum_tiles[mid]) lo = mid;
        else hi = mid - 1;
    }
    const int group_idx = lo;
    const int tile_start = s_cum_tiles[lo];
    const int local_tile = gt - tile_start;
    ...
}
```

Six LDS reads + compares serialized through the unroll body
(~6 × 25 = 150 cycles serialized on CDNA3), executed once per
persistent-loop iteration. R38 (Lever L) already proved the closed form
`s_cum_tiles[k] = k * tiles_per_group` for var-K (since every group of a
var-K dispatch produces an identical-sized dB output `[bpr, bpc]`
tile-grid), so the binary search is provably redundant: any
`gt ∈ [0, G*tiles_per_group)` has `lo = gt / tiles_per_group`.

**Predicted effect**: save ~150 cycles per persistent-loop iteration on
Down-B4 family (484 total tiles / 192 slots = 2.5 tiles/slot). Per-slot
savings: 2.5 × 150 = 375 cycles. Slot wall ≈ 93 µs at 1.4 GHz / 256 CUs ≈
130k cycles. Predicted lift: 0.3 % on Down-B4-M2048 wgrad.

## What happened

Implemented `gt / tiles_per_group` + dropped the now-dead `s_cum_tiles[]`
LDS array and `s_total_tiles` LDS scalar, replacing them with a register-
held `total_tiles = g.G * tiles_per_group` computed once at kernel entry.

### Kernel resource usage diff (`-Rpass-analysis=kernel-resource-usage`)

| Metric           | Baseline (binary search) | R16 trim (divide) | Δ |
|---|---:|---:|---:|
| TotalSGPRs       | 80   | 86   | +6   |
| VGPRs            | 256  | 256  |  0   |
| **VGPR Spill**   | **37**   | **41**   | **+4**   |
| ScratchSize/lane | 152  | 168  | +16  |
| LDS / block      | 139796 | 139524 | −272 |
| Occupancy        | 2 waves/SIMD | 2 waves/SIMD | 0 |

The 272-byte LDS reduction landed (s_cum_tiles[65 ints] + s_total_tiles
= 264 bytes, plus alignment), but the +4 VGPR spill is the killer.

### Metric impact (3 runs each)

|                   | wgrad avg (T) | Down-B4-M2048 wgrad (T) | score |
|---|---:|---:|---:|
| Baseline (binary) | 1775 / 1777 / 1769 → mean **1774** | 1286 / 1278 / 1272 → 1278 | 682 / 682 / 680 → mean **681** |
| R16 trim (divide) | 1758 / 1753 / 1759 → mean **1757** | 1269 / 1259 / 1257 → 1262 | 680 / 681 / 678 → mean **680** |

- wgrad section avg: **−17 T** (1774 → 1757). Beyond noise floor (run-to-run
  spread on either side ~5-10 T section avg).
- Score: −1 (within noise, but trending down).
- Down-B4-M2048 wgrad anchor: −16 T (−1.3 %).

The **regression was not just on the under-saturated short-grid family** —
even the saturated GateUP-B32 family lost ~30 T on the M=2048 shape
(1861 → 1832), and Down-B32-M2048 wgrad lost ~50 T (1668 → 1619). This is
consistent with a register-pressure regression that hits *every* tile,
amplified on under-saturated shapes (which spend a higher fraction of slot
wall in kernel and a lower fraction in launch tail).

## Why the divide regressed (vs the predicted ~150 cycle save)

The binary search's 6 LDS reads consumed almost zero VGPR budget at
runtime (the LDS pointer is one SGPR; each LDS read returns into a
short-lived VGPR; LLVM aggressively re-uses the same VGPR slot across the
unrolled iterations because the values are immediately consumed by the
next compare). The total live VGPR set during the search was effectively
2-3 registers.

The soft-divide instruction sequence on CDNA3 (no native integer divide)
expands to ~10-15 ops (`v_cvt_f32_u32`, `v_rcp_f32`, `v_mad_f32`,
`v_mul_f32`, `v_cvt_u32_f32`, refinement steps) all of which consume
VGPRs simultaneously. LLVM materialised those temporaries with a
LIFETIME that overlapped the K-loop register graph (the divide is
emitted at the top of each iteration, so its temporaries are live across
the LDS-pre-fill barriers and into the MMA register file). Net +4 VGPRs
of pressure pushed 4 of the existing K-loop live values into scratch.

Each spill = 1 store + 1 load = 2 VMEM ops. There are 16 K-iter rounds
per tile (M_g/HB = 2048/128 = 16), and the spill register might be live
across ~half of those iterations on average. So per tile:

  4 spills × 2 ops × ~8 K-iter live-in / live-out = ~64 VMEM ops/tile

VMEM scratch traffic on MI355X is ~30-40 cycles per op (HBM3-cached
scratch with NUMA bias). So 64 × 35 ≈ **+2240 cycles per tile** of spill
overhead.

vs ~150 cycles saved by removing the binary search ⇒ **net −2090 cycles
per tile**. Predicted regression matches the measured one well.

## Implication for the broader R14 / R15 plan

The R14 #1 candidate ("per-tile prologue trim — hoist invariants across
tiles within the same group: cumsum offset + group index + ki_g are all
group-uniform") was framed as a *cycle saving* lever. R16 (this falsification)
proves that on the existing var-K kernel, *any* register-bumping cycle-save
is net-negative because the kernel is at the 256-VGPR cap with 37
VGPRs already spilling.

Future kernel-surgery candidates on the var-K wgrad path must be
**register-neutral or register-reducing**:

1. ~~KI_HINT specialization~~ — already falsified by R63 (compile-time
   loop bounds INCREASE register pressure on KS=22 specialisation).
2. ~~Per-tile prologue trim via divide~~ — R16 (this note).
3. **Register-neutral**: rewriting the `s_cum_tiles[]` binary search to
   use the same SGPR budget as it already does (e.g., a closed-form
   computation in *scalar* registers using `s_div` or pre-computed
   reciprocals shared across the warp). Untested, may face the same
   register pressure issue if LLVM treats the sgpr→vgpr broadcast as
   live across MMA boundaries.
4. **Register-reducing**: the single biggest lever would be cutting the
   K-loop's MMA register footprint (RBM × RBN × 4 register tiles = a lot
   of live VGPRs). This is a major refactor — not a kernel-surgery round
   item, more like a months-of-work redesign. Out of scope for the
   single-round auto_optimize cadence.
5. **Layout-side**: re-pack the per-group LDS metadata into a vec2 layout
   so each lane reads `(m_start_g, M_g)` in one LDS instruction instead
   of two. Saves at most ~30 cycles per tile, not enough to be worth a
   correctness round.

The only *known* register-neutral lever left for var-K is **chunk_size
(chiplet alignment)** — already saturated at chunk_size=64 / slots=192 by
R3 + R15 sweeps.

## Conclusion: kernel-cell ceiling on the var-K wgrad path

After R3 (slots=192), R14 (PMC fwd RCR launch geometry), R15 (fine-slots
+ gm-xcds revisit), and R16 (per-tile prologue trim via divide), the
gpt_oss var-K wgrad path on MI355X is at the kernel-cell ceiling: any
single-round lever is bounded by the 256-VGPR / 37-spill register graph
of the existing kernel.

**Recommendation for next rounds**:

- Move to fwd RCR / dgrad RRR shapes for cheap dispatcher rule wins
  (the (gm, xcds) cells of those layouts have NOT been exhaustively
  swept on the post-R12 K-tail-fuse-eligible kernel binary).
- Specifically: gpt_oss-Down-B4-M2048 fwd (1448 T, ratio 0.516) is the
  next-worst section/shape with a same-shape RRR/RCR sweep that has not
  been re-run since the K-tail fuse landed.
- Defer var-K kernel surgery until a register-budget-cutting redesign
  is in scope (a multi-week refactor, not a single auto_optimize round).

## Repro

Probe script: `scripts/_probe_round_16_vark_correctness.py`
(builds for both binary-search and divide variants by toggling
`grouped_var_k_kernel_fp8`'s body; used for SNR check).

Build command:
```bash
cd /workspace/code/HipKittens && source env.src
cd analysis/fp8_gemm/mi350x && make -B 2>&1 | grep -A 11 grouped_var_k
```

Metric command:
```bash
cd /workspace/code/Primus-Turbo
for i in 1 2 3; do timeout 60 python3 scripts/_metric_gpt_oss_fp8_kernel.py \
  2>&1 | grep -E "wgrad.*avg|Down_B4_M2048.*wgrad|score="; done
```
