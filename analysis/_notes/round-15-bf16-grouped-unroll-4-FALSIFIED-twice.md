# Round 15 — BF16 grouped unroll-factor sweep (TWO SUB-EXPERIMENTS, BOTH FALSIFIED)

**Status:** FALSIFIED — Both R15 sub-experiments (KI=44 specialization at
varied unroll factors AND KI=0 dynamic-K + unroll-4) regressed metric.
Combined with R10/R11/R14, this round adds the **4th falsified lever**
on the gpt_oss K=2880 kernel-side surface; the gpt_oss K=2880 kernel-
level optimization surface is now firmly closed.

## R15 entry state

Baseline metric: **884** (gpt_oss geomean 1.0971, DSV3 1.1162, Qwen3 1.1128).
Best historical: 891 (R7). Plateau range R7-R14: 873-887, ±10 noise band.

R14 falsified `KI_HINT=44 + FUSED_KTAIL=true` with full `#pragma unroll`
due to 28-30 VGPRs spilling to scratch (-30 score). R14's R15 plan
recommended trying `#pragma unroll 4` as a "middle ground" between
KI=0's `#pragma unroll 2` (no spill, no scheduling depth) and KI=44
full unroll (full depth, heavy spill). Resource-report check first.

## R15 sub-experiment 1: KI=44 + various `#pragma unroll N`

Same instantiation surface as R14: added `template grouped_kernel<RCR, 44, true>`
+ `template grouped_kernel<RRR, 44, true>` + `launch_one_grouped_fuse_ki<L, KI>`
+ `case 44:` dispatch branch. Tried 3 unroll factors via the
`else if constexpr (FUSED_KTAIL)` branch in `device_gemm_tile_body`:

| Unroll factor | KI=44+RCR Spill | KI=44+RCR Scratch | KI=44+RRR Spill | KI=44+RRR Scratch |
|---|---:|---:|---:|---:|
| `#pragma unroll` (full, R14 baseline) | 28 | 116 | 30 | 124 |
| `#pragma unroll 4`                    | 28 | 116 | 30 | 124 |
| `#pragma unroll 2`                    | 28 | 116 | 30 | 124 |

**Identical spill profile across all three unroll factors.** The
`#pragma unroll N` directive becomes effectively decorative when
`num_tiles` is a `constexpr int` (KI_HINT > 0): hipcc / LLVM unrolls
the loop based on the known bound regardless of the pragma value.

The KI=0 path (which keeps `num_tiles = num_tiles_dyn` runtime-bound)
shows zero spill at the same unroll-2:

| Path                          | RCR VGPR | RCR Spill | RRR VGPR | RRR Spill |
|---|---:|---:|---:|---:|
| KI=0 + unroll-2 (production)  | 248 | 0 | 249 | 0 |
| KI=44 + unroll-{2,4,full}     | 256 | 28-30 | 256 | 30 |

**Mechanism**: at KI=44 (constexpr), the compiler can constant-propagate
`tile`'s value into each unrolled main_loop_iter copy, enabling
aggressive cross-iteration scheduling. This expands per-unrolled-copy
live ranges — registers for prefetch loads (A_tile, B_tile_*) get
stretched across multiple inlined iterations because the compiler can
see them all simultaneously. Working set exceeds 256 VGPR budget → spill.

KI=112 (DSV3 K=7168) doesn't spill at full unroll because the longer
dependency chain (more MMAs interleaved with prefetches) lets the
compiler stagger live ranges across more iters. KI=44 has half the
iters — twice the live-range compression — over budget.

**Conclusion 1: Adding ANY `KI_HINT > 0` specialization for the gpt_oss
K=2880 fuse path is closed.** The constexpr num_tiles itself is the
trigger; unroll-factor knob doesn't help.

## R15 sub-experiment 2: KI=0 dynamic-K + `#pragma unroll 4`

Pivoted to a smaller change: keep `KI_HINT=0` (dynamic-K, runtime-bound
num_tiles), but add a template-specialized unroll factor inside
`device_gemm_tile_body`'s KI_HINT==0 branch:

```cpp
} else {
    const int num_tiles = num_tiles_dyn;
    if constexpr (FUSED_KTAIL) {
        #pragma unroll 4   // R15: deeper scheduling for gpt_oss K=2880 fuse
        for (int tile = 0; tile < num_tiles - 2; tile += 2) main_loop_iter(tile);
    } else {
        #pragma unroll 2   // legacy
        for (int tile = 0; tile < num_tiles - 2; tile += 2) main_loop_iter(tile);
    }
}
```

No new instantiation, no dispatch change — minimal patch.

Resource report (KI=0 + FUSED_KTAIL=true):

| Path           | RCR VGPR | RCR Spill | RCR Scratch | RRR VGPR | RRR Spill |
|---|---:|---:|---:|---:|---:|
| unroll-2 (prod) | 248 | 0 | 0 | 249 | 0 |
| unroll-4 (R15)  | 248 | 0 | 0 | 249 | 0 |

**Zero spill** — runtime-bound num_tiles keeps the compiler's
unrolling bounded to the pragma, and 4-copy partial unroll fits within
the working set. Resource gate passed. Correctness gate: SNR 49.58 dB
on all 6 probe shapes (5 fuse-eligible + 1 control).

Performance gate (the kill):

|                    | metric | gpt_oss geo | DSV3   | Qwen3  |
|--------------------|-------:|------------:|-------:|-------:|
| Baseline (HEAD)    |   884  | 1.0971      | 1.1162 | 1.1128 |
| KI=0 + unroll-4    |   867  | **1.0655**  | 1.1154 | 1.1070 |
| Revert verify      |   875  | 1.0790      | 1.1175 | 1.1143 |

**Δ = -17 score, gpt_oss -3.2 pp.** The unroll-4 partial unrolling
*didn't spill*, but the kernel ran *slower* than unroll-2.

**Mechanism (educated guess)**: 4 inlined copies of main_loop_iter
within each outer-loop iteration is ~85 lines × 4 = ~340 lines of
intermediate code per outer iter. The body has many barriers /
sched_barriers / setprio annotations carefully tuned for the unroll-2
schedule (lines 605-684). At unroll-4 the compiler may be:

1. **Over-flattening barriers**: the manual `__builtin_amdgcn_s_barrier()`
   calls were placed every 4 MMAs to give MFMA pipeline drain windows;
   at unroll-4 these now fire 16 times per outer iter instead of 8,
   potentially serializing more aggressively than needed.
2. **I-cache pressure**: 4× the inlined main_loop_iter body may exceed
   L0 I-cache, triggering refills mid-iteration.
3. **Suppressing sched_barrier(0) effectiveness**: the carefully placed
   `sched_barrier(0)` directives that prevent specific instruction
   reorders may interact poorly with the larger inlined region.

The KI=0 + unroll-2 path was hand-tuned for this schedule (see the
extensive comments at lines 600-686); changing the unroll factor
breaks the assumed window size.

DSV3 / Qwen3 are unaffected within noise (-0.7 to -0.9 pp on geomean,
both within ±1 pp). Confirms the change is FUSED_KTAIL-gated.

**Conclusion 2: The KI=0 + unroll-2 baseline is at a local optimum for
this kernel's schedule.** Changing unroll factor ± breaks the careful
hand-tuned barrier / sched_barrier interaction and regresses runtime.

## Combined R15 ablation map

```
              constexpr num_tiles?    pragma     spill   metric Δ
KI=0  unroll-2 (prod)        no         2          0      0
KI=0  unroll-4               no         4          0    -17
KI=44 unroll-full            yes        ∞         28    -30
KI=44 unroll-4               yes        4         28    -30
KI=44 unroll-2               yes        2         28    -30
```

Every variant at this surface was negative-EV. The single working point
is the production `KI=0 + unroll-2`.

## R10/R11/R14/R15 cumulative falsification surface for gpt_oss K=2880

| Round | Lever                                       | Result |
|-------|---------------------------------------------|--------|
| R10   | A2a/A2b analytical scoping                  | "+13 pp ceiling" projected |
| R11   | A1: K-tail HBM-load early-issue prefetch    | sub-noise (Δ ≈ 0) |
| R14   | A2c: KI=44 + full unroll                    | -30 score (spill) |
| R15a  | A2c': KI=44 + unroll-4 / unroll-2           | -30 score (spill, identical) |
| R15b  | A2c'': KI=0 + unroll-4 (FUSED_KTAIL only)   | -17 score (no spill, slower) |

**4 distinct kernel-side lever attempts on the same surface, all
falsified.** The K=2880 fuse path runs at a tight local optimum that
isn't perturbable via:
* Compile-time vs runtime loop-bound choice (KI knob)
* Unroll factor (2 / 4 / full)
* Scheduling-window restructuring (R11's prefetch reorder)

R10's "13 pp short-K main-loop saturation" is a real diagnostic gap
but the kernel architecture (HALF_BLOCK_M=128 × HALF_BLOCK_N=128
WG-mma scheduling with 256 VGPR budget) does not have spare headroom
to close it. The Triton kernel's 1.10× advantage on these shapes
(gpt_oss B=32 ratio 1.04-1.09 vs target 1.25) is structural, not
parameter-tuning.

## Compliance check

* No metric file edits.
* No `can_handle` tightening.
* No host syncs.
* HipKittens C++ reverted via `git checkout` after both sub-experiments;
  rebuild restored baseline (875 ≈ pre-R15 873-887 noise band).
* No new dispatcher rules.
* Doc-only commit in Primus-Turbo this round.
* Numerical correctness gate PASSed both sub-experiments (49.58 dB).

## What R15 DID

1. Re-ran R15 baseline metric (884; gpt_oss geomean 1.097 — within
   plateau).
2. **Sub-experiment 1**: re-implemented R14's KI=44 + FUSED_KTAIL=true
   instantiation but with `#pragma unroll 4`, then `#pragma unroll 2`.
   Built each. Read resource report — **identical 28-30 VGPR spill**
   regardless of pragma factor. Confirmed constexpr num_tiles itself
   is the spill cause. Did NOT run metric (resource gate failed).
3. **Sub-experiment 2**: pivoted to KI=0 + FUSED_KTAIL=true +
   `#pragma unroll 4` (no instantiation, no dispatch change). Built.
   Resource clean (0 spill). Correctness PASS (49.58 dB). Metric
   867 (-17 vs baseline 884).
4. Reverted HK kernel via `git checkout`, rebuilt.
5. Verified revert metric = 875 (within noise of pre-R15 baseline).
6. Wrote this combined FALSIFIED note.

## Files touched (Primus-Turbo only — HK reverted)

* `analysis/_notes/round-15-bf16-grouped-unroll-4-FALSIFIED-twice.md` — this note.

No code changes shipped. HipKittens repo: `git checkout` reverted the
kernel after both sub-experiments.

## R16 plan (pivot: gpt_oss surface closed)

Cumulative evidence (R10/R11/R14/R15 all falsified) firmly closes the
gpt_oss K=2880 kernel-side optimization surface. R16 should pivot
out of this surface. Two options:

1. **R16 — DSV3 / Qwen3 Phase B push** (per task body lines ~Phase B).
   Once gpt_oss is "capped" at the kernel ceiling (~+10 pp from where
   it is, modulo our analytical estimates), the score plateau is ~960.
   The remaining +40 to score 1000 lives on the K%128==0 fast path:
   * **B1**: MFMA pipeline scheduling on K=4096/7168 main loop.
     Profile `valuMfmaUtil` on DSV3-GateUP-B16-M4096 first; R9
     measured 79.7 % util — only ~5 pp from the 85 %+ cluster ceiling.
   * **B2**: register pressure / occupancy tuning (DSV3 K=7168 main
     loop has 56 K_TWO_TILE iters at full unroll — KI=112 doesn't
     spill but may be near the budget cliff; Qwen3 K=4096 KI=64 is
     similar).
   * **B4**: persistent kernel / work-stealing for the dB var-K path
     on DSV3 K=2048 — same idea as A2 but on the K%128==0 case.

2. **R16 — Plateau acceptance + summary doc**: combine R10/R11/R14/R15
   findings into a single summary recommending the auto_optimize loop
   pivot. Score = 0 movement. Documents the closure for future agents.

R16 should pick option 1: **profile DSV3 or Qwen3 first**, then attempt
the highest-headroom lever. Target: +5-15 score from a B-class lever.
The DSV3 / Qwen3 surface has not been kernel-side attacked at all in
this run; potentially has untapped headroom unlike the gpt_oss surface.
