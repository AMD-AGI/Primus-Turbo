# Round 95 (auto-loop round 19) — BF16 grouped GEMM: Lever B1 step 1 (sched_barrier mask relaxation in `main_loop_iter`) FALSIFIED — kernel is hand-tune-saturated; compiler scheduler has no usable slack

## Context

R94 PMC (auto-loop R17/R18) measured `grouped_kernel<RCR, FUSED=true>`
and `grouped_var_k_kernel<0>` on the lowest-progress shape
`gpt_oss-Down-B4-M2048` (K=2880, K-tail path):

| kernel        | MfmaUtil | VALUUtil | MemStall | MeanOcc/CU | LdsBC | dur_us |
|---------------|---------:|---------:|---------:|-----------:|------:|-------:|
| grouped       |   42.2 % |   88.2 % |   0.15 % |   5.00 / 8 |  0 %  |  151   |
| grouped_var_k |   40.4 % |   85.9 % |   0.10 % |   5.04 / 8 |  1 %  |  166   |

R94's diagnosis: kernel is **MFMA-undersaturated** (42 % MFMA-util on a
nominally MFMA-bound workload), **not memory-bound** (MemStall 0.15 %).
Levers R83-R93 had all been chasing memory-bound bottlenecks; R94
falsified that framing.

R94 recommended R19/R95 try **Lever B1 step 1**:
> Insert `__builtin_amdgcn_sched_barrier(0x18)` (= mask out
> `SCHED_BARRIER_MASK_VALU | SCHED_BARRIER_MASK_TRANS`) before each MFMA
> in `device_gemm_tile_body` to let the compiler hoist V_ALU above
> pending MFMAs. No correctness risk, no VGPR delta, ~1 hour to
> implement and gate. Expected +2-5 score if it works; falsification
> note if not.

## Mask semantics — clarified mid-round

LLVM AMDGPU `__builtin_amdgcn_sched_barrier(mask)` semantics: bits
**SET** = instruction categories **ALLOWED** to cross the barrier in
either direction. Verified via
[LLVM AMDGPUUsage.rst](https://rocm.docs.amd.com/projects/llvm-project/en/latest/LLVM/llvm/html/_sources/AMDGPUUsage.rst.txt):

```
0x0001 ALU,  0x0002 VALU,  0x0004 SALU,  0x0008 MFMA/WMMA,
0x0010 VMEM, 0x0020 VMEM_RD, 0x0040 VMEM_WR,
0x0080 DS,   0x0100 DS_RD,   0x0200 DS_WR,  0x0400 TRANS
```

So `mask=0` (current code in `main_loop_iter`, 8 occurrences after
each DO_MMA cluster) = **strict** (nothing crosses); `mask=0x18` =
MFMA + VMEM allowed to cross (VALU is **blocked**). R94's stated intent
("let V_ALU be hoisted above pending MFMAs") would actually require
`mask=0x02` (VALU bit set), not `mask=0x18`. The R94 plan had a
mask-direction inversion in its narrative; this round tests the
underlying lever (= "any non-zero mask helps"), independent of the
specific bit-pattern muddle.

## What was tried — 4 mask variants, full A/B series

Single `BF16_MAIN_LOOP_SCHED_MASK` macro added at
`kernel_bf16_dynamic.cpp:528`; the 8 `__builtin_amdgcn_sched_barrier(0)`
calls in `main_loop_iter` (lines 630-707) became
`__builtin_amdgcn_sched_barrier(MAIN_LOOP_SCHED_MASK)`, with the
inner constexpr:

```cpp
constexpr unsigned MAIN_LOOP_SCHED_MASK =
    (L == Layout::RCR) ? (unsigned)BF16_MAIN_LOOP_SCHED_MASK : 0u;
```

The Layout-RCR gate is **necessary**: see "What faults" below.
Epilog 1 / epilog 2 / K-tail blocks left at `mask=0` (untouched).

### Mask = 0x06 (VALU + SALU) — CRASH on first probe

Memory access fault on first `grouped_gemm` call (any layout). Compiler
hoisted `a_coord(...) / b_coord(...)` VALU work across the LDS
A_tile / B_tile read-write boundary (`load_a_subtile` / `load_b_subtile`
internally decompose into VALU-side address computation that depends on
the previous iteration's LDS write completion). Reverted before metric.

### Mask = 0x04 (SALU only), all-layouts — CRASH on backward

Forward fwd-only probe passes; backward (calling
`grouped_kernel<Layout::RRR>` for dA + `grouped_var_k_kernel<0>` for dB)
faults. Root cause: `s_waitcnt vmcnt(N)` and `s_waitcnt lgkmcnt(N)`
are emitted by inline `asm volatile(...)` in the kernel; the LLVM
scheduler classifies them under SALU. Allowing SALU to cross the
post-DO_MMA barrier moves an `s_waitcnt vmcnt` past a `buffer_load`,
so the next iteration's HBM data is consumed before it lands. The CRR
(`grouped_var_k_kernel`) and RRR (`grouped_kernel<Layout::RRR>`)
main loops both anchor their pipelines on `s_waitcnt vmcnt(N)`
(K-aligned-LDS prefetch idiom); RCR forward anchors on
`s_waitcnt lgkmcnt(N)` (LDS-side dep) which the scheduler respects via
data-dep tracking even when SALU is unblocked.

→ Gated `mask=0x04` to `Layout::RCR` only. Verified backward path
runs cleanly (metric correctness 24/24 PASS).

### Mask = 0x04, RCR-only — NOISE BAND (4 samples)

```
runs:  891, 882, 882, 884   mean = 884.75   std = 4.3
```

Single-run 891 was a noise spike (inside historical metric variance
±2-4 score for this benchmark on a contended pool, see R91 noise
distribution analysis: R86-R90 mean = 882.6 ± 0.5 with single-runs
spanning 882-886).

### Mask = 0x10 (VMEM only), RCR-only — NOISE BAND (5 samples)

Hypothesis: VMEM-cross would let the compiler hoist HBM `buffer_load`
issues into the MFMA shadow without touching VALU/SALU schedule.

```
runs:  882, 883, 884, 885, 885   mean = 883.8   std = 1.3
```

Distinguishable from baseline at 0σ. No improvement.

### Mask = 0x18 (MFMA + VMEM = R94 literal), RCR-only — NOISE BAND (5 samples)

```
runs:  881, 883, 882, 888, 884   mean = 883.6   std = 2.7
```

Single-run 888 noise spike, otherwise flat. R94's literal
recommendation tested as-stated; mean delta ≈ 0.

### Baseline (mask=0), all-layouts — NOISE BAND (3 samples)

```
runs:  884, 882, 884             mean = 883.3   std = 1.2
```

(Re-baselined after each rebuild; the macro infrastructure with
`MAIN_LOOP_SCHED_MASK = 0` produces bit-identical codegen to the
pre-R95 source — see post-revert verification at the end.)

## Aggregate verdict

| build                              | n | mean  | std  | Δ vs baseline |
|------------------------------------|--:|------:|-----:|--------------:|
| baseline (mask=0, original)        | 3 | 883.3 | 1.2  | -             |
| RCR mask=0x04 (SALU)               | 4 | 884.75| 4.3  | +1.5          |
| RCR mask=0x10 (VMEM)               | 5 | 883.8 | 1.3  | +0.5          |
| RCR mask=0x18 (MFMA+VMEM, R94)     | 5 | 883.6 | 2.7  | +0.3          |
| All-layout mask=0x04               | - | CRASH | -    | -             |
| All-layout mask=0x06               | - | CRASH | -    | -             |

Pooled noise σ ≈ 2.5 score across the 17 samples. None of the +1.5 /
+0.5 / +0.3 deltas survive significance: t < 1.0 on each, well below
the **+5 score commit gate** (per task body's "Score ≥ prior best + 5"
rule; prior best = 889).

## Why the lever is exhausted — kernel hand-tune saturation

The forward `grouped_kernel<RCR, FUSED_KTAIL=true>` `main_loop_iter`
lambda is heavily orchestrated:

```cpp
asm volatile("s_waitcnt lgkmcnt(0)");   // wait for LDS A/B
__builtin_amdgcn_s_setprio(1);          // boost wave priority
DO_MMA(C_accum[i][j], A_tile, B_tile_X, C_accum[i][j]);  // 4 MFMA insns
__builtin_amdgcn_s_setprio(0);          // drop priority
__builtin_amdgcn_s_barrier();           // HW wave sync
__builtin_amdgcn_sched_barrier(0);      // strict compiler fence (R95 target)

[load_subtile (LDS read) + G::load (HBM→LDS write) for next tile pair]
[s_waitcnt + s_barrier]
```

R94's diagnosis (42 % MFMA util) is correct but the leverage is
**not** in the compiler scheduler. The HW wave barriers
(`__builtin_amdgcn_s_barrier()`), inline-asm `s_waitcnt vmcnt/lgkmcnt`,
and the DO_MMA pipeline ordering are all enforced by hardware /
inline-asm volatile semantics — the `sched_barrier(0)` is just the
**compiler-side** mirror of constraints the kernel already enforces.

Concretely, what mask=non-zero allows the compiler to do is a
**tiny** instruction-window reordering that:

1. Cannot move `s_waitcnt`-anchored loads earlier (because LLVM
   conservatively keeps `asm volatile` pinned even when SALU is
   unblocked — proven by RCR-fwd correctness with mask=0x04).
2. Cannot move LDS reads earlier (DS bit not set in any tested mask).
3. Cannot move MFMA earlier across HW barriers (HW barrier is opaque
   to the compiler scheduler).

The remaining slack is on the order of single-cycle scheduling of
loop-counter SALU, address-recompute VALU, and `s_setprio` wave-priority
flips — all of which sum to ~1-3 cycles per K-step (~0.5-1 % per
kernel) which falls inside the 2-3 score noise floor.

This is the same mechanism that defeated R83-R93 LDS swizzle / VGPR
rematerialization / cross-permute attacks: the kernel was saturated
on a different bottleneck (then assumed memory-bound, now confirmed
MFMA-undersaturated due to **occupancy** = 5 / 8 waves resident, not
scheduler bandwidth). Releasing more compiler reordering doesn't
release any new MFMA-issue throughput.

## What R94 step 2 was — and why it's the next attempt

R94's R20 plan, deferred since R95 step 1 needs to be falsified first:

> R20 tries B1 step 2: split `C_accum[2][2]` into two register banks
> `C_accum_a[2][2]` and `C_accum_b[2][2]`, with DO_MMA writing
> alternating banks. Halves the RAW chain length; should double
> effective MFMA throughput. Risk: may exceed 256 VGPR ceiling
> (currently 250 → ~282 after split). Mitigation: drop one of the
> intermediate B-stage register banks to LDS to make room.

Note: the data does **not** actually support a "RAW-chain stall"
hypothesis on `C_accum`. R94's PMC table showed VALUUtilization 86-88 %
which means the V_ALU / RAW-chain slack is busy doing useful
non-stall work; the 60 % MFMA-idle window is on the **MFMA-pipe** side,
not the V_ALU dependency chain. R94 step 2 may also under-deliver
when actually attempted.

The deeper data point from R94 PMC was occupancy: 5 / 8 waves
resident per CU. The kernel is at 250 VGPR vs the 256-VGPR / 8-wave
ceiling. To get from occupancy 5 to occupancy 8 requires VGPR ≤ 192,
a 58-VGPR drop. That's a major refactor (the current
`C_accum[2][2]` is a 16x16 tile array @ ~16 VGPR per tile = 64 VGPR
just for the accumulator). The R94 step 2 split DOUBLES the
accumulator footprint to 128 VGPR — moving in the wrong direction
for occupancy.

## Recommended R20+ plan (revised, post-R95)

Three vectors are still un-falsified, all higher-effort than R95:

### Lever B1 step 3 (untested) — `__launch_bounds__` lowering

`grouped_kernel` is `__launch_bounds__(NUM_THREADS, 2)` (= 2 waves
per CU minimum). The compiler aims for occupancy 2 to satisfy this.
If we lower the second arg to 1 (= 1 wave per CU minimum, occupancy
1+ acceptable), the compiler may drop the VGPR ceiling enforcement
and produce a denser schedule with higher single-wave throughput.

Risk: occupancy 1 means 1 wave per CU = no latency hiding via
wave switching. PMC data suggests this may be net-negative for
B=4 shapes (low tile count / CU = no wave reuse anyway, so dense
single-wave wins) but net-negative for B=32 (high tile count
benefits from wave switching).

Cost: 1 attribute change, 1 build, 1 metric run. ~3 minutes.

### Lever C (still applicable, partial-FROZEN risk) — uniform-M dB BMM dispatch

R91 / R92 noted: when `M_g` is uniform across groups (= the metric
environment with `balance=True`), `dB[g] = dout[g].T @ x[g]` is
mathematically a regular batched GEMM with no var-K geometry. The
dB var-K kernel (var_k_kernel) carries 256-VGPR baggage from the
on-device `compute_var_k_group_lookup` arithmetic; a flat BMM
kernel could drop to 192 VGPR (occupancy 3+) and gain speed
linearly with occupancy.

FROZEN-list status: requires checking group_lens uniformity (would
need `.tolist()` host sync — banned), unless the kernel binding
exposes a "static-uniform-M fast path" knob. Per R92, **HK BF16
binding does NOT currently expose this** — Lever C is multi-round
HK kernel work. Defer past R30.

### Lever D — DSV3 / Qwen3 K%128==0 path PMC profile

R87 / R94 PMC are both on the K-tail path (gpt_oss). The K%128==0
path (DSV3 K∈{2048, 7168}, Qwen3 K∈{1536, 4096}) hasn't been
profiled. With both gpt_oss K-tail levers (R94 / R95) at ceiling,
the K-aligned path is the next score-headroom zone (sits at
geomean 1.12 → 1.25 = 12 pp gap on 16 shapes × weight 1 = 24 score
ceiling).

Cost: 1 PMC round (no commit), then 1-2 lever rounds.

### R20 pick

**Lever B1 step 3** is the cheapest & safest next test (~3 min total
runtime, single attribute change, easy revert). If it falsifies,
**Lever D PMC profile** opens the next zone of headroom. Lever C
deferred indefinitely (multi-round HK kernel work).

## Falsification verification

Post-revert (HK working tree clean, kernel restored to git HEAD
`40be51de`):

```
metric_grouped_bf16_weighted_wall.py: score = 883
correct_fail = 0/24, reject = 0/24, all 24 PASS
```

883 sits inside the R86-R95 noise band (882.6 ± 2.5 across all
sampled rounds). No regression introduced.

## Round result

* Lever: B1 step 1 — relax `sched_barrier(0)` mask in `main_loop_iter`
* Variants tested: 0x04 / 0x06 / 0x10 / 0x18, with and without
  Layout::RCR gating
* Files touched (then reverted):
  - `HipKittens/analysis/bf16_gemm/mi350x/kernel_bf16_dynamic.cpp:494-708`
* Build: clean, no spill / occupancy regression on any tested mask
  (`grouped_kernel<RCR>` VGPR=250, `grouped_var_k_kernel` VGPR=256
  unchanged across mask variants)
* Correctness: 24/24 PASS for all in-pool mask variants (0x04 RCR-only,
  0x10 RCR-only, 0x18 RCR-only); 0x04 and 0x06 all-layouts crash on
  backward path (memory access fault → reverted before metric)
* Metric A/B (17 samples total across 4 builds):
  - Baseline mean = 883.3, σ = 1.2
  - Best lever variant mean = 884.75 (mask=0x04 RCR), Δ = +1.5
  - All variants within 1σ of baseline
  - **Below the +5 commit gate by 3.5+ score** in every case
* Action: **REVERTED**. Final post-revert metric: 883 (in noise band).

## Updated patience / streak tracking

Cumulative falsification streak now R83-R95 = **13 rounds**, all
attempts on the kernel-internal compute / scheduler axis falsified or
noise-bound. Best-ever 889 (R14 single-run noise spike) holds.

## SHA pointers

* Primus-Turbo HEAD pre-round: `8fe2be2b38f153ed47fb1b9f1472428b3bb560c8`
* HipKittens HEAD: `40be51de` (unchanged — kernel reverted clean)
* This note committed in Primus-Turbo only; HipKittens working tree
  clean post-revert (the 4 build cycles in this round all targeted
  the same `kernel_bf16_dynamic.cpp` and were reverted via
  `git checkout --`).
