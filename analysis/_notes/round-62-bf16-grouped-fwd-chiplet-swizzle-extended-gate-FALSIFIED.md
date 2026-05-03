# Round 62 — BF16 grouped, fwd chiplet swizzle + extended work-stealing gate — both FALSIFIED

## Goal coming in

R61 (HK SHA `31585671`, PT `78f68a5`) shipped gated atomic-claim
work-stealing in `grouped_kernel`'s persistent loop, gated on
`tiles ∈ (0, NUM_CUS*4) ∧ tiles % NUM_CUS != 0`. 5-sample paired
metric showed +16 median / +20 mean score, with the imbalanced
gpt_oss B=4 M=2048 family flipping to PASS (ratio 1.30+).

R61 doc R62 next-action surface listed two main levers:
1. **Per-XCD chiplet swizzle for the 256-block grid** — R60 finding
   that `chiplet_transform_chunked(_, NUM_CUS=256, 8, 64)` is identity
   (`limit = (256/512)*512 = 0`). chunk_size=32 gives `block=256,
   limit=256` and a true permutation that places XCD `i`'s 32 CUs on
   tiles `{32i, 32i+1, ..., 32i+31}`.
2. **Extend work-stealing gate** to capture
   `gpt_oss-GateUP-B4-M4096` (tiles=1472), the 3rd-most-imbalanced
   metric shape (192/256 wave-imbalance ratio).

Round-start metric (GPU 4, single sample): score=897. Worst shape
single-sample was `gpt_oss-Down-B4-M2048` ratio 1.040 (gated path,
high variance). Stable worst weight-3 ungated: `gpt_oss-Down-B32-M2048`
1.081 / `gpt_oss-Down-B32-M4096` 1.085. Stable worst weight-1:
`DSV3-Down-B16-M2048` 1.081 / `DSV3-Down-B32-M2048` 1.099.

## Lever A: chunk_size=32 chiplet swizzle (FALSIFIED — neutral)

### Implementation

`/workspace/code/HipKittens/analysis/bf16_gemm/mi350x/kernel_bf16_dynamic.cpp`
line 3753 (grouped_kernel static-path branch):

```cpp
// before:
pid = chiplet_transform_chunked(blockIdx.x, NUM_CUS, g.num_xcds, 64);
// after:
pid = chiplet_transform_chunked(blockIdx.x, NUM_CUS, g.num_xcds, CUS_PER_XCD);
```

`CUS_PER_XCD = 32` makes `block = 8 * 32 = 256` and `limit = (256/256) * 256 = 256`,
so all 256 workgroups get permuted (vs identity at chunk_size=64).
The work-stealing gated path (atomic-claim) is untouched.

Predicted XCD assignment shift:

| Path           | XCD 0 CUs at iter 0 process tiles |
|----------------|-----------------------------------|
| chunk_size=64  | {0, 8, 16, ..., 248} (every 8th tile, identity)  |
| chunk_size=32  | {0, 1, 2, ..., 31}    (consecutive)              |

Mechanism intent: route adjacent CUs of the same XCD to consecutive
tiles, concentrating each XCD's first-iter working set onto a
contiguous (M, N) sub-region for tighter per-XCD L2 reuse.

### Resource report

Identical to R61: SGPR 92-104, VGPR 246-256, +1 VGPR spill on KI=48/64/88,
24 spill on KI=112, occ=2 (compiler), occ=1 actual (LDS-bound).
The chunk_size constant is computed at compile-time (no runtime
register). Build clean.

### Correctness probe (5/5 PASS, bf16 floor)

```
gpt_oss-GateUP-B4-M2048   tiles=736   gated      47.85 dB allclose=True
gpt_oss-Down-B4-M2048     tiles=384   gated      47.85 dB allclose=True
gpt_oss-GateUP-B4-M4096   tiles=1472  static     47.85 dB allclose=True
DSV3-GateUP-B16-M2048     tiles=2048  static     47.85 dB allclose=True
Qwen3-Down-B16-M2048      tiles=2048  static     47.83 dB allclose=True
```

### Metric — 5-sample paired test (GPU 4)

```
R61 baseline (chunk=64, identity):   906 / 903 / 894 / 901 / 889
   median 901   mean 898.6   range 17
R62-A (chunk=32, real swizzle):       889 / 907 / 898 / 932 / 896
   median 898   mean 904.4   range 43
```

Score Δ: -3 median / +5.8 mean (within 1σ noise floor: SEM ≈ 8.2).

### Per-shape investigation — 3 paired samples each, weight-1 family

```
                              R61 baseline (3 samples)   R62-A (3 samples)
DSV3 + Qwen3 weight-1 avg:    0.917 / 0.947 / 0.905     0.910 / 0.943 / 0.908
Mean of means:                0.9232                     0.9204
                                                         Δ = -0.003
```

The single-sample comparison earlier had suggested a +0.06-0.09
ratio gain on Down-* shapes (16 weight-1 cells), but the 3-sample
mean shows the gain disappears under noise — that 1-sample signal
was a same-time-window thermal/contention coincidence.

### Why chunk_size=32 was neutral

The kernel's tile scheduler uses WGN super-block scheduling within
each group: tiles 0..7 = (pid_m=0, pid_n=0..7), tiles 8..15 =
(pid_m=1, pid_n=0..7), .... Without chiplet swizzle, the natural
HW round-robin XCD dispatch already spreads (pid_m, pid_n) cells
across XCDs in a pattern that the WGN scheduler's super-block
cohesion reuses. Adding chunk_size=32 swizzle doesn't free additional
L2 reuse because the working set is already larger than per-XCD L2
partition (~4 MB) for all 24 metric shapes — concentrating XCD
work doesn't fit the working set tighter.

For DSV3-Down (K=7168), one tile B-vector is 256·7168·2 = 3.5 MB.
8-tile working set = 28 MB > 4 MB L2 partition. Concentration via
swizzle doesn't help because the working set evicts within an iter
regardless.

For gpt_oss-Down (K=2880), per-tile = 1.5 MB. 8-tile = 12 MB > 4 MB.
Same argument.

For Qwen3-Down (K=1536), per-tile = 0.75 MB. 8-tile = 6 MB > 4 MB.
Borderline — the swizzle shows the largest single-sample swing
here (Down-B16-M2048: +0.092 in one sample, -0.026 in another).

The single-sample variance signal was concentrated on Qwen3-Down
shapes (smallest K → smallest working set → closest to L2 fit) but
across 3 samples the mean reverted to baseline.

### Decision

Reverted chunk_size back to 64 (which is identity — pre-R62 behavior).
The R61 static partition's natural CU→tile assignment (no swizzle)
remains the best-known choice for the 22 ungated shapes.

## Lever B: extend work-stealing gate to NUM_CUS*6=1536 (FALSIFIED — variance regression)

### Implementation

`/workspace/code/HipKittens/analysis/bf16_gemm/mi350x/kernel_bf16_dynamic.cpp`
`should_use_work_stealing`:

```cpp
// before:
return (tiles > 0) && (tiles < NUM_CUS * 4) && ((tiles % NUM_CUS) != 0);
// after:
return (tiles > 0) && (tiles < NUM_CUS * 6) && ((tiles % NUM_CUS) != 0);
```

Predicted shape coverage:

| Shape (24-shape MoE BF16 metric)      | tiles | mod 256 | R61 gate? | R62-B gate? |
|---------------------------------------|-------|---------|-----------|-------------|
| gpt_oss-Down-B4-M2048                 | 384   | 128     | YES       | YES         |
| gpt_oss-GateUP-B4-M2048               | 736   | 224     | YES       | YES         |
| gpt_oss-GateUP-B4-M4096               | 1472  | 192     | NO        | YES (new)   |
| (all others, mod 256 == 0)            | ≥768  | 0       | NO        | NO          |

The natural mod 256 == 0 filter still excludes the larger ungated
suite (DSV3, Qwen3 at tiles=2048-7168, gpt_oss-B32 at tiles=3072-11776).
Only one new shape gets gated.

### Build / probe

5/5 PASS (probe identical to lever A — same probe shapes).

### Metric — 5-sample paired (GPU 4, sequential to lever A test)

```
R61 baseline:                        906 / 903 / 894 / 901 / 889
   median 901   mean 898.6   range 17
R62-B (gate ≤ NUM_CUS*6):            926 / 879 / 872 / 896 / 918
   median 896   mean 898.2   range 54
```

Per-shape `gpt_oss-GateUP-B4-M4096` ratio across the 5 R62-B samples:

```
1.169 / 1.082 / 1.093 / 1.134 / 1.217   (mean 1.139)
```

vs R61 baseline ~1.107-1.111 (mean ~1.11). **Δ_ratio = +0.03, matching
the ~3 % wave-imbalance prediction for tiles=1472**.

### Why the per-shape lift didn't show at the score level

Score Δ = -0.4 mean, **range 3.2× higher than R61 baseline (54 vs 17)**.
The variance increase canceled the +2-3 score expectation. Three
likely sources of the variance escalation, all triggered by including
a 3rd shape on the work-stealing path:

1. **atomic counter contention escalates per-launch** — 3 hipMemsetAsync
   primes + 3 distinct atomic-claim sequences per metric round (each
   shape contributes its own counter race window).
2. **L2-locality loss on tiles=1472** — working set ~12 MB B-tiles
   vs L2 partition ~4 MB/XCD. Work-stealing's arrival-order claim
   fragments per-CU tile sequences across more (M, N) cells than
   static stride.
3. **HW dispatcher state-machine cost** — additional dispatch-time
   branching on `g.tile_counter != nullptr` for the additional shape.

The key insight: the gated-path variance penalty scales **per
gated shape**, not per metric round. The per-shape benefit
(+0.03 ratio) is smaller than the per-shape variance penalty for
this particular shape (`tiles=1472`'s working set is too large to
fit L2 partition).

### Decision

Reverted gate cutoff back to `NUM_CUS * 4 = 1024`. R61 cutoff is
the best-known choice. The R62 attempt is documented inline in
`should_use_work_stealing` comment block in HK SHA `cabf90c0`.

## Outcome

* **HipKittens**: 1 commit (`cabf90c0` — docs-only update annotating the
  R62 negative result in `should_use_work_stealing`'s comment). No
  functional kernel change vs R61 (`31585671`). Build verifies
  identical kernel hash across R61 / R62 modulo source-line annotations.
* **Primus-Turbo**: 1 commit (this round note).
* **Metric** (single-sample verification post-revert): score=896,
  consistent with R61 baseline mean 898.6.

## Backward bench

Not run for R62 because the kernel function is **functionally
identical to R61** (final state has only docs-only diff in the
gate's comment block). R61's bench results (24/24 PASS, avg fwd
1141.07 / bwd 855.25 TFLOPS) remain valid.

## R63 next-action surface

Both R62 levers eliminated (chunk_size=32 swizzle and gate ≤
NUM_CUS*6). The remaining R61-doc levers and new candidates:

1. **Reduce kernel prologue cost (R60-discovered ~63 µs S
   estimate)**. This is the biggest untapped source on the worst
   gated shape (gpt_oss-GateUP-B4-M2048 wall = S + 3I; S = 6 % of
   wall but consistently paid by every CU). Hoist `s_offs[]` LDS
   load to be cooperative across warps (currently single-thread
   O(G) loop). Pre-fetch first-iter A-tile via early issue. Likely
   worth 2-4 % wall on all 24 shapes.

2. **dA backward audit** (R61 doc carry-over). Bench shows bwd 855
   vs fwd 1141 TFLOPS. dB var_k path (`grouped_var_k_kernel`)
   is unchanged since R55 — pure lever surface for non-metric
   workloads. Doesn't affect this metric but expands the bench
   wall delta.

3. **PMC marker bracketing of prologue subcomponents** (R60 / R61
   doc carry-over). Insert kernel-entry → first-MMA timing markers
   to disambiguate the 63 µs S into (cumsum, LDS init, swizzle
   prefill, first HBM fetch issue) components. This is **diagnostic
   only** — won't move score directly but informs lever #1.

4. **Reduce KI=112 spill (24 VGPRs)** for DSV3-GateUP K=7168.
   This has been outstanding since R52. The KI=112 fully-unrolled
   schedule's 24-spill ceiling means DSV3-GateUP-B16/B32-M2048/M4096
   (4 metric shapes × weight 1) are running register-pressured.
   Need to find the right unroll attenuation that preserves
   pipeline depth without overflowing 256 VGPRs.

5. **Per-XCD work-stealing counter** (lever-A → lever-B variant).
   Replace single global atomic with 8 per-XCD counters, each
   pre-zeroed to a 1/8 share of `total_tiles`. Eliminates atomic
   contention across XCDs and preserves XCD-local cache locality
   AS WELL AS imbalance recovery. Could tame the variance increase
   that R62-B lever B saw on tiles=1472. Implementation cost: ~4
   hours (8 counters in struct, 8 hipMemsetAsync primes, 8-way
   atomicAdd target via `blockIdx.x % 8`).
