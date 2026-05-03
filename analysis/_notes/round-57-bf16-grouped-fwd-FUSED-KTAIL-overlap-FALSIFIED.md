# Round 57 — BF16 grouped, fwd FUSED_KTAIL overlap with EPILOG 2 — FALSIFIED + R52 STALE-CLAIM CORRECTION

## Goal coming in

R56 PMC diagnostic identified `gpt_oss_20B-GateUP-B4-M2048` (tiles=736,
MFMA util=50.4%) as the worst metric shape and proposed R57 lever =
**persistent-grid sizing for sub-1024-tile workloads**. R57 starting
metric (1 sample, GPU 3 contention high) confirmed it's still the
worst by a wide margin: ratio = 0.846, weight 3, HK 685.1 vs TRT 809.6.

## Critical correction to R52's "FUSED_KTAIL is dead code" claim

R52 (sha ebffbf4) doc:
> The dispatcher gates fuse on `K_rem_for_fuse == K_STEP` (= 32) at line 4211.
> For metric shapes:
>   - gpt_oss K=2880: K_rem = 64 → not eligible
>   - DSV3, Qwen3: K_rem = 0 → not eligible
> The `FUSED_KTAIL=true` template is dead code for the entire metric.

**This is now stale.** The current kernel has `K_STEP = 64` (line 21:
`constexpr int K_STEP = 64`), so `K_rem_for_fuse == K_STEP` is `64 == 64`
for gpt_oss K=2880 → **fuse_ktail_eligible = TRUE** for all 8 gpt_oss
shapes. R56 PMC capture confirms it: the kernel name in the rocprofv3
SQLite output for `gpt_oss-GateUP-B4-M2048` was

```
_Z14grouped_kernelIL6Layout0ELi0ELb1EEv22grouped_layout_globals
                                     ^   ^
                                     KI  FUSED_KTAIL
                          → grouped_kernel<RCR, KI=0, FUSED=true>
```

So the FUSED_KTAIL block IS hot for the metric's worst-performing
family (gpt_oss, weight 3). R52 may have been correct under K_STEP=32,
but a later K_STEP doubling re-routed gpt_oss into the fuse path
without R52's claim being re-verified.

This re-opens FUSED_KTAIL block tweaks as a legitimate attack surface
for the next several rounds.

## R57 Hypothesis (FALSIFIED)

R55 added `__builtin_amdgcn_sched_barrier(0)` at the END of EPILOG 2
(`device_gemm_tile_body` line 786, after `s_barrier()`). For the
non-FUSED path (DSV3/Qwen3), this pinning prevents LLVM from hoisting
the post-`device_gemm_tile_body` C-store coordinate compute / store
prologue UP into EPILOG 2's MMA-burst window — a load-bearing pin
that contributed to R55's +9.5 median across 8B/10P paired runs.

For the FUSED=true path (gpt_oss), the next instructions are NOT a
C-store — they are the FUSED_KTAIL block's `load_a_kt(0)` /
`load_b_kt(B_tile_0/1)` `buffer_load_b128` HBM loads. R55's
sched_barrier(0) BLOCKS LLVM from hoisting those loads into EPILOG 2's
MMA-burst window, even though A_tile / B_tile registers are dead
post-`DO_MMA(C_accum[1][1], …)` and the loads could safely overlap
with in-flight MMAs.

**Hypothesized fix**: gate the line-786 sched_barrier(0) on
`!FUSED_KTAIL` so the FUSED=true template gets MMA / K-tail-load
overlap while DSV3/Qwen3 retain R55's pin.

## Implementation

`/workspace/code/HipKittens/analysis/bf16_gemm/mi350x/kernel_bf16_dynamic.cpp`
(NOT committed — falsified):

```cpp
        DO_MMA(C_accum[1][1], A_tile, B_tile_1, C_accum[1][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
-       __builtin_amdgcn_sched_barrier(0);
+       if constexpr (!FUSED_KTAIL) {
+           __builtin_amdgcn_sched_barrier(0);
+       }
    }

    /********** Round-4 path A: fused K-tail epilog (RCR only) **********/
    if constexpr (FUSED_KTAIL) {
```

Resource impact (pre-R57 vs post-R57): `grouped_kernel<RCR, 0, FUSED=1>`
unchanged at SGPR=96, VGPR=248, 0/0 spill, occupancy=2. Non-FUSED
hot KIs (RCR KI=48/64/88/112) all unchanged (gate is constexpr-false
for those templates).

## Correctness

`/tmp/probe_r55_correctness.py` (5 representative shapes, SNR threshold
30 dB):

```
Qwen3-Down-B16-M2048(KI48-RCR)             47.83 dB  allclose=True
Qwen3-Down-B32-M4096(KI48-RCR)             47.83 dB  allclose=True
gpt_oss-Down-B4-M2048(K2880)               47.86 dB  allclose=True   ← FUSED=true path
DSV3-GateUP-B32-M2048(KI112)               47.85 dB  allclose=True
Qwen3-GateUP-B16-M2048(KI64)               47.83 dB  allclose=True
```

47.83-47.86 dB = bf16-rounding floor.

## Metric (10×R57 paired with 10×baseline, GPU 3, single shell session)

| batch        | runs                              | median | mean  |
|--------------|-----------------------------------|-------:|------:|
| R57 batch 1  | 926, 935, 903, 932, 909           |  926   | 921.0 |
| Baseline 1   | 918, 895, 897, 919, 925           |  918   | 910.8 |
| R57 batch 2  | 912, 906, 909, 924, 900           |  909   | 910.2 |
| Baseline 2   | 918, 936, 912, 928, 922           |  922   | 923.2 |
| **R57 all**  | (10 runs)                         | **910.5** | **915.6** |
| **Base all** | (10 runs)                         | **918.5** | **917.0** |
| **Δ**        |                                   | **−8** | **−1.4** |

Pair-by-pair (interleaved, not strictly paired):
+8, +40, +6, +13, −16, −6, −6, −30, −3, −22 → 4 positive / 6 negative.
First batch's +8 median was contention-favored; the larger sample is
clearly **noise-level neutral / slightly negative**.

## Why R57 was NEUTRAL (most likely explanation)

LLVM was likely **already overlapping** the FUSED_KTAIL loads with
EPILOG 2's last MMAs DESPITE the sched_barrier(0). The sched_barrier
mask 0 prevents reordering of all instruction CLASSES, but the LLVM
GCN scheduler may treat in-flight buffer_load_b128 instructions issued
EARLIER in the EPILOG 2 block (e.g., from the `load_a_subtile` LDS
load) as the "scheduling boundary marker", and the FUSED_KTAIL HBM
loads remain in their textual position past the s_barrier (a hardware
op that cannot move) — already in the MMA shadow.

The s_barrier() at line 785 is the actual schedule wall here — the
sched_barrier(0) at 786 is structurally redundant for the FUSED=true
case, neither blocking helpful reorder nor enabling any new one.

## Bookend: levers that CANNOT help K-tail more (analytical bound)

K-tail block per-iter cost analysis (gpt_oss K=2880, fuse path):

* HBM traffic: 20× `buffer_load_b128` per lane = 320 bytes/lane.
  Main loop: 44 K-tile passes × ~24 b128/pass = ~1056 b128 = ~16.5KB/lane.
  → K-tail = 320/16800 = **1.9 % of HBM traffic**.
* MMA cycles: 4 MFMA per lane vs main-loop 352 MFMA per lane.
  → K-tail = **1.1 % of MMA work**.
* vmcnt(0) latency: 2× ~50-cycle waits (best case) = 100 cycles ≈ 0.2 µs/iter.
  Per-iter wall ~150 µs → **0.13 %**.

**Bound: even fully eliminating the K-tail block lifts gpt_oss family
by ≤ 2 %**, i.e. ratio 0.846 → 0.863 — still nowhere near 1.25 target.
The K-tail block is structurally NOT the dominant overhead for the
worst-performing metric shape; further FUSED_KTAIL micro-optimizations
should be deprioritized.

## R58 next-action surface

The R57 falsification + R56 PMC diagnostic narrow the dominant
overhead for `gpt_oss-GateUP-B4-M2048` (worst shape, ratio 0.846,
weight 3) to one of:

1. **Per-block fixed overhead amortization** — at 2.88 iters/block,
   the persistent kernel pays 1× prologue (cumsum scan, SRD setup,
   chiplet swizzle) + 1× C-store-finalize over only 2-3 work iters.
   Per-iter wall ≈ 150 µs; fixed overhead estimated ~30-50 µs (R56
   model), → 17-25 % wall overhead/block. Compare to B=32 M=2048
   (23 iters/block): same fixed cost amortized over 23 → 1.5-2.5 %
   overhead. **Largest leverable gap.**

   Sub-levers:
   - (a) Hoist `s_cum_tiles` cumsum scan from kernel-entry (LDS init,
     1 thread, O(G) scan) to host-side (G is known on host: g.G).
     But: device-side scan reads `g.group_offs` HBM — replicating on
     host requires HBM→host sync, BANNED. Would need the dispatcher
     to be passed the cum-tiles array via a separate kernel launch
     OR via Primus-Turbo storing it in the GroupedGEMM context. Both
     are infrastructure-heavy but legitimate.
   - (b) Skip the cum-tiles scan ENTIRELY for uniform-group cases:
     each group has the same M_g, so `s_cum_tiles[gi] = gi *
     (M_g/BLOCK_SIZE) * num_pid_n`, computable from `g.M_total / g.G`.
     But "uniform groups" check is BANNED (FROZEN list "no uniform
     judgement").
   - (c) Reduce SRD setup count: currently 2 SRDs (a_srsrc_base
     once, b_srsrc_curr cached on group change). Already minimal.
   - (d) Defer chiplet_transform_chunked(blockIdx.x) to a precomputed
     LDS lookup — but blockIdx.x is constant per block and the
     transform is ~5 cycles. Negligible.

2. **Wave-imbalance work stealing** — for tiles=736, NUM_CUS=256,
   32 CUs do 2 iters and 224 do 3 iters (R56 model). Imbalance =
   1 iter / 3 iters = 33 % of the trailing CUs' wall, 12 % of total
   wall. Work stealing (atomicAdd-based global tile counter)
   eliminates this entirely. Implementation cost: 1 device atomic
   counter (zero-init in dispatcher), 1 atomicAdd per persistent
   iter (~50 cycles, negligible). Risk: atomic contention across
   256 CUs serializes counter access, may add 256× ~50 = 12.8K
   cycles per workload. Net win unclear. **Worth testing.**

3. **Smaller M-tile for low-tile workloads** — BLOCK_SIZE=128 on M
   doubles tile count, makes wave imbalance smaller in absolute
   terms. But HALF_BLOCK_SIZE=64 cascades through the entire
   kernel (LDS layout, register tiles, MMA shape). Major surgery,
   high risk of breaking RCR/RRR/CRR layout invariants. Defer.

**Recommended R58**: option 2 (work stealing). Smallest surface
change, falsifiable in 1 round, addresses an R56-confirmed dominant
overhead.

## Outcome

* No kernel commit. Working tree reverted to R55 baseline (HK SHA
  237ca6b1).
* This doc + R52-stale-claim correction + R58 lever ranking is the
  R57 deliverable.
