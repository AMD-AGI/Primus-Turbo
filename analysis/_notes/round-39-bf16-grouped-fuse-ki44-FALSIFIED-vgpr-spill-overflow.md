# Round-39 BF16 grouped — fuse KI=44 specialization FALSIFIED

## Lever

Add `KI_HINT=44` template specialization to `launch_one_grouped_fuse<L>` so
the gpt_oss forward main loop (and the gpt_oss-Down dA H4-rerouted RCR
fuse path) full-unrolls 21 `main_loop_iter` calls instead of running with
`#pragma unroll 2` over a dynamic K count.

Rationale: the non-fuse switch in `dispatch_grouped<L>` already
specializes `KI_HINT ∈ {56, 64, 112, 128, 172, 224, 256, 296, 448, 462,
832}` and the fast K%128==0 metric shapes (DSV3-GateUP K=7168 → ki=112,
Qwen3-GateUP K=4096 → ki=64) sit on those compile-time cases. The fuse
path was the only K-tail target left at `KI_HINT=0` (dynamic), and the 12
gpt_oss-family shapes hitting fuse (8 forward + 4 Down dA via H4) are
weight-3 in the metric — biggest expected per-shape leverage.

## Mechanism (predicted)

Compile-time `KI_HINT > 0` enables `#pragma unroll` over `(num_tiles - 2)
/ 2` iters in the RCR/RRR branch (line 700). For ki=44 that's 21
unrolled `main_loop_iter` calls, eliminating the dynamic loop branch and
giving LLVM full instruction-level scheduling across all K-iterations.
The expected win was main-loop MFMA pipelining + load-issue interleaving
better than what 2x unroll achieves.

## Implementation

`/workspace/code/HipKittens/analysis/bf16_gemm/mi350x/kernel_bf16_dynamic.cpp`:

1. Added explicit instantiations:
   ```cpp
   template __global__ void grouped_kernel<Layout::RCR, 44, true>(...);
   template __global__ void grouped_kernel<Layout::RRR, 44, true>(...);
   ```
2. Made `launch_one_grouped_fuse<L>` templated on `KI_HINT` (default 0).
3. Replaced the unconditional fuse-launch in `dispatch_grouped<L>` with
   ```cpp
   switch (g.ki) {
       case 44: launch_one_grouped_fuse<L, 44>(g); break;
       default: launch_one_grouped_fuse<L, 0>(g);  break;
   }
   ```

## Falsification (mechanism — register pressure overflow)

Build resource report (`-Rpass-analysis=kernel-resource-usage`):

| Kernel                        | VGPRs | VGPR Spill | ScratchSize/lane | Occupancy |
|------------------------------|-------|-----------|-----------------|----------|
| `RCR, 0,  true`  (KI=0 fuse) | 248   | **0**     | 0               | 2        |
| `RRR, 0,  true`              | 249   | **0**     | 0               | 2        |
| `RCR, 64, false` (existing)  | 256   | **0**     | 0               | 2        |
| `RRR, 64, false`             | 256   | **0**     | 0               | 2        |
| `RCR, 44, true`  (R39 NEW)   | 256   | **28**    | 116 B/lane      | 2        |
| `RRR, 44, true`  (R39 NEW)   | 256   | **30**    | 124 B/lane      | 2        |

The fuse template (`FUSED_KTAIL=true`) has a K-tail epilog block that
loads `A_tile`, `B_tile_0`, `B_tile_1` for the K-tail K-iteration and
accumulates into `C_accum[2][2]`. Those tile registers are live across
the entire main loop (LLVM keeps them in registers for the post-loop
epilog) → adds ~8 VGPRs of live state on top of what the non-fuse
template needs. Combined with full-unroll over 21 `main_loop_iter`
(each holds its own `A_tile/B_tile_0/B_tile_1` for 2 K-tiles) the
register file ceiling (256) is exceeded → 28-30 VGPR spill.

Non-fuse `KI=64` doesn't pay this tax — same VGPR=256 ceiling but no
K-tail epilog block → 0 spill, clean full-unroll.

## Empirical impact

```
Baseline (KI=0 fuse):  score=886, gpt_oss geomean=1.0981, all 24 PASS
R39 KI=44 fuse spec:   score=845, gpt_oss geomean=1.0129, all 24 PASS
                       Δ=-41 points (kernel-side regression)
```

Per-shape gpt_oss damage:
```
  shape                                pre     post    Δ
  gpt_oss_20B-GateUP-B4-M2048         1.116   1.066   -5.0pp
  gpt_oss_20B-Down-B4-M2048           1.129   1.014  -11.5pp
  gpt_oss_20B-GateUP-B4-M4096         1.122   1.051   -7.1pp
  gpt_oss_20B-Down-B4-M4096           1.123   1.015  -10.8pp
  gpt_oss_20B-GateUP-B32-M2048        1.060   0.993   -6.7pp  *worst pre
  gpt_oss_20B-Down-B32-M2048          1.062   0.985   -7.7pp
  gpt_oss_20B-GateUP-B32-M4096        1.090   1.007   -8.3pp
  gpt_oss_20B-Down-B32-M4096          1.085   0.975  -11.0pp
```

Every gpt_oss shape regressed ≥5pp; Down family worst (-7.7 to -11.5pp,
matches Down's tighter HBM-band utilisation pre-change). 12-shape
regression for what should have been a precise +0.1-2pp scheduling gain
proves the spill cost dominates the unroll benefit.

## Implication for future rounds

1. **Do NOT** specialize the fuse template with `KI_HINT > 0` — the
   K-tail epilog block + main-loop unroll-21 jointly exceed the VGPR
   ceiling. Any future `KI_HINT > 0` for fuse needs a structural change
   (e.g., spill the K-tail data to LDS instead of registers, or split
   the K-tail block into a separate launch — undoing the fuse).
2. **Do** consider `KI_HINT > 0` for NON-fuse paths missing from the
   switch (ki=24 Qwen3-Down fwd, ki=32 DSV3-Down fwd, ki=48 Qwen3-GateUP
   dA RRR, ki=88 gpt_oss-GateUP dA H4 RCR). KI=64 non-fuse compiled
   clean (0 spill); KI=88 < KI=112 (existing case) so it should also
   compile clean. Expected per-shape lift is small (these shapes are
   fast K%128==0 path) but multi-shape coverage may aggregate to a few
   points.
3. **Pivot away from K-tail unroll** — R37/R38/R39 all confirm
   K-tail-related kernel-body changes are bounded by the small wall
   fraction of the K-tail (R38 falsification note Option C still
   pending). The remaining low-hanging fruit is on the K%128==0 path
   (DSV3/Qwen3 push, Lever B in the task body).

## Revert

`launch_one_grouped_fuse` returned to `template<Layout L>` form, no `g.ki`
switch in fuse dispatch, KI=44 explicit instantiations removed.
`git diff` of `kernel_bf16_dynamic.cpp` is empty after revert. Rebuild
verified score=879 (within ±5 noise of baseline 886; not committed
HipKittens-side).

## Suggestion for R40

Pivot to one of:
- **Phase B Lever** — `KI_HINT > 0` non-fuse cases for ki ∈ {24, 32, 48, 88}.
  Low risk (mirrors existing 56/64/112 pattern, no fuse epilog =
  spill-clean); modest expected lift (4 weight-1 fwd + 4 weight-1 dA
  shapes = 8 weight-1 shapes × 0.5-1.5pp each).
- **Option C from R38** — rocprofv3 marker-bracketed K-tail wall-fraction
  measurement on gpt_oss B=32 GateUP. Diagnostic round (no metric move)
  but tells us whether further K-tail attack is worth it.
- **Deeper analysis** — per-shape breakdown of fwd vs bwd wall
  contribution in the metric. If bwd dB var-K dominates, there's a
  separate code path to attack (`grouped_var_k_kernel`, currently 0
  spill / 256 VGPRs / 2 occ — unchanged for many rounds).
