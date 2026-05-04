# Round 75 — BF16 grouped bwd var-K work-stealing FALSIFIED (noise band, +1 VGPR spill)

**Status:** FALSIFIED — atomic-claim work-stealing on `grouped_var_k_kernel`
moves the score by **Δ -2 within ±2 noise band**, with a +1 VGPR spill
on the static-stride path that explains the small drift. Reverted to
baseline.

## Hypothesis (R75 lever)

R74 closed the var-K LDS-padded swizzle direction (P1 from R72/R68)
because the single-line shape swap broke 24/24 correctness. The
remaining R72 priority list:
- P1: var-K CRR LDS swizzle (R74 falsified)
- P2: FUSE pipeline prefetch (VGPR spill risk per R67)
- P3: `__launch_bounds__` (R73 falsified — already at max)

**R75 new lever (orthogonal to P1-P3):** mirror the R61 fwd-grouped
work-stealing (atomic-claim persistent loop) into `grouped_var_k_kernel`.
Same predicate (`tiles < NUM_CUS*4 && tiles % NUM_CUS != 0`) — only
fires for the 2 shapes whose dB var-K total_tiles is in the
sub-NUM_CUS*4 imbalanced regime:

| Shape | total_tiles | tiles/CU | imbalance % |
|---|---|---|---|
| gpt_oss-Down-B4-M2048 dB | 4×12×12 = 576 | 2.25 | 33 % CUs do 3, 67 % CUs do 2 |
| gpt_oss-Down-B4-M4096 dB | 4×12×12 = 576 | 2.25 | (same) |
| gpt_oss-GateUP-B4-M*  dB | 4×23×12 = 1104 | 4.31 | > NUM_CUS*4 → no gate |
| Qwen3-Down-B16 dB | 16×16×6 = 1536 | 6.0 | > NUM_CUS*4 → no gate |
| (all other 22 shapes) | > 1536 | > 6 | → static partition |

Lowest-progress shape coming in: gpt_oss-Down-B4-M2048 (ratio 1.051,
progress 0.841, weight 3x). Expected lift on the 2 gate-firing shapes:
3-10 % via the same mechanism that R61 captured for the fwd kernel
(+25/+33% on tiles=384/736 fwd cases).

## Implementation

`/workspace/code/HipKittens/analysis/bf16_gemm/mi350x/kernel_bf16_dynamic.cpp`:

1. Added `int* tile_counter` field to `grouped_var_k_layout_globals`
   (mirror the R61 fwd struct field). nullptr-default for legacy
   direct callers.
2. Added `__shared__ int s_claim` in `grouped_var_k_kernel` — coupled
   to a kernel-entry seed claim (atomicAdd or chiplet swizzle) and a
   per-iter advance (atomicAdd or +NUM_CUS).
3. Refactored the persistent loop from
   `for (gt = pid; gt < total_tiles; gt += NUM_CUS)` →
   `int gt = pid; while (gt < total_tiles) { ... }` with the advance
   block at the bottom and a `bool do_body` flag wrapping the body
   (since the original code's `continue` paths needed to advance gt,
   and the for-clause did this implicitly).
4. Added `should_use_var_k_work_stealing(total_tiles)`,
   `grouped_var_k_tile_counter_buffer()`,
   `prime_grouped_var_k_tile_counter(g)` host helpers (mirror R61's
   `prime_grouped_tile_counter` — separate counter buffer to allow
   independent fwd / var-K stream issuance).
5. `dispatch_grouped_var_k` calls `prime_grouped_var_k_tile_counter`
   inside `launch_grouped_var_k` before the launch.

## Build resources

| Kernel | TotalSGPRs | VGPRs | ScratchSize | VGPR Spill | Occupancy |
|---|---|---|---|---|---|
| Baseline `grouped_var_k_kernel<0>` | 95 | 256 | 0 B/lane | 0 | 2 waves/SIMD |
| R75 `grouped_var_k_kernel<0>` | 100 | 256 | 8 B/lane | **1** | 2 waves/SIMD |

The `do_body` refactor (and the additional `__shared__ int s_claim` +
the per-iter atomic-vs-static branch on `g.tile_counter`) increased
TotalSGPRs by 5 and introduced a single VGPR spill (8 bytes/lane
scratch). The compiler appears unable to fold the `bool do_body`
register cleanly into the existing pid_m / pid_n register lifetime.
Occupancy unchanged at the per-SIMD max.

## Metric (5-run paired test)

```
                    run-1  run-2  run-3   run-4  run-5  mean  range
Baseline            874    875    873     —      —      874   2
R75 (work-stealing) 872    873    871     —      —      872   2
Δ                   -2     -2     -2     —      —      -2    0
```

**Δ_mean = -2, range matched at 2.** Within the ±5 noise band but
**directionally negative**. No improvement.

## Per-shape breakdown (R75 vs baseline, single-run snapshot)

```
                                   baseline  R75    Δ ratio
gpt_oss-Down-B4-M2048    (WS-ON)    1.051    1.051   0.000   ← target, no improvement
gpt_oss-Down-B4-M4096    (WS-ON)    1.103    1.092  -0.011   ← target, slightly down
gpt_oss-GateUP-B4-M2048  (WS-OFF)   1.081    1.069  -0.012   ← static, +1 VGPR spill drag
gpt_oss-GateUP-B4-M4096  (WS-OFF)   1.110    1.105  -0.005
gpt_oss-Down-B32-M2048   (WS-OFF)   1.054    1.049  -0.005
DSV3-GateUP-B16-M4096    (WS-OFF)   1.137    1.144  +0.007   (noise)
```

The 2 WS-ON target shapes are flat / slightly down. The WS-OFF static
shapes are slightly down by 0-1 % — consistent with the +1 VGPR spill
on the kernel hurting all shapes uniformly.

## Why work-stealing didn't help

R61's win on the FORWARD `grouped_kernel` was +16-20 score because:
1. fwd kernel's per-tile body is large (full-K-loop GEMM at K=2880),
   so a 33 % imbalance penalty translates to a real wall-second cost
   that work-stealing recovers.
2. fwd has lower per-block VGPR pressure (256 cap minus epilog), so
   work-stealing's +5 SGPR cost folds in cleanly.

The var-K kernel differs:
1. **Per-tile body is smaller**: ki_g = M_g/K_STEP = 32 K-iters for
   M=2048 (vs ki_max=KI_HINT=44/48/64/88 in fwd FUSE path). Per-tile
   wall is ~50 % of fwd's, so a 33 % imbalance penalty is ~16 % of
   the original wall — but the **per-tile prologue cost** (LDS-cache
   group_offs scan + chiplet pid_m/pid_n compute + per-group_idx
   accumulator zero) is almost the same as fwd, so the
   imbalance-recoverable fraction shrinks proportionally.
2. **VGPR ceiling already pinned at 256** (vs fwd's 254-256 range).
   Adding the work-stealing state pushed 1 VGPR over the per-wave
   cap into scratch — direct wall cost on every iter (extra
   ds_read/ds_write per iter).
3. **Atomic counter contention at NUM_CUS=256**: 256 blocks racing
   on a single 4-byte atomic counter on every tile boundary. For
   total_tiles=576, ~600+ atomic ops; on the small per-tile body
   the atomic latency (8-12 cycles uncontended; ~30-50 cycles
   contended at NUM_CUS=256) is a non-trivial fraction of the
   per-tile wall. R65's per-XCD-counter attempt was meant to
   address this but introduced its own VGPR spill.

The net of these three: WS's wall-balancing benefit ≈ WS's per-tile
overhead, and the +1 VGPR spill tips it slightly negative.

## Reverted to baseline

- HK: `git checkout -- kernel_bf16_dynamic.cpp`, rebuilt, baseline
  resource report restored (VGPR Spill: 0, ScratchSize: 0).
- Baseline metric verified after revert: 873/875/874 (mean 874).
- No code committed in HK; no Primus-Turbo code changed; this round
  note is the only artifact.

## R76 direction

R74 + R75 have now closed both the LDS-padded-swizzle (P1) and the
work-stealing (R75-only, structural-orthogonal-to-P1-P3) levers. The
remaining levers from R72:

**P2 (FUSE pipeline prefetch — R76 candidate, +5-10 expected):**
In the FUSE K-tail epilog (`kernel_bf16_dynamic.cpp:808-946`),
overlap slab-1 A-HBM-load with slab-0 MMAs. Risk: VGPR spill on
the already-tight FUSE path (R67 found KI=44 close to threshold).
Need a SASS-level prefetch order audit before patching.

**Lever B2 (occupancy / `__launch_bounds__` for fwd grouped_kernel
— NOT yet probed for KI=44 fuse, only var-K):** R73 closed P3 for
var-K. The fwd grouped_kernel's KI=44/48/64/88 specs may have
different occupancy profiles. Need a per-KI resource report scan
before committing to a probe.

**Lever C1 (sched_barrier extension to MAIN — *uncovered*):**
R55 extended sched_barrier(0) to EPILOG 1/2 (+9.5 median). Did
NOT extend to the inner main loop's MFMA scheduling. R55 note
hints "doubling density in main_loop_iter from R54 was a +5/+7
score win" — there may be more density-headroom in the body.

**P3-bis (alternative — re-examine R65 per-XCD-counter):**
R65 closed per-XCD-counter for the FORWARD grouped_kernel. The
analysis applied: variance was not atomic-contention-driven,
+5/+20 VGPR spill on hot KIs. But R75 just confirmed the
var-K kernel is contention-sensitive (own spill +1 even with the
simple atomic). Maybe a per-XCD-counter on var-K alone (which has
no FUSE/KI multiplicities) can reduce contention without VGPR
penalty. Risk: same kind of drift R65 saw on fwd (+0.x within
noise). Lower priority than P2/B2/C1.

## Compliance check

* No metric file modified.
* No `can_handle` tightening.
* No CPU sync / host-pad / per-group launch introduced.
* HIPKITTEN registered with `autotune=False`.
* All 24 shapes correctness PASS during patched run (correct_fail=0).
* HK working tree clean after revert; PT working tree has only
  this round note.

## Chat-window note

R75 used a fresh resume window. Chat at ~30 min on entry; R76 will
likely be a continuation in the same chat. R74 → R75 transition
worked smoothly via the round-note path.
