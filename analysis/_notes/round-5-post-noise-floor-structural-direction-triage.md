---
name: round-5-post-noise-floor-structural-direction-triage
description: R5 triage of the 5 remaining structural directions (A1/A3/B/E/G) after R29 noise-floor closes the dispatcher/macro micro-tweak axis. Closes B and G a-priori; ranks A1, A3, E by EV vs round-cost; picks A3 (decoupled-warps producer-consumer) for R6 entry.
type: project
---

# Round-5 — Post-noise-floor structural direction triage

## TL;DR

After R29's 23-sample noise-floor study (cluster σ ~3-4 score, ±5 = noise,
±15 = minimum detectable single-sample effect), the past 4 rounds of this
session (R1-R4) confirm operationally that **dispatcher/macro micro-tweaks
are unidentifiable from noise**:

  | R | commit                                 | metric | type             |
  |---|----------------------------------------|--------|------------------|
  | 1 | 301c51c  R47 var-K uniform-M decode    | 692    | docs FALSIFIED   |
  | 2 | 5e20a3e  R48 var-K K-loop asm inv      | 695    | docs FALSIFIED   |
  | 3 | f3e151b  R46-port xcds=8 GateUP-B32-M4096 | 694 | perf +0.78 % cell |
  | 4 | d15d718  RCR_KTAIL_VMCNT=16            | 693    | docs FALSIFIED   |

  Δ across 4 rounds = ±2 score = pure noise. R3 shipped a +0.78 % single-
  cell win that mathematically should net +0.5 score, but the daemon's
  693 vs 695 reading cannot distinguish that signal from cluster
  variation. **All five available levers within the dispatcher/macro/
  wait-counter/template axis are now operationally exhausted.**

Per task md "TARGET: 900 score (per user, 2026-05-09). NV achieves this
same shape × same hardware. Gap from 696 = +204 score = kernel-template /
algorithm-level breakthroughs required, NOT dispatcher tweaks (those
have been audited)."

This round produces the triage that converts the task md NEW DIRECTIONS
list (A-G) into a ranked round-6+ work program with concrete entry
plans, after closing two of the seven directions a-priori.

## Closed a-priori this round

### Direction B (cross-stream parallelism) — CLOSED

Premise: dgrad and wgrad on separate streams could give nearly-2× wall
on bwd sections.

**Inapplicable to the metric.** `_metric_gpt_oss_fp8_kernel.py:130-196`
times each section's GEMM via `hk_ratio._time_op(_call)` where
`_call()` is a single `grouped_gemm_fp8_impl` invocation. The TFLOPS
formula `2*B*M*N*K / ms / 1e9` measures one kernel call's wall, not
section concurrency. Cross-stream parallelism would change end-to-end
training wall but yield zero TFLOPS lift on this metric.

The task md flagged this risk explicitly ("Verify the metric script's
behavior first — if it times them serially with explicit syncs, this
won't help"). Confirmed: serial single-call timing per section. **B
yields 0 score.**

### Direction G (cross-shape co-optimization via default tweaks) — LOW EV

Premise: a uniform default change that nets +X % across multiple shapes
could win even if it's −0.5 % on one.

`primus_turbo/pytorch/kernels/hipkitten/config.py` already has dense
per-cell coverage for the 8 gpt_oss FP8 shapes (RCR rules at ~1303-1660,
RRR at ~2230-2570, var-K CRR at ~2440-2620). The only un-overridden
defaults are `_FP8_DEFAULT_GROUP_M = 4` and `_FP8_DEFAULT_KERNEL = None`,
both of which apply only to cells NOT covered by per-shape rules — i.e.
**not the 8 metric shapes**.

Cross-shape co-opt would therefore require either (a) a brand-new
predicate that overrides multiple per-cell rules with a unifying rule
(but the per-cell rules were tuned individually as winners — replacing
them with a single rule mathematically can't beat the union of their
individual optima), or (b) an algorithmic change that changes what
"optimum" means per cell, which is just A1/A3/E by another name.

**G is dominated by A1/A3/E**, not a separate axis. Closed.

## The three remaining viable directions

### A1 — Stream-K / persistent + work-stealing

**Mechanism**: replace static slot assignment in the persistent grouped
kernel with dynamic work-stealing. Decouples per-CTA work from CTA count;
reduces tail effects on small grids.

**Where it bites — tile-count audit per shape (BLK=256, MI355X SMs=304)**:

  | shape                   | total_tiles | tiles / 304 | tail wave fill | tail effect |
  |-------------------------|-------------|-------------|----------------|-------------|
  | Down-B4-M2048           |   352       | 1.16        | 48/304 = 16 %  | **HIGH**    |
  | Down-B4-M4096           |   704       | 2.32        | 96/304 = 32 %  | **HIGH**    |
  | GateUP-B4-M2048         |   704       | 2.32        | 96/304 = 32 %  | **HIGH**    |
  | GateUP-B4-M4096         |  1408       | 4.63        | 192/304 = 63 % | medium      |
  | Down-B32-M2048          |  2816       | 9.26        | 80/304 = 26 %  | low         |
  | Down-B32-M4096          |  5632       | 18.5        | 160/304 = 53 % | low         |
  | GateUP-B32-M2048        |  5632       | 18.5        | 160/304 = 53 % | low         |
  | GateUP-B32-M4096        | 11264       | 37.1        | 32/304 = 11 %  | low         |

  (B=4 cells: 1-2 waves only — last wave runs at 16-63 % CU occupancy
  for ~half of the kernel runtime → tail dominates.)

**Score envelope**: the 4 B=4 cells are 4/24 of the total metric weight
(8 shapes × 3 sections, but the per-section means are independent).
Stream-K's win mechanism = recover the idle CU fraction in the trailing
wave. Pessimistic upper bound on B=4 cells: +20 % of the 16-32 % idle
fraction = +3-10 % per cell, × 4/24 weighting = **+15 to +50 score
envelope**. Crosses the 1σ noise floor cleanly.

**Cost**: 2-4 rounds (preflight + atomic-counter scaffolding + steal
loop + bit-equiv + metric).

**Risk**: the only existing atomicAdd usage in HK is R61's tile-counter
work-stealing for the fused-act post-kernel (`tile_counter` int32
compare-and-claim — see R33 doc). Stream-K's dynamic dispatch loop is
structurally similar and **would reuse that exact primitive** rather
than build new infrastructure. R33's split-K closure cited "no precedent"
as a falsification reason; Stream-K does not have that problem because
it uses int32 control-flow atomics, not value-accumulator atomics.

### A3 — Decoupled-warps / producer-consumer

**Mechanism**: dedicate 1-2 warps in each CTA to HBM→LDS load (producer),
rest to MFMA (consumer). Decouples load latency from MFMA pipe.
Avoids the CTA-wide `s_barrier()` that pins the schedule.

**Where it bites — direct attack on R21 etiology**:

R21 PMC identified all 6 measured cells as **issue-rate bound** with
60-70 % of cycles completely idle and MfmaUtil 31-49 %. Root cause:
the 4× CTA-barrier-per-iter schedule pin (R21 doc, still standing).
LDS/SQ_busy is 31-70 %, SALU/SQ_busy 39-85 % — the SIMD is busy doing
non-MFMA work (LDS load issue, address arith, barrier wait) and the
MFMA pipe sits idle.

A3 directly removes the barrier pin: producer warps issue HBM loads
ahead of consumer demand; consumers just check `lds_ready[stage]` flag
and proceed. The 4× CTA barriers per iter become 0 CTA barriers
(replaced by per-stage flag spin or s_setprio handoff). Per the
prevailing literature (CUTLASS ping-pong, ThunderKittens warpgroup
specialization), expected MFMA-pipe idle reduction is 30-50 %, which
maps to ~+15-25 % per-cell TFLOPS lift. With the lift applying to ALL
8 shapes × 3 sections (the bottleneck is universal), the score envelope
is **+100 to +200 score** — by far the largest of any candidate.

**Cost**: 4-6 rounds (warp-role taxonomy + LDS double-buffer flag scheme
+ producer body + consumer body + bit-equiv + tuning).

**Risk**: high implementation complexity. The kernel currently uses
8 warps with WARPS_M=2, WARPS_N=4. Splitting into producer (1-2 warps)
+ consumer (6-7 warps) reshapes register/LDS budget. The kernel body
is 700+ LOC across `kernel_fp8_layouts.cpp` (`grouped_rcr_kernel`
lines ~3438+, `grouped_var_k_kernel_fp8` line 8204+). Existing
`s_setprio` instrumentation (line 368-369, 2068) is partial scaffolding
only; full producer/consumer requires LDS-flag synchronization not yet
present. R26-R28 single-barrier-drop attempts all failed (per FORBIDDEN
PATHS), confirming the barrier scheme is non-trivially load-bearing.

### E — Different barrier scheme (s_setprio + per-warp-group)

**Mechanism**: replace CTA-wide `s_barrier()` with `s_setprio` priority
hints + per-warp-group sync (using HW LDS flag or asm scratch).

**Status**: partial — the kernel already uses `s_setprio(1)/s_setprio(0)`
to bracket MFMA issue (line 2068, `CRR_MMA_BEGIN/END` macros at
line 368-369). The remaining 215 `__builtin_amdgcn_s_barrier()` call
sites in `kernel_fp8_layouts.cpp` are CTA-wide. Direction E is therefore
an **incremental-barrier-replacement** axis — pick individual barriers,
prove they can be replaced with warpgroup-local sync without breaking
correctness, repeat.

**Cost**: 3-5 rounds, but the prior R23-R28 single-barrier-drop work
(four falsified rounds) suggests that EACH barrier replacement is its
own multi-sample falsification effort. The score envelope per barrier
is bounded by the single-iter barrier latency contribution to MFMA
idle: ~10-30 cy/iter × 5-15 % of total cycles ≈ ±0.5-2 % per cell, i.e.
**+5 to +20 score per replaced barrier**. To net +200 score this would
need to replace 10-40 barriers, which exceeds the implementation budget
without producing more lift than A3's single restructure.

**E is dominated by A3** (same mechanism end-state, but A3 makes the
restructure once vs E making it 10-40 times). Keep E in inventory but
DEPRIORITIZE.

## Triage scoreboard

| Direction | EV envelope | Cost (rounds) | EV / round | Risk | Prerequisite |
|---|---|---|---|---|---|
| **A3 decoupled-warps** | +100 to +200 | 4-6 | +20 to +50 | high | none |
| A1 Stream-K            | +15 to +50   | 2-4 | +4 to +25  | medium | re-check R61 atomic primitives |
| E barrier-by-barrier   | +5 to +20 / replacement, ~10-40 needed for +200 | 10-40 | +0.5 to +5 | medium | dominated by A3 |
| B cross-stream         | 0            | -   | -          | -    | metric-incompatible (CLOSED) |
| G cross-shape co-opt   | dominated by A1/A3/E | - | - | - | (CLOSED) |

## Round-6 entry plan — A3 (decoupled-warps producer-consumer)

The entry round (R6) does NOT touch the kernel body. It produces the
**warp-role taxonomy preflight** that decides whether A3 is
implementable on the existing 8w / WARPS_M=2 / WARPS_N=4 / 256 VGPR
budget without busting register/LDS limits. Concretely R6 should:

1. **Enumerate producer-consumer split candidates** for `grouped_rcr_kernel`:
   - 1 producer warp + 7 consumer (WARPS_M=2, WARPS_N=4 → 1P + 7C is
     odd; 2P + 6C maps cleanly to {1×WARPS_M, 3×WARPS_N} consumer grid).
   - 2 producer + 6 consumer (preferred — geometry aligns).
   - Compute per-warp register budget impact: producer needs HBM tensor
     descriptors only (~32-48 VGPR); consumer needs accumulator
     fragments (current 256 VGPR / 8 warps = 32 VGPR/lane; consumer
     would jump to 256/6 ≈ 43 VGPR/lane → still under 256-lane limit
     but reduces occupancy 8 → ~6 waves).

2. **LDS-flag synchronization scheme** — sketch the multi-stage LDS
   buffer with per-stage `ready` and `consumed` flags. AMD CDNA has no
   HW barrier-arrive primitive equivalent to NVIDIA mbarrier; the AMD
   pattern is `__atomic_store_n(&lds_flag, 1, __ATOMIC_RELEASE)` for
   producers + `while(__atomic_load_n(&lds_flag, __ATOMIC_ACQUIRE) == 0)`
   for consumers. Round-trip latency on LDS atomic is ~16-24 cy on
   CDNA3/4 — well below the ~120 cy HBM-load latency we're trying to
   hide.

3. **Occupancy + barrier-count delta projection** — current 8w kernel
   has 4 `s_barrier()` per main-loop iter. Producer-consumer end-state
   has 0 CTA-wide barriers per iter (only LDS flag wait, which is a
   per-warp polling loop). With ~10-30 cy/barrier × 4 = 40-120 cy/iter
   freed, against the current main-loop iter cost of ~140-200 cy
   (estimated from R21 PMC: MfmaUtil 35 % means MFMA is 35 % of
   ~400 cy/iter at 4 mfma × ~35 cy each = 140 cy MFMA), the iter cost
   reduction is **+25-60 %**, which matches the +15-25 % per-cell TFLOPS
   estimate after dispatch-/store-/prologue-overhead amortization.

4. **Falsification gate**: if any of the three sub-points above can't
   be made to fit (register pressure pushes spill > 50, LDS scheme
   needs >2× LDS budget, barrier-count delta < +20 % of iter cost),
   **document the preflight FALSIFICATION** and rotate to A1 (Stream-K)
   for R6. Do NOT commit a kernel-body edit until preflight clears all
   three gates.

R7+ executes the implementation: M1 type system, M2 producer body,
M3 consumer body, M4 LDS flag scheme integration, M5 correctness
(7-seed × 1500-iter bit-equiv + SNR), M6 metric (10-sample dbg_remote
to clear noise floor).

## Code state this round

No HK changes. No PT changes. Single docs commit. Bit-equivalent to R4
(d15d718).

Daemon's R5 metric will be sample #N+1 in the same noise distribution
as R1-R4: prior probability ≈ 9 % it lands in tail-mode (per R29). If
the daemon prints anything below 685, it's noise (not a regression);
above 705, also noise (not a win).

## Why this round is not a wasted commit

Past sessions have repeatedly attempted ad-hoc structural directions
(R32 alt-tile, R33 split-K, R39b 4w port, R40 2cta-per-tile, R41 MFMA-
32x32x64, R42 var-K RCR variant) and falsified each one a-priori
before reaching code. The pattern is: pick one direction → preflight
→ falsify → forward-pointer → next round picks the forward-pointer
→ falsify → ... — five rounds spent on five preflight falsifications
that, in retrospect, were all dominated by the same "per-tile fixed
overhead doesn't shrink with tile" or "no precedent / multi-round
build" mechanisms.

This round produces the **direction-level decision tree** instead of
yet another sub-direction preflight. R6+ executes A3 directly, with
A1 as the documented fallback if A3 preflight gates fail. No more
direction dithering.

## Forward pointers

* **R6**: A3 preflight — register/LDS/barrier-count audit per
  enumerated split (1P+7C, 2P+6C). Forward-pointer to R7 if all
  three gates clear; rotate to A1 if any gate fails.
* **R7+**: A3 implementation, OR A1 preflight + implementation if R6
  rotated.
* **R12+ (if A3 ships)**: re-do PMC on the post-A3 kernel; check whether
  MFMA-util pushes from current 31-49 % toward 70-80 %; if not, the
  remaining gap is structural compute-pipe (MFMA chain length, wave
  occupancy) and E (incremental barrier replacement) becomes a
  cleanup pass for any residual barriers.
