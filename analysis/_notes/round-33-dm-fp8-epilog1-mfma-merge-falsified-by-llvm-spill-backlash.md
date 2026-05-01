# Round 33-dm (R6 DM probe) — FP8 grouped: epilog-1 stage-3/4 MFMA merge is CORRECTNESS-SAFE but falsified by LLVM VGPR-spill backlash on DSV3-Down (same failure pattern as R3 sched_group_barrier)

Status: **FALSIFIED** — correctness PASS across all 32 shapes, but score 917 → 911 (−6). DSV3-Down shapes regressed 3-6 pp each; geomean grp_FP8 dropped 1.0241 → 1.0118 (−1.2 pp). Reverted.

## What was changed

Single structural edit to epilog 1 of `grouped_rcr_kernel` in
`analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp` (~line 2205-2232):
the stage-3 `mfma(cC,a,b0)` and stage-4 `mfma(cD,a,b1)` — which share the
same `a` (loaded once in stage 3) and use different, earlier-loaded b-regs
(b0 from stage 1, b1 from stage 2) — were collapsed into a single setprio
bracket with one leading `s_waitcnt lgkmcnt(0)` and one trailing `s_barrier`.
The stage-4 `load_b(b0, b_tile(toc, 0), wn)` prefetch for epilog 2 moved
AFTER the merged mfma block; its ds_read is drained by epilog 2 stage 1's
own leading `s_waitcnt lgkmcnt(0)`.

Exactly the same pattern as epilog 2's own shipped merged-final-pair at
lines ~2250-2254 (which is byte-unchanged).

## Why this *should* have been a safe win

1. Epilog 1 does NOT contain the `if (wm == 1) __builtin_amdgcn_s_barrier();`
   staggered half-barrier idiom. The R31-dm/R32-dm prologue-invariant trap
   cannot apply here.
2. Data dependencies are transparent: both mfmas read the same `a`, different
   b-regs; no WAR/WAW hazards; LDS reads all drain by the leading lgkmcnt(0).
3. Epilog 2 already ships this exact pattern (final `rcr_mma(cC,a,b0); rcr_mma(cD,a,b1);` under one setprio). So the pattern itself is known-good at the
   hardware-correctness level.
4. Per-tile savings (on paper): −1 s_barrier, −1 s_waitcnt lgkmcnt(0),
   −1 setprio-bracket pair. That's ~3 cycles of serial latency per tile.

## Why it regressed anyway

Resource-usage comparison on the 4 `grouped_rcr_kernel` template specs:

```
Spec                   Baseline (ScratchSize / VGPRs-Spill) → Probe
<0, FUSED_KTAIL=0, LONG=0>    272 / 67  →  288 / 71   (+16 scratch, +4 spill)
<0, FUSED_KTAIL=1, LONG=0>    308 / 76  →  320 / 79   (+12 scratch, +3 spill)
<0, FUSED_KTAIL=0, LONG=1>    196 / 48  →  196 / 48   (unchanged)
<0, FUSED_KTAIL=1, LONG=1>    236 / 58  →  236 / 58   (unchanged)
```

The LONG=0 specs (short-K path used by DSV3) saw +3-4 additional VGPR spills.
Mechanism: merging stages 3+4 stretches `a`'s live-range from
"load → 1 mfma → spill-kill" into "load → 2 mfmas → spill-kill". LLVM's
greedy register allocator observes a longer interval on an
already-pressured path, chooses to spill an adjacent temp to stay within
the 256-VGPR/wave budget for 2-wave occupancy, and the spill's scratch-load
latency overwhelms the 3-cycle barrier saving.

Per-shape delta (baseline → probe):
```
DSV3-Down-B16-M2048    0.991 → 0.958   −3.3 pp
DSV3-Down-B16-M4096    1.008 → 0.956   −5.2 pp
DSV3-Down-B32-M2048    1.025 → 0.967   −5.8 pp
DSV3-Down-B32-M4096    1.014 → 0.955   −5.9 pp
DSV3-GateUP-B32-M2048  1.050 → 1.034   −1.6 pp
gpt_oss-GateUP-B32-M2048  0.982 → 1.002  +2.0 pp
(others within ±1 pp)
```

DSV3-Down (ki=16, short-K, per-tile overhead-dominated) gets hit hardest
because (a) those shapes use the LONG=0 spec where spills increased, and
(b) short-K means the spill-load surfaces as a larger fraction of
per-tile cost. gpt_oss-GateUP-B32-M2048 (LONG=1 spec, no spill change)
actually BENEFITED from the barrier save, supporting the diagnosis.

## Pattern match: this is the same failure as R3 (sched_group_barrier)

Both R3 and R33 show the same shape-class asymmetry:

| Probe | Spill delta | DSV3 wall-clock delta | gpt_oss wall-clock delta |
|---|---|---|---|
| R3 sched_group_barrier   | −1 to −10 per spec (WIN)     | −5 to −9.5 pp (LOSS) | small + or flat |
| R33 epilog-1 merge       | +3 to +4 per LONG=0 spec (LOSS) | −3 to −6 pp (LOSS) | +2 pp on 1 shape |

The deeper invariant: **LLVM's register allocator is not monotone in
our favor**. Any change that perturbs the live-range graph — even one
that REDUCES cycle-count on paper — can trigger a spill-pattern shift
that wipes out the gain on short-K shapes. Spill count alone is not
a predictor; the ALLOCATED per-VGPR LIFETIME is what matters, and
that is not exposed by `-Rpass-analysis=kernel-resource-usage`.

## Corollaries for future rounds

1. **Any mfma-merging change must be co-designed with the register-
   allocator hints.** A naked merge is insufficient. Candidate pairings:
   - Merge + explicit `__restrict__` on all pointers to collapse aliasing
     intervals. (Note: R28-dm already tested plain compiler hints as no-op;
     the combo with a structural change was not tested.)
   - Merge + `asm volatile` fence to pin `a` to a specific VGPR range.
   - Merge + splitting `a` into two aliased references (`a_c`, `a_d`) so
     LLVM sees two half-intervals instead of one long interval.
   None of these are cheap single-commit probes; each is a multi-round
   co-design.

2. **Epilog 1 cannot be attacked in isolation by structural simplification
   alone.** The theory-predicted win is real (~3 cycles/tile) but is
   smaller than the LLVM-induced tax (−3 to −6 pp on 4 DSV3-Down shapes
   × 2 = 8-24 pp of negative contribution to geomean).

3. **Barrier-count reduction is now 0-for-2 as a clean lever.** R3 and
   R33 were both barrier-reduction attempts at different regions (main
   loop via sched_group_barrier vs epilog 1 via structural merge); both
   falsified the same way (shape-asymmetric LLVM reg-alloc response).
   Barrier count and allocator behavior are coupled on the
   short-K/DSV3-Down path.

4. **The remaining viable levers (per SKILL.md "architectural rewrite
   required"):** Lever A (async global→LDS), Lever B (dual LDS ping-pong),
   Lever D (32x32x64 cell shape) — all of which intrinsically ALTER the
   register-allocation graph (different load/operand types entering
   MFMAs) rather than trying to game LLVM's existing response. These
   should be the priority for rounds 34-40+.

## Concrete commit trail

- HipKittens: no change (revert byte-identical to HEAD `19ce45a1`). Doc-only
  commit in Primus-Turbo.
- Post-revert metric: 914 (within R6 baseline noise of 915-917).

## Next-round recommendations

1. **ISA-inspection tooling FIRST.** Without seeing the emitted ISA for
   both spec variants (LONG=0 vs LONG=1), we cannot predict which
   structural change LLVM will respond to gracefully. Try `hipcc
   --save-temps` + `readelf -d` on the `.so`, or `/opt/rocm/llvm/bin/llvm-objdump
   --disassemble-all --triple=amdgcn--amdhsa --mcpu=gfx950` on the
   device-side `.o`. Even partial ISA coverage of the `grouped_rcr_kernel`
   epilog region would materially improve the odds of the next structural
   probe succeeding.

2. **Lever D (32x32x64 cell shape) with co-designed LDS/load**, matching
   what R15-dm couldn't achieve naively. Expected: the different mma shape
   produces a different register-tile count per mfma, which fundamentally
   redraws the live-range graph rather than nudging it. Multi-round.

3. **LOWEST-RISK fallback:** stop attacking barrier count entirely. Focus
   on data-flow paths where the load-to-use distance can be SHORTENED
   (opposite of what R33 tried) — specifically prologue→main-loop
   handoff latency (how long is the first `load_b/load_a` waiting after
   the prologue barrier triple?). But this has the R31/R32-dm prologue
   invariant trap to navigate.
