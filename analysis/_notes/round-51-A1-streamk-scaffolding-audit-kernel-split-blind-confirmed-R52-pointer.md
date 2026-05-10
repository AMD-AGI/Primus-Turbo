round-51-A1-streamk-scaffolding-audit-kernel-split-blind-confirmed-R52-pointer.md
====================================================================================

Round: 51 / 100
Date: 2026-05-10
Pre-SHA: a1309825 (R50 docs)
Task: gpt_oss_fp8_kernel_score (8 fp8 shapes, kernel-only TFLOPS)

## TL;DR

R50 forward-pointer asked R51 to audit the Stream-K (Direction A1)
scaffolding (HK ``bc5df92d`` ``43f37f8b`` ``4e9f6b62`` ``49ffb984``) +
PT-side WorkspaceCache (``grouped_gemm_fp8_impl.py`` lines 107-181)
to determine whether the kernel-side K-split branch — deferred to "R18"
in the parallel run that landed the scaffolding — has actually shipped,
or whether the existing scaffolding is alloc-only.

**Verdict: alloc-only. Kernel is split-BLIND.** Both code-read and
end-to-end probe agree:

  * **Code-read**: ``grouped_rcr_kernel`` body (``kernel_fp8_layouts.cpp``
    lines 3024-3742) contains ZERO references to ``sk_split_n`` or
    ``sk_partial_buf``. Verified via
    ``awk 'NR>=3024 && NR<=3742 && /sk_split_n|sk_partial_buf/{print}'``
    → no matches. The kernel ignores both fields completely.

  * **Empirical** (``scripts/_probe_round_51_streamk_audit.py``,
    Down_B4_M2048 fwd via grouped_rcr_dscale, GPU 3 / MI355X, R17
    caller-allocated workspace path):
    | sk_split_n | max_abs_diff | SNR (dB) | any_nan |
    |---|---|---|---|
    | 0 (baseline) | — | — | False |
    | 2 | 0.000000 | inf | False |
    | 4 | 0.000000 | inf | False |
    Bit-identical output across all sk_split_n values, no crash, no NaN.
    The 96.0 MiB caller-allocated workspace slab is wired through but
    the kernel never touches it.

This rules out the R50 hypothesis "If SNR > 25 dB, the kernel is
already split-aware" — bit-identical output here means split-BLIND,
not split-aware. A split-aware kernel writing partial-tile fp32
accumulators to ``sk_partial_buf`` without a paired reduce kernel
would either NaN (uninitialized partials read back) or produce wrong
answers (each split CTA writing 1/N of the result with no fold-down).
Bit-identity at the launched output dtype means the K-loop is the
unmodified single-CTA-per-tile reduction.

## What is in scaffolding vs. what is missing

LANDED (parallel-run R12-R17, HK + PT):

| Layer | Where | Status |
|---|---|---|
| ``grouped_layout_globals`` struct field ``int sk_split_n`` | HK l.2937-2969 | ✓ |
| ``grouped_layout_globals`` struct field ``int* sk_partial_buf`` | HK l.2944-2970 | ✓ |
| Host alloc/zero/free in ``dispatch_grouped_rcr`` | HK l.7884-7900, 8132-8138 | ✓ (gated, NEUTRAL) |
| Pybind kwarg ``sk_split_n`` on ``grouped_rcr{,_dscale}`` | HK l.9057, 9094, 9841, 9860 | ✓ |
| Pybind kwarg ``sk_workspace_ptr`` (caller-allocated) | HK l.9058, 9095, 9078-9079, 9114-9115, 9849, 9862 | ✓ |
| PT ``_FP8WorkspaceCache`` singleton + LRU | PT l.139-181 | ✓ |
| PT ``_fp8_sk_workspace_bytes`` mirror of HK formula | PT l.184-200 | ✓ |
| PT dispatch wires cache → ``sk_workspace_ptr`` kwarg | PT l.742-773 | ✓ (gated on ``cfg.sk_split_n > 0``) |
| ``HipKittenConfig.sk_split_n`` dataclass field (default 0) | PT config l.239 | ✓ |

ABSENT (R18 work that never shipped):

| Item | Required for | Notes |
|---|---|---|
| Kernel control-flow branch on ``g.sk_split_n > 0`` | A1 K-split working at all | The kernel must partition K-iters across CTAs, accumulate fp32 partials to ``g.sk_partial_buf`` via atomicAdd (or a per-tile lock-and-add), and skip the regular C-store on partial CTAs. |
| Per-tile reduce kernel | Folding fp32 partials → fp8/bf16 output | Must run after main kernel, reads ``sk_partial_buf``, writes the user-facing C tensor with the correct epilogue (rescale + cast). |
| Per-cell dispatcher rule that sets ``cfg.sk_split_n > 0`` | Activating A1 in production | Must be cell-conditioned: the R11 cost decomp shows projected lift on B=4 cells (23-42% per-cell theoretical) but only 0.04-3% on B=32 cells; activating universally would regress B=32. |

## Why the scaffolding stopped at R17

Reading the R11→R17 doc trail in
``analysis/_notes/round-{11,12,13,14,15,16,17}*A1prime*``:

  * **R11** (cost decomp): GREEN-LIT R12-R17, projected mean lift 13.9%
    (B=4 cells 23-42%, B=32 cells 3-7%), atomic contention <1%.
  * **R12-R13a**: struct fields + host alloc, NEUTRAL.
  * **R14**: alloc-cost probe falsified per-call ``hipMallocAsync`` at
    2.9-9.1 ms (vs ~3 µs R11 premise → would erase the entire envelope).
  * **R15-R17**: pivot to caller-allocated workspace via PT-side cache;
    R17 commit message states "The kernel control-flow K-split branch
    that consumes ``g.sk_partial_buf`` lands in R18; this commit only
    replaces the R13a alloc primitive with the R14-followup caller-
    allocated path. R19 ships the +25-30 verdict (or A1' falsification)."
  * **R18-R50**: R18+ never landed. The arc was abandoned mid-stream
    after the R17 NEUTRAL ship; subsequent rounds (R20-R50 on the
    fp8 task) explored other directions (kernel-template, dispatcher
    sweeps, macro flags, KREM spec).

## EV reassessment for actually doing R18+

Given the patience streak (9 rounds plateau at 691-699) and round
budget (51/100), the question is whether R52-R57 should land the
kernel branch + reduce kernel.

**Pros (per R11 cost decomp)**:

  * Theoretical max wall lift mean = 13.9% across 8 cells (B=4 cells
    23-42% individually).
  * Atomic-contention overhead bounded at <1% per tile.
  * Targets the *unaddressed* tail-wave underfill on small-grid B=4
    cells. Other levers (dispatcher, macro, kernel-template) don't
    move tail-wave wall-time.
  * Scaffolding cost already paid; only kernel branch + reduce
    remain.

**Cons (R51 reassessment)**:

  * R11 cost decomp models grid-occupancy lift, not issue-rate lift.
    The PMC bottleneck is **issue-rate starvation** (60-70% MFMA-pipe
    idle on barrier-pin schedule, R21 etiology). K-split adds
    cross-CTA reduction work — neutral for issue-rate per-CTA, may
    harm via reduce-kernel launch overhead and partial-buf HBM
    traffic.
  * "Smaller tiles" prototypes (small-tile 4w, 256x128 asymmetric)
    were both falsified — those also tried to reduce per-CTA work
    and failed because per-CTA fixed costs dominated. K-split has
    similar risk profile (per-CTA prologue/epilogue amortized over
    fewer K-iters).
  * Estimated 4-6 round investment for R52 (kernel branch with
    atomicAdd) + R53 (per-tile reduce kernel) + R54 (dispatcher
    rule + per-cell tuning) + R55-R57 (correctness/SNR tighten +
    multi-sample verify + fallback if any cell regresses).
  * If A1 produces only +5-10 score after the cost-decomp's overhead
    realities (vs the +25-30 claim), it consumes 4-6 rounds for a
    sub-10 lift — same EV as a single dispatcher tweak.

**Recommendation for R52**: do NOT immediately leap into kernel
branch implementation. Instead, R52 should be a 1-round preflight
that re-probes the R21 PMC etiology vs the R11 grid-occupancy model:

  * Measure per-CTA wall on Down_B4_M2048 fwd (PMC ``s_endpgm``
    timestamps) to confirm whether the second wave underfill is
    actually 42% of total wall (R11 model) or absorbed by the
    barrier-pin schedule (PMC R21 etiology suggests barriers are
    issue-rate bound, not grid-bound).
  * If second-wave underfill is <10% of wall, A1 EV ceiling is
    <5% lift across 8-shape mean → A1 closes.
  * If second-wave underfill is >25% of wall, A1 EV survives the
    PMC reality check → R53 lands the kernel branch.

This 1-round preflight saves the 4-6 round downside if the issue-rate
diagnosis dominates over the grid-occupancy model.

## R52 forward-pointer — A1 PMC reality check

Concrete R52 step:

  1. Write ``scripts/_probe_round_52_a1_pmc_reality_check.py`` that
     runs ``rocprofv3 --pmc s_endpgm,SQ_BUSY_CY_TIME,GRBM_GUI_ACTIVE``
     on Down_B4_M2048 fwd ×3 trials, splits CTAs into "wave 1" (304
     CTAs) and "wave 2" (48 CTAs), reports:
       * mean(wave1 CTA wall) vs mean(wave2 CTA wall)
       * fraction of total wall spent in wave2-only (no wave1 CTA active)
       * MfmaUtil during wave2-only window (should approach ~6% if all
         304 CUs idle except 48 wave2 → 48/304 = 15.8% × MfmaUtil_per_CTA)
  2. **Gate**: if wave2-only fraction > 25% of wall, A1 EV survives;
     R53 lands kernel branch. If <10%, A1 closes; pivot R52+ to
     Direction A3 (decoupled-warps) per task md NEW DIRECTIONS.

If preflight passes, R53-R56 implement the kernel branch + reduce +
dispatcher rule + verify, with metric expected at +10-15 score on
gpt_oss 8-shape mean (R11 lift × 0.5 overhead × 0.6 cell-weight
dilution).

## Falsification register update (gpt_oss FP8 task)

| Lever                                            | Status        | Source              |
|--------------------------------------------------|---------------|---------------------|
| C-1 (restrict / lifetime hints)                  | SATURATED     | R12, R54            |
| C-3 `+a` inline-asm AGPR hint                    | FALSIFIED     | parallel R48 (bf16) |
| C-3 step 1 art_base typedef (R47 plan)           | A-PRIORI FALS | R49                 |
| C-4 `-mllvm -amdgpu-mfma-vgpr-form=0`            | FALSIFIED     | R48 this run        |
| Other 6 amdgpu mllvm flags                       | A-PRIORI FALS | R48 this run        |
| C-1' KREM=64 collapse on grouped_rcr_kernel      | SHIPPED       | parallel R49 (HK 6c52d017) |
| C-1' KREM=64 collapse on grouped_var_k_kernel    | N/A           | R50                 |
| C-2 warp-tile restructure (4w occ=1)             | NOT STARTED   | R52+ pending        |
| **A1 Stream-K scaffolding end-to-end audit**     | **AUDIT-DONE** | **R51 (this round)** |
| **A1 kernel-side K-split branch (R18 work)**     | **NOT IMPL**  | **R51 (this round)** |
| K-tail micro-knobs (vmcnt / reorder)             | SATURATED     | R3-R55, R31b        |
| RCR_KTAIL_VMCNT ≠ 8                              | FALSIFIED     | R31b                |
| Direction D step 1 var-K SALU coord-decode       | SHIPPED       | HK b3a5c8db (R9)    |

## What this round changed

* ``analysis/_notes/round-51-A1-streamk-scaffolding-audit-kernel-split-blind-confirmed-R52-pointer.md``
  — this file (Primus-Turbo only).
* ``scripts/_probe_round_51_streamk_audit.py`` — end-to-end audit probe
  (Primus-Turbo only).
* No code changes in either repo's kernels / dispatchers.
* No metric / test edits.

## Scoring expectation

NEUTRAL round, daemon-canonical metric expected within the 691-699
window characterised by R29 (23-sample bit-equivalent baseline). No
code shipping ⇒ score should land near R47-R50 (693/692/691/692). The
value is (a) closing the audit step R50 explicitly forward-pointed
(scaffolding is alloc-only, kernel is split-blind, R18 work never
shipped) and (b) replacing the R50 plan's "audit + immediately leap
to kernel branch" with a 1-round PMC reality check at R52 that gates
the 4-6 round kernel-implementation arc on whether grid-occupancy
or issue-rate dominates the actual measured wall.

## Attribution

* HipKittens HEAD: ``49ffb984`` — UNCHANGED this round
* Primus-Turbo: only this doc note + audit probe script
* No ``config.py`` / ``dispatch.py`` / kernel changes
* No metric / test edits
