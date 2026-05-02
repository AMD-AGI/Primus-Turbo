# Round 17 — FP8 grouped: Lever R (stage-level pipelining) microbench gate FALSIFIED

**Status**: Lever R (R16 plan) FALSIFIED via single-wave microbench with
asm-barrier-controlled comparison. Hand-rolled pre-issue of K-tail
buffer_load_b128 BEFORE epilog 2 mfma chain provides **-0.07 %** delta
vs LLVM's auto-scheduled current code — same fate as Lever E (R11).

**Auto-optimize round**: 17 / 100
**Date**: 2026-05-02
**HK SHA at round start / end**: `0f14b165` → microbench commit (this round)
**PT SHA at round start**: `41866da4`
**Reported best (forward)**: 966 (R15, high-tail of noise band)
**R17 baseline metric**: **963** (median of round, noise-band consistent)

---

## R17 plan (per R16 round note)

R16 proposed Lever R: pre-issue 24 K-tail buffer_load_b128 BEFORE epilog
2's 4 mfma chain to overlap the ~80-cy HBM round-trip with the ~64-128
cy epilog 2 compute. Estimated geomean impact: +0.83 pp on grp_FP8 →
+6 score points if microbench gate ≥ +3 pp.

R17 = build microbench, run gate, decide.

---

## Microbench design

`/workspace/code/HipKittens/analysis/fp8_gemm/mi350x/lever_r_microbench.cu`
(366 lines, mirrors `lever_e_microbench.cu` skeleton).

Per-iter workload (single wave, 64 lanes, 1 CTA):
* 4 mfma_scale_f32_16x16x128 using register-resident A/B (no vmcnt dep)
  — mirrors epilog 2 cA/cB/cC/cD at lines 2484-2506.
* 24 raw_buffer_load_b128 from a 64 KB HBM region (cached, mostly L2-hit)
  — mirrors `load_b_kt(b0,b1) + load_a_kt(a, a_kt1)` at lines 2709-2712.
* `s_waitcnt vmcnt(8)` drain → 2 mfma using freshly-loaded HBM data.
* `s_waitcnt vmcnt(0)` drain → 2 mfma using freshly-loaded HBM data.

Two paths:
* **(a) "current"**: 4 mfma → 24 buffer_load → vmcnt → mfma chain.
* **(b) "pre-issued"**: 24 buffer_load → 4 mfma → vmcnt → mfma chain.

11-trial median per path; 2 run orders (a-first vs b-first) to defeat
warmup bias.

---

## Initial result (with asm volatile barrier in path (a)) — apparent +14.21 %

```
Run-order #1 (a-first, then b): delta = +13.84 %
Run-order #2 (b-first, then a): delta = +14.58 %
Mean delta = +14.21 %  (3 reps: +13.94, +14.11, +14.43)
Verdict (initial): PASS — Lever R confirmed
```

The initial microbench inserted an `asm volatile("" : "+v"(cA[0]), ...)`
barrier between the 4 mfma's and the 24 buffer_loads in path (a) to
force textual ordering. This made path (a)'s SQ stall on mfma issue
before any buffer_load could be issued.

---

## Falsification audit — barrier removed → +0.29 %

The asm volatile barrier in path (a) was an **artifact**, not a faithful
model of the real kernel. In the real kernel's epilog 2:
* No equivalent ``asm volatile("" : "+v"(cA[0]),...)`` exists.
* Epilog 2 (lines 2484-2506) is a **mix** of LDS reads (`load_a`,
  `load_b`) + lgkmcnt drains + 4 scattered mfma's + 4 wave-sync barriers.
* SQ is **not** monopolised by the mfma chain — it is also issuing
  ds_reads and waiting on lgkmcnt. The vector ALU does the mfma work
  in parallel with SQ activity. SQ remains AVAILABLE to issue HBM
  loads throughout epilog 2's ~150-200 cy duration.
* Therefore LLVM's auto-scheduler is free to interleave the K-tail
  buffer_loads with epilog 2's instruction stream — without any
  hand-rolled hoist.

REMOVING the asm barrier from path (a) and re-running:

```
Run-order #1 (a-first, then b): delta = +0.37 %  (then -0.09 %)
Run-order #2 (b-first, then a): delta = +0.20 %  (then -0.05 %)
Mean delta (rep 1) = +0.29 %
Mean delta (rep 2) = -0.07 %
Verdict: FAIL — plateau accepted (LLVM auto-schedules overlap)
```

**+0.29 % then -0.07 %** = within microbench noise. Both runs are
inside the ±3 pp gate band. Lever R FALSIFIED.

---

## Architectural confirmation (cycle accounting)

Even setting aside the microbench evidence, the proposed pre-issue
hits a CORRECTNESS wall in the real kernel:

**Current K-tail block (lines 2709-2718)**:
```
issue order: b0(4) → b1(4) → a(8) → a_kt1(8)   # 24 total
s_waitcnt vmcnt(8)        # 16 retired = b0+b1+a; a_kt1 in-flight
mfma cA = a · b0          # uses a/b0/b1 — all retired
mfma cB = a · b1          # overlaps with last 8 a_kt1 retirement
s_waitcnt vmcnt(0)        # drains a_kt1 (≤ 13 cy after cA/cB)
mfma cC = a_kt1 · b0
mfma cD = a_kt1 · b1
```

The 2-stage drain (vmcnt(8) → vmcnt(0)) overlaps cA/cB mfma with the
LAST 8 loads' retirement. This is a **deliberate optimisation** (R12-dm
+ R37-dm reordering) that depends on a_kt1 being LAST in issue order.

**With pre-issued a_kt1** (Lever R proposal):
```
PRE-ISSUE: a_kt1(8)            # at start of epilog 2
... epilog 2 mfmas ...
K-tail issue: b0(4) → b1(4) → a(8)   # 16 total in K-tail
                                # 24 total counting pre-issued
```

Now vmcnt(8) at end of K-tail issue:
* a_kt1 (8): pre-issued earliest, retired by now (vmcnt -= 8).
* In-flight: b0+b1+a = 16. vmcnt = 16. Wait ≤ 8 → wait for 8 to retire.
* In-issue-order on AMDGCN: first 8 retire = b0(4) + b1(4).
* a (8) still in-flight.
* mfma cA = a · b0 — **CORRUPTED** (a not retired).

Reordering to issue a first → cA needs b0/b1 not retired. Reordering
to issue b0/b1 last → cA needs a not retired. NO ORDER works with
vmcnt(8) drain when a_kt1 is pre-issued.

Forcing single vmcnt(0) drain (no 2-stage):
* All 24 retired before any K-tail mfma fires.
* End time = T_e2_start + max(pre_issue_done, K_tail_issue_done) + 100.
* = T_e2_start + 150 + 25 + 100 = T_e2_start + 275.
* cA/cB/cC/cD all start T_e2_start + 275, end T_e2_start + 301.

Compare to current code with 2-stage drain:
* T_e2_start + 150 (epilog 2 done) + 23 (K-tail issue) + 92 (b0+b1+a
  retire) = T_e2_start + 265.
* cA/cB end T_e2_start + 291. cC/cD end T_e2_start + 317.

Hmm, current code's cD finishes at +317, new code at +301. Sounds like
new code wins by 16 cy! BUT: the SQ issue time for the pre-issued 8
loads (~32 cy) is added at T_e2_start + 5..37, which extends epilog 2's
duration. T_e2_start + 150 → T_e2_start + 182. New end = T_e2_start +
311 → only +6 cy faster than current. Within noise.

This +6 cy/output-tile estimate × ~370 tiles/CU ≈ 2200 cy ≈ 1.1 us, and
gpt_oss main-kernel wall is ~140-200 us. Net: **+0.5-0.7 % main-kernel
TFLOPS gain** on gpt_oss, geomean impact +0.18-0.25 pp ≈ **+1-2 score
points**. Inside metric noise band (±2 score points across 5 trials).

The microbench evidence (+0.29 %, -0.07 %) is consistent with this
small-but-positive expected gain in the real kernel — but the gain is
small enough that:
1. Not worth the implementation risk (refactor of a_kt1 declaration
   scope, hoisting K-tail constants, breaking the 2-stage drain
   invariant — all of which could perturb LLVM's register allocation
   on other templates).
2. Below the +3 pp microbench gate threshold.

---

## Cumulative lever falsification (R17 update)

| Lever | Verdict | Falsified | Mechanism |
|---|---|---|---|
| **A** Async global→LDS | FALSIFIED | R2 | Already shipped via inline ASM |
| **B** Triple LDS slab | FALSIFIED | R2 | LDS at 137/160 KB cap |
| **C-X** N_MASKED helper SENTINEL | FALSIFIED | R4 | Spill neutral, -1.8 pp regression |
| **C-2** K-tail capture refactor | FALSIFIED | R3 | Already correctly scoped |
| **D** mfma_32x32x64 cell-shape | FALSIFIED | R5 | Microbench -0.03 % |
| **E** ASM software pipelining (iter-level) | FALSIFIED | R11 | Microbench -7.28 % |
| **R** Stage-level pipelining (NEW) | FALSIFIED | R17 | Microbench -0.07 % (LLVM auto-overlaps) |
| **F** Per-shape dispatcher rules | LANDED | R6-R10 | 5 rules landed, saturated |

---

## Why hand-scheduling consistently fails on this kernel

R5/R11/R17 all share a single failure mode:

> LLVM's instruction scheduler has been observed to find the same
> overlap pattern that hand-scheduling proposes — when the data flow
> permits the overlap. Hand-scheduling can ONLY win when there is a
> data flow constraint that LLVM cannot infer.

For Lever D (R5): the data flow IS the same regardless of cell shape
(both 16×16×128 and 32×32×64 process the same 64×32×128 region per
warp); LLVM scheduled both equivalently → -0.03 %.

For Lever E (R11): iter-level pipelining requires duplicating the
LDS double-buffer state to permit cross-iter prefetch; LLVM found this
a worse trade and produced a tighter schedule → -7.28 %.

For Lever R (R17): stage-level pipelining requires no data flow change
(a_kt1 ⊥ epilog 2's working set); LLVM was already overlapping → -0.07 %.

The pattern: when LLVM has the same information as the human, it picks
the same schedule. When LLVM has MORE information (about data flow,
register pressure, vmcnt accounting), it picks BETTER. Hand-scheduling
has no leverage on this kernel.

---

## Plateau acceptance (final, with R17 evidence)

After 17 rounds of disciplined falsification on the 24-shape suite:

* Architectural ceiling: 256 dw/lane VGPR cap vs ~250-290 dw/lane working
  set → unavoidable spill on `<0,T,T>` template (gpt_oss).
* Forward kernel binary unchanged since R5 (HK ecbead9a) — verified via
  HK commit log inspection in R16.
* All 6 architectural levers (A/B/C/D/E/R) FALSIFIED with cycle-level
  evidence + microbench gates.
* Score band 960-966 (5-trial median 962) is the kernel plateau.
* No untried forward-path architectural lever remains within the
  current grouped-rcr-kernel design.

To break the plateau, the next user-budget option would be one of:

1. **Different kernel architecture** (e.g., persistent block-CCR layout
   instead of RCR; explicit warp-group MMA scheduling). Estimated
   effort: 5-15 rounds of new-kernel work, high risk of regressing
   non-gpt_oss subsets.

2. **Wave-Specialisation rewrite** (split warps into producer/consumer
   roles for K-tail). Estimated 4-8 rounds, novel for HipKittens.

3. **Accept the plateau** and shift remaining budget to backward-track
   optimisations (Lever H Direction B fused dA-transpose, estimated
   +5-8 % bwd TFLOPS on gpt_oss reroute subset — does NOT move forward
   metric but improves real-world training wall-time).

R18 should pick option (3) — backward track — as the highest-EV,
lowest-risk remaining lever. The forward metric is noise-bound; further
forward-path work has diminishing returns.

---

## What this round changed in code

**HipKittens repo** (1 file, microbench only):
* `analysis/fp8_gemm/mi350x/lever_r_microbench.cu` (NEW, 380 lines)
  Falsification gate microbench. Build + run via:
  ```
  cd analysis/fp8_gemm/mi350x
  /opt/rocm/bin/hipcc lever_r_microbench.cu -o lever_r_microbench --offload-arch=gfx950 -O3
  ./lever_r_microbench
  ```

**Primus-Turbo repo** (1 file, docs only):
* `analysis/_notes/round-17-fp8-grouped-lever-r-stage-pipeline-falsified.md`
  (this file).

**No kernel binary change.** Forward `grouped_rcr_kernel` remains
bit-identical since R5.

---

## Hard-constraint compliance

- [x] No metric / benchmark / config edits
- [x] No dispatcher / can_handle changes
- [x] No quantize fuse, no host-side `.item()` / `.tolist()`
- [x] No per-model branches in `select_default_config`
- [x] HIPKITTEN remains `BackendEntry(..., autotune=False)`
- [x] One focused HK commit (microbench), one focused PT commit (docs)
- [x] No BF16 grouped touch
- [x] Correctness 0/48 fail across baseline trials
- [x] BF16 grouped baseline geomean 1.187 (R16: 1.187, no regression)
- [x] FP8 grouped baseline geomean 1.125 (R16 reported, consistent)

---

## DoD smoke status

R17 not on the 5/10/15/20 cadence. DoD harness next runs at R20.
Last DoD score recorded: 608 (SHA `97cbda86`).

---

## R18 explicit recommendation

**Option (3): switch to backward-track Lever H Direction B (fused
dA-transpose).**

The forward path is at architectural plateau. R18 should pick up the
backward optimisation thread last attempted in R12-R15:

1. **Lever H Direction B**: replace the `triton_fp8_transpose` +
   per-group-matmul sequence in
   `primus_turbo/pytorch/kernels/grouped_gemm/grouped_gemm_fp8_impl.py::dA_path`
   with a single fused HipKittens kernel that does on-the-fly transpose
   inside the GEMM main loop.
2. Expected gain: +5-8 % bwd TFLOPS on gpt_oss reroute subset (per R12
   wall decomposition).
3. Verification: `bench_grouped_gemm_turbo.py --dtype fp8` (NOT in
   metric — agent self-runs and pastes fwd+bwd TFLOPS + correctness
   into commit message per task body).

Alternatively, **Option (1)/(2)**: novel kernel architecture rewrite —
high risk, multi-round, may regress non-gpt_oss subsets. Defer unless
R18+ confirms backward track is also saturated.

DO NOT:
* Retry any of A / B / C-X / C-2 / D / E / R levers — all falsified
  with microbench evidence (cite R2/R3/R4/R5/R11/R17).
* Implement Lever R as a "small wins are still wins" play. Architectural
  analysis shows +1-2 score points expected, well within metric noise
  (±2 across 5 trials), with implementation risk of perturbing LLVM
  register allocation on other templates.

---

## Round-end summary (本轮目标 / 改了什么 / before-after metric / commit SHA / 下一轮建议)

- **本轮目标**: 实证 R16 计划 Lever R 提案 (stage-level pipelining,
  pre-issue K-tail buffer_load before epilog 2 mfma chain).
- **改了什么**:
  - HK: `analysis/fp8_gemm/mi350x/lever_r_microbench.cu` (NEW)
  - PT: `analysis/_notes/round-17-fp8-grouped-lever-r-stage-pipeline-falsified.md` (NEW)
- **Before-after metric**: 963 → 963 (no kernel change, baseline only).
- **Commit SHA**: filled at commit time.
- **下一轮建议**: R18 = switch to backward-track (Lever H Direction B
  fused dA-transpose). Forward path at architectural ceiling, all 7
  forward levers (A/B/C-2/C-X/D/E/R) falsified.
