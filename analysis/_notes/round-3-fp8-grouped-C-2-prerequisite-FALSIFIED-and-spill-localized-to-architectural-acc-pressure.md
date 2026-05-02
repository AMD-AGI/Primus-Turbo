# Round 3 — FP8 grouped: Lever C-2 prerequisite FALSIFIED + spill localization shows architectural ceiling

**Status**: NO KERNEL CHANGE — R3 = source-code audit (C-2 prerequisite
falsification) + LLVM optimization-record dump + spill localization
analysis → re-route R4 to Lever C-X (N_MASKED helper) or accept plateau
**Auto-optimize round**: 3 / 100
**Date**: 2026-05-02
**HK SHA at round start / end**: `ecbead9a` (unchanged; rebuild was
bit-identical: 728352 bytes both before and after, since
`-fsave-optimization-record` doesn't affect codegen)
**PT SHA at round start**: `849ae8c`
**Round time**: ~25 min (2 metric runs + 1 instrumented build + opt.yaml parse)
**Score before**: 960 (R2 baseline)
**Score after**: 956 (R3 noise sample; bit-identical .so, ±4 noise band)

---

## Lever C-2 prerequisite FALSIFIED — captures already in if-branch

R2's plan said Lever C-2 = "refactor `kernel_fp8_layouts.cpp:2535-2719`
FUSED_KTAIL=true block to drop by-reference lambda captures whose
liveness LLVM extends across the entire persistent loop even when
g.fast_k == g.k (DSV3/Qwen K%128==0 dead branch)".

**That premise is wrong.** Lines 2588-2615 (the `const int laneid`,
`const int K_REM`, `const fp8e4m3* a_base_ptr`, `const i32x4 a_srsrc_kt`,
`const uint32_t K_tail_base_bytes`, etc.) are **all declared INSIDE
the `if (g.fast_k < g.k)` block** (line 2540 opens it, line 2719
closes it). The lambdas `load_a_kt` / `load_b_kt` (lines 2627-2673)
capture by reference but ALL captures are already inside the if-branch
scope. K%128==0 cases hit the `false` arm at line 2540 and ALL of
this state has zero liveness contribution.

The only piece of state that does survive across the if-branch is
`A_row_reg a_kt1;` declared at line 2539 (above the inner if), but
this was deliberately kept by R56-dm because it gives LLVM a better
liveness graph for the FUSED_KTAIL=true template (see line 5260-5265
comment: "FUSED_KTAIL=true template specs have HALF to a THIRD the
number of interleaved scratch_store/mfma pairs in epilog 1 (28 vs 62
for GateUP spec, 22 vs 44 for Down spec)").

**Conclusion**: there's nothing to refactor. Lever C-2 is a no-op.

---

## Empirical spill localization (LLVM optimization remarks)

Rebuilt with `-fsave-optimization-record` (yields a 65 MB
`kernel_fp8_layouts-hip-amdgcn-amd-amdhsa-gfx950.opt.yaml`). Parsed
all `SpillReloadCopies` / `LoopSpillReloadCopies` remarks and grouped
by function:

```
grouped_rcr_kernel<0, false, false> (DEAD - never dispatched)
  L 2223 [FUNC] NumVR= 268  Cost=576.0  (SpillReloadCopies)
  L 2313 [LOOP] NumVR= 266  Cost=556.0  (LoopSpillReloadCopies)
  L 2223 [FUNC] NumVR=  22  Cost=112.5  (SpillReloadCopies)
  L 2313 [LOOP] NumVR=  15  Cost= 98.0  (LoopSpillReloadCopies)

grouped_rcr_kernel<0, false, true> (DSV3+Qwen3, 16/24 case) ← TARGET
  L 2223 [FUNC] NumVR= 273  Cost=580.0  (SpillReloadCopies)  ← architectural cA-cD pressure
  L 2313 [LOOP] NumVR= 271  Cost=560.0  (LoopSpillReloadCopies)
  L 2223 [FUNC] NumVR=  19  Cost= 51.5  (SpillReloadCopies)  ← LOWEST secondary cluster
  L 2313 [LOOP] NumVR=  10  Cost= 36.0  (LoopSpillReloadCopies)

grouped_rcr_kernel<0, true , false> (DEAD)
  L 2223 [FUNC] NumVR= 273  Cost=576.0
  L 2313 [LOOP] NumVR= 271  Cost=556.0
  L 2223 [FUNC] NumVR=  66  Cost=136.5  ← N_MASKED+ NO_FUSE: 47 NumVR worst-case helper inflation
  L 2313 [LOOP] NumVR=  59  Cost=122.0

grouped_rcr_kernel<0, true , true> (gpt_oss, 8/24 case)
  L 2223 [FUNC] NumVR= 286  Cost=600.0
  L 2313 [LOOP] NumVR= 284  Cost=580.0
  L 2223 [FUNC] NumVR=  63  Cost= 75.5  ← N_MASKED helper adds +44 NumVR vs DSV3 spec
  L 2313 [LOOP] NumVR=  54  Cost= 60.0

grouped_var_k_kernel_fp8<0> (dB backward, correctness-only)
  L 5573 [FUNC] NumVR= 275  Cost=317.0
  L 5573 [FUNC] NumVR= 162  Cost=137.9   ← BIG secondary cluster, dB pressure
  L 5633 [LOOP] NumVR= 150  Cost= 96.0
```

### Key empirical findings

1. **Primary spill (~560-600 cost, 266-286 NumVR) sits at L2313** =
   the persistent-tile outer loop. It is **uniform across all 4
   template specs** (variation < 5 %). This means the primary spill
   is **independent of FUSED_KTAIL and N_MASKED_STORE** — it is
   driven by the structural cA/cB/cC/cD accumulator pressure +
   a/b0/b1 register tile holding pattern.

2. **Secondary spill cluster** (10-66 NumVR) shows the controllable
   delta:

   | Template | Secondary NumVR (loop+func) | Spill remark dw | Source |
   |---|--:|--:|---|
   | `<0,F,T>` DSV3+Qwen | 10+19 = 29 | 32 | **architectural minimum** |
   | `<0,F,F>` (dead)    | 15+22 = 37 | 39 | +6 dw vs F,T (a_kt1 missing) |
   | `<0,T,F>` (dead)    | 59+66 = 125 | 43 | +73 NumVR from N_MASKED helper alone |
   | `<0,T,T>` gpt_oss   | 54+63 = 117 | 39 | +88 NumVR vs `<0,F,T>` |

3. **The `N_MASKED_STORE=true` template adds +44 NumVR / +24 cost**
   to the secondary cluster (compare `<0,F,T>` vs `<0,T,T>`: 29 → 117
   NumVR, mostly from the `store_c_tile_n_masked` helper at line
   909-972). This **IS a localized hot spot** addressable in principle.

4. **For DSV3+Qwen3 (16/24 cases, the `<0,F,T>` template)**, only 32 dw
   spill / 132 B/lane scratch / 29 secondary NumVR — there's no
   localized hot spot to attack. This is the **architectural floor**
   for the current main-loop shape (4 accumulators × 64×32 fp32 +
   3 register tiles).

### Architectural ceiling derivation

Working set inside `for (int gt = pid; gt < total_tiles; ...)` body:

| Variable | Size (dw/lane) | Held when |
|---|--:|---|
| cA, cB, cC, cD (rt_fl<64,32>) | **128** | entire main loop + epilog |
| a (A_row_reg, rt_fp8<64,128>) | 16 | main loop + K-tail |
| b0, b1 (B_row_reg, rt_fp8<32,128>) | 16 | main loop + K-tail |
| soA, soB swizzled offsets | 4 | entire kernel body |
| lds_addrs[3] in helper inline | ~3 (transient) | each load helper invoke |
| laneid + warpid + tic/toc | ~4 | main loop |
| Subtotal core working set | **~171** | persistent loop body |
| **Available VGPR after subtotal (cap=256)** | **~85** | for everything else |
| Helper temporaries + iter index + spill-tag overhead | ~80-120 | main loop + helper |
| **Net overflow → VMEM scratch** | **~32-43** | what we measure |

The 32 dw spill on `<0,F,T>` is **exactly the overflow** of working set
beyond VGPR cap. Not addressable without architectural change.

---

## What does this mean for R4+ planning?

### Lever C-X (N_MASKED helper refactor, gpt_oss-only): viable but small impact

The `<0,T,T>` spec adds 44 NumVR vs `<0,F,T>` from `store_c_tile_n_masked`
helper (line 909-972). This helper is the **only localized spill source
left for ANY template**. It only affects gpt_oss (8/24 case).

**Mechanism**: refactor `store_c_tile_n_masked` to (a) read
`r_tile`/`c_tile`/`n_limit` once into local SGPR-friendly variables
before the nested `for i,j,k,l` loop, (b) hoist `dst_ptr`/`row_stride`
outside the inner loops, (c) replace the per-lane `if (n0 + col >=
n_limit) continue;` divergent skip with a per-lane mask + unconditional
buffer_store with masked-OOB-zero (since the SRD bounds the C tensor
range_bytes anyway, OOB stores no-op). This last change drops the
divergent-control-flow live-state that the per-lane mask carries.

**Expected impact**: 7 dw spill drop on `<0,T,T>` spec → 2-3 % main-loop
runtime improvement on gpt_oss → ~+1.5 pp on gpt_oss-{Down-N=2880,
GateUP-N=5760} 4 cases each. 8/24 case × +1.5 pp = +0.5 pp geomean →
**+5 score points**, just clears noise.

**Cost**: 1-2 rounds (refactor + microbench + revert if regression).

### Lever D microbench gate (R64-dm leftover): mandatory before any cell-shape work

R64-dm landed `st_32x64` shared-memory tile type as scaffolding but
explicitly required a microbench gate before committing 4-6 rounds to
the full mfma_323264 port. Microbench has not been run yet. R3 spill
data shows the structural ceiling, which strengthens R64's argument
that the only way to break the ceiling is to halve the per-iter
accumulator count via mfma_323264 (1 mfma per 32×32 cell vs 4 mfma per
16×16 cell across the 64x32 output, with mfma_323264 at ~2× per-call
cost = same MFMA throughput, but **half the accumulator working set
held simultaneously**).

The cost-model from R64-dm said equal MFMA cycle count and equal
accumulator dw/lane for the same K volume — but it didn't account
for the SCHEDULING freedom of having half the simultaneous live
accumulators. The microbench is needed to test if the scheduling
change actually wins.

### Lever F (Qwen-Down K=1536 short-K): geometrically limited

4/24 case × even +5pp = +0.8 pp geomean → +8 score points. Marginal,
defer.

### Lever E (ASM software pipelining): last resort, high risk

Defer to R20+.

---

## R4 recommendation: **Lever C-X (N_MASKED helper)** first, then **Lever D microbench gate**

Rationale:
- C-X is the LAST localized spill source left to attack.
- Risk is bounded: refactor is gpt_oss-only (`<0,T,T>` template), DSV3+Qwen
  cases use `<0,F,T>` template which is untouched.
- Expected +5 score (just clears noise) is small but **provable** vs.
  Lever D's 4-6 round speculative investment.
- After C-X commits, R5+ runs the D microbench gate. If gate passes
  (mfma_323264 ≥ 3pp single-warp throughput advantage), commit to full
  port. If fails, **accept the 956-962 plateau as final** for FP8
  grouped and redirect remaining rounds to BF16 grouped (whose
  geomean=1.18 is also < 1.20 target — though task body marks it as
  [watch], hitting 1.20 there would still be useful for end-to-end).

Wait — task body explicitly forbids touching BF16 grouped this run
("不许碰 dense / BF16"). So if Lever C-X + Lever D both fall short,
the only remaining options are Lever E (ASM software pipeline, high
risk) or accepting the plateau and running out the 100-round budget
on docs / cleanup. This is the realistic ceiling we're hitting.

---

## What this round changed

**Nothing in code.** Only:
- New round note (this file)
- Rebuilt `tk_fp8_layouts.cpython-312-x86_64-linux-gnu.so` twice
  (bit-identical 728352 bytes both times — `-fsave-optimization-record`
  doesn't affect codegen). Final `.so` is unchanged from R2 end-of-round.
- Generated then deleted the 65 MB opt.yaml (parsed in-memory before
  unlinking — see the FUNC/LOOP table above).

**HipKittens repo**: no change (HK SHA stays at `ecbead9a`).
**Primus-Turbo repo**: this `.md` is the only diff.

---

## What R4 should do (concrete first step)

1. Read `kernel_fp8_layouts.cpp:909-972` (the `store_c_tile_n_masked`
   helper). Identify the variables held across the nested
   `for i, j, k, l` loops.
2. Refactor the helper to:
   - Hoist `dst_ptr`, `row_stride`, `laneid`, `row_offset`, `col_offset`,
     `buffer_size`, `srsrc` outside ALL nested loops (compute once).
   - Replace `if (n0 + col >= n_limit) continue;` divergent skip with
     a precomputed per-lane mask × unconditional `buffer_store_b16`
     using SENTINEL voffset for OOB (mirrors the K-tail SENTINEL
     pattern used in lines 2637/2638 already in the same file).
   - Mark the helper `__attribute__((always_inline))` (already
     `__device__ __forceinline__` per line 909, may need stronger
     hint).
3. Rebuild + check `-Rpass-analysis=kernel-resource-usage`:
   - target on `<0,T,T>` spec: VGPR Spill 39 → ≤32, ScratchSize 160
     → ≤132 B/lane (matching `<0,F,T>` floor)
4. If spill drops AND correctness passes → run metric.
   If score ≥ +5 pts → commit. Else revert.
5. If spill UNCHANGED → revert immediately, jump to Lever D microbench
   gate (R5).

Hard rules unchanged from R2:
- No metric/dispatch/can_handle edits
- One focused commit per repo per round
- BF16 grouped is `[watch]`, FP8 grouped is the only optimization target
- HIPKITTEN must remain `BackendEntry(..., autotune=False)`

---

## Updated lever roadmap (replace R2 table)

| Lever | Status | R3 verdict |
|---|---|---|
| **A** Async global→LDS copy | FALSIFIED R2 (already shipped) | — |
| **B** Dual/triple LDS slab  | FALSIFIED R2 (LDS 137/160 KB used, dual already shipped) | — |
| **C-1** LDS hand-spill | DEFERRED — R3 data shows spill is architectural, not localized → LDS hand-spill won't help unless we can identify a SINGLE expression to redirect | gated on Lever C-X first |
| **C-2** K-tail capture refactor | **FALSIFIED R3** — captures already in if-branch (line 2540-2719), no liveness leak | — |
| **C-3** spill source localization | **DONE R3** — primary spill is architectural (cA-cD + a/b0/b1 = ~171 dw working set, overflows 256 VGPR cap by 32-43 dw), secondary cluster is N_MASKED helper +44 NumVR | data feeds C-X |
| **C-X** N_MASKED helper refactor | **OPEN — R4 candidate** | gpt_oss-only +0.5 pp geomean expected |
| **D** mfma_323264 cell-shape | DEFERRED — R5+ if C-X exhausted; R64-dm scaffolded, microbench gate not run | mandatory gate before full port |
| **E** ASM software pipelining | LAST RESORT — only if D microbench passes AND port itself fails | R20+ |
| **F** Qwen3-Down K=1536 short-K | DEFERRED — only +0.8 pp geomean even if successful | R15+ |

---

## DoD smoke status

Not run this round (docs + rebuild only, no shared-code change).
Last run was at R64 SHA `94fc3121`. Will re-run at R5 checkpoint per
task-body cadence.
