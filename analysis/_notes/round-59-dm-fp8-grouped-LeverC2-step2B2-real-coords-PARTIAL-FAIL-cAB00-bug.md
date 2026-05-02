# Round 59 (auto-optimize) — FP8 grouped Lever C-2 step 2B-2 PARTIAL FAIL

**Status**: 🟡 PARTIAL — kernel built, **AGPR retained at 256**, but
correctness FAIL on cAB[0][0] specifically (cells [0][1], [1][0], [1][1]
all CORRECT). Real-coord plumbing works; cell-0 register aliasing bug
to debug in R60.

**Score**: 982 (vs R58 baseline 983, noise band, no regression).

**HipKittens commit (auto-detected at end)**: see `git log -n 1 HEAD`.

---

## Hypothesis under test (from R58 step-2B-1 follow-up)

The R58 round note set up step 2B-2 to:

> Replace placeholder coords in test kernel with real `(br, bc, k)`
> indexing into `g.a / g.b / g.c` (correct GEMM output, no group binary
> search yet — fixed B=1 launch). Acceptance gate: build clean, AGPR
> ≥ 200, `max_abs ≤ 0.5` and `SNR ≥ 22 dB` vs torch fp32 ref.

The hypothesis was that AGPR retention (the R57+R58 finding) survives
the addition of full coordinate arithmetic + persistent tile loop +
4 cAB cells with proper sub-tile mapping. If yes → R60 ports the group
binary search prologue. If no → C-2 path likely dead.

**Outcome**: AGPR retention ✓, correctness ✗ (specific cell only).
The hypothesis is **half-confirmed** — AGPR survives, but a register
allocation issue in the cAB[0][0] tile blocks correctness validation.

---

## Implementation summary

New code added (HK `kernel_fp8_layouts.cpp`, namespace
`lever_c2_round_59_step2b2_real_coords`):

* `test_grouped_rcr_kernel_4w_real_coords<KI_HINT=0>`: persistent
  4w-style grouped FP8 kernel with real `(br, bc, k)` coord plumbing,
  single-buffer LDS (As[2] M-slabs + Bs[2] N-slabs, no ping-pong),
  4 cAB cells per warp (256 fp32/lane footprint), 4 mma_ABt per K-iter,
  scale epilog + 4 stores per tile.
  - Single-group only (G=1, m_subtile_A=0 hardcoded). Group binary
    search deferred to R60.
  - Caller contract: M%256==0, N%256==0, K%128==0.
  - LDS budget: 64 KB/block (2 As + 2 Bs of 16 KB each).
* `dispatch_test_4w_real_coords`: minimal host launcher (sets
  g.bpc, g.M_total, g.k from C/A; launches NUM_CUS-grid).
* Pybind binding `tk_fp8_layouts.test_4w_real_coords(a, b, c, sa, sb, group_offs)`:
  the **debug-only** entry the probe script uses. NOT in dispatch path.
* Probe script `analysis/fp8_gemm/mi350x/probe_4w_real_coords.py`:
  4 single-group probe shapes (M=512/768/512/1024, N=256/256/512/256,
  K=256/128/384/512), checks `max_abs ≤ 0.5` and `SNR ≥ 22 dB` vs
  torch fp32 ref, with cell-by-cell dump for the first shape.

Total lines added: ~210 (kernel + binding + probe).

---

## Resource report (HK build)

| Metric              | R57 step-2A | R58 step-2B-1 | **R59 step-2B-2** | grouped_rcr_kernel<0,T,T> (gpt_oss) |
|---------------------|-------------|---------------|-------------------|-------------------------------------|
| VGPRs               | 256         | 256           | **256**           | 256                                 |
| AGPRs               | 256         | 256           | **256** ✓         | 0                                   |
| Scratch (B/lane)    | 0           | 36            | **324**           | 152                                 |
| Spill               | 0           | 8             | **80**            | 37                                  |
| Occupancy (W/SIMD)  | 1           | 1             | **1**             | 1                                   |
| LDS (B/block)       | 65792       | 69632         | **69632**         | 139796                              |

**AGPR allocation is robust** — survives placeholder G::load → real
4-warp cooperative load (R58 step-2B-1) → real coord plumbing + cAB
cells (R59 step-2B-2). The 256 AGPR allocation is the empirical
mechanism the dense `rcr_4w::kernel` uses to hit 0 spill on FP8.

**Spill jumped 10× from R58** (8 → 80). Hypothesis: the persistent
tile loop body (4 loads + 4 register-tile loads + 4 mma + 4 stores +
scale epilog) ramps the live-set above what AGPR can absorb,
forcing 80 dwords/lane to scratch. This is **on par with production
grouped's 37 spill** so not unreasonable for an unoptimized first
real-coord port. R60+ tuning should recover the gap.

---

## Correctness probe (FAIL)

Probe shape (M=512, N=256, K=256, B=1):

```
  shape M=512 N=256 K=256: max_abs=294.0117 SNR=-35.81 dB → FAIL
  shape M=768 N=256 K=128: max_abs=348.1243 SNR=-39.38 dB → FAIL
  shape M=512 N=512 K=384: max_abs=264.4390 SNR=-33.65 dB → FAIL
  shape M=1024 N=256 K=512: max_abs=253.1458 SNR=-32.78 dB → FAIL
```

### Per-cell dump (shape M=512, N=256, K=256, block 0 = M[0,256), N[0,256))

| Region (M, N)       | chunk_max | diff_max  | Verdict | Owner          |
|---------------------|-----------|-----------|---------|----------------|
| [0:64, 0:64]        | 175.0     | 174.8     | **WRONG** | warp 0 cAB[0][0] |
| [0:64, 64:128]      | 175.0     | 174.6     | **WRONG** | warp 1 cAB[0][0] |
| [64:128, 0:64]      | 294.0     | 293.7     | **WRONG** | warp 2 cAB[0][0] |
| [64:128, 64:128]    | 201.0     | 201.4     | **WRONG** | warp 3 cAB[0][0] |
| [0:64, 128:192]     | 0.645     | 0.004     | OK      | warp 0 cAB[0][1] |
| [0:64, 192:256]     | 0.641     | 0.004     | OK      | warp 1 cAB[0][1] |
| [64:128, 128:192]   | 0.621     | 0.003     | OK      | warp 2 cAB[0][1] |
| [64:128, 192:256]   | 0.617     | 0.004     | OK      | warp 3 cAB[0][1] |
| [128:192, 0:64]     | 0.656     | 0.002     | OK      | warp 0 cAB[1][0] |
| [128:192, 64:128]   | 0.715     | 0.004     | OK      | warp 1 cAB[1][0] |
| [128:192, 128:192]  | 0.516     | 0.004     | OK      | warp 2 cAB[1][0] |
| [128:192, 192:256]  | 0.668     | 0.003     | OK      | warp 3 cAB[1][0] |
| [192:256, 0:64]     | 0.625     | 0.004     | OK      | warp 0 cAB[1][1] |
| [192:256, 64:128]   | 0.602     | 0.003     | OK      | warp 1 cAB[1][1] |
| [192:256, 128:192]  | 0.629     | 0.004     | OK      | warp 2 cAB[1][1] |
| [192:256, 192:256]  | 0.617     | 0.002     | OK      | warp 3 cAB[1][1] |

**12 / 16 cells correct** (diff ~0.003 = pure fp8 quantization noise).
**4 / 16 cells wrong** — exactly the **cAB[0][0] tile of all 4 warps**.

### Logical impossibility analysis

cAB[0][0] = a_reg[0] @ b_reg[0]^T (4 base mma over base_tile rows of d).
cAB[0][1] = a_reg[0] @ b_reg[1]^T → CORRECT → **a_reg[0] is fine**.
cAB[1][0] = a_reg[1] @ b_reg[0]^T → CORRECT → **b_reg[0] is fine**.

So both inputs to cAB[0][0] are valid yet the output is garbage. The
bug is in the **destination register tile cAB[0][0] itself** — either
zero() doesn't actually clear it, mma writes to the wrong VGPRs, mul
clobbers it, or the store reads garbage from it.

### Reorder probe: NOT a "first-mma-in-sequence" issue

Reordered to `mma cAB[1][1]` first, `mma cAB[0][0]` second. Result:
**same pattern** — cAB[0][0] still WRONG, cAB[1][1] still CORRECT.

This rules out:
* Compiler picking a degenerate codegen for the first mma in a sequence.
* LDS read race that affects only the first mma after `s_waitcnt`.

### Most likely root cause (hypothesis for R60)

**Register aliasing between cAB[0][0]'s VGPRs and the LLVM-allocated
spill traffic**. With 80 spill slots and Scratch=324 B/lane, the
compiler is interleaving heavy spill/reload activity through 80+ VGPRs.
cAB[0][0] is the FIRST register tile declared (`C_acc_4w cAB[2][2]`)
so its VGPR slots are first-allocated and most likely to overlap with
spill traffic.

**R60 debug plan**:
1. Dump ISA for `test_grouped_rcr_kernel_4w_real_coords<0>` and trace
   the VGPR slots holding cAB[0][0].tiles[0..3][0..3]. Check for
   scratch_load/scratch_store touching those slots.
2. Try a sacrificial dummy declaration:
   ```cpp
   C_acc_4w cAB_sacrificial;
   zero(cAB_sacrificial);   // never used, just shifts the register layout
   C_acc_4w cAB[2][2];
   ```
   If correctness improves, confirms register-allocation hypothesis.
3. Try `__attribute__((noinline))` on the host launcher's call to
   force a different codegen perspective.
4. If those don't work, drop the cAB cell count from 4 → 2 (per warp
   covers 128 M × 128 N as 1×2 cells, block tile = 256 M × 128 N) —
   this halves the per-warp acc footprint to 128 fp32/lane (likely
   loses AGPR allocation) but gives a known-good correctness baseline
   to bisect from.

---

## Metric impact

```
[metric_grouped_only]   grp_BF16  vs triton geomean=1.1883 (n=24)
[metric_grouped_only]   grp_FP8   vs triton geomean=1.1691 (n=24)
[metric_grouped_only] model_focus=all segment_focus=all  weighted_progress=0.9822
                       correct_fail=0/48  reject=0/48  below_target=34/48
                       goals=0/2  score=982
```

R58 baseline = 983, R59 = 982 → within noise band, **no regression**.

The test kernel is gated behind a separate pybind binding
(`test_4w_real_coords`) that is ONLY called by `probe_4w_real_coords.py`.
The production grouped kernel and dispatch path are byte-identical to
R58 — confirmed by the metric staying flat.

---

## R60+ roadmap (revised)

The R58 roadmap had R59 = step 2B-2 (coords), R60 = step 2B-3 (group
binary search), R61 = step 2B-4 (K-tail), R62 = step 2B-5 (N-mask),
R63 = step 2B-6 (dispatcher wire). R59's PARTIAL FAIL **inserts a
debug round** before R60:

| Round | Task | Target |
|-------|------|--------|
| **R60** | **Debug cAB[0][0] register allocation issue.** Try ISA dump → identify spill-overlap. Try sacrificial dummy / cell-count drop. Goal: `max_abs ≤ 0.5` on probe shape with AGPR ≥ 200 retained. | New gating round |
| R61 | Group binary search prologue (was R60 plan) | (was R60) |
| R62 | K-tail FUSED_KTAIL block (was R61 plan) | (was R61) |
| R63 | N-masked C-store (was R62 plan) | (was R62) |
| R64 | Dispatcher wire + 24-shape metric (was R63 plan) | geomean ≥ 1.18 |

If R60 debug fails (= correctness can't be recovered without losing
AGPR), the C-2 path is closed and we pivot to alternative levers
(Lever D fan-out cost falsified previously; remaining options are
async load + LDS pipelining, see task body).

---

## Files changed

* HK `analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp`:
  + new namespace `lever_c2_round_59_step2b2_real_coords` (+ kernel
    + dispatch + force-instantiation)
  + new pybind binding `test_4w_real_coords` (debug-only)
  + ~190 lines added
* HK `analysis/fp8_gemm/mi350x/probe_4w_real_coords.py` (NEW):
  4-shape correctness probe with cell dump
* Primus-Turbo `analysis/_notes/round-59-dm-...-PARTIAL-FAIL-cAB00-bug.md`
  (this file)

No HK or Primus-Turbo production code touched. Metric path
byte-identical to R58. No commit pushed to remote.
