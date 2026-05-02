# Round 18 — FP8 grouped: plateau re-confirmation (5-trial noise band) + R19 Direction-B scoping

**Status**: R18 metric **964** (median of 5 trials: 964/962/964/964/959, range
5 points) sits inside the documented 960-966 noise band. **No statistical
signal of regression OR improvement** since R15's high-tail 966.

After R17 falsified the last forward-path architectural lever (Lever R,
stage-level pipelining; -0.07 % microbench), R18 = plateau re-confirmation
with multi-trial evidence + concrete scoping for the R19+ multi-round
project (Lever H Direction B, fused dA-transpose).

**Auto-optimize round**: 18 / 100
**Date**: 2026-05-02
**HK SHA at round start / end**: `0f14b165` (no kernel rebuild this round)
**PT SHA at round start**: `aa91e04b`
**Reported best (forward)**: 966 (R15, high-tail of noise band)
**R18 baseline metric**: 964 (single trial); **5-trial median = 964**
**R18 patience**: 3 rounds at noise floor (R16=964, R17=963, R18=964 [×5])

---

## R18 evidence: 5-trial noise band

```
trial 0 (initial): 964
trial 1:           962
trial 2:           964
trial 3:           964
trial 4:           959
─────────────────────
range:             5 score points  (959 .. 964)
median:            964
```

Combined with R14-R17 history:
* R14 = 962, R15 = 966, R16 = 964, R17 = 963.
* Cumulative 9-trial range: **959 .. 966** (7 score points), median ≈ 963.

This is the kernel forward plateau, not a transient dip. Per `min(geomean /
1.20, 1.0) × 1000` scoring, the score-grade ceiling for the current
forward kernel is ~966; the path to 1000 requires geomean ≥ 1.20 which
needs +6.7 % on average across 24 shapes (or +16-21 % on the worst gpt_oss
6-case cluster) — far exceeding any single in-place optimisation.

---

## Per-shape FP8 ratios (R18 baseline trial)

```
worst 8 (sorted asc, all gpt_oss + 1 Qwen-Down):
  1.030  gpt_oss-GateUP-B32-M4096   hk=1979  trt=1922
  1.050  gpt_oss-Down-B32-M4096     hk=1883  trt=1794
  1.053  gpt_oss-GateUP-B32-M2048   hk=1929  trt=1832
  1.062  gpt_oss-GateUP-B4-M4096    hk=1901  trt=1790
  1.066  gpt_oss-Down-B32-M2048     hk=1804  trt=1692
  1.083  gpt_oss-GateUP-B4-M2048    hk=1728  trt=1595
  1.084  Qwen3-Down-B16-M4096       hk=1790  trt=1652
  1.091  gpt_oss-Down-B4-M4096      hk=1754  trt=1608

best 4 (above target):
  1.226  DSV3-Down-B32-M2048       hk=2123  trt=1731
  1.228  DSV3-Down-B32-M4096       hk=2178  trt=1773  
  (only 2 of 24 cases ≥ 1.20)
```

The 6 worst slots are all gpt_oss `<0,T,T>` template (FUSED_KTAIL=true +
N_MASKED_STORE=true) at K=2880 K_REM=64. These match the R3 spill
analysis: 67 VGPR spilled, ~250-290 dw working set vs 256 dw VGPR cap.

---

## Why no forward lever moves the metric (R10-R17 cumulative)

| Lever | Verdict | Round | Mechanism |
|---|---|---|---|
| **A** Async global→LDS | FALSIFIED | R2 | Already shipped via inline ASM |
| **B** Triple LDS slab | FALSIFIED | R2 | LDS at 137/160 KB cap |
| **C-X** N_MASKED helper SENTINEL | FALSIFIED | R4 | Spill neutral, -1.8 pp regression |
| **C-2** K-tail capture refactor | FALSIFIED | R3 | Already correctly scoped |
| **D** mfma_32x32x64 cell-shape | FALSIFIED | R5 | Microbench -0.03 % |
| **E** ASM software pipelining | FALSIFIED | R11 | Microbench -7.28 % |
| **R** Stage-level pipelining | FALSIFIED | R17 | Microbench -0.07 % (LLVM auto-overlaps) |
| **F** Per-shape dispatcher rules | LANDED & SATURATED | R6-R10 | 5 rules landed, R10-dm audit confirmed top-1 |
| **K** var_k epilog spill-trim | FALSIFIED | R14 | -1 pp metric (sees backward only) |
| **Q** fp8_transpose_3d block-shape | LANDED | R15 | +1.1 % gpt_oss bwd |
| **H/A** Triton fp8_transpose_3d | LANDED | R13 | +9.3 % bwd avg |

Forward kernel binary `grouped_rcr_kernel<0,T,T>` has been bit-identical
since HK commit `ecbead9a` (R5). All lever-class dimensions exhaustively
explored; no untried architectural rewrite remains within the
single-grouped-rcr-kernel design.

---

## Why R18 is docs (not perf)

The user FROZEN list and task body explicitly forbid the remaining
forward "knobs":

* (gm, num_xcds) further sweeps — saturated, R10-dm audit re-verified.
* Kernel template id (4-wave / 8-wave) — R32+R39 verified binding
  auto-pick is bit-equivalent.
* `#pragma unroll` enumeration on main loop / K-tail — forbidden.
* WARPS_M / WARPS_N flip — forbidden.
* Host-pad K / per-group launch / CPU sync / quantize fuse — forbidden.
* Host-overhead trim — R11 reduced to ~1 µs, "not a lever" per task body.

The only remaining scope-allowed direction within "architectural
rewrite / important data-flow change" is to step OUT of the current
single-grouped-rcr-kernel design — see R19 proposal below.

---

## R19+ proposal: Lever H Direction B (fused dA-transpose, multi-round)

R13 landed Direction A (Triton `fp8_transpose_3d` outside the GEMM).
Direction B = move the transpose INSIDE the GEMM kernel's K loop, so
the dA backward call:

```
# current (R13/R15)
b_t = fp8_transpose_3d(b)        # B*K*N bytes rd+wr at ~3 TB/s eff
out = grouped_rcr_kernel(grad_out, b_t)

# Direction B (proposed R19+)
out = grouped_rcr_b_inline_t(grad_out, b)  # one launch, no transpose pass
```

The HK kernel already has all-load-via-`buffer_load_b128`-on-SRD
infrastructure (lines 2596-2719 of `kernel_fp8_layouts.cpp`). Adding a
template parameter `B_INLINE_TRANSPOSE = true` that swaps the `(k_idx,
n_idx)` voffset computation to `(n_idx, k_idx)` AT LOAD TIME is
mechanically straightforward; the LDS layout / mfma cell shape stay
identical because the `tl.trans` happens implicitly in the SRD voffset
expression.

**Estimated impact** (per R12 wall decomposition, R13 transpose subtractive
analysis):
* gpt_oss reroute subset (8 of 24 cases) bwd wall: -4 .. -8 % per
  case (transpose pass eliminated).
* No effect on metric (forward path unchanged).
* No correctness risk on aligned shapes (Direction B kernel is gated
  on `K_RRR % 128 != 0 || N_RRR % 256 != 0` — same gate as R13's
  Direction A reroute).

**Estimated effort**: 3-5 rounds.
* R19: design doc + new template parameter wired through the binding.
* R20: write the inline-transpose voffset expression + bit-eq verify
  on a single shape (gpt_oss-GateUP B=32 M=4096).
* R21: extend to all 8 reroute shapes; bench wall + correctness gate.
* R22+ (if needed): perf tuning, dispatcher rule integration.

**Risk**: re-doing voffset arithmetic on the load helpers can perturb
LLVM's register allocation on the `<0,T,T>` template; need to verify
forward kernel binary bit-identity post-rebuild via
`md5sum tk_fp8_layouts.so` before/after. Failure mode = forward
regression on the FUSED_KTAIL/N_MASKED template. Mitigate via
template specialisation (`B_INLINE_TRANSPOSE = true` is a NEW spec, not
a modification of the existing `<0,T,T>`).

---

## Alternative R19+ paths (lower priority)

1. **Wave-specialisation rewrite** (R17's option 2). 4-8 rounds. Splits
   warps into "load wave" (issues all K-tail HBM loads) + "compute wave"
   (does mfma chain). High novelty, no precedent in HipKittens. Risk of
   not landing within budget.

2. **Different kernel architecture** (block-CCR layout). 5-15 rounds.
   Replaces the entire RCR kernel with a CCR variant. Higher reward
   ceiling but exposes a much larger surface to LLVM register
   allocation regression.

3. **Accept plateau, no further forward work**. Allocate remaining
   83 rounds (100 - 17) entirely to backward + utility code paths. The
   metric stays at 962-966 noise band; the bench wall improves
   monotonically. Reasonable if the "forward metric ≥ 1.20" goal is
   reframed as aspirational rather than hard.

R19 should pick **option Direction B** (this note's primary proposal):
highest EV/round, scoped to the 8 gpt_oss reroute shapes that already
pay the transpose cost, and within the R12-R17 lever framework's
budget projection (each Direction-B sub-task < 1 round of HK kernel
work).

---

## R18 metric data (5-trial archive)

```
$ for i in 1 2 3 4; do python3 scripts/_metric_grouped_only.py 2>&1 | tail -1; done
962
964
964
959
$ python3 scripts/_metric_grouped_only.py 2>&1 | tail -1   # initial
964
```

R14-R18 cumulative score history: 962, 966, 964, 963, 964, 962, 964, 964, 959.
Min = 959, max = 966, range = 7, median = 964. **Confirmed plateau:
forward kernel ceiling 962-966 ± 3.**

---

## Files touched in R18

* `analysis/_notes/round-18-fp8-grouped-plateau-confirmation-r19-direction-b-scoping.md` (NEW)

No HK kernel changes (no rebuild). No PT runtime changes. R18 score
delta vs R17: +1 (964 vs 963), within noise band.
