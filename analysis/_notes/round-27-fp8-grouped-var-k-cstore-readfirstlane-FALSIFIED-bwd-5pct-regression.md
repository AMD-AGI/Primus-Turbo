# Round 27 — FP8 grouped: var_k C-store readfirstlane FALSIFIED (-5.5 % dB backward despite -8 dw spill drop)

**Status**: KERNEL CHANGE TRIED + REVERTED — `__builtin_amdgcn_readfirstlane` on
the 4 wave-uniform store coords (r0/r1/c0/c1) of `grouped_var_k_kernel_fp8`'s
C-store epilog ON ITS OWN dropped the kernel's VGPR spill from 37 → 29 dw
(-8 dw, comparable to R24/R25 magnitudes) but the dB backward bench
**regressed -3 % to -9 % on all 24 metric shapes** vs the pre-R27 baseline.
Fully reverted. Forward metric unchanged at 979 (var_k is dB-only).
**Auto-optimize round**: 27 / 100
**Date**: 2026-05-02
**HK SHA at round start**: `4caa6d9a`
**HK SHA at round end**: `4caa6d9a` (unchanged after revert; tree was stashed
and the stash dropped after backward bench falsified the change)
**PT SHA at round start**: `0a8719c0`
**Round time**: ~30 min (1 build + 1 forward metric + 2 bench A/B + 1 noise
re-run + 1 revert)
**Score before** (R26 baseline median): 977
**Score during probe** (forward metric): 979 (within ±2 noise band)
**Score after revert**: 979 (baseline retained)

---

## Hypothesis (from R26 doc — extension of R24/R25 readfirstlane lever class)

R24 landed `__builtin_amdgcn_readfirstlane(r0, r1, c0, c1)` in the
`grouped_rcr_kernel` C-store epilog (forward, +15 score points). R25
ported the identical pattern to `grouped_rrr_kernel` (dA backward,
-11 dw spill, +0.20 % dA TFLOPS). The third grouped FP8 forward/backward
kernel is `grouped_var_k_kernel_fp8` (dB backward, CRR layout, K-variable
per group). It carries the highest single-spec spill profile in the FP8
grouped binary at **37 dw / 152 B scratch** (pre-R27), which is near the
gpt_oss-Down `<0,T,T>` rcr template's 37 dw. R26 already exhausted the
K-tail SRD uniformization lever class on the FP8 grouped forward path,
but R27 is a fresh probe on a **different kernel entry point** (var_k,
not rcr). The single missing application of the R24 pattern.

Mechanism applied (line 5904-5911 in `kernel_fp8_layouts.cpp`):

```cpp
// PRE (the original):
store_c_tile_mn_masked_grouped(g.c, cA, group_idx,
    br*WARPS_M*2+wm,         bc*WARPS_N*2+wn,         g.n, g.k);
store_c_tile_mn_masked_grouped(g.c, cB, group_idx,
    br*WARPS_M*2+wm,         bc*WARPS_N*2+WARPS_N+wn, g.n, g.k);
store_c_tile_mn_masked_grouped(g.c, cC, group_idx,
    br*WARPS_M*2+WARPS_M+wm, bc*WARPS_N*2+wn,         g.n, g.k);
store_c_tile_mn_masked_grouped(g.c, cD, group_idx,
    br*WARPS_M*2+WARPS_M+wm, bc*WARPS_N*2+WARPS_N+wn, g.n, g.k);

// POST (the R27 probe):
const int r0 = __builtin_amdgcn_readfirstlane(br*WARPS_M*2+wm);
const int r1 = __builtin_amdgcn_readfirstlane(br*WARPS_M*2+WARPS_M+wm);
const int c0 = __builtin_amdgcn_readfirstlane(bc*WARPS_N*2+wn);
const int c1 = __builtin_amdgcn_readfirstlane(bc*WARPS_N*2+WARPS_N+wn);
store_c_tile_mn_masked_grouped(g.c, cA, group_idx, r0, c0, g.n, g.k);
store_c_tile_mn_masked_grouped(g.c, cB, group_idx, r0, c1, g.n, g.k);
store_c_tile_mn_masked_grouped(g.c, cC, group_idx, r1, c0, g.n, g.k);
store_c_tile_mn_masked_grouped(g.c, cD, group_idx, r1, c1, g.n, g.k);
```

`group_idx` was DELIBERATELY left as VGPR — R22 V-A established that
`readfirstlane(group_idx)` triggers ~+15-21 dw VGPR spill cascade
through the kernel's hot path (the persistent `for (gt = pid; ...)` loop
extends SGPR liveness through K-loop). r/c are localised to the
post-K-loop epilog so should have shorter SGPR liveness — same rationale
that worked for R24/R25.

---

## LLVM resource-usage probe (var_k spec, ScratchSize / VGPRs Spill)

`-Rpass-analysis=kernel-resource-usage` on the rebuilt `.so`:

| Kernel        | Pre-R27 ScratchSize | Pre-R27 Spill | Post-R27 ScratchSize | Post-R27 Spill | Δ |
|---|--:|--:|--:|--:|--:|
| `grouped_rcr_kernel<0, F, F>` (dead) | 220 B | 54 dw | 220 B | 54 dw | 0 |
| `grouped_rcr_kernel<0, T, F>` (dead) | 156 B | 38 dw | 156 B | 38 dw | 0 |
| `grouped_rcr_kernel<0, F, T>` (DSV3+Qwen) | 140 B | 34 dw | 140 B | 34 dw | 0 |
| `grouped_rcr_kernel<0, T, T>` (gpt_oss) | 152 B | 37 dw | 152 B | 37 dw | 0 |
| `grouped_rrr_kernel<0>` (dA bwd, post-R25) | (264 B) | 65 dw | 264 B | 65 dw | 0 |
| **`grouped_var_k_kernel_fp8<0>` (dB bwd) ← TARGET** | **152 B** | **37 dw** | **120 B** | **29 dw** | **-32 B / -8 dw** |

**The change is structurally clean** — it isolates entirely to the var_k
kernel's resource report, with no spill backlash on the rcr/rrr templates.
The var_k spill drop is in the same magnitude as R24 / R25 wins.

**Yet the bench regressed.** This is the key falsification.

---

## Bench falsification (the hard signal)

Pre-R27 baseline (HK SHA `4caa6d9a`, var_k spill 37 dw):

```
Run 1: Average Forward TFLOPS: 1366.24    Average Backward TFLOPS: 1371.37
Run 2: Average Forward TFLOPS: 1365.93    Average Backward TFLOPS: 1370.30
                                                                 ↑ noise band ±1 TF (±0.07%)
```

Post-R27 (var_k spill 29 dw, var_k C-store coords readfirstlane'd):

```
Run 1: Average Forward TFLOPS: 1366.00    Average Backward TFLOPS: 1296.04
                                                                 ↑ -75 TF (-5.5%) vs baseline
```

Per-shape backward delta (post-R27 vs pre-R27, all 24 shapes):

```
Case                                     B     M     pre_bwd    post_bwd    delta (pp)
DeepSeek-V3-GateUP                      16  2048    1430.4      1315.3      -8.05%
DeepSeek-V3-Down                        16  2048    1356.7      1269.4      -6.44%
DeepSeek-V3-GateUP                      16  4096    1678.0      1597.1      -4.82%
DeepSeek-V3-Down                        16  4096    1610.7      1552.0      -3.65%
DeepSeek-V3-GateUP                      32  2048    1429.2      1299.7      -9.06%
DeepSeek-V3-Down                        32  2048    1368.4      1269.5      -7.23%
DeepSeek-V3-GateUP                      32  4096    1676.8      1584.2      -5.53%
DeepSeek-V3-Down                        32  4096    1609.6      1540.7      -4.28%
gpt_oss_20B-GateUP                       4  2048    1148.4      1101.5      -4.08%
gpt_oss_20B-Down                         4  2048     920.7       871.6      -5.33%
gpt_oss_20B-GateUP                       4  4096    1495.5      1441.2      -3.63%
gpt_oss_20B-Down                         4  4096    1181.3      1148.4      -2.79%
gpt_oss_20B-GateUP                      32  2048    1304.6      1194.5      -8.44%
gpt_oss_20B-Down                        32  2048    1082.1      1030.9      -4.73%
gpt_oss_20B-GateUP                      32  4096    1569.0      1453.6      -7.35%
gpt_oss_20B-Down                        32  4096    1296.8      1210.7      -6.64%
Qwen3-235B-A22B-GateUP                  16  2048    1284.0      1206.8      -6.01%
Qwen3-235B-A22B-Down                    16  2048    1164.7      1110.1      -4.69%
Qwen3-235B-A22B-GateUP                  16  4096    1501.2      1435.8      -4.36%
Qwen3-235B-A22B-Down                    16  4096    1376.5      1329.5      -3.41%
Qwen3-235B-A22B-GateUP                  32  2048    1288.4      1207.2      -6.30%
Qwen3-235B-A22B-Down                    32  2048    1203.6      1126.3      -6.42%
Qwen3-235B-A22B-GateUP                  32  4096    1527.1      1452.9      -4.86%
Qwen3-235B-A22B-Down                    32  4096    1409.0      1356.3      -3.74%
                                                                       ────────
                                                                       median -5.10%
                                                                       max    -9.06% (DSV3-GateUP-B32-M2048)
                                                                       min    -2.79% (gpt_oss-Down-B4-M4096)
                                                                       sign:  24/24 negative
```

**24 of 24 shapes regress monotonically -3 to -9 %.** This is 70× the
noise band (±0.07 % from the 2 pre-R27 runs); the regression is
unambiguously real.

Forward metric (run separately, 24 fwd cases): 979 → 979 (no change,
expected — var_k is dB-only, not on the forward path).

Correctness: All 24 shapes PASS allclose check on both pre-R27 and
post-R27. The change is *correct* but slower.

---

## Why a -8 dw spill drop maps to a -5 % runtime regression

This is the third occurrence of the same anti-pattern (R22 V-A, R25
attempt-1, R26 K-tail probes, now R27). The pattern:

1. `__builtin_amdgcn_readfirstlane(expr)` materializes `expr` in a VGPR
   first (LLVM cannot execute readfirstlane directly on a non-uniform
   computation), then issues `v_readfirstlane_b32 sgpr, vgpr` to copy
   to SGPR.
2. The SGPR target value (`r0`/`r1`/`c0`/`c1`) replaces a VGPR-resident
   coord. **This is what makes the spill remark drop**: 4 SGPR-resident
   coords vs 4 VGPR-resident coords = 4 dw less VGPR budget needed in
   the C-store window × 2 epilog passes ≈ 8 dw spill saved.
3. **However**, the SOURCE expressions (`br*WARPS_M*2+wm` etc.) remain
   VGPR until the readfirstlane consumes them. LLVM extends both
   liveness (VGPR source + SGPR result) across the 4 store calls.
4. The 4 `v_readfirstlane_b32` ops also occupy issue slots that
   previously were available for `mfma` overlap or scratch_load
   completions. On a kernel where the C-store epilog is on the
   critical path (var_k has no extra fanout work after C-store), the
   issue-slot pressure dominates the spill savings.
5. Net runtime regresses despite spill report improving.

**Why R24 / R25 succeeded but R27 fails**: R24 (rcr) and R25 (rrr) both
have a **multi-warp scale + 2-axis-mask + N-strip duplicating** epilog
that already saturates the issue pipeline; the readfirstlane ops
overlap with the existing scale / store latency. Var_k's epilog is
**leaner** — 4 mul + 4 store, with no N-strip duplication
(M-axis is per-group not per-batch in CRR layout) — so the readfirstlane
ops sit on the critical path with no cycle to hide them.

The spill metric is the WRONG proxy for runtime improvement on
already-saturated epilogs. R24 / R25 wins were spillaccompanied by
schedule overlap; R27's spill drop has nowhere to overlap and pays a
critical-path tax.

---

## Why this falsification is structural (the readfirstlane lever class
is now formally exhausted on FP8 grouped)

R22 (group_idx prologue) — falsified by spill backlash.
R24 (rcr C-store coords) — LANDED +15 pts (spill drop + free schedule).
R25 attempt 1 (rcr K-tail b_per_group_bytes) — falsified by spill backlash.
R25 attempt 2 (rcr K-tail partial readfirstlane) — falsified by same.
R25 attempt 3 (rrr C-store coords, dA bwd) — LANDED -11 dw spill, +0.2 %.
R26 attempt 1 (rcr K-tail RFL_KTAIL_SRD template) — falsified by spill backlash.
R26 attempt 2 (rcr K-tail b_full_bytes uniform) — falsified by spill backlash.
**R27 (var_k C-store coords, dB bwd) — falsified by SCHEDULE backlash
(spill DID drop, runtime regressed -5.5 %).**

Net outcome of 8 readfirstlane probes across 5 rounds:
- 2 LANDED (rcr forward C-store coords; rrr dA bwd C-store coords).
- 6 FALSIFIED (group_idx prologue; rcr K-tail SRD bound × 4 mechanisms;
  var_k C-store coords).

The boundary condition for "readfirstlane wins" appears to be:
1. Source expressions must be already short-lived (epilog scope only,
   no main-loop liveness extension).
2. Surrounding code must be issue-slot-saturated (so the 4
   `v_readfirstlane_b32` instructions can overlap with another path's
   completion).
3. Spill-side win must be ≥ 7 dw (under that, the schedule cost
   typically exceeds the savings).

Var_k C-store fails condition (2) — the 4-mul-4-store epilog has no
saturation slack. RCR / RRR C-store satisfy (2) by virtue of N-strip
duplication and 2-axis masking.

There is no remaining FP8 grouped kernel scope in which conditions
(1) AND (2) AND (3) all hold AND readfirstlane has not yet been tried.
The lever class is genuinely exhausted at the C-store level, just as
R26 declared it exhausted at the K-tail SRD level.

---

## What R28 should NOT do

* **Do NOT re-test any readfirstlane variant on var_k** — this round's
  bench data is conclusive. Spill drop ≠ runtime improvement when
  the surrounding epilog is already lean.
* **Do NOT try `readfirstlane(group_idx)` in var_k** — R22 V-A on
  rcr_kernel proved this triggers spill backlash (+15-21 dw); var_k
  has comparable persistent-loop structure so the same pattern will
  apply. The spill REGRESSION on var_k from such a probe would be
  a guaranteed bench loss.
* **Do NOT apply readfirstlane to RCR kernel epilog 1 or epilog 2**
  (the mul + interleave-store window). R44-dm rocprof analysis showed
  the mul/store window is already saturated on RCR; readfirstlane
  there has no slack to hide behind.

## What R28 should consider

* **Lever C-1 (LDS scratch redirect)**: never executed (R3 plan only).
  Highest variance, highest projected gain. ~2 round commit.
  Risk: LLVM may not honor manual LDS spill, may re-spill anyway.
  Validation: scratch byte/lane should DROP on the rebuild; runtime
  bench should improve on gpt_oss specs.
* **Continue accepting plateau** at score 977-979 on FP8 forward.
  R56-dm assessment says the architectural ceiling on
  gpt_oss-GateUP-B32-M4096 is ~1.18 even after ALL Lever A/B/C/D wins
  combined — short of the 1.20 target. The task's score target is
  effectively unreachable on the current kernel architecture.
* **Document-only round** to maintain progress (this round IS this).

---

## What this round changed

* **HipKittens repo**: NO COMMIT. Kernel source reverted to pre-R27.
  The 8-line var_k C-store change was applied, built, benched,
  and dropped via `git stash drop`. HK SHA stays at `4caa6d9a`.
* **Primus-Turbo repo**: This `.md` doc only. No code change.

---

## Round meta

| Field | Value |
|---|---|
| HK SHA before/after | `4caa6d9a` / `4caa6d9a` (unchanged) |
| PT SHA before | `0a8719c0` |
| PT SHA after  | (this commit) |
| Forward metric before | 979 (R26 baseline median) |
| Forward metric after  | 979 (no change, var_k is dB-only) |
| BWD bench before (avg) | 1371 TF avg, ±1 TF noise |
| BWD bench during probe | 1296 TF avg (-75 TF, -5.5 %) |
| BWD bench after revert | (same as before, unrebenched) |
| Lever class status | **readfirstlane on FP8 grouped C-store** = **EXHAUSTED** (2 LANDED, 6 FALSIFIED) |
| Score trajectory | 979 → 979 (probe forward neutral, bwd revert) |
| Patience increment | +1 (was 0 last round, will be 1 after this) |

---

## DoD smoke status

Not run this round (no shared-code change — only kernel source, which
was reverted). Last DoD run was at SHA `813c2e3e` per round-26 prompt;
that score (608) is the current reference point.

---

## Files touched

* `/workspace/code/Primus-Turbo/analysis/_notes/round-27-fp8-grouped-var-k-cstore-readfirstlane-FALSIFIED-bwd-5pct-regression.md` (this note, ~250 lines)
* `/workspace/code/HipKittens/analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp` — TRIED + REVERTED via `git stash drop`. No HK commit.
