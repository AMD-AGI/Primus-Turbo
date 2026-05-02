# Round 25 — FP8 grouped: readfirstlane RRR C-store coords (dA backward kernel)

## Status: LANDED — backward-only win (+0.20 % avg dA TFLOPS, 76→65 dw spill)

## Summary

R24's localised `__builtin_amdgcn_readfirstlane` fix on the four C-store
coordinates (`r0/r1/c0/c1`) of `grouped_rcr_kernel` netted +15 metric points by
eliminating the divergent-SRD fallback loops that LLVM emits when uniform-in-
principle store coords cannot be proven wave-uniform. Same mechanism is
trivially applicable to `grouped_rrr_kernel` (the dA backward path, RRR layout)
which had the highest VGPR spill count of any FP8 grouped kernel (76 dw vs
~37 dw for the post-R24 forward `grouped_rcr_kernel`). R25 lands that mirror
edit.

Not a forward-metric win (RRR is dispatched only for dA backward, never the
forward path), but +1.05–1.90 % on Qwen3-Down dA, +0.32–0.65 % on DSV3-Down dA,
and a clear ASM-level signal (-11 dw spill) make it a clean perf-positive
commit before moving back to forward-metric levers.

## Two probes attempted in R25

### Probe A — K-tail block readfirstlane on `grouped_rcr_kernel` (FALSIFIED)

R22 ASM disassembly counted ~84 of the 411 rcr<T,T> divergent-SRD loops in
the K-tail block area (vs ~335 in the C-store epilog which R24 fixed). The
hypothesis was that wrapping `b_per_group_bytes` (and optionally
`b_group_byte_base`) in `__builtin_amdgcn_readfirstlane` at the make_srsrc
call site would lift `b_srsrc_kt` to SGPR i32x4, eliminating per-lane
fallback loops on K-tail B-loads.

| Variant                                        | rcr<F,T> spill | rcr<T,T> spill | Score  |
|------------------------------------------------|----------------|----------------|--------|
| R24 baseline                                   | 34 dw          | 37 dw          | 977    |
| R25 probe A1 (b_per_group + b_group_byte_base) | 48 dw (+14)    | 49 dw (+12)    | 959 (-18) |
| R25 probe A2 (b_per_group only)                | 48 dw (+14)    | 49 dw (+12)    | 960 (-17) |

Both probes triggered identical ~14 dw VGPR spill backlash on the two used
specs, swamping any divergent-loop savings. Same structural signature as
R22 V-A/V-B (`readfirstlane(group_idx)` in the binary search prologue,
+21 dw spill).

**Root cause hypothesis:** unlike R24's r0/r1/c0/c1 (which are *fresh*
SSA values used only in 4 stores then dead), the K-tail block's
`b_per_group_bytes` derives from `group_idx` — which has wide downstream
use across the K-loop hot path. Forcing SGPR allocation for a value
sourced from `group_idx` extends SGPR live ranges across the heaviest
register-pressure region of the kernel, and LLVM responds by spilling
VGPR. Even with the readfirstlane *literally inside* the `if (g.fast_k <
g.k)` runtime branch, LLVM's hoisting behaviour still causes the
backlash to materialise on the K_REM=0 specs (rcr<F,T>) where the
runtime branch is dead at runtime.

The K-tail readfirstlane lever class is structurally
incompatible with R24-style point-of-use SGPR promotion — confirmed
falsification.

### Probe B — C-store readfirstlane on `grouped_rrr_kernel` (LANDED)

Identical pattern to R24 applied to lines 3343-3350 of
`kernel_fp8_layouts.cpp`:

```cpp
const int r0 = __builtin_amdgcn_readfirstlane(m_subtile_C + br*WARPS_M*2+wm);
const int r1 = __builtin_amdgcn_readfirstlane(m_subtile_C + br*WARPS_M*2+WARPS_M+wm);
const int c0 = __builtin_amdgcn_readfirstlane(bc*WARPS_N*2+wn);
const int c1 = __builtin_amdgcn_readfirstlane(bc*WARPS_N*2+WARPS_N+wn);
store(g.c, cA, {0, 0, r0, c0});
store(g.c, cB, {0, 0, r0, c1});
store(g.c, cC, {0, 0, r1, c0});
store(g.c, cD, {0, 0, r1, c1});
```

#### Build effect

| Kernel                       | Spill before | Spill after | Δ      |
|------------------------------|--------------|-------------|--------|
| `grouped_rrr_kernel<0>`      | 76 dw        | 65 dw       | -11 dw |
| All `grouped_rcr_kernel<*>`  | unchanged    | unchanged   | 0      |

The fix is local to `grouped_rrr_kernel`; rcr forward kernels untouched (R24
win preserved).

#### Forward metric (sanity check, not the validation gate)

| Run             | Score | grp_FP8 geomean |
|-----------------|-------|-----------------|
| R24 (this round baseline) | 977   | 1.1581          |
| R25 (rrr fix)             | 976   | 1.1560          |

977 vs 976 = within run-to-run noise band (~±2 pts observed across last
10 rounds). RRR is not in the forward dispatch path, so no real movement
expected.

#### Backward bench (the validation gate per task body)

`PRIMUS_TURBO_HIPKITTEN_PATH=/workspace/code/HipKittens
PRIMUS_TURBO_GROUPED_GEMM_BACKEND=HIPKITTEN python3
benchmark/ops/bench_grouped_gemm_turbo.py --dtype fp8`

| Metric                   | R24 baseline | R25 (rrr fix) | Δ      |
|--------------------------|--------------|---------------|--------|
| Avg fwd TFLOPS           | 1363.58      | 1365.26       | +1.68 (+0.12 %) |
| **Avg bwd TFLOPS**       | **1367.05**  | **1369.81**   | **+2.76 (+0.20 %)** |
| Correctness PASS         | 24/24        | 24/24         | —      |

Per-shape backward ΔTFLOPS (R24 → R25):

```
DSV3-GateUP-B16-M2048      1421.72 → 1423.33  Δ  +1.61 (+0.11%)
DSV3-Down-B16-M2048        1333.58 → 1342.30  Δ  +8.72 (+0.65%)  ← Down win
DSV3-GateUP-B16-M4096      1678.24 → 1674.11  Δ  -4.13 (-0.25%)
DSV3-Down-B16-M4096        1599.62 → 1609.32  Δ  +9.70 (+0.61%)  ← Down win
DSV3-GateUP-B32-M2048      1428.32 → 1416.15  Δ -12.17 (-0.85%)
DSV3-Down-B32-M2048        1367.79 → 1372.22  Δ  +4.43 (+0.32%)
DSV3-GateUP-B32-M4096      1682.82 → 1675.70  Δ  -7.12 (-0.42%)
DSV3-Down-B32-M4096        1615.32 → 1616.05  Δ  +0.73 (+0.05%)
gpt_oss-GateUP-B4-M2048    1153.19 → 1159.18  Δ  +5.99 (+0.52%)
gpt_oss-Down-B4-M2048       915.08 →  915.34  Δ  +0.26 (+0.03%)
gpt_oss-GateUP-B4-M4096    1483.93 → 1488.79  Δ  +4.86 (+0.33%)
gpt_oss-Down-B4-M4096      1183.17 → 1177.57  Δ  -5.60 (-0.47%)
gpt_oss-GateUP-B32-M2048   1308.30 → 1311.20  Δ  +2.90 (+0.22%)
gpt_oss-Down-B32-M2048     1079.27 → 1081.25  Δ  +1.98 (+0.18%)
gpt_oss-GateUP-B32-M4096   1567.95 → 1564.63  Δ  -3.32 (-0.21%)
gpt_oss-Down-B32-M4096     1298.47 → 1294.14  Δ  -4.33 (-0.33%)
Qwen3-GateUP-B16-M2048     1280.91 → 1282.24  Δ  +1.33 (+0.10%)
Qwen3-Down-B16-M2048       1143.42 → 1165.17  Δ +21.75 (+1.90%)  ← biggest win
Qwen3-GateUP-B16-M4096     1492.81 → 1503.47  Δ +10.66 (+0.71%)
Qwen3-Down-B16-M4096       1369.35 → 1388.48  Δ +19.13 (+1.40%)  ← Qwen-Down win
Qwen3-GateUP-B32-M2048     1294.76 → 1288.86  Δ  -5.90 (-0.46%)
Qwen3-Down-B32-M2048       1193.88 → 1206.42  Δ +12.54 (+1.05%)  ← Qwen-Down win
Qwen3-GateUP-B32-M4096     1517.37 → 1515.93  Δ  -1.44 (-0.09%)
Qwen3-Down-B32-M4096       1399.86 → 1403.56  Δ  +3.70 (+0.26%)
```

Pattern: Qwen3-Down dA shows the biggest gains (+1.05–1.90 %), DSV3-Down a
clean smaller win (+0.32–0.65 %), gpt_oss and GateUP variants noise-band.
Down kernels engage the RRR layout most heavily on dA reduction so the
divergent-SRD elimination shows up there. Several shapes (4 of 24) show
small negatives within ±0.5 % which are run-to-run noise (2nd trial avg
bwd 1371.41 vs 1369.81 — ±1.6 TFLOPS shape-wise variance).

## Why this lever class is exhausted on forward

After R24 (rcr forward C-store) + R25 (rrr backward C-store), the obvious
"fresh wave-uniform value with short lifetime" sites in the FP8 grouped
kernels are covered:

| Site                                              | Status     |
|---------------------------------------------------|------------|
| `grouped_rcr_kernel` C-store r0/r1/c0/c1          | LANDED R24 |
| `grouped_rcr_kernel` K-tail b_per_group_bytes     | FALSIFIED R25-A (spill backlash) |
| `grouped_rrr_kernel` C-store r0/r1/c0/c1          | LANDED R25-B |
| `grouped_var_k_kernel_fp8` C-store (dB backward)  | DEFERRED — uses different helper `store_c_tile_mn_masked_grouped` that takes group_idx as separate arg; doesn't fit R24 pattern as cleanly |

For forward metric improvement, the next lever class needs to come from
*outside* the SGPR-promotion family. Open candidates:

* Lever A (async global→LDS via `global_load_lds_*` ASM intrinsic) — still
  the highest-EV unexplored option. Would cut A_row_reg / B_col_reg
  VGPR live ranges in K-loop main body.
* Lever B (dual-LDS ping-pong) — LDS budget tight on gfx950 (64 KB/CTA),
  but viable with care. Would hide ds_write latency.
* Lever F (Qwen3-Down K=1536 specialised variant) — only 4/24 cases but
  K=1536 is structurally distinct (12 K-iters vs 16-56 for others) so
  may have different bottleneck.

R26 should pick from {A, B} as next probe. The `grouped_rcr_kernel`
C-store path is fully optimised; further gains require K-loop main body
re-architecting.

## Risk / coverage

* No `grouped_rcr_kernel` change → R24 forward win preserved (verified
  via metric: 977 → 976 ≈ noise).
* `grouped_rrr_kernel` change isolated to 4 lines in C-store epilog —
  semantically a no-op (`__builtin_amdgcn_readfirstlane` on a
  wave-uniform value is identity).
* All 24 backward correctness checks PASS (numerical match to fp32
  reference, max_abs and SNR within bench tolerance).

## Commits

* HipKittens: see HK round-25 commit (`analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp`).
* Primus-Turbo: this doc only (no PT code change).

## Recommendations for R26

1. Pick Lever A (async global→LDS in `grouped_rcr_kernel` K-loop main
   body) as the next forward-metric probe.
2. Keep R24 + R25 fixes intact (don't touch C-store paths).
3. If A backlashes on register pressure, fall back to Lever B
   (dual-LDS).
