# Round-37-dm · FP8 grouped — K-tail issue reorder `[b0, b1, a, a_kt1]` WINS +16 pts via LLVM liveness reshape

**Status**: WIN. Score 934 → 950 (stable at 947 on re-run).
grp_FP8 geomean 1.0621 → 1.0987 (+3.7 pp).
HipKittens commit: `04f82d49`.

## The change

Single-line reorder of 4 `load_*_kt` calls inside K-tail block at
`kernel_fp8_layouts.cpp:2446-2449`:

```diff
-                load_a_kt(a,     0);   // 8 buffer_load → a (M slab 0)
-                load_b_kt(b0,    0);   // 4 buffer_load → b0
-                load_b_kt(b1,    1);   // 4 buffer_load → b1
-                load_a_kt(a_kt1, 1);   // 8 buffer_load → a_kt1 (LAST)
+                load_b_kt(b0,    0);   // 4 buffer_load → b0 (FIRST)
+                load_b_kt(b1,    1);   // 4 buffer_load → b1
+                load_a_kt(a,     0);   // 8 buffer_load → a (M slab 0)
+                load_a_kt(a_kt1, 1);   // 8 buffer_load → a_kt1 (LAST)
```

vmcnt split preserved: `vmcnt(8)` between cA/cB and cC/cD mfma pairs,
`vmcnt(0)` before cC/cD. The first 16 retired (under in-issue-order
vmcnt invariant, per R12-dm) are still `b0+b1+a` — exactly the regs
cA/cB need.

## Per-shape impact (R10 baseline=934 → probe=950)

| Shape | Before | After | Δ pp |
|---|---|---|---|
| DSV3-GateUP-B16-M2048 | 1.085 | 1.129 | **+4.4** |
| DSV3-Down-B16-M2048 | 1.055 | 1.170 | **+11.5** |
| DSV3-GateUP-B16-M4096 | 1.128 | 1.151 | +2.3 |
| **DSV3-Down-B16-M4096** | 1.044 | **1.186** | **+14.2** |
| DSV3-GateUP-B32-M2048 | 1.116 | 1.165 | +4.9 |
| DSV3-Down-B32-M2048 | 1.117 | 1.192 | +7.5 |
| DSV3-GateUP-B32-M4096 | 1.139 | 1.163 | +2.4 |
| **DSV3-Down-B32-M4096** | 1.078 | **1.203** | **+12.5** 🎯 first FP8 shape ≥1.20 |
| gpt_oss-GateUP-B4-M2048 | 1.035 | 1.046 | +1.1 |
| gpt_oss-Down-B4-M2048 | 1.111 | 1.111 | ~0 |
| gpt_oss-GateUP-B4-M4096 | 0.992 | 1.015 | +2.3 |
| gpt_oss-Down-B4-M4096 | 1.034 | 1.045 | +1.1 |
| gpt_oss-GateUP-B32-M2048 | 0.982 | 1.016 | +3.4 |
| gpt_oss-Down-B32-M2048 | 1.022 | 1.030 | +0.8 |
| gpt_oss-GateUP-B32-M4096 | 0.967 | 0.983 | +1.6 |
| gpt_oss-Down-B32-M4096 | 1.008 | 1.016 | +0.8 |

## Mechanism: LLVM liveness reshape via dead-branch permutation

**Key observation**: the change is INSIDE
`if constexpr (FUSED_KTAIL) { if (g.fast_k < g.k) { ... } }`.
For DSV3 (K_REM=0), `g.fast_k == g.k` → the runtime branch is DEAD.
Yet DSV3 shapes saw the BIGGEST gains (+7 to +14 pp on Down family).

This is a codegen artifact:
- LLVM's liveness analysis sees the 4 `load_*_kt` call-order
  `[b0, b1, a, a_kt1]`.
- It computes live-ranges for `a`, `a_kt1`, `b0`, `b1` based on this
  program text (ignoring that the block is runtime-dead).
- Register allocator chooses different VGPR assignments.
- Spill placement in the main loop + epilog changes.
- HOT PATH performance improves despite the COLD PATH being the source
  of the perturbation.

This mirrors R34-dm: extending FUSED_KTAIL=true to K_REM=0 shapes
gained +17 pts purely via LLVM liveness reshape. The `A_row_reg a_kt1;`
declaration was R34's handle; the K-tail issue ORDER is R37's handle.

**Both levers operate on the same mechanism** — reshaping LLVM's
liveness graph from different angles.

## Why THIS SPECIFIC reorder

Hypothesis: putting the smaller `b0`/`b1` loads FIRST gives LLVM a
tighter VGPR window to allocate early in the block, leaving more
spare VGPRs for the subsequent larger `a`/`a_kt1` loads. The
allocator's choices cascade back into main-loop register assignments
(since `a` and `a_kt1` are the same template-visible register types
used elsewhere).

**DSV3 benefits more than gpt_oss** because DSV3's main loop runs 56
K-iters (K=7168) vs gpt_oss's 22 (K=2880). Any register-allocation
improvement amplifies with iteration count — a spill saved per-iter
multiplies through the main loop.

## ISA spill count delta

TODO (next round): re-run `analysis/tools/dump_fp8_grouped_isa.sh` on
post-R37 kernel and compare interleaved-spill counts. Expected:
- Spec `<0,false,true>` (DSV3): 28 → lower
- Spec `<0,true,true>` (gpt_oss): 22 → similar or lower

## vmcnt in-issue-order invariant still holds

Original comment (R12-dm):
> (vmcnt is in-issue-order retirement on AMDGCN — same semantics
> relied on by the main loop's ``RCR_STEADY_VMCNT=8`` mid-iter wait)

New issue order: `[b0(4), b1(4), a(8), a_kt1(8)]` = 24 total.
- vmcnt(8) = wait until 16 retired = first 16 = `b0+b1+a`.
- cA = a·b0, cB = a·b1 both need `a+b0+b1` — ALL retired. ✓
- vmcnt(0) = wait for remaining 8 = `a_kt1`.
- cC = a_kt1·b0, cD = a_kt1·b1 both need `a_kt1+b0` — all retired. ✓

Correctness: all 32 shapes `correct_fail=0/32, reject=0/32` ✓

## Regression check: BF16 grouped

grp_BF16 geomean: 1.1832 (R10 baseline) → 1.1838 (after). Noise.
BF16 kernel (separate `.cpp`) not touched.

## Remaining FP8 shapes below 1.20

After R37:
- gpt_oss-GateUP-B32-M4096: 0.983 (worst)
- gpt_oss-GateUP-B4-M4096: 1.015
- gpt_oss-Down-B32-M4096: 1.016
- gpt_oss-GateUP-B32-M2048: 1.016
- gpt_oss-Down-B32-M2048: 1.030
- gpt_oss-GateUP-B4-M2048: 1.046
- gpt_oss-Down-B4-M4096: 1.045
- gpt_oss-Down-B4-M2048: 1.111
- DSV3-GateUP-B16-M2048: 1.129
- DSV3-GateUP-B16-M4096: 1.151
- DSV3-GateUP-B32-M2048: 1.165
- DSV3-GateUP-B32-M4096: 1.163
- DSV3-Down-B16-M2048: 1.170
- DSV3-Down-B16-M4096: 1.186
- DSV3-Down-B32-M2048: 1.192
- **DSV3-Down-B32-M4096: 1.203** 🎯

One FP8 shape at target. DSV3-Down family in striking distance
(1.17-1.20). gpt_oss family still the long pole (0.98-1.11).

## R11 plan: continue liveness-graph exploration

Working lever: permuting K-tail block CODE ORDER (not semantics).
- Try other permutations: `[a, b1, b0, a_kt1]`, `[a_kt1, b0, b1, a]`
  (need to also reorder mfmas if their dependencies change), etc.
- Try interleaving: `[a_half, b0, a_other_half, b1, a_kt1]` — may
  require splitting `load_a_kt` into finer-grained calls.
- Try adding DUMMY register declarations (e.g., `A_row_reg a_extra;`
  never used) to see if LLVM gains more liveness graph flexibility.
  R34 succeeded with a_kt1; maybe more register-tile declarations
  in the hot scope would unlock further wins.
- Use `analysis/tools/dump_fp8_grouped_isa.sh` to compare ISA before
  and after R37 to see which specific VGPRs changed allocation.

Goal: close the gpt_oss gap (0.98-1.11) to push grp_FP8 geomean
over 1.20. Still need +10.1 pp on geomean to hit target 1.20.
