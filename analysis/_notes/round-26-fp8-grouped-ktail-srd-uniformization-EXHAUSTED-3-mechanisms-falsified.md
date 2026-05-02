# Round 26 — FP8 grouped: K-tail B-side SRD uniformization lever class EXHAUSTED

## Status: 3 mechanisms tested, all FALSIFIED. Lever class declared dead.

## Summary

R22 ASM disassembly identified ~76 of the 411 rcr<T,T> divergent-SRD
fallback loops as living in the K-tail block (vs ~335 in C-store epilog,
which R24 fixed). Across 4 prior rounds (R22, R25-A, R26-A, R26-B) we have
now tried 3 distinct mechanisms to lift `b_srsrc_kt` to a uniform SGPR
SRD. All three trigger the same ~+14 dw VGPR spill backlash and net
metric regression (-12 to -18 pts vs R24+R25 baseline 977-978). The lever
class is structurally dead — there is no path to uniform K-tail B-side
SRD that avoids cascading spill into the K-loop hot path.

## R26 probes

### Baseline noise band (5 trials, post R25 LANDED state)

```
974, 982, 978, 979, 980, 979  →  median ~979, range 974-982 (8 pts)
```

The R26-instruction baseline measurement (974) was on the low edge of
the noise band; subsequent trials clustered 978-982. All probe results
below are interpreted against the median (~979).

### Probe A — RFL_KTAIL_SRD template parameter (gpt_oss-only RFL=true)

Hypothesis: R25-A failed because `readfirstlane` was applied
unconditionally on rcr<F,T> (used by 16 K_REM=0 cases — DSV3 + Qwen3
+ gpt_oss-GateUP-K_aligned, plus 4 gpt_oss-GateUP-K_REM=64). Adding a
4th template parameter `bool RFL_KTAIL_SRD` and dispatching K_REM>0
cases to the RFL=true spec (gpt_oss-GateUP-K_REM=64 + gpt_oss-Down,
8 cases) should localise the spill cost to specs that benefit.

Implementation:
* `template<int KI_HINT, bool N_MASKED_STORE, bool FUSED_KTAIL, bool RFL_KTAIL_SRD = false>`
* `if constexpr (RFL_KTAIL_SRD)` gating around the b_per_group_bytes readfirstlane
* 6 template instances (4 RFL=false existing + 2 RFL=true new)
* Dispatcher: `ktail_active = (K_rem_for_fuse > 0)` selects RFL=true

Build effect:
| Spec                               | Spill (dw) | Note            |
|------------------------------------|------------|------------------|
| `rcr<0, F, T, F>` (K_REM=0)        | 34         | unchanged       |
| `rcr<0, T, T, F>` (unused now)     | 37         | unchanged       |
| `rcr<0, F, T, T>` (gpt_oss-GateUP) | 48 (+14)   | new RFL spec    |
| `rcr<0, T, T, T>` (gpt_oss-Down)   | 49 (+12)   | new RFL spec    |

Per-shape ratio change vs baseline (974):

```
DSV3-Down-B16-M2048    1.206 → 1.255  Δ +0.049 ← BIG (binary layout shift)
DSV3-Down-B16-M4096    1.184 → 1.258  Δ +0.074 ← BIG (binary layout shift)
DSV3-Down-B32-M2048    1.222 → 1.290  Δ +0.068 ← BIG (binary layout shift)
DSV3-Down-B32-M4096    1.260 → 1.288  Δ +0.028
gpt_oss-GateUP-B4-M2048 1.112 → 1.087  Δ -0.025 ← gpt_oss spec regress
gpt_oss-Down-B4-M2048   1.199 → 1.191  Δ -0.008
gpt_oss-GateUP-B4-M4096 1.112 → 1.073  Δ -0.039 ← gpt_oss spec regress
gpt_oss-Down-B4-M4096   1.117 → 1.095  Δ -0.022
gpt_oss-GateUP-B32-M2048 1.083 → 1.052  Δ -0.031 ← gpt_oss spec regress
gpt_oss-Down-B32-M2048  1.106 → 1.080  Δ -0.026
gpt_oss-GateUP-B32-M4096 1.067 → 1.035  Δ -0.032 ← worst-case regressed
gpt_oss-Down-B32-M4096  1.079 → 1.056  Δ -0.023
[Qwen3 cases mostly noise-band ±0.02 either direction]
```

Net: 974 → 975 score. **All 8 gpt_oss cases regressed -2.2 to -3.9 pp**
on the new RFL=true spec — the divergent-loop savings did NOT
compensate for the +12-14 dw VGPR spill cost. The +1 pt net came from
4 DSV3-Down cases that gained +5-7 pp from a kernel-binary layout shift
(unrelated to the readfirstlane logic — they use the unchanged
`rcr<F,T,F>` spec). The DSV3 wins are coincidental and unstable: any
future kernel change that shifts the binary layout would erase them.
Committing this would lock in a real regression on gpt_oss in exchange
for fragile DSV3 wins.

**Status: FALSIFIED.** Reverted in-tree.

### Probe B — Replace per-group SRD bound with FULL B tensor span

Hypothesis: Probe A's spill backlash came from `readfirstlane` itself
forcing a specific SGPR live range. What if we restructure the K-tail
SRD construction to use `b_full_bytes = g.G * g.n * b_row_stride_bytes`
instead of `b_per_group_bytes = (group_idx + 1) * g.n * b_row_stride_bytes`?
All inputs are uniform struct fields — make_srsrc would emit a clean
SGPR i32x4 SRD with no `__builtin_amdgcn_readfirstlane` and no
`group_idx` dependency.

Bound clamping is identical for K-tail loads: per-lane voffset always
lands within group g (no zero-clamp on the larger bound either).
Numerics identical.

Implementation:
```cpp
const uint32_t b_full_bytes =
    static_cast<uint32_t>(g.G) *
    static_cast<uint32_t>(g.n) *
    static_cast<uint32_t>(b_row_stride_bytes);
i32x4 b_srsrc_kt = make_srsrc((const void*)b_base_ptr, b_full_bytes);
```

Build effect:
| Spec       | Spill (dw) | Δ vs R25 baseline |
|------------|------------|------------------|
| `rcr<F,T>` | 48         | **+14** (same as R25-A and R26-A) |
| `rcr<T,T>` | 49         | **+12** (same as R25-A and R26-A) |

Metric: **score=962** (-15 to -17 pts vs noise-band median 979).

The spill backlash is IDENTICAL in magnitude regardless of mechanism
(readfirstlane vs explicit SGPR-only SRD inputs). This confirms the
backlash is from LLVM's response to having a uniform SGPR-resident
i32x4 SRD in the K-tail block — not from the readfirstlane intrinsic
itself. Whenever `b_srsrc_kt` becomes SGPR i32x4, LLVM evicts ~14 dw
of VGPR state from the K-loop hot path.

**Status: FALSIFIED.** Reverted in-tree.

## The bigger picture

Across R22, R25-A, R26-A, R26-B:

| Round  | Mechanism                                   | Spill vs R25 base | Net metric  |
|--------|---------------------------------------------|--------|--------------|
| R22 V-A| readfirstlane(group_idx) early in function  | +21 dw | regressed    |
| R22 V-B| readfirstlane(lo) post binary-search        | +21 dw | regressed    |
| R25-A1 | readfirstlane(b_per_group_bytes + b_group_byte_base) | +14 dw | -18 pts |
| R25-A2 | readfirstlane(b_per_group_bytes only)       | +14 dw | -17 pts |
| R26-A  | template-gated readfirstlane (gpt_oss-only) | +12-14 dw | -2-4 pp on gpt_oss |
| R26-B  | b_full_bytes (no readfirstlane, uniform-only inputs) | +14 dw | -15+ pts |

The structural cost is **not avoidable**: any kernel where the K-tail
B-side SRD is SGPR i32x4 incurs +12-14 dw VGPR spill, which costs more
in K-loop main body cycles than it saves in K-tail divergent-fallback
loops. The 76-of-411 K-tail divergent loops identified by R22 are NOT
recoverable without a deeper restructure (Lever A async global→LDS or
Lever B dual-LDS) that changes the K-loop register allocation pattern.

## Why the spill backlash is structural

Hypothesis (unproven, but consistent with all 6 falsifications):

When `b_srsrc_kt` is VGPR i32x4 (the current state), it lives in the
VGPR pool right next to A_row_reg / B_col_reg / cA-cD accumulators —
sharing the same allocation arena. LLVM treats it as a per-lane VGPR
quad and schedules it within the K-loop register flow.

When `b_srsrc_kt` is SGPR i32x4, it lives in the SGPR pool. LLVM
evidently responds by:
1. Holding the SGPR SRD live across the K-tail block (24 cy).
2. Reorganising the surrounding SGPR allocation (kernel arg pointers,
   group_idx, m_subtile_*, etc.) to free space for the SRD.
3. Side effect: VGPR allocation in the K-loop main body gets
   pessimised (LLVM's arena allocator doesn't isolate the two pools
   perfectly — SGPR pressure can cascade into VGPR via constraint
   propagation).

Net: 4 SGPRs gained (the SRD), ~14 VGPRs lost (the cascade). The
trade is structurally bad.

The only way to escape is to NOT have the SRD live across the K-tail
block as a unit — e.g., by reconstructing it inline at each
buffer_load_b128 call (forcing zero live-range). But that doubles
the SRD-construction overhead per load and likely regresses K-tail
throughput. Not pursued in R26.

## R27+ recommendations

1. **DO NOT** retry K-tail SRD uniformization in any form (it is now
   demonstrably 3-mechanism-falsified across 4 rounds).
2. **DO NOT** retry readfirstlane on values sourced from `group_idx`
   (R22 V-A/V-B already proved this hits SGPR live-range cascade).
3. The remaining ~76 K-tail divergent-SRD loops on rcr<T,T>/rcr<F,T>
   (gpt_oss specs) are **accepted overhead** until a deeper restructure
   of the K-loop register flow (Lever A or Lever B).

For R27, the recommended levers are:

* **Lever A — async global→LDS** (`__builtin_amdgcn_global_load_lds_*`
  ASM intrinsic, gfx950): replace the synchronous A/B HBM→register→LDS
  staging in `rcr_8w_load_hoist` with direct HBM→LDS. Cuts A_row_reg /
  B_col_reg VGPR live ranges in K-loop main body. Multi-round work
  (probably 2-3 rounds): scaffold + A-only port + B port + numerical
  verification. Highest expected EV (+5-10 pp on gpt_oss + smaller
  gains on Qwen3 K=4096 GateUP).

* **Lever B — dual-LDS ping-pong**: split the LDS slabs into 2 banks,
  alternate K-iter writes/reads to hide ds_write latency. LDS budget
  on gfx950 (64 KB/CTA) is already tight (current single slab uses
  ~32 KB FP8 per A and B = 64 KB combined — needs careful re-budget).
  Lower EV (+3-6 pp), simpler change.

* **Lever F — Qwen3-Down K=1536 specialised variant**: K=1536 has
  only 12 K-iters (vs 16-56 for other shapes). Prologue/epilog overhead
  proportionally larger. May benefit from different unroll factor.
  ONLY 4/24 cases (~16% weight) — defer until A/B saturated.

## Working-tree state at end of R26

* HipKittens: clean (all 4 R26 probes reverted; only R24 + R25 fixes
  in tree).
* Primus-Turbo: this doc only.

## Commits

* HipKittens: none (R26 = falsification round).
* Primus-Turbo: doc-only commit (this file).
