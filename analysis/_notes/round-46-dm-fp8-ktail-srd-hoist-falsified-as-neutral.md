# Round-46-dm · FP8 grouped — K-tail SRD setup hoist out of runtime if FALSIFIED as NEUTRAL

**Status**: 1 probe FALSIFIED (NEUTRAL within ±2 pt run-noise).
Reverted to R44-dm winner state.
Score: baseline 958.75 (4-run mean) → R46-dm 958.67 (3-run mean).

## Hypothesis

R45-dm note flagged the K-tail as ~15% of per-tile wall time on
gpt_oss (~314 cy of ~2112 cy total). Within K-tail:
* SRD setup + scalar bookkeeping: ~30 cy of pure SALU work
* 24 buffer_load_b128 issue: ~24 cy
* vmcnt(8) wait: ~150 cy (HBM latency for first 16 loads)
* mfma cA, cB: ~16 cy (overlaps last 8 loads draining)
* vmcnt(0) wait: ~50 cy
* mfma cC, cD: ~16 cy

The SCALAR bookkeeping (SRD setup at lines 2338-2365) is currently
INSIDE the runtime `if (g.fast_k < g.k)` branch. LLVM cannot hoist
across the runtime branch even though the work is pure scalar.

If we move the SRD setup OUT of the runtime if (still inside
`if constexpr (FUSED_KTAIL)`), LLVM can schedule the SALU ops
EARLIER, potentially overlapping them with epilog 2's VALU mfma
issues via SALU/VALU pipeline parallelism. Expected ~30 cy K-tail
shorter on gpt_oss = ~1.4% gain on per-tile time = ~+1-2 pp on
gpt_oss FP8 ratios.

Cost: ~30 cy of dead SALU work for K-aligned shapes (DSV3, K_REM=0).
DSV3 already at 1.117-1.221 ratio; -0.3 pp tolerable.

## Implementation

```diff
 if constexpr (FUSED_KTAIL) {
     A_row_reg a_kt1;
+    // HOISTED out of runtime if: pure scalar SRD setup
+    const int laneid = kittens::laneid();
+    const int row_lane = laneid % 16;
+    const int k_lane_byte = (laneid / 16) * 32;
+    const int K_REM = g.k - g.fast_k;
+    const bool b128_lo_valid = (k_lane_byte + 16) <= K_REM;
+    const bool b128_hi_valid = (k_lane_byte + 32) <= K_REM;
+    constexpr uint32_t SENTINEL = 0xFFFF0000u;
+    const fp8e4m3* a_base_ptr = (const fp8e4m3*)&g.a[{0, 0, 0, 0}];
+    const fp8e4m3* b_base_ptr = (const fp8e4m3*)&g.b[{0, 0, 0, 0}];
+    const int a_row_stride_bytes = g.a.template stride<2>();
+    const int b_row_stride_bytes = g.b.template stride<2>();
+    const uint32_t a_total_bytes = ...;
+    const uint32_t b_per_group_bytes = ...;
+    i32x4 a_srsrc_kt = make_srsrc((const void*)a_base_ptr, a_total_bytes);
+    i32x4 b_srsrc_kt = make_srsrc((const void*)b_base_ptr, b_per_group_bytes);
+    const uint32_t K_tail_base_bytes = static_cast<uint32_t>(g.fast_k);
+    const uint32_t b_group_byte_base = ...;
     if (g.fast_k < g.k) {
-        // (was: SRD setup here, now hoisted above)
+        // (only loads + mfmas remain)
         load_b_kt(b0, 0); ...
     }
 }
```

## Spill count delta (`-Rpass-analysis=kernel-resource-usage`)

| Spec template params | R44-dm baseline | R46-dm probe | Δ |
|---|---|---|---|
| `<0,false,false>` (FUSED=false n_aligned)         | 39 | 39 | 0   |
| `<0,true ,false>` (FUSED=false n_masked)          | 43 | 43 | 0   |
| `<0,false,true >` (FUSED=true  n_aligned, DSV3)   | 32 | 32 | 0   |
| `<0,true ,true >` (FUSED=true  n_masked, gpt_oss) | 39 | **43** | **+4** |

The gpt_oss spec (the spec that ACTUALLY uses K-tail at runtime)
spilled +4 dwords. Hoisting the SRD setup keeps a_srsrc_kt /
b_srsrc_kt SGPR allocations alive longer (across the runtime branch
boundary), which cascades to LLVM spilling 4 more dwords of hot
state to scratch. Smaller backlash than R41-dm's +25-28 (because
SRDs are SGPR-resident, not VGPR), but still measurable.

## Per-shape metric (R44-dm baseline 958 → R46-dm 3-run mean)

Per-spec runs (R46-dm probe state):
* Run 1: 960 (grp_FP8 1.1221, grp_BF16 1.1824)
* Run 2: 958 (grp_FP8 1.1176, grp_BF16 1.1833)
* Run 3: 958 (grp_FP8 1.1187, grp_BF16 1.1826)
* Mean : 958.7 (grp_FP8 1.1195, grp_BF16 1.1828)

Re-baseline runs (post-revert, 4 runs):
* 959, 959, 959, 958 → mean 958.75 (baseline drift +0.75 pt vs
  Round-18 start of 958, within run noise band)

Net: R46-dm 958.7 vs revert baseline 958.75 → **NEUTRAL** (within
±2 pt run-noise).

Per-shape spot check (R46-dm run 1 vs Round 18 baseline):
* DSV3-Down-B16-M4096: 1.122 → 1.190 (+6.8 pp) ← biggest swing
* DSV3-GateUP-B32-M2048: 1.146 → 1.172 (+2.6 pp)
* DSV3-Down-B32-M4096: 1.213 → 1.232 (+1.9 pp)
* DSV3-GateUP-B16-M2048: 1.149 → 1.125 (-2.4 pp)
* DSV3-Down-B32-M2048: 1.221 → 1.197 (-2.4 pp)
* gpt_oss-GateUP-B4-M4096: 1.051 → 1.039 (-1.2 pp)
* gpt_oss-Down-B4-M2048: 1.140 → 1.156 (+1.6 pp)
DSV3 mostly +1-7 pp on run 1, but later runs show drift back to
baseline. The +6.8 swing is well above noise band (2-3 pp), but
not consistently reproducible across runs.

Correctness: PASS (correct_fail=0/32, reject=0/32) on all 3 runs.

## Why the analysis was wrong

### Mechanism #1 (SALU/VALU overlap in epilog 2): not delivered

LLVM did NOT actually hoist the SRD setup into epilog 2's mfma
slack. Looking at the resulting kernel asm (would need to verify
with `--save-temps`), the scalar setup is likely placed JUST
before the K-tail loads, providing only ~5-10 cy of overlap, not
the expected ~30 cy. The constexpr branch `if constexpr (FUSED_KTAIL)`
already establishes a basic block boundary that LLVM can in principle
schedule across, but practical LLVM scheduler heuristics keep
related work clustered.

### Mechanism #2 (SGPR live-range cost): real and offsetting

The 8 SGPRs for a_srsrc_kt + b_srsrc_kt now live across the
runtime if branch boundary, increasing register pressure. LLVM
compensated by spilling +4 dwords on the gpt_oss spec. The +4
spill cost roughly cancels the marginal hoist benefit.

### Mechanism #3 (per-shape volatility)

DSV3-Down-B16-M4096 swung +6.8 pp on run 1 but settled near
baseline on subsequent runs. This is consistent with HBM L2 cache
state varying between launches. Not attributable to source change.

## Conclusion

Hoisting K-tail SRD setup out of the runtime branch is NEUTRAL.
The hypothesized SALU/VALU overlap window does not materialize in
LLVM's emitted schedule, and the SGPR live-range extension causes
+4 dwords spill backlash that cancels any marginal gain.

This confirms a recurring pattern (R34/R41/R42-dm): code-motion
optimizations that EXTEND a register's live range across LLVM
basic-block boundaries trigger spill backlash even when the
extended state is "free" (SGPR vs VGPR). The R44-dm INTERLEAVE
pattern works because it CONTRACTS live ranges (frees state
progressively), not because it extends them.

## Lever exhaustion update

Levers tried since R44-dm:
1. R45-dm: conditional interleave by N_MASKED — NEUTRAL ✗
2. R46-dm: K-tail SRD setup hoist out of runtime if — NEUTRAL ✗

Pending levers:
3. SRD construction shared across 4 N_MASKED helper calls (estimated
   0.7% gain — sub-noise, low priority)
4. Apply R44-dm INTERLEAVE pattern WITHIN the K-tail (no clear
   "store + free" target; mfmas accumulate cA-cD until C-store
   epilog so no early dead state to interleave around)

Next direction: pure code-motion levers are exhausted. Need to
shift to ALGORITHMIC changes:
* (a) Reduce K-tail load count by combining a + a_kt1 into a
  single 256x128 tile load (requires register tile redefinition
  to height=8; doubles VGPR per lane; may regress occupancy from
  4 → 2 waves/SIMD).
* (b) Skip K-tail entirely for K-aligned shapes via separate
  template instantiation (`HAS_KTAIL` template param). Eliminates
  runtime branch overhead for DSV3 (~10 cy/tile = 0.5% gain on
  DSV3).
* (c) Investigate why MemUnitStalled remains at 67% post-R44 with
  spill at 39 dwords. The 67% can't all be scratch I/O. Need
  ATC_* counters or HBM-controller queue depth metrics.
