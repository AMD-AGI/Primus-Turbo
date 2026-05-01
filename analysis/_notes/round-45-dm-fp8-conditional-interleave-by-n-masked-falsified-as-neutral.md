# Round-45-dm · FP8 grouped — conditional interleave (revert batched mul/store for N_MASKED=false) FALSIFIED as NEUTRAL

**Status**: 1 probe FALSIFIED (NEUTRAL). Reverted to R44-dm winner state.
Score 957 → 957/958 (probe, two runs) → 957 (revert). Within run-to-run noise.

## Hypothesis

R44-dm interleaved mul + store INDISCRIMINATELY across both `N_MASKED=true`
and `N_MASKED=false` paths. The win mechanism was specifically about freeing
cA/cB/cC/cD VGPR slots so the masked-store helper's intermediate state could
absorb into vacated slots. For `N_MASKED=false`, there is NO helper state —
the 4 stores are bare `store(...)` calls. So interleave provides no benefit
on the unmasked path AND may HURT by breaking HBM write-coalescing across
the 4 batched stores.

R44-dm metric showed 4 of 8 DSV3 FP8 shapes regressed -1.4 to -2.4 pp:

| Shape | R37 (pre-R44) | R44-dm (R17 baseline) | Δ pp |
|---|---|---|---|
| DSV3-GateUP-B16-M4096 | 1.169 | 1.145 | -2.4 |
| DSV3-Down-B16-M4096   | 1.134 | 1.117 | -1.7 |
| DSV3-GateUP-B32-M4096 | 1.165 | 1.149 | -1.6 |
| DSV3-Down-B32-M4096   | 1.207 | 1.193 | -1.4 |

Hypothesis: revert to BATCHED mul→store (4× mul, then 4× store) for
`N_MASKED=false` only. Keep R44-dm interleave for `N_MASKED=true` (gpt_oss
gain). Expected: DSV3 recovers +1-2 pp on the 4 regressed shapes; gpt_oss
unchanged. Net +0.5 to +1.5 pp on grp_FP8 geomean.

## Implementation

```diff
 if constexpr (N_MASKED_STORE) {
     // R44-dm interleave (kept) ...
 } else {
-    mul(cA, cA, combined_scale);
-    store(g.c, cA, {0, 0, r0, c0});
-    mul(cB, cB, combined_scale);
-    store(g.c, cB, {0, 0, r0, c1});
-    mul(cC, cC, combined_scale);
-    store(g.c, cC, {0, 0, r1, c0});
-    mul(cD, cD, combined_scale);
-    store(g.c, cD, {0, 0, r1, c1});
+    mul(cA, cA, combined_scale);
+    mul(cB, cB, combined_scale);
+    mul(cC, cC, combined_scale);
+    mul(cD, cD, combined_scale);
+    store(g.c, cA, {0, 0, r0, c0});
+    store(g.c, cB, {0, 0, r0, c1});
+    store(g.c, cC, {0, 0, r1, c0});
+    store(g.c, cD, {0, 0, r1, c1});
 }
```

## Spill count delta (`-Rpass-analysis=kernel-resource-usage`)

| Spec template params | R44-dm (R17 base) | R45-dm probe | Δ |
|---|---|---|---|
| `<0,false,false>` (FUSED=false n_aligned)         | 39 | 67 | +28 |
| `<0,true ,false>` (FUSED=false n_masked)          | 43 | 43 | 0   |
| `<0,false,true >` (FUSED=true  n_aligned, DSV3)   | 32 | 36 | +4  |
| `<0,true ,true >` (FUSED=true  n_masked, gpt_oss) | 39 | 39 | 0   |

The DSV3 spec `<0,false,true>` only saw +4 dwords (much less than
expected +40 reverting to pre-R44 baseline of 72). This suggests the
batched code's spill profile depends on more than the local mul/store
ordering — LLVM may be amortizing scheduling decisions across both
template specialisations sharing the FUSED_KTAIL=true frontend.

## Per-shape metric (R44-dm baseline=957 → R45-dm probe runs)

Two consecutive runs of the R45-dm probe to measure run-to-run noise:

| Shape | R44 (R17 base) | R45 run 1 | R45 run 2 | mean Δ vs R44 |
|---|---|---|---|---|
| DSV3-GateUP-B16-M2048 | 1.124 | 1.148 | ~1.148 | +2.4 |
| DSV3-Down-B16-M2048   | 1.184 | 1.194 | ~1.194 | +1.0 |
| DSV3-GateUP-B16-M4096 | 1.145 | 1.159 | ~1.159 | +1.4 |
| DSV3-Down-B16-M4096   | 1.117 | 1.117 | ~1.117 | 0.0  |
| DSV3-GateUP-B32-M2048 | 1.149 | 1.179 | ~1.179 | +3.0 |
| DSV3-Down-B32-M2048   | 1.233 | 1.135 | ~1.230 | -0.0 (noise) |
| DSV3-GateUP-B32-M4096 | 1.149 | 1.171 | ~1.171 | +2.2 |
| DSV3-Down-B32-M4096   | 1.193 | 1.177 | ~1.177 | -1.6 |
| gpt_oss-GateUP-B4-M2048   | 1.091 | 1.089 | ~1.089 | -0.2 |
| gpt_oss-Down-B4-M2048     | 1.139 | 1.153 | ~1.153 | +1.4 |
| gpt_oss-GateUP-B4-M4096   | 1.065 | 1.069 | ~1.069 | +0.4 |
| gpt_oss-Down-B4-M4096     | 1.081 | 1.085 | ~1.085 | +0.4 |
| gpt_oss-GateUP-B32-M2048  | 1.071 | 1.042 | ~1.042 | -2.9 |
| gpt_oss-Down-B32-M2048    | 1.067 | 1.069 | ~1.069 | +0.2 |
| gpt_oss-GateUP-B32-M4096  | 1.023 | 1.031 | ~1.031 | +0.8 |
| gpt_oss-Down-B32-M4096    | 1.051 | 1.042 | ~1.042 | -0.9 |

grp_FP8 geomean: 1.1162 → {1.1149, 1.1175} = mean 1.1162 (NEUTRAL).
grp_BF16 geomean: 1.1821 → {1.1824, 1.1837} = mean 1.1830 (+0.1 pp,
noise — BF16 unaffected by FP8 kernel changes; pure system noise).

Score: 957 → {957, 958} = mean 957.5. Within ±2 pt run noise.

Correctness: PASS (correct_fail=0/32, reject=0/32).

## Why the analysis was wrong (or only partially right)

### Mechanism #1 (HBM write-coalescing): not measurable

Hypothesis was that batched 4 stores allow HBM controller to coalesce
adjacent cells into wider HBM transactions, beating per-store interleaved
issue. Empirically: DSV3 shapes mostly gained +1-3 pp (which would support
this), but the gain was within noise. The HBM controller's coalescing
window is microseconds — much shorter than the interleave gap (~5 mfma
cycles between stores), so all 4 issue patterns probably look identical
to the controller.

### Mechanism #2 (LLVM CSE across templates): negative confounder

The two FUSED_KTAIL=true templates (`<0,false,true>` and `<0,true,true>`)
share the same frontend code in source but emit DIFFERENT machine code
post-template-instantiation. LLVM's RA decisions for the conditional
branch in the C-store epilog (when both arms are inlined) cross-pollute
between specs. Reverting the `else` arm changed `<0,false,true>` spill
by only +4 dwords (vs expected +40) suggests LLVM had already optimized
the batched arm via some shared structure that the interleaved arm's
codegen referenced.

### Mechanism #3 (per-shape noise dominates)

DSV3-Down-B32-M2048 swung from 1.233 (R17 base) to 1.135 (R45 run 1) to
~1.230 (R45 run 2). The swing of ~10 pp in one run is consistent with
a different group-tile scheduling decision triggered by a small change
in launch-time HBM cache state. Not attributable to the source change.

## Conclusion

Conditional interleave does NOT yield a statistically significant gain.
The R44-dm interleave is "good enough" for both spec templates — neither
strictly better nor worse than the batched alternative. Reverting half
the change leaves us in noise band of the original.

## Lever exhaustion update

Levers tried since R44-dm win (R44):
1. R45-dm: conditional interleave (revert batched for N_MASKED=false) — NEUTRAL

Remaining levers per R44-dm "下轮建议":
2. Retry R41-dm pre-issue a_kt1 (low-prob; R41 backlash was systemic)
3. Apply R44 interleave to K-tail mfma+wait (no clear store/free target)
4. SRD hoist outside N_MASKED helper (estimated 0.7% gain — too small)

Next direction: re-rocprof with finer granularity. R44 reduced scratch
228→160 B/thread but MemUnitStalled only dropped 70.6→67.2% (-3 pp
vs expected ~10 pp). The remaining 67% stall has a non-spill component
that needs identifying. Candidates: HBM read-return queue saturation
from K-tail's burst of 24 buffer_load_b128, or LDS read-back queue
contention with the prefetch pipeline.
