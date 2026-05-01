# Round 53-dm — FP8 grouped: K-tail micro-tuning EXHAUSTED

**Status**: BOTH PROBES FALSIFIED / no commit on HK kernel
**Score before**: 959 (HK SHA `6a93fa32`, R50-dm winner)
**Score after**:  959 (revert)
**HK SHA**: clean (no commit)
**Round time**: ~25 min, 3 build cycles, 7 metric runs

---

## Goal

After R52 documented all 3 architectural probes as FALSIFIED, looked for
remaining K-tail micro-knobs not exhausted by R23/R49/R51.

Worst shape: gpt_oss-GateUP-B32-M4096 = 1.024 (unchanged from R52).

---

## Probe 1: K-tail load reorder + 3-stage vmcnt pipelining

### Hypothesis

Current order (R37-dm winner): `[b0, b1, a, a_kt1]` with 2-stage `vmcnt(8)` /
`vmcnt(0)`. mfma cA fires after 16 retire (b0+b1+a all done), bunching cA + cB.

Reorder to: `[a, b0, b1, a_kt1]` with 3-stage `vmcnt(12)` / `vmcnt(8)` /
`vmcnt(0)`:
- `vmcnt(12)`: `a + b0` done → mfma cA (12 retire boundary, ~4 retire earlier
  than R37's vmcnt(8))
- `vmcnt(8)`: + `b1` done → mfma cB
- `vmcnt(0)`: + `a_kt1` done → mfma cC, cD

Hypothesized cA fires ~40 cy earlier, overlapping cA's 16 cy MFMA with b1's
last 4 retires.

### Result: NEUTRAL across 3 runs

```
Score: 956 / 959 / 959 (mean 958, baseline 959 — within ±2 noise band)
correct_fail: 0/32 (PASS)
spill: unchanged (39 / 43 / 32 / 39 dwords for 4 specs)
```

### Root cause

K-tail end-to-end latency is gated by **cD completion** (last mfma needed by
C-store epilog), NOT cA start. cD requires `vmcnt(0)` + cD's own MFMA latency
(~16 cy). The reorder shifts cA earlier but cD's path is unchanged:

- OLD: t=160 (vmcnt(8)) + 32 cy (cA + cB) → t=192. Then idle 48 cy until
  t=240 (vmcnt(0)) + 32 cy (cC + cD) → t=272.
- NEW: t=120 (vmcnt(12)) + 16 cy (cA) → t=136. cB starts at max(t=136,
  t=160 vmcnt(8)) = t=160 + 16 = t=176. Then idle 64 cy until t=240
  (vmcnt(0)) + 32 cy (cC + cD) → t=272.

Same total. cA earlier doesn't help because cD is the bottleneck.

The 3-stage vmcnt also adds ~5-10 cy per extra wait (pipeline drain
overhead), partially offsetting the cA savings.

### Falsification

**FROZEN: K-tail load issue order is saturated at R37's [b0, b1, a, a_kt1]
with 2-stage vmcnt(8) / vmcnt(0).** Reordering to enable finer-grained
vmcnt does not help because the K-tail critical path is cD, not cA.

---

## Probe 2: Add `s_setprio(1)` around K-tail mfma block

### Hypothesis

K-tail's 4 mfmas have NO setprio bracketing (unlike main loop which uses
`setprio(1); mfma; setprio(0)` per mfma). Hypothesized adding `setprio(1)`
before cA and `setprio(0)` after cD would give the K-tail wave priority
in the SQ for faster MFMA issue.

### Result: CATASTROPHIC -120 pts across 3 runs

```
Score: 840 / 839 / 842 (mean 840, baseline 959, -119 pts)
grp_FP8: 1.1186 → 0.86
correct_fail: 0/32 (PASS)
```

### Root cause

`setprio(1)` is a **per-wave priority hint** in the SQ. In the K-tail block,
ALL 8 waves of the workgroup execute the K-tail simultaneously (uniform
SIMT branch — all warps see `g.fast_k < g.k`). When all 8 waves call
`setprio(1)`, there's no relative priority differentiation among them.

But the `setprio(1)` likely **interferes with concurrent activity in OTHER
waves on the same SIMD that are doing useful work**:
- Other CUs' waves (executing the persistent kernel for other groups) are
  unaffected (different SIMD).
- Within the same WG, all waves have prio=1 → no advantage.
- The **side effect**: when MFMA waves are at prio=1, the SQ may starve
  buffer_load issue from OTHER warps still in K-tail load issue (some warps
  may be slightly behind in the K-tail load sequence). This serialises
  what was previously parallelizable.

Alternative theory: `setprio(1)` may suppress async memory pipeline drain,
delaying retirement of in-flight HBM operations from other CUs sharing the
HBM controller. ~120 pts loss is consistent with significant HBM pipeline
stall.

Either way, **setprio inside K-tail's uniform-branch block is harmful**.

### Falsification

**FROZEN: K-tail mfmas must NOT be bracketed with `setprio(1)`/`setprio(0)`.**
Setprio is only useful for asymmetric work distribution among waves in the
same SIMD (which K-tail is not — it's uniform).

---

## Cumulative state after R53

### K-tail micro-tuning EXHAUSTED

```
R23  (R51): VMCNT main-loop INIT0/INIT1   - exhausted
R37  (won): K-tail load order             - shipped [b0, b1, a, a_kt1]
R49  (lost): K-tail internal C-store      - falsified spill backlash
R52  (lost): wm==0 barrier removal        - load-bearing
R52  (lost): M2N2 K-tail launch swap      - too small-grained
R52  (lost): lgkmcnt(0) drop              - neutral
R53  (lost): K-tail load reorder + 3-stage - critical path cD-bound
R53  (lost): setprio on K-tail mfmas      - uniform branch interference
```

Score plateau at 959 (R50 winner ceiling). Three consecutive rounds (R51,
R52, R53) have failed to find further wins via micro-tuning.

### Remaining lever

**Lever D (FULL main kernel cell shape migration)** is the only unfalsified
architectural option. As documented in R52, it's a 2-3 round commitment
with high regression risk on DSV3.

### Recommended next round

R54-dm should commit to Lever D Round-A:
1. Define new `cAB_32 = rt_32x64<float>` accumulator type (unified slab-0/1
   accumulator).
2. Add new `mma_AB_base<rt_32x64, ...>` specialization using
   `__builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4`.
3. In K-tail block ONLY (not main loop), replace 4× `rcr_mma(cX, ...)` with
   2× `rcr_mma_32(cAB_32_X, ...)` then fan-out to existing cA-cD via
   register copies.
4. Verify correctness on gpt_oss (where K-tail fires); compare metric.

If correctness OR perf regresses, revert and ACCEPT 959 as plateau ceiling.
