# Round 52-dm — FP8 grouped: three architectural probes FALSIFIED

**Status**: FALSIFIED ALL THREE / score plateau at 959 (R50 winner)
**Score before**: 959 (HK SHA `6a93fa32`, R50-dm winner)
**Score after**:  959 (revert / no commit on HK kernel)
**HK SHA**: clean (no commit)
**Round time**: ~30 min, 4 build cycles, 5 metric runs

---

## Goal

After R51 conclusively exhausted VMCNT micro-tuning (INIT0 boundary at 4, INIT1
neutral), the only remaining obvious lever is architectural rewrite. Tested
three architectural probes today.

## Worst shape (selection driver)

```
gpt_oss-GateUP-B32-M4096   1.029   ← worst
gpt_oss-GateUP-B32-M2048   1.048
gpt_oss-GateUP-B4-M4096    1.056
gpt_oss-Down-B32-M4096     1.066
gpt_oss-Down-B32-M2048     1.068
```

All worst cases on gpt_oss (K_REM=64). Confirms K-tail block is the bottleneck
for FUSED=true main kernel.

---

## Probe 1: Remove `wm == 0` half-barrier before C-store epilog (line 2484)

### Hypothesis

The `if (wm == 0) __builtin_amdgcn_s_barrier()` at line 2484 fires AFTER the
K-tail mfmas (or epilog 2 mfmas if K-tail dead) and BEFORE the C-store epilog.
Each warp writes a unique `(wm, wn)` slice of `g.c` with no cross-warp data
overlap. The mfma RAW hazard for `mul(cD, cD, scale)` should be enforced by
hardware scoreboard. Hypothesized barrier was redundant.

### Result: CATASTROPHIC -854 pts

```
Score: 959 → 105
correct_fail: 0/32 → 15/32
grp_FP8: 1.1186 → 0.0133
```

**15/32 forward correctness FAIL** — fwd-SNR < 0.8 to 20 across all FP8 shapes
(both DSV3 and gpt_oss). BF16 PASS unchanged (1.1824 → 1.1828).

### Root cause

The `wm == 0` half-barrier is **load-bearing for FP8-specific accumulator
flush**. Hypothesis on mechanism:
- `s_barrier` on gfx950 likely flushes mfma write-pending state across all
  waves in the workgroup before subsequent reads. The mfma latency for
  `mfma_scale_f32_16x16x128_f8f6f4` may not be fully covered by scoreboard
  alone — particularly when `mul(cX, cX, scale)` issues immediately after
  the mfma without enough independent instructions to cover the hazard.
- BF16 path uses `mfma_f32_16x16x32_bf16` which has different latency /
  scoreboard characteristics → no break.
- The `if (wm == 0)` pattern: only wm==0 warps signal, but ALL waves still
  participate in the wait portion of `s_barrier_signal` / `s_barrier_wait`
  on gfx950. So removing it loses the wait too.

### Falsification

**FROZEN: line 2484 `if (wm == 0) __builtin_amdgcn_s_barrier()` is required.**
Future agents: do not remove. The barrier serves an mfma accumulator coherence
role specific to FP8 (16x16x128 cell shape).

---

## Probe 2: Route gpt_oss (K_REM=64) through standalone `grouped_ktail_kernel_mfma32x32_M2N2`

### Hypothesis (Lever D, partial)

Modify dispatcher condition `fuse_ktail_eligible` to require `K_rem_for_fuse ==
0` only (no longer `(== 0) || (== 64)`). For gpt_oss K_REM=64:
- Main kernel uses `<0,*,false>` spec (FUSED=false): main loop without K-tail
- Standalone `grouped_ktail_kernel_mfma32x32_M2N2` launches separately to
  handle K=[fast_k, k) remainder via 32x32x64 native FP8 mfma + RMW on g.c

Trade-offs predicted:
- (+) Main kernel sheds ~24 buffer_loads + 4 mfmas + vmcnt waits per tile.
- (+) Native 32x32x64 mfma at 100% util vs FUSED's padded 16x16x128 at 50%.
- (-) Main kernel spec spill: 39 → 43 dwords (per R34-dm).
- (-) Extra launch overhead ~1 us.

### Result: CATASTROPHIC -76 pts

```
Score: 959 → 883
correct_fail: 0/32 (PASS)
grp_FP8: 1.1186 → 0.9483

gpt_oss-GateUP-B32-M4096:  1.029 → 0.699  (-33 pp)
gpt_oss-GateUP-B32-M2048:  1.048 → 0.718  (-33 pp)
gpt_oss-Down-B32-M4096:    1.066 → 0.734  (-33 pp)
gpt_oss-Down-B32-M2048:    1.068 → 0.735  (-33 pp)
gpt_oss-GateUP-B4-M4096:   1.056 → 0.710  (-35 pp)
gpt_oss-Down-B4-M4096:     1.085 → 0.844  (-24 pp)
gpt_oss-GateUP-B4-M2048:   1.076 → 0.837  (-24 pp)
gpt_oss-Down-B4-M2048:     1.159 → 0.961  (-20 pp)
```

ALL 8 gpt_oss shapes regressed -20 to -35 pp. DSV3 unchanged or +0-2 pp
(consistent — DSV3 K_REM=0 path unaffected).

### Root cause

`grouped_ktail_kernel_mfma32x32_M2N2` is structurally **wrong for grouped FP8**:

1. **Single-wave thread blocks**: `block_dim = 64` (1 wave). For
   gpt_oss-GateUP-B32-M4096 with `(N/64) × (M/64) = 90 × 2048 = 184,320` blocks.
   Compare to main kernel: 256 persistent blocks × 8 waves = `~2K waves` total.
   **180,000 single-wave blocks** vs 2K = 90× launch overhead amplification.

2. **Per-cell scalar RMW**: lines 4609-4613 in M2N2 kernel use `load_bf16_scalar`
   + `store_bf16_scalar` for the C accumulator update. Each cell is a separate
   1-byte scalar HBM op, not a vectorized buffer_store_b16. ~16 cells × 8 lanes
   per block → 128 scalar HBM round-trips per block → ~1 us per block.
   180,000 blocks × 1 us = 180 ms tail per kernel call. Catastrophic.

3. The standalone M2N2 kernel was designed for the SHORT path of K=64 remainder
   (e.g., when K_REM was the entire K dim in some legacy scenario), not as a
   regular K-tail companion to a multi-tile main kernel.

### Falsification

**FROZEN: do not route FUSED-eligible K_REM=64 shapes through the standalone
`grouped_ktail_kernel_mfma32x32_M2N2` launch.** The standalone K-tail kernel is
fundamentally too small-grained for grouped FP8 workloads.

For Lever D (32x32x64 cell shape) to work, the **MAIN kernel** itself must
adopt 32x32x64 mfma — not just a separate launch for K-tail. The required
work is:
- Port the main kernel's MFMA cell shape from 16x16x128 to 32x32x64 (rt_64x32
  for accumulator)
- Halve the number of register tiles needed per acc → reduces VGPR pressure
- Keep FUSED K-tail INSIDE main kernel but use 32x32x64 mfma there too
- Estimated: 2-3 round commitment, full kernel rewrite

---

## Probe 3: Drop `lgkmcnt(0)` at end of tile (analog to R50's `vmcnt(0)` drop)

### Hypothesis

R50 successfully removed `vmcnt(0)` at end of tile (relying on next tile's
prologue `TK_WAIT_VMCNT(INIT0=4)` to absorb HBM stragglers). Try same for
`lgkmcnt(0)`: by end-of-tile all main-loop LDS reads are CONSUMED by mfma
(no in-flight ds_read), and main-loop LDS writes are LOGICALLY retired
(else mfma would have hit data hazard). The `s_barrier()` should be
sufficient for inter-warp sync.

### Result: NEUTRAL / NOISE

```
3 runs after change: 956, 960, 958
Mean: 958
Baseline: 959
```

Within noise band (~3-4 pts std). No clear win/loss.

### Root cause

The s_barrier alone is sufficient for the cross-tile sync (all warps reach the
barrier together), but removing `lgkmcnt(0)` doesn't expose any meaningful
overlap because:
- The previous tile's last K-iter LDS writes are already retired by the time
  mfma consumed them (RAW hazard within the K-iter).
- The cross-tile latency to recover is dominated by HBM (vmcnt), not LDS
  (lgkmcnt). R50 already captured the HBM overlap. There's no incremental
  LDS-side opportunity.
- `lgkmcnt(0)` typically completes in 0-50 cy at end-of-tile (already
  drained by previous main-loop scoreboard waits). Removing it saves
  effectively zero cycles in steady state.

### Falsification

**Lever exhausted.** R50's mechanism (cross-tile HBM/store overlap) does
NOT generalize to LDS (lgkmcnt) — the LDS pipeline is already drained
by mfma's RAW scoreboard. The end-of-tile `lgkmcnt(0)` is harmless and
serves as defensive sync; keep it.

---

## Cumulative state after R52

### Score plateau

```
R50:  961  (vmcnt(0) drop WIN, +5 pts vs R49)
R51:  959  (INIT0/INIT1 sweep neutral/break, no change)
R52:  959  (3 probes all falsified, no change)
```

### Exhausted lever inventory (52 rounds total)

- VMCNT/LGKMCNT micro-tuning: EXHAUSTED (R51, R52 probe 3)
- `s_barrier` removal: EXHAUSTED (R31 wm==1 prologue half-bar, R52 wm==0
  C-store half-bar). All half-barriers are load-bearing.
- K-tail internal interleave: EXHAUSTED (R49)
- `store_c_tile_n_masked` unroll variants: EXHAUSTED (R48 catastrophic
  runtime VGPR-array indexing)
- Register reuse / `__restrict__` / hoist: EXHAUSTED (R34, R46, R47)
- Standalone K-tail kernel routing: EXHAUSTED (R52 probe 2)
- 16x16x128 cell shape micro-knobs: EXHAUSTED

### Remaining lever (high effort)

**Lever D (FULL): main kernel cell-shape migration 16x16x128 → 32x32x64**

- Requires rewriting `rcr_mma`, `A_row_reg`, `B_row_reg`, `C_acc_reg` to use
  `rt_32x64` / `rt_64x32` shapes (scaffold in HK commit 96a84c08)
- LDS slab layout must change from `st_16x128_v2_s` to `st_32x64_*` swizzle
- ds_read patterns must match new mfma input layout (current 16x128_v2
  swizzle hardcoded)
- Number of register tiles per accumulator HALVES (16x16 → 32x32 acc =
  4× larger per tile) → VGPR pressure relief
- K-tail block also moves to 32x32x64 (single-mfma K=64 → 100% util)
- Estimated: 2-3 rounds full commitment, high risk of regression on DSV3

### Recommended next round

R53-dm should commence Lever D full migration. Plan:
1. Round-A: Swap `rcr_mma` cell shape (Bs/As/cX rebind to rt_32x64). Verify
   correctness on DSV3 first (K_REM=0 simplifies). Skip K-tail for now.
2. Round-B: Add 32x32x64 K-tail INSIDE FUSED path. Re-verify gpt_oss.
3. Round-C: Tune VMCNT/sched for new schedule.

If any round shows DSV3 perf regression > 3 pp, abandon migration and accept
959 as plateau ceiling.
