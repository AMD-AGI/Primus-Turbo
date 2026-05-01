# Round-25 — RCR_EPILOGUE_VMCNT saturated + epilog RCR_SCHED_BARRIER is required

**Date**: 2026-05-01  
**Repo / HEAD**: Primus-Turbo `3ce72bb` (round-24); HipKittens `62cebd5` (no net change this round)  
**Focus**: gpt_oss 16 shapes (8 grp_BF16 + 8 grp_FP8); DSV3 = `[watch]`  
**Result**: no net code change (both probes falsified, restored to baseline)

---

## Goal of this round

Round-24 falsified BN=128 path (Triton uses identical BM=256/BN=256 on
gpt_oss) and saturated `RCR_INIT0_VMCNT` / `RCR_INIT1_VMCNT` (raising
breaks correctness via `vmcnt(N)` no-op race; lowering ≤ noise).

Round-24 docs flagged **two remaining unprobed levers** in
`HipKittens/analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp`:

1. `RCR_EPILOGUE_VMCNT` (currently `4`) — only direction left is *lower*
   (raising would race like INIT). One unswept counter.
2. `RCR_SCHED_BARRIER()` in **epilog** (lines 2199, 2217, 2230). Round-4
   already removed all 4 from the main loop body (+0.45 pp grp_FP8); the
   3 epilog instances were untouched.

This round probes both, in order.

---

## Probe 1 — `RCR_EPILOGUE_VMCNT` 4-cell sweep (lower direction only)

### Setup

Baseline `RCR_EPILOGUE_VMCNT = 4`. Sweep `{4 baseline, 3, 2, 1}`. One
metric run per cell (single-run noise ≈ ±2 score from prior history).
GPU pinned `HIP_VISIBLE_DEVICES=2` via `auto_optimize.py`.

### Results

| EPILOGUE_VMCNT | grpFP8 GateUP-B4-M2048 | Down-B4-M2048 | GateUP-B4-M4096 | Down-B4-M4096 | score |
|---|---|---|---|---|---|
| 4 (baseline)   | 0.926 | 0.988 | 0.948 | 0.924 | **883** |
| 3              | 0.927 | 0.973 | 0.940 | 0.924 |  881  |
| 2              | 0.927 | 0.976 | 0.940 | 0.924 |  881  |
| 1              | 0.920 | 0.968 | 0.944 | 0.924 |  882  |

All 4 cells within ±2 score (single-run noise band ≈ ±2 confirmed by
round-23/24 history). No correctness FAIL, no rejects.

### Conclusion (Probe 1)

`RCR_EPILOGUE_VMCNT` is **saturated** in both directions.

- Raise direction was already known unsafe (race condition: `vmcnt(N)`
  becomes a no-op when the actual outstanding vmem op count `< N`,
  letting the kernel read stale LDS — same mechanism that crashed
  round-24 INIT sweep at `(6,8)` and `(8,10)`).
- Lower direction (this probe) shows no measurable signal because
  `RCR_EPILOGUE_VMCNT` only fires **once per output tile** (after Epilog
  1's `load_a`), not per K-iteration. With ki_dyn ≈ 22 main-loop iters
  amortizing the cost, a 1–3 cycle epilog wait shave is below noise.

Restored to baseline `4`. No HK code change for this probe.

---

## Probe 2 — Remove `RCR_SCHED_BARRIER()` from grouped Epilog 1 + Epilog 2

### Setup

Removed 3 calls in `grouped_rcr_kernel` at lines 2199, 2217, 2230
(Epilog 1 cA/cD MMAs and Epilog 2 cB MMA). Kept all `s_setprio`,
`s_barrier`, `s_waitcnt` instructions — only the compiler reorder hint
is removed. Identical edit to round-4 main-loop change which gave
+0.45 pp grp_FP8 (commit `f04d6b8`).

5-run metric verify (single ÷s, no warmup of warmups):

| run | grpFP8 GateUP-B4-M2048 | Down-B4-M2048 | GateUP-B4-M4096 | Down-B4-M4096 | grp_BF16 geomean | grp_FP8 geomean | score |
|---|---|---|---|---|---|---|---|
| 1 | 0.910 | 0.962 | 0.927 | 0.930 | 1.1635 | 0.9430 | 873 |
| 2 | 0.909 | 0.981 | 0.920 | 0.929 | 1.1687 | 0.9437 | 875 |
| 3 | 0.913 | 0.957 | 0.924 | 0.932 | 1.1613 | 0.9416 | 871 |
| 4 | 0.912 | 0.970 | 0.929 | 0.928 | 1.1700 | 0.9442 | 876 |
| 5 | 0.913 | 0.964 | 0.920 | 0.928 | 1.1656 | 0.9426 | 874 |
| **mean** | **0.911** | **0.967** | **0.924** | **0.929** | **1.1658** | **0.9430** | **873.8** |

Baseline (last 4 runs from Probe 1 + post-restore): scores 880, 881,
881, 882, 883 → mean ≈ 881.4.

**Δ = 873.8 − 881.4 = −7.6 score (5/5 runs below baseline minimum)** —
unambiguous regression, well outside noise band.

### Why it regresses (despite main-loop removal helping)

The main-loop body (round-4) and the epilogs differ in scheduling
constraints:

- **Main loop**: 4 MMAs interleaved with `load_b` / `load_a` /
  `rcr_8w_load_hoist` for the *next* K-tile. Compiler reorder is *good*
  here because (a) the prefetch load addresses are independent of the
  MMA accumulator, and (b) the loop body is unrolled by
  `RCR_MAIN_UNROLL` so the compiler sees the full schedule and can move
  loads earlier across the `__builtin_amdgcn_s_barrier()` boundary
  (`sched_barrier(0)` was *blocking* this beneficial reorder).

- **Epilog 1 / Epilog 2**: The schedule is **terminal** — there's no
  next K-tile to prefetch, only the last A/B loads for the final
  accumulation. The hand-written sequence
  `load_b → s_barrier → s_waitcnt(lgkm,0) → setprio(1) → mma →
  setprio(0) → s_barrier` is already optimal for the GFX950 issue
  pipeline (mfma issue ↔ vmem completion). Without `sched_barrier(0)`,
  the compiler is free to reorder the next `load_a` / `load_b` *across*
  the MMA, which (in this measurement) introduces a vmem-issue stall
  that the hand-tuned schedule was masking. The loss aligns with
  round-2's −5.6 % regression when both `setprio` *and* `sched_barrier`
  were removed together: setprio was the dominant signal, but
  sched_barrier independently contributes ~1 pp grp_FP8 in the epilog.

### Conclusion (Probe 2)

Epilog `RCR_SCHED_BARRIER()` is **load-bearing** — opposite sign to the
main-loop case. Reverted all 3 removals. **Do not retry this lever** in
the grouped FP8 RCR kernel; it is now formally falsified.

---

## Updated saturated-knobs inventory (cumulative through round 25)

For `grouped_rcr_kernel` in
`HipKittens/analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp` and gpt_oss
shapes only:

| knob | direction tested | result | note |
|---|---|---|---|
| `RCR_INIT0_VMCNT`         | both, round 24  | saturated; raise races | safe range = `4` |
| `RCR_INIT1_VMCNT`         | both, round 24  | saturated; raise races | safe range = `6` |
| `RCR_STEADY_VMCNT`        | both, round 5   | saturated              | per-K-iter; main-loop critical path |
| `RCR_PREFETCH_LGKM`       | both, round 5   | saturated              | per-K-iter |
| `RCR_EPILOGUE_VMCNT`      | lower this rd   | saturated; raise races | only-once-per-tile, no leverage |
| `RCR_TWO_TILE_MID_VMCNT`  | rounds 6, 15    | dead-end (grouped path) | 2-tile path catastrophic for grouped |
| `RCR_TWO_TILE_MIN_KI`     | round 15        | no-op for ki_dyn=22    | grouped not 2-tile-eligible at gpt_oss K |
| main-loop sched_barrier   | round 4         | **removable +0.45 pp** | **already removed** |
| epilog sched_barrier      | this round      | **load-bearing** (-1 pp)| **must keep** |
| `chunk_size` (chiplet)    | round 22        | 64 baseline optimal    | 32 regresses B=4 |
| `(group_m, num_xcds)`     | rounds 21, 23   | per-shape rules wired  | per-shape table converged |
| `BLOCK_SWIZZLE_NUM_XCDS`  | round 12-14     | 8 = MI355X HW          | hard constraint |
| BN=128 dispatch path      | round 24        | falsified              | Triton uses BM/BN=256/256 too |
| Host overhead             | round 22        | 1.7 µs Python ≈ 1 % B=4 wall | Triton 0.04 µs; remaining trim ≤ 0.5 µs |

**Every micro-knob currently exposed to the kernel is now saturated for
gpt_oss.** No 1-round single-knob lever remains.

---

## What's left (multi-round, structural)

Confirmed unchanged from round-24 docs:

1. **BF16 grouped: BK=64 → BK=32 + num_stages=3 main-loop port**  
   Triton uses `BM=256, BN=256, BK=32, ns=3, gm=4` for grp_BF16 RCR
   forward (audited round 24 via `_get_gg_bf16_fwd_config`). HK uses
   `BK=64, ns=2`. Smaller BK + deeper pipeline reduces vgpr spill
   pressure and tightens K-iter latency. **3-4 rounds**: must port
   `kernel_bf16_dynamic.cpp` main loop to alternating BK=32 with double
   the K-iter count, re-derive register tile, then re-tune
   `(group_m, num_xcds, chunk_size)`. Round-5 spill lesson: do this
   incrementally, single subroutine at a time.

2. **FP8 grouped: MFMA cell-shape `16x16x128` → `32x32x64`**  
   HK uses `mfma_f8f6f4_16x16x128_f8f8` (8w warp tile, 8 MMA / wave).
   Triton uses `mfma_scale_f32_32x32x64_f8f6f4` (4 MMA / wave, 2× Tile-K
   per MMA, 4× MFMA accum vgpr per warp). Re-deriving `RBM/RBN` and
   register tile is non-trivial: cell-shape change alters which lanes
   own which (m,n) accum coords, so `St_subtile`'s 4-lane HW transpose
   layout must be re-mapped. **2-3 rounds**.

3. **K-tail epilog single-load merge**  
   Currently path B (round-3 commit `07354791`) does an extra
   `rcr_8w_load_hoist` for the K-tail block before fold-into-cA/cB/cC/cD.
   For ki_dyn ≈ 22 + 1 K-tail iter (= ~5 % of total work), the extra
   vmem fetch is ~1 % wall on small-B cases. Folding the K-tail vmem
   into the last-K-tile prefetch slot inside Epilog 2 would amortize.
   **2 rounds**: requires path B and Epilog 2 schedule co-design.

---

## Commit plan

Primus-Turbo only — HK has no net code change (both probes restored).

```
docs(round-25): RCR_EPILOGUE_VMCNT saturated + epilog RCR_SCHED_BARRIER required
```

## Next-round suggestion

All single-knob 1-round levers are now exhausted. Round-26 must commit
to one of the 3 multi-round structural projects above. Recommend
**option 1 (BF16 BK=64 → BK=32 port)** because:

- Triton's BF16 advantage on gpt_oss is smaller (geomean 1.166 vs HK
  baseline) than its FP8 advantage gap, so BF16 has more headroom.
- BK port is a contained edit (one main-loop subroutine), unlike the
  MFMA cell-shape change which touches the whole register-tile
  derivation chain.
- Failure mode is graceful: if intermediate compile breaks, partial
  progress is committable as a `WIP` round.

Round-26 step 1 should be: read the BF16 RCR kernel main loop, list the
exact register-tile / load-hoist sites that hard-code BK=64, and split
into 2-3 incremental commits.
