# Round 2-dm — FP8 grouped LDS layout ST_v3 swap FALSIFIED (correctness)

**Date**: 2026-05-01 (Round 2 of 100)
**Primus-Turbo HEAD (entry)**: `08b307a` (round-1-dm docs)
**HK HEAD (entry)**: `3c679aa9` (round-1-dm docs)

## Round-2 summary

- **Baseline metric**: score 812 (geomean 0.9744), lowest shape
  `grpFP8_gpt_oss_20B-GateUP-B4-M2048 @ 0.925` — same lowest-ratio
  target as round 1.
- **Probe**: swap `grouped_rcr_kernel`'s `ST_rcr` LDS type from
  `st_16x128_v2_s` → `st_16x128_v3_s` (the only unused variant,
  `(r & 15) << 3` swizzle, `subtile_padding=0`).
- **Result**: correctness fails on all 16 FP8 shapes (`fwd-snr < -0.6`);
  score 812 → 8. Reverted; metric restored to 812.
- **Lesson**: ds_read_b64_tr_b8 address generation hard-codes ST_v2's
  `(r & 7) << 4` swizzle granularity. An ST-layout swap is NOT a
  one-liner — it requires full prefill + ds_read + rt co-port
  (2-4 rounds of work, high risk).

## Why this was the round-2 shot

- All single-knob levers formally saturated (rounds 4, 5, 8, 14-16,
  22-25, 27). Task-body lever **B** (LDS bank conflicts / swizzle /
  padding) was unlabeled as tested in the round-25 inventory.
- ST_v3 is the only `st_16x128_v*` variant currently unused anywhere
  in `kernel_fp8_layouts.cpp` — v2, v2a, `st_16x128_s` (no suffix) are
  all live in dense 4-wave/8-wave/RRR/CRR paths.
- Round-1 falsified the config-rule route for DSV3 (all (gm,xcd)
  within noise), so round-2 needed a kernel-level probe.

## What a "real" LDS-layout rework would require

Three wiring sites would need co-update for any non-ST_v2 layout:

1. `using ST_rcr = …` (one line, done here)
2. `prefill_swizzled_offsets<ST_rcr>` — already ST-parameterized (OK).
3. **`load<reg, ST_rcr>` + `subtile_inplace<RBM, BK>`** — THE hard
   piece. The ds_read_b64_tr_b8 cooperative stride formula in
   `include/ops/warp/memory/tile/global_to_register.cuh` is written for
   ST_v2's 8-row×16-byte XOR granularity; ST_v3's 16-row×8-byte
   granularity needs a new lane-to-(r,c) inversion.
4. `rcr_mma` operand layout — likely unaffected (tile-positional), but
   needs audit against the new lane mapping.

Estimate: 2-4 rounds, with a round-27-class VGPR-live-range cliff risk
when the new addressing cascades through register allocation.
**Not recommended.**

## Cumulative knobs table (updated with this round)

Added one row to round-25's "saturated knobs" inventory:

| knob | direction tested | result | note |
|---|---|---|---|
| `ST_rcr` LDS layout (v2 → v3) | one-liner swap (this round) | **CORRECTNESS FAIL** | ds_read hard-codes v2 swizzle; cannot swap without full co-port |

Other tested & saturated:

- Main-loop `sched_barrier` (rounds 4, 8, 25)
- Epilog `sched_barrier` (rounds 8, 25; **load-bearing**)
- `RCR_PREFETCH_LGKM` {2,4,8} (round 5)
- `RCR_STEADY_VMCNT` {4,8,12} (round 8)
- `RCR_INIT0/INIT1_VMCNT` (round 24; race-prone on raise)
- `RCR_EPILOGUE_VMCNT` {1,2,3,4} (round 25)
- `RCR_TWO_TILE_MIN_KI` 28 → 20 (round 15; no-op for ki=22)
- `RCR_TWO_TILE_MID_VMCNT` (rounds 6, 15; 2-tile path catastrophic)
- Chiplet `chunk_size` 64 → 32 (round 22; B=4 regresses)
- BN=128 dispatch (round 24; Triton uses BN=256 too)
- `(group_m, num_xcds)` per-shape (rounds 21, 23, 1-dm; saturated)
- `BLOCK_SWIZZLE_NUM_XCDS` = 8 (hard MI355X HW, round 12-14)
- Host Python trim (rounds 11, 22; tapped out)
- K-tail `load_a_kt(a_kt1)` hoist (round 27; VGPR live-range cliff)
- FP8 RRR fuse path A (round 27; compiler alias bug)

## What remains (multi-round structural only)

1. **FP8 K-tail single-load merge + multi-tile-M amortize** — 2-3
   rounds. Share one K-tail call across an M-slab of output tiles
   (round-8 note §1, round-25 note §3). Risk: VGPR pressure (we're at
   the 256 ceiling on grouped main loop).

2. **FP8 direct HBM→reg main loop** (skip A-tile LDS staging) —
   task-body lever **E**, round-4 §9.1(c). 4-8 rounds; highest
   documented yield (+5-7 pp per tile on DSV3-Down K=2048 aligned
   shapes). Risk: A-tile HBM traffic multiplied by WARPS_N=4 if no
   cross-warp sharing, cancelling the LDS-sync savings.

3. **FP8 MFMA cell-shape rework** (16x16x128 → 32x32x64 scaled) —
   2-3 rounds. Round-12 rocprof partially falsified (same MMA count),
   but the register-tile & LDS layout implications may still unlock
   perf. Round-25 flagged this.

4. **BF16 BK=32 + ns=3 port** — out of scope (`[watch]`, no score
   move per task-body RED line).

## Round-3 suggestion

Commit to **structural project (1)**: FP8 K-tail amortize across
multi-tile-M. Rationale:

- Affects 8 of 16 FP8 shapes (all gpt_oss, K=2880). Average lift
  estimate +1.5-3 pp per shape = +1 pp overall geomean = +10 score.
  Smaller than (2) but much less risky.
- Round-8 note §1 has the design sketch (shared accumulator + single
  end-of-slab K-tail call). Never attempted.
- Failure modes are graceful: if VGPR pressure blows up, revert with
  no side effects on DSV3 (`FUSED_KTAIL=false` variant).

Round-3 step 1 would be a read-only audit of `grouped_rcr_kernel`'s
outer persistent loop + FUSED_KTAIL epilog (lines 2045-2331 in
`kernel_fp8_layouts.cpp`) to identify the M-slab boundary and the
shared-accumulator insertion point. No kernel edit in round-3;
design doc commit only.

## Files touched

- HipKittens: `analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp:1974`
  (ST_v3 probe + revert; net no change this round).
  `analysis/_notes/round-2-dm-fp8-grouped-lds-st-v3-swap-falsified.md`
  (new).
- Primus-Turbo: `analysis/_notes/round-2-dm-fp8-lds-st-v3-falsified.md`
  (this file).

## Commits

- HipKittens: `docs(round-2-dm): FP8 grouped LDS ST_v2 → ST_v3 swap falsified (SNR fail, ds_read hard-codes v2 swizzle)`
- Primus-Turbo: `docs(round-2-dm): FP8 grouped LDS layout ST_v3 swap falsified`
