# Round 7 — FP8 grouped launch_bounds MIN=4 ignored + a_kt1 scoping cleanup (mirror)

See `/workspace/code/HipKittens/analysis/_notes/round-7-dm-fp8-launch-bounds-min4-noop-and-akt1-scoped.md`
for the full resource table + round-8 4-wave port plan.

## Baseline (round-7 entry)

`score=815, geomean=0.9781, n=16`. Worst shapes:
`gpt_oss-Down-B4-M4096 @ 0.929`, `gpt_oss-GateUP-B4-M2048 @ 0.930`.

## Probes this round

1. **`__launch_bounds__(_NUM_THREADS, 4)` hint** (tightened from
   round-5's `→2` test, which was also no-op). Compiler ignores:
   bit-identical resources (VGPRs=256, spill=67/76/45/54, occ=2).
   Occupancy ceiling is VGPR-bound, not hint-bound. **Falsified.**

2. **`A_row_reg a_kt1` scoping cleanup.** Moved from function scope
   (line 1995) into the `if constexpr (FUSED_KTAIL)` block where it's
   actually used (line ~2290). Compiler DCE was already handling this
   cleanly — resources bit-identical — but the refactor makes intent
   match code. 3-run metric: 813/817/816 (median 816, baseline median
   816, σ≈2). Shipped as cleanup; no score impact.

## Round-8 plan

Begin 4-wave grouped RCR kernel port. Dense has a 4-wave variant
(`rcr_4w::kernel` line 805; WM=2, WN=2, 256 threads, **4 blocks/CU =
4 waves/SIMD occupancy**) dispatched for
`aligned_grid ≥ 3200 && k ≤ 8192`. 7 of 16 metric FP8 shapes qualify
(all DSV3 except GateUP-B16-M2048 which is below grid threshold).
gpt_oss does NOT qualify (grid ≤ 1500). DSV3-Down subset currently at
0.97-0.98 ratio — prime candidate for 4-wave occupancy uplift.

Porting constraints:

- Must preserve persistent single-launch architecture (RED LINE).
  Launch `NUM_CUS * 4 = 1024` blocks, each 256 threads, persistent
  stride 1024.
- Per-tile per-group base pointer recomputation for A/B (rcr_4w
  assumes single-matrix HBM base).
- Happy path only initially: gate `N aligned + K aligned + grid
  thresholds` → `grouped_rcr_4w_kernel<0, false, false>` variant.

Scope estimate: 2-3 rounds (round 8 skeleton+compile+dispatch; round 9
schedule tuning; round 10+ extend to FUSED_KTAIL/N_MASKED_STORE).

## Commits

- HipKittens: a_kt1 scoping + note (single commit).
- Primus-Turbo: this mirror.
