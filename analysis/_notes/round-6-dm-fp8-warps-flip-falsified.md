# Round 6 — FP8 grouped WARPS_M/WARPS_N flip catastrophically falsified (mirror)

See `/workspace/code/HipKittens/analysis/_notes/round-6-dm-fp8-warps-flip-falsified.md`
for the full resource delta + SNR table.

## Baseline (round-6 entry)

`score=820, geomean=0.9837, n=16`.

## Probe

`WARPS_M=2, WARPS_N=4 → WARPS_M=4, WARPS_N=2` (auto-derives `RBM=64,
RBN=32 → RBM=32, RBN=64`). Round-5 recommended this as "1-line
Option A".

## Outcome: FALSIFIED — catastrophic SNR failure

- Build: **clean**. Resource remarks: VGPRs=256, occ=2 (unchanged);
  spill INCREASED (+14/+23/+26 across the 3 SNR-failed instances).
- Metric post-flip: **score=8, geomean=0.01** (ALL 16 shapes fail
  fwd-snr < 25 dB, most < 5 dB — numerical garbage).
- **Reverted in-round.** Post-revert metric: 815 (baseline band σ≈2).

## Why 1-line flip doesn't work

`rcr_8w_load_hoist<_NUM_THREADS>` + `G::prefill_swizzled_offsets` have
thread-to-swizzled-offset mappings hand-derived for 2×4 warp layout.
These don't auto-derive from `WARPS_M/WARPS_N` constants — they need
a manual LDS cooperative-load rewrite. Round-5's "1-line Option A"
recommendation was incorrect.

## What auto-derives (confirmed clean)

`rt_fl<RBM,RBN,...>`, `A/B_row_reg` shapes, `subtile_inplace<RBM,BK>`,
`rcr_mma` MFMA shapes.

## Round-7 plan

Shift to Option 1 from the HK note: **port `rcr_4w` 4-wave dense kernel
to grouped**. Dense uses 4-wave for high-grid low-K shapes (3200 ≤ grid,
K ≤ 8192) with 4 blocks/CU occupancy. DSV3 shapes qualify. Grouped has
no 4-wave variant — porting would lift occupancy 2→4 waves/SIMD for
DSV3 and open 15–25% TFLOPS headroom on the main-loop-gap shapes.

Initial scope: `grouped_rcr_4w_kernel` with `N_MASKED_STORE=false,
FUSED_KTAIL=false` only (DSV3 happy path); reuse existing
`s_offs/s_cum_tiles` machinery; gate on `aligned_grid ≥ 3200 && k ≤
8192 && K_rem == 0 && N aligned` in `dispatch_grouped_rcr`.

## Commits

- HipKittens: `analysis/_notes/round-6-dm-fp8-warps-flip-falsified.md`
  (kernel reverted; doc-only commit).
- Primus-Turbo: this mirror.
