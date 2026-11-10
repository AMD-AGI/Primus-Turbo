# Round 16 — ACCEPTED (trade-off) — 2026-11-10 09:40 — knob re-sweep on the r15 base: GEMM3 kstep ring depth 6 -> 4 (g3d)

## Hypothesis
Round 15 removed the loop tail's division and changed the hot loop's instruction mix (877 -> 600
instructions), so the builder knobs last swept in round 9 may have a different optimum on this
code. A per-process re-sweep of the knobs that compile on the MLA path is one config-sweep round.

## Single change
`ops_bwd._dkdv_kwargs`: `g3d=4` (was the builder default 6 = G3_KSTEPS, the full band). No kernel
source change. The new default compiles to the same dkdv ISA as the timed `g3d=4` override
(md5 c587bf720b; r15 9aff03ccb5). Snapshot: kernel_snapshot/.

## Screening (ISA, artifacts/isa_r16.txt)
17 configs. g3d 2..5, g2_half=False, g3_sb=1 keep 94 spills / 380 B scratch; g2d=2 98;
g3_st_n=2 95 (hot loop scratch_load 3 -> 10); g3_st_at 2/3 124-125; iglp=3 144; g3_dbat=6 169;
g1_ks_outer=True 46 (round 9's trade-off, timed again below); g3_dbat=4, dv_pin=True and
g3_kreg=True fail builder asserts on MLA.

Hot-loop structure (dev/hot_runs.py): g3d=4 leaves the GEMM3 tail and the MFMA count (156)
unchanged; the leading non-MFMA run before the first MFMA shrinks 84 -> 77 instructions and the
GEMM1 section's 17-instruction VALU / ds_read clump plus four back-to-back waits is replaced by an
even 2-3-instruction interleave. g3d=5 has the same structure as r15.

## Sweep (dev/sweep_proc.py, one process per config, rotated)
artifacts/sweep_r16t.raw (3 rounds; B2 S4K / B1 S8K / B1 S16K, gm):

| config | B2 S4K | B1 S8K | B1 S16K | gm |
|---|---|---|---|---|
| g3d=2 | -0.04 % | -0.70 % | -0.68 % | -0.48 % |
| g3d=3 | -0.54 % | -0.85 % | -0.75 % | -0.71 % |
| g3d=4 | +1.22 % | +1.66 % | +1.28 % | +1.39 % |
| g3_sb=1 | +0.13 % | +0.14 % | +0.20 % | +0.16 % |
| g1_ks_outer=True | -1.97 % | -4.22 % | -5.10 % | -3.77 % |
| g2d=2 | -0.27 % | -1.02 % | -0.78 % | -0.69 % |
| g3_st_n=2 | -24.14 % | -25.01 % | -26.50 % | -25.22 % |

artifacts/sweep_r16u.raw (4 rounds): g3d=1 -0.30 %, g3d=4 +1.13 % (+0.81 / +1.68 / +0.90 %),
g3d=5 +0.12 %. In round 9 (r9 code) the same g3d=4 measured +0.40 %.
artifacts/sweep_r16v.raw (6 rounds, explicit g3d=6 vs g3d=4): B1 S4K +2.54 % (2.571 -> 2.507 ms),
B2 S4K +1.28 %.

## Correctness
- Full set 7 / 7 PASS in all three candidate runs; dK / dV SNR identical to the r15 reference
  (dK 49.54-49.66, dV 51.08-51.38 dB), dQ 44.68-47.02 dB (within 0.01 dB, bf16 atomic order);
  dK/dV bitwise repeat True.
- The ring depth only moves when GEMM3's transpose-reads are issued; the MFMA operands and order
  are unchanged.

## Results
Full set (n02-29 GPU 0, rotated vs aiter a16). Order: candidate run1
(`dev/bench_over.py '{"g3d": 4}'`), r15 reference (`quick_test_bench.py`, r15 defaults), run2,
run3. Bwd ms:

| shape | run1 | run2 | run3 | r15 ref (same session) | R15 run2 | Step run3 | d vs r15 ref | step ratio vs aiter |
|---|---|---|---|---|---|---|---|---|
| b1_s4k | 2.486 | 2.570 | 2.553 | 2.514 | 2.511 | 771.96 | -1.234 % | 0.893 |
| b1_s8k | 8.370 | 8.455 | 8.448 | 8.536 | 8.539 | 918.65 | +0.824 % | 0.942 |
| b1_s16k | 30.983 | 31.045 | 31.034 | 31.451 | 31.504 | 995.94 | +1.075 % | 0.964 |
| b1_s32k | 119.992 | 120.140 | 120.266 | 120.529 | 121.634 | 1030.15 | +0.140 % | 0.950 |
| b1_s64k | 479.681 | 479.892 | 480.369 | 482.476 | 483.034 | 1035.74 | +0.274 % | 0.938 |
| b2_s4k | 4.972 | 4.979 | 4.974 | 5.016 | 5.002 | 788.27 | +0.629 % | 0.888 |
| b2_s16k | 62.395 | 62.584 | 62.344 | 63.262 | 63.390 | 992.74 | +1.146 % | 0.945 |

Step geomean: run1 931.61 (ratio 0.9339), run2 925.93 (0.9295), run3 927.33 (0.9311); median
927.329 (+0.564 % vs R15 922.132; +0.405 % vs the same-session r15 reference 923.59, ratio
0.9269). Trend row uses run3 (the median): Fwd 1205.403 (kernel unchanged), Bwd 851.870 (ratio
0.9148), Step 927.329.

b1_s4k: run1 +0.8 %, runs 2-3 -1.2 / -1.8 % against the reference, with a per-shape stddev of
6.0-7.6 TFLOPS in runs 2-3 against 2.1 in the reference. The focused per-process A/B at that shape
(sweep_r16v, 6 rotated processes per config) measures +2.54 % for g3d=4, so the full-set dip is
the shape's per-process spread (round 9 bimodality; round 15 run1 was 5 % off at the same shape),
not a property of g3d=4.

## Decision
ACCEPTED (trade-off, by Rule 3): combined step +0.405 % against the same-session reference, all
three candidate processes above it; b1_s4k -1.2 % in the median run (< 5 %), +2.5 % in the
focused per-process A/B.

## Real-training transfer check
- Bucket: K1 (codegen: LDS read issue depth in GEMM3)
- Cache key (if any): compile-time knob, part of the launcher cache key via the builder kwargs
- id(...) audit: n/a
- 4-step pen-and-paper trace hit rate: n/a
- Benchmark gain on this round: +0.405 % combined-step vs same-session reference (median of 3)
- Estimated real-training gain: +0.1..+1.1 % on the MLA attention backward at these shapes (the
  per-shape full-set deltas; the 4K shapes +0.6..+2.5 % per the focused A/B). No data-dependent
  path: same spills (94), same scratch (380 B / lane), same MFMA order.
- Decision: ACCEPT-as-real

## Next direction
ATT of the r15 code (profiles/att_r15_windows.txt): cycles 40-570 of the iteration, from the
head-step's WAR barrier to the first transpose-read after the publish barrier, hold no MFMA (WAR
barrier 101, Q/dO commit + next prefetch 344, lgkmcnt(0) 50, publish barrier 31). GEMM3 reads only the band-resident K and the
previous trip's dS slot, and that dS is already retired by the lgkmcnt(0) the WAR barrier's fence
plants, so GEMM3 can run inside the rendezvous (before the publish barrier) on the VGPR-staged
Q_PREF route, which has no LDS DMA pending to force vmcnt(0).
