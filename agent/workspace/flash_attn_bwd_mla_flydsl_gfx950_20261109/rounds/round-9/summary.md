# Round 9 — ACCEPTED (trade-off) — 2026-11-10 01:50 — per-process knob re-sweep: g1_ks_outer True -> False

## Hypothesis
Every sweep since round 7 timed its configs inside one process, under round 7's 16 KB per-wave
scratch (round 8 finding), so its ranking is not trustworthy. Re-sweep the schedule knobs with
one process per config (dev/sweep_proc.py) and take what holds up.

## Single change (config sweep = one round)
`ops_bwd._dkdv_kwargs`: add `g1_ks_outer=False` (the builder's default is on for D >= 128: emit
GEMM1's D-contraction outermost so the accumulator chains interleave). Kernel source unchanged.

## Sweep (dev/sweep_proc.py sweep_r9.py, 3 rounds, B2 S4K + B1 S16K; artifacts/sweep_r9.raw)

| config | B2 S4K d% | B1 S16K d% | gm d% |
|---|---|---|---|
| g3d=2 / 3 / 4 | -0.43 / -0.26 / -0.00 | +0.30 / +0.24 / +0.80 | -0.07 / -0.01 / +0.40 |
| iglp=0 / 3 | -26.60 / -21.35 | -28.32 / -23.12 | -27.46 / -22.24 |
| g3_st_n=2 | -11.09 | -9.43 | -10.26 |
| g2_fill | -15.48 | -16.13 | -15.81 |
| g2_half=False | +0.06 | +0.30 | +0.18 |
| g1_ks_outer=False | -0.05 | +3.25 | +1.59 |
| mfma_tie=1 | -45.30 | -43.51 | -44.42 |
| g3_sb=2 / 3 | -6.21 / -5.65 | -6.45 / -5.87 | -6.33 / -5.76 |

g3_dbat 1 and 3 trip the builder ("a D-tile group holds whole pairs"). iglp=0 was -1.8 % in the
round-8 in-process sweep and is -27 % alone: the in-process sweep mis-ranked it by 25 points.

## Resources (dev/isa_r9/g1ko0, S 1024)
512 VGPR, 98 VGPRs spilled (R8: 46), 396 B scratch per lane = 25 KB per wave (not 16 / 32 KB),
84 scratch instructions (R8: 45), v_accvgpr_read 175 (R8: 242), v_accvgpr_write 220 (R8: 269),
MFMA and ds_read counts unchanged, LDS 147712 B.

## Correctness
Full set 7 / 7 PASS in every run; SNR identical to R8 to 0.01 dB; dK/dV bitwise repeat True.
D128 regression not re-run (only `ops_bwd` changed, which the D128 path does not use).

## Results (n02-29 GPU 0, full set, rotated vs aiter a16)
Three full-bench processes with this config; one landed in a slow state on the >= 8K shapes:

| shape | bwd ms run1 | run2 | run3 | R8 bwd ms (R8 run / same-session ref) | Step run3 | delta vs R8 | step ratio vs aiter |
|---|---|---|---|---|---|---|---|
| b1_s4k | 2.708 | 2.741 | 2.749 | 2.693 / 2.690 | 726.63 | -1.662 % | 0.844 |
| b1_s8k | 9.228 | 9.091 | 9.118 | 9.248 / 9.242 | 863.41 | +1.045 % | 0.889 |
| b1_s16k | 34.432 | 33.384 | 33.502 | 34.457 / 34.616 | 936.83 | +2.234 % | 0.910 |
| b1_s32k | 132.434 | 129.126 | 129.337 | 132.838 / 132.792 | 971.76 | +2.033 % | 0.901 |
| b1_s64k | 527.482 | 513.051 | 513.657 | 527.571 / 528.993 | 981.37 | +2.005 % | 0.892 |
| b2_s4k | 5.249 | 5.253 | 5.260 | 5.251 / 5.261 | 752.69 | -0.291 % | 0.852 |
| b2_s16k | 69.193 | 66.113 | 66.264 | 69.290 / 69.341 | 945.19 | +3.515 % | 0.904 |

Step geomean: run1 866.17 (ratio 0.8714), run2 878.97 (0.8843), run3 876.92 (0.8843); median
876.92 (+1.255 % vs R8 866.046). R8 config in the same session (artifacts/r8_ref.raw): 864.96
(0.8716). Trend row uses run3 (the median): Fwd 1198.185 (kernel unchanged), Bwd 795.023
(ratio 0.8576), Step 876.918.

## Decision
ACCEPTED (trade-off): median +1.255 % combined-step; b1_s4k -1.66 %, b2_s4k -0.29 % (both < 5 %).

## The slow state (measured, not explained)
- Seen in 1 of 13 processes that ran this config at >= 8K: the round-9 full bench run1
  (S16K 34.43 ms, the same as R8). The other 12 (3 sweep children, 4 timing probes, full-bench
  runs 2-3, one --no-aiter bench, two rocprofv3 full-bench runs) are at 33.3-33.5 ms wall
  (32.6 ms kernel). 4K does not move.
- Not aiter: g1ko0 rotated with aiter bwd in one process 33.28 ms; after one aiter call 33.30 ms
  (artifacts/timing_probe_r9.raw). Not the scratch layout of another kernel: in the full bench our
  dkdv has the largest scratch (aiter's three bwd kernels and our odo/dqa16 have 0, our forward 48,
  one hipBLASLt GEMM 132). Not a scratch re-allocation: --scratch-memory-trace shows one 24 MiB and
  one 200 MiB allocation, both in the first second, in a (fast) full-bench run.
- R8 shows no such state in 17 processes without a 16 / 32 KB scratch kernel (34.27-34.62 ms).
Hypothesis (not verified): the base address of the one scratch allocation decides it, and twice
the spill traffic makes this variant sensitive to it.

## Real-training transfer check
- Bucket: K1 (codegen order of GEMM1's contraction)
- Cache key (if any): no cache
- id(...) audit: n/a
- 4-step pen-and-paper trace hit rate: n/a
- Benchmark gain on this round: +1.255 % combined-step (median of 3 processes)
- Estimated real-training gain: +1.0..+1.3 % at >= 8K in most processes; 0 % in the slow state
  (1 of 13 processes), never below R8.
- Decision: ACCEPT-as-real

## Next direction
Spills are now 98 VGPRs and scratch traffic is the common factor in both process-dependent
effects seen in rounds 8-9. Remove them: find which live ranges spill at NT 3 (K register packs,
dS packs, transpose-read bases parked in AGPRs) and restage one of them through LDS.
