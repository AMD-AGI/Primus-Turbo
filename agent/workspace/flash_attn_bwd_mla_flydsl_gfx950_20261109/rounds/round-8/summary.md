# Round 8 — ACCEPTED — 2026-11-10 01:05 — schedule-knob sweep at kv tile 192: g2d 2 -> 1, g3_sb 1 -> 0

## Hypothesis
Round 7 moved to a 192-row kv tile (NT 3) and left 60 VGPRs spilled. The schedule knobs were
tuned at NT 2; at NT 3 a shallower GEMM2 transpose-read ring (g2d) frees registers and
issue slots, and the GEMM3 sched_barrier (g3_sb) may now cost more than it orders.

## Single change (config sweep = one round)
`ops_bwd._dkdv_kwargs`: `g2d` 2 -> 1 (GEMM2 transpose-read prefetch ring depth),
`g3_sb` 1 -> 0 (no sched_barrier after the deferred GEMM3 issue). Kernel source unchanged.

## Sweep (dev/sweep.py, B2 S4K + B1 S16K; artifacts/sweep_r8.raw, sweep_r8b.raw)
First pass, 5 repeats, gm %: g2d=1 +2.59, g3_sb=0 +1.39, g3d=3 +0.63, iglp=0 -1.83,
g3_st_at 0 / 2 about -11 / -12, mfma_tie=2 -43.4, g3_dbat=6 -35.2.
Confirmation, 11 repeats: g2d=1 +2.99, g2d=1 + g3_sb=0 +3.44, g2d=1 + g3d=3 +1.45,
all three +3.31. Chosen: g2d=1 + g3_sb=0.

## Resources (dev/isa_r8, S 1024)
512 VGPR (256 arch + 256 acc), 46 VGPRs spilled, 188 B scratch per lane (round 7: 60, 244 B),
LDS 147712 B.

## Correctness
Full set 7 / 7 PASS; dQ/dK/dV SNR identical to round 7 to 0.01 dB on every shape; dK/dV bitwise
repeat True on every shape. D128 regression not re-run: it builds the dkdv module with its own
kwargs (dev/d128_regress.py) and does not go through `ops_bwd`, the only file changed.

## Results (n02-29 GPU 0, full set, rotated vs aiter a16)

| shape | Step TFLOPS | delta vs R7 | Bwd TFLOPS | Bwd ms | step ratio vs aiter | SNR dq/dk/dv |
|---|---|---|---|---|---|---|
| b1_s4k | 738.91 | +6.081 % | 663.67 | 2.693 | 0.856 | 47.02/49.57/51.21 |
| b1_s8k | 854.48 | +6.853 % | 772.86 | 9.248 | 0.874 | 46.01/49.54/51.27 |
| b1_s16k | 916.36 | +6.523 % | 829.72 | 34.457 | 0.884 | 45.01/49.66/51.08 |
| b1_s32k | 952.40 | +6.318 % | 860.85 | 132.838 | 0.879 | 45.08/49.62/51.16 |
| b1_s64k | 962.08 | +6.910 % | 867.00 | 527.571 | 0.871 | 44.68/49.59/51.16 |
| b2_s4k | 754.89 | +6.215 % | 680.66 | 5.251 | 0.851 | 46.98/49.66/51.38 |
| b2_s16k | 913.09 | +7.774 % | 825.20 | 69.290 | 0.869 | 44.87/49.54/51.31 |

Geomean: Step 866.046 (+6.666 % vs R7; aiter 996.63, ratio 0.8690), Bwd 781.734 (ratio 0.8384),
Fwd 1203.640 (unchanged kernel).

Same-session full-bench A-B-A (dev/bench_over.py; artifacts/aba_*.raw), one config per process:
R7 config 813.89 -> R8 868.06 -> R7 config 811.97 Step geomean (aiter 996.95 / 995.37 / 995.69).

## Decision
ACCEPTED: +6.666 % combined-step, every shape +6.1..+7.8 %, correctness unchanged.

## Measurement finding: the in-process sweep under-measured this change by half
The sweep predicted +3.4 % and the full bench measured +6.7 %. Measured, B1 S16K, bench sampler
(artifacts/timing_probe_r8*.raw, ktrace summary below):
- Timer is not the cause: CUDA events with 3 or 16 iterations and the bench's perf_counter
  sampler give the same numbers within 0.2 %.
- Build order is not the cause (sweep_r8c / sweep_r8d: same +3 % with either config built first).
- R8 alone in a process: 34.27 / 34.36 / 34.39 ms. R8 in a process that also runs R7: 36.81 /
  36.91 ms. R7: 37.42 ms alone, 37.36 / 37.48 ms with R8.
- One R7 call at B1 S1K before any timing is enough: R8 then runs 36.66 ms for the whole process.
  One call of `g2d=1, g3_sb=1` (also 188 B scratch) first leaves R8 at 34.55 ms.
- rocprofv3 kernel trace: the dkdv kernel itself is slower (median 33.41 ms alone, 36.07 ms
  after R7); odo, dqa16 and the torch kernels are unchanged. Scratch_Size 188 (R8) vs 244 (R7).
- Consequence for past sweeps: every config in one sweep process was timed after the largest
  scratch any config in that process used. A config with less scratch than that was penalized.

Isolating it from R7 (dev/scratch_probe.py: one throwaway HIP kernel with a chosen per-lane
scratch size runs once, then R8 is timed alone; artifacts/scratch_probe*.raw):

| earlier dispatch, scratch per lane (per wave) | R8 B1 S16K ms |
|---|---|
| none | 34.48 / 34.30 |
| 176 B (11 KB) | 34.22 |
| 256 B (16 KB) | 36.85 / 37.00 |
| 272 B (17 KB) | 34.47 |
| 304 B (19 KB) | 34.35 |
| 400 B (25 KB) | 34.36 |
| 512 B (32 KB) | 36.26 |
| 528 B (33 KB) | 34.41 |
| 1024 B (64 KB) | 34.02 |
| 256 B then 528 B | 34.40 |
| 528 B then 256 B | 34.47 |

Measured: only a 16 KB or 32 KB per-wave scratch slows R8 (+5.2..+7.9 %); R7's 244 B rounds to
16 KB per wave at 1 KB granularity, which is why one R7 call is enough. The largest earlier
private segment decides (256 then 528, and 528 then 256, both fast). So R7 itself always ran
with a 16 KB per-wave scratch, and part of this round's gain is R8 leaving it (R8: 188 B, 12 KB
per wave): with both on a 16 KB layout in one process R8 is +1.5 % (36.81 / 36.91 vs
37.36 / 37.48 ms).
Hypothesis (not verified): with a power-of-two per-wave stride every wave's same spill slot maps
to the same memory channels. Not explained: 64 KB per wave is fast.

## Attribution
Measured in isolation (timing_probe_r8c.raw, B1 S16K): `g2d=1, g3_sb=1` 34.38 ms vs
`g2d=1, g3_sb=0` 34.45 ms, so at S16K the gain is the g2d change; g3_sb=0 was +1.4 % at B2 S4K in
the (equally contaminated) confirmation sweep. Not separated: how much of g2d's gain is the
14 fewer spilled VGPRs vs the shorter read-ahead burst.

## Real-training transfer check
- Bucket: K1 (launch configuration / schedule knobs)
- Cache key (if any): no cache
- id(...) audit: n/a
- 4-step pen-and-paper trace hit rate: n/a
- Benchmark gain on this round: +6.666 % combined-step
- Estimated real-training gain: +6.7 % if nothing earlier in the training process dispatched a
  kernel whose scratch is 16 or 32 KB per wave (241-256 or 497-512 B per lane) without a larger
  one; under such a kernel R8 slows, R7 does not (it already runs on 16 KB), and R8's gain is
  +1.5 % at B1 S16K (+3.4 % gm with B2 S4K in the sweep). The full bench runs
  our forward (44 B) and aiter next to the backward, like a training step.
- Decision: ACCEPT-as-real (gain positive in both scratch states; size depends on process history)

## Next direction
1. Sweep harness: one process per config (dev/sweep_proc.py), rotated across processes, so
   scratch state cannot leak between configs. Every round-7/8 sweep was timed under R7's 16 KB
   scratch; re-run the near-misses (g3d=3, iglp, g3_st_at, g3_dbat) with it.
2. Remove the remaining 46 spilled VGPRs: zero scratch removes the spill traffic and the
   dependence on other kernels' scratch.
