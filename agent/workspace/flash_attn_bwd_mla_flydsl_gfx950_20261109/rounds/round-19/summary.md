# Round 19 — ACCEPTED — 2026-11-11 09:14 — knob re-sweep on the r18 base: GEMM2 transpose-read prefetch depth g2d 1 -> 2

## Machine
Screening (ISA and per-process timing) ran on n01-33 before that allocation expired; the full-set
A/B runs on smci355-ccs-aus-n02-29 again (new allocation), container xiaoming-dev, GPU 0, same
toolchain (torch 2.14.0+rocm10.1.0, flydsl 0.3.4.1, primus_turbo 0.6.1.dev4). r18's dkdv ISA hash
reproduces on n02-29 (ae8b324271; artifacts/isa_r19_n0229.txt), so r18 is re-measured in the same
session as the reference.

## Hypothesis
Round 18's cost came from the register allocator, not from the DS_PAR instructions: spills 94 -> 98
and GEMM1a's third hot-loop K-pack scratch reload lost its MFMA cover (5 -> 0). A builder knob that
reshapes the live ranges around GEMM1a/GEMM2 can move that reload back under MFMA cover without
reopening the race. Expected: the knob that restores a nonzero minimum reload cover wins back most
of the -1.3 %.

## Single change
`ops_bwd.py` `_dkdv_kwargs`: `g2d=1` -> `g2d=2` (artifacts/g2d.diff). Kernel source unchanged
(flash_attn_bwd_mla.py == r18).

## Screening
- ISA screen 1 (artifacts/sweep_r19.py, isa_r19.txt), 22 configs on r18: only three restore a
  nonzero minimum hot-loop reload cover: g2d=2 (12), llvm_opts=L2 only (13), llvm_opts=none (13);
  all others keep a cover-0 reload, and g3_st_n / g3_st_at / g3_dbat / mfma_tie variants raise
  spills to 103-183.
- Timed screen 1 (sweep_r19t.py / .raw, per process, 3 shapes): g2d=2 +2.12 % gm, L2 only +1.56 %,
  none +1.83 %; g2d=2 + L2 only -0.31 % (knobs do not add); r16 (racy reference) +2.07 %.
- ISA screen 2 (sweep_r19b.py, isa_r19b.txt): g2d=3 cover 0; g2d=2 with g3d 2/3 or L1 only keep
  cover 12-13; with g3d 5/6 or llvm_opts=none cover falls to 1.
- Timed screen 2 (sweep_r19c.py / .raw, 6 rounds): g2d=2 +2.23 %, g2d=2 g3d=2 +1.68 %,
  g2d=2 g3d=3 +1.02 %, g2d=2 L1 only +1.03 %.

ISA of the chosen config (n02-29):

| | md5 | spills | scratch | hot loop scratch_load / vmcnt(0) | hot reload MFMA cover (offset:cover) |
|---|---|---|---|---|---|
| r18 | ae8b324271 | 98 | 396 B | 3 / 1 | 0:35 16:18 208:0 |
| r19 (g2d=2) | a87e141e6e | 98 | 396 B | 3 / 2 | 0:35 16:37 256:12 |

## Correctness
- artifacts/bitwise_r19_vs_r18.log (r18 package vs r19, 3 repeats each), B1 S1024 / 2112 / 4096 /
  4224, B2 S4096, B1 S8192: dK and dV bitwise equal to r18 on every shape and repeat; dQ SNR of r19
  vs r18 49.8-62.0 dB, within r18's own repeat-to-repeat spread (bf16 atomic order).
- dev/d128_regress.py: REGRESS PASS (artifacts/d128_regress_r19.log).
- Full set 7 / 7 PASS in all three r19 runs; dQ / dK / dV SNR 44.68-47.02 / 49.54-49.66 /
  51.08-51.38 dB, the same as r18.

## Results
Full set, rotated r19 / r18 / r19 / r18 / r19 (artifacts/run_full_ab.sh, full_r19_run{1,2,3},
full_r18_run{1,2}, per_shape_ab.txt):

| shape | r19 bwd ms run1/2/3 | r18 bwd ms run1/2 | bwd d | r19 step (median) | r18 step (mean) | step d | step ratio vs aiter (r19) |
|---|---|---|---|---|---|---|---|
| b1_s4k | 2.484/2.594/2.589 | 2.556/2.666 | +2.16 % | 762.66 | 757.66 | +0.66 % | 0.884 |
| b1_s8k | 8.387/8.442/8.445 | 8.586/8.610 | +2.06 % | 918.57 | 905.28 | +1.47 % | 0.943 |
| b1_s16k | 30.997/31.056/31.115 | 31.692/31.702 | +2.06 % | 994.84 | 978.87 | +1.63 % | 0.961 |
| b1_s32k | 119.889/120.172/120.246 | 122.476/122.717 | +2.08 % | 1030.27 | 1014.46 | +1.56 % | 0.951 |
| b1_s64k | 480.869/480.589/480.886 | 489.554/489.468 | +1.82 % | 1035.54 | 1020.46 | +1.48 % | 0.938 |
| b2_s4k | 4.962/4.988/4.980 | 5.044/5.060 | +1.50 % | 786.93 | 778.19 | +1.12 % | 0.886 |
| b2_s16k | 62.244/62.177/62.326 | 63.352/63.321 | +1.75 % | 993.57 | 980.05 | +1.38 % | 0.946 |

Step geomean: r19 931.415 / 925.220 / 924.950 (median 925.220, ratio vs aiter 0.9292); r18
915.936 / 910.670 (mean 913.303, ratio 0.9170). Bwd geomean: r19 849.957 (median run, ratio
0.9125) vs r18 836.164 (+1.650 %). Step +1.305 % vs same-session r18; fwd +0.044 % (noise).
r19 is -0.23 % against the n02-29 R16 row (927.329), i.e. the race fix now costs about nothing.
b1_s4k is the noisiest shape (r19 run1 2.484 ms vs 2.59 ms in runs 2/3); its step delta (+0.66 %)
also carries fwd noise.

## Attribution
g2d=2 deepens the GEMM2 transpose-read prefetch, which changes the live ranges around GEMM1a; the
allocator then spills the third K-pack to offset 256 with 12 MFMAs between its reload and the
guarding vmcnt, and the second reload's cover goes back 18 -> 37. Spill count and hot-loop
scratch_load count are unchanged (98 / 3), so the gain is the restored cover, consistent with the
round-18 diagnosis. On r9 code g2d=1 had won (round 8: spills 60 -> 46); the optimum moved with the
hot loop again. Exact cycle cost not ATT-measured (hypothesis level).

## Decision
ACCEPTED: step +1.305 % vs r18, every shape +0.66 .. +1.63 % step, no regression. Rollback count 0.

## Real-training transfer check
- Bucket: K1 (kernel-internal prefetch depth)
- Cache key (if any): no cache
- id(...) audit: n/a
- 4-step pen-and-paper trace hit rate: n/a
- Benchmark gain on this round: +1.31 % combined-step
- Estimated real-training gain: +1.31 % combined-step (compile-time knob of the same kernel)
- Decision: ACCEPT-as-real

## Next direction
The remaining gap to aiter is about 7 % step / 9 % bwd, largest at 4K (step 0.88). With r16's
performance back, the next lever is the 98-VGPR spill itself (scratch 396 B): free the 4 extra
spills DS_PAR added (derive the GEMM3 read base from the write base instead of carrying both),
then look at the 4K shapes (prologue / masked-loop share).
