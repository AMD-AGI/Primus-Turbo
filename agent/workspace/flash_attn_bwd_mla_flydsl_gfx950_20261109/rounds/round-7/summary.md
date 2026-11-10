# Round 7 — ACCEPTED (trade-off) — 2026-11-10 00:35 — kv tile 128 -> 192

## Hypothesis
Re-profile after rounds 5-6: the issue cost matches aiter's and the gap is stall, led by the dQ
atomics (every atomic is executed past L2 on both kernels). aiter issues exactly 2/3 of our atomic
wavefronts for the same dQ volume; its config table lists the kernel as `ts=192` (kv tile 192).
A 192-row band cuts the (band, q-row) partials -- dQ atomics, and Q/dO fetches -- by one third and
amortizes the per-head-step fixed costs (Q/dO commit, barriers) over 1.5x the MFMA work. Risk: the
dK/dV accumulators need 240 registers per lane instead of 160.

## Single change
`ops_bwd.mla_bwd_sbhd` default `block_kv` 128 -> 192 (NT 2 -> 3 kv 16-tiles per wave), with what
the larger tile needs to run at all:
- `dv_pin=False` in `_dkdv_kwargs` (DV_PIN auto-enables at NT >= 3 and is not wired for MLA; at
  NT 2 it was already off, so block_kv 128 is unchanged);
- the host accepts Skv % block_kv != 0 (the last band is partial: it lies wholly inside the causal
  mask, its K/V rows past Skv read 0, its dK/dV rows past Skv clip on the buffer bound), and the
  dQ image count uses the rounded-up band count.
Kernel source unchanged.

## Resources (dev/isa_r7, S 3072)
512 VGPR (256 arch + 256 acc), 60 VGPRs spilled (244 B scratch per lane), LDS 147712 B
(block_kv 128: 432 VGPR, 0 spill, 106752 B).

## Correctness
- Partial last band (dev/tail_band_check.py, B2 S4K heads 0/63/127, last band rows 4032-4095):
  dK/dV SNR 48.0-50.3 / 50.4-51.5 dB, the same as full bands; dQ of the last q rows 43.1-43.4 dB,
  the same as the band before it (later q rows collect more bands).
- Full set 7 / 7 PASS; dK/dV bitwise repeat True on every shape; D128 regression PASS.
- dQ SNR +0.3..+1.0 dB on every shape (one third fewer bf16 adds per dQ element).

## Pre-check (divisible S, dev/sweep.py, 5 repeats; artifacts/sweep_r7.raw)
B2 S6144 11.832 -> 11.555 ms (+2.40 %), B1 S12288 22.513 -> 20.550 ms (+9.55 %).

## Results (n02-29 GPU 0, full set, rotated vs aiter a16)

| shape | Step TFLOPS | delta vs R4 | Bwd TFLOPS | Bwd ms | step ratio vs aiter | SNR dq/dk/dv |
|---|---|---|---|---|---|---|
| b1_s4k | 696.55 | -2.080 % | 616.68 | 2.898 | 0.805 | 47.02/49.57/51.21 |
| b1_s8k | 799.68 | +2.421 % | 711.14 | 10.051 | 0.820 | 46.01/49.54/51.27 |
| b1_s16k | 860.25 | +5.408 % | 766.62 | 37.292 | 0.829 | 45.01/49.66/51.08 |
| b1_s32k | 895.80 | +7.676 % | 797.70 | 143.353 | 0.826 | 45.08/49.62/51.16 |
| b1_s64k | 899.90 | +6.873 % | 798.10 | 573.120 | 0.815 | 44.68/49.59/51.16 |
| b2_s4k | 710.72 | -1.053 % | 630.97 | 5.665 | 0.800 | 46.98/49.66/51.38 |
| b2_s16k | 847.23 | +3.933 % | 751.95 | 76.040 | 0.807 | 44.87/49.54/51.31 |

Geomean: Step 811.921 (+3.252 % vs R4; aiter 996.79, ratio 0.8145), Bwd 721.270 (ratio 0.7735),
Fwd 1206.162 (unchanged kernel).

## Decision
ACCEPTED (trade-off): aggregate +3.252 %; b1_s4k -2.080 % and b2_s4k -1.053 % (both < 5 %).

## Attribution
Measured: the gain grows with S (+2.4 % at 8K to +7.7 % at 32K), as the per-band fixed work and
atomics shrink against the MFMA work. Not measured yet: how much the 60 spilled VGPRs cost, and
why 4K loses. Plausible (unverified): at 4K the partial last band and the wider masked span
(6 masked q-blocks of 32 per band instead of 4) are a larger share, and MASK_ALIGN turns off
(it needs 32 kv rows per wave, now 48).

## Real-training transfer check
- Bucket: K1 (launch configuration / tile size)
- Cache key (if any): no cache
- id(...) audit: n/a
- 4-step pen-and-paper trace hit rate: n/a
- Benchmark gain on this round: +3.252 % combined-step
- Estimated real-training gain: +3.252 % (no dependence on tensor reuse)
- Decision: ACCEPT-as-real

## Next direction
Remove the spill at NT 3: candidates are k_reg False (K for GEMM1a from LDS instead of 72 VGPRs
per lane), a shallower g2d / g3d ring, or pinning more transpose-read bases. Then look at the 4K
loss (masked span / MASK_ALIGN at 48 rows per wave).
