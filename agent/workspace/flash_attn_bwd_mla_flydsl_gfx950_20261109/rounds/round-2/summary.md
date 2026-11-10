# Round 2 — ACCEPTED — 2026-11-09 23:05 — band-interleaved a16 dQ images

## Hypothesis
dQ error on the a16 path comes from bf16 atomics rounding the running sum, and grows with the
number of kv bands that add into an element (measured round 1: about -2 dB per doubling of S at
S >= 16K, i.e. error power ~ n^1.66). Spreading the bands over K images (band b -> image b % K),
summed in fp32 by the un-permute, should give about 0.66 * 10*log10(K) dB: +2 dB at K 2, +4 dB at K 4.
K is picked so an image takes at most 128 adds per element (cap 4): K 1 up to S16K (no change),
K 2 at S32K, K 4 at S64K. Cost: K x image zeroing and un-permute read (estimate +0.9 % at S64K).

## Single change
- dkdv builder `a16_imgs` (A16_IMGS): the a16 descriptor base becomes
  `((kv_tile_idx % K) * B + batch) * |image slice|` (64-bit base offset).
- dqa16 builder `n_img`: one descriptor per image (base offset i * B * |image|), fp32 sum of the
  K images before the scale. K = 1 emits the same arithmetic as before.
- ops_bwd: `_a16_imgs_for(Skv, block_kv)`; image stack [K, |image|]; the kernels get image 0 as the
  tensor argument (a flat argument's length is packed as int32; the stack is one allocation).

## Correctness
- `dev/d128_regress.py`: REGRESS PASS (bitwise vs Turbo, unchanged).
- `dev/imgs_check.py` (forced K): dK/dV bitwise equal to K 1; dq S2K 47.19 / 47.87 / 48.28,
  B2 S4K 46.42 / 47.37 / 47.98 for K 1 / 2 / 4 (smaller gains where few adds per element).
- Full set: 7 / 7 PASS.

## Results (n02-29 GPU 0, full set, rotated vs aiter a16)

| shape | K | Step TFLOPS | delta vs R1 | Bwd TFLOPS | Bwd ms | dq SNR (R1 -> R2) | dk / dv | aiter dq |
|---|---|---|---|---|---|---|---|---|
| b1_s4k | 1 | 658.31 | +0.109 % | 576.13 | 3.102 | 46.41 -> 46.42 | 48.46 / 49.65 | 49.09 |
| b1_s8k | 1 | 747.52 | +0.113 % | 655.46 | 10.905 | 45.16 -> 45.16 | 48.43 / 49.71 | 47.59 |
| b1_s16k | 1 | 792.90 | -0.147 % | 694.49 | 41.166 | 43.97 -> 43.97 | 48.55 / 49.63 | 46.23 |
| b1_s32k | 2 | 816.71 | +0.802 % | 713.23 | 160.331 | 41.97 -> 44.06 | 48.51 / 49.73 | 44.03 |
| b1_s64k | 4 | 839.67 | +0.892 % (R1 FAIL) | 734.13 | 623.058 | 40.00 -> 44.39 | 49.60 / 51.16 | 41.95 |
| b2_s4k | 1 | 660.33 | +0.161 % | 577.68 | 6.187 | 46.36 -> 46.37 | 48.53 / 49.79 | 49.02 |
| b2_s16k | 1 | 778.73 | -0.100 % | 679.09 | 84.198 | 43.82 -> 43.82 | 48.44 / 49.79 | 46.06 |

Geomean (7 PASS): Step 753.256 (aiter 995.96, ratio 0.7563), Bwd 658.817 (ratio 0.7069), Fwd 1200.849.
vs R1's 6-PASS geomean: step +1.985 % (the PASS set grew by b1_s64k, a high-TFLOPS shape); over the
6 shapes PASS in both rounds: step +0.156 %, worst shape -0.147 %.

## Decision
ACCEPTED. Primary aggregate improved on both readings, no shape regressed >= 5 %, and the round
does what it was for: every shape now passes the absolute 40 dB gate, b1_s32k / b1_s64k dQ now at
or above aiter a16 (44.06 vs 44.03, 44.39 vs 41.95).

## Attribution
The dQ gains match the running-sum model (+2.09 dB at K 2, +4.39 dB at K 4). The expected
zeroing / extra read cost did not show: the two K > 1 shapes got ~1 % faster in the backward.
Hypothesis (not measured): adjacent bands of one head run concurrently and hit the same q rows,
so splitting them over images halves / quarters the atomics per L2 line.

## Real-training transfer check
- Bucket: K1 (kernel-internal accumulation layout; per-call allocation)
- Cache key (if any): no data cache (launcher cache adds K, a shape scalar)
- id(...) audit: no tensor ids
- 4-step pen-and-paper trace hit rate: n/a
- Benchmark gain on this round: +0.156 % combined-step (common PASS set); +1.985 % headline incl. b1_s64k
- Estimated real-training gain: same as benchmark (no benchmark-idiom dependence); the headline part
  from b1_s64k becoming countable is an accuracy fix, not speed
- Decision: ACCEPT-as-real

## Next direction
q_split 1 by default (probe: -10.4 % kernel time at b2_s4k, -2.5 % at b1_s16k; removes the torch
slot sums and the bf16 slot round-trip, +1.1 dB dK/dV expected).
