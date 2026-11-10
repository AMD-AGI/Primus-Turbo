# Round 1 — BASELINE — 2026-11-09 22:40

## Hypothesis
Starting point: the Turbo FlyDSL fused dK/dV + a16-dQ backward, generalized to D_QK 192 / D_V 128
(`flash_attn_bwd_mla.py`, a fork of `primus_turbo/flydsl/attention/flash_attn_bwd.py`) with the
host flow in `ops_bwd.py`. No optimization yet.

## Code state (what the baseline is)
- dkdv builder: `head_dim_v`; Q/K LDS rows padded to 256 (XOR swizzle needs a power-of-two row);
  dO/V paths on D_V; GEMM2 skips dV for d-tiles >= DT_V; GEMM3 d-tile run of 6 per wave taken as
  interleaved pairs (`_g3_toff`) because the K^T read adds the tile by XOR.
- Config: block_kv 128, block_q 32, flat_wg 256, waves_per_eu 1, q_pref, g3_defer, k_reg, g2d 2,
  g3_dbat 2, wsq_a16 (bf16 dQ image, one image).
- Host: prescale Q, odo (delta, D_V), torch.zeros image, lse prescale, dkdv, dqa16 un-permute,
  dK/dV slot sum (torch) when q_split 2. q_split 2, falling back to 1 when the dK slot workspace
  would exceed int32 elements (b1_s64k).
- Resources (`FLYDSL_DUMP_IR`, B1 S1K): dkdv 430 VGPR, 0 spill, LDS 106752 B, 70 SGPR.
- D128 regression (`dev/d128_regress.py`): bitwise vs Turbo (det path dq/dk/dv, a16 dk/dv).

## Results (n02-29, GPU 0, full set, rotated A/B vs aiter a16 in one session)

| shape | Check | Bwd TFLOPS | Bwd ms | aiter a16 Bwd | bwd ratio | Step TFLOPS | aiter Step | step ratio | SNR dq/dk/dv | aiter SNR dq/dk/dv |
|---|---|---|---|---|---|---|---|---|---|---|
| b1_s4k | PASS | 575.0 | 3.108 | 809.5 | 0.710 | 657.6 | 865.2 | 0.760 | 46.41/48.46/49.65 | 49.08/52.17/52.60 |
| b1_s8k | PASS | 654.1 | 10.927 | 913.0 | 0.716 | 746.7 | 975.6 | 0.765 | 45.16/48.43/49.71 | 47.58/52.20/52.81 |
| b1_s16k | PASS | 695.4 | 41.113 | 965.5 | 0.720 | 794.1 | 1036.5 | 0.766 | 43.97/48.55/49.63 | 46.23/52.51/52.60 |
| b1_s32k | PASS | 706.2 | 161.927 | 1010.9 | 0.699 | 810.2 | 1081.6 | 0.749 | 41.97/48.51/49.73 | 44.03/52.38/52.65 |
| b1_s64k | FAIL (dq 39.996) | 726.4 | 629.695 | 1032.4 | 0.704 | 832.3 | 1102.6 | 0.755 | 40.00/49.60/51.16 | 41.95/52.48/52.65 |
| b2_s4k | PASS | 576.0 | 6.205 | 830.8 | 0.693 | 659.3 | 886.8 | 0.743 | 46.36/48.53/49.79 | 49.01/52.24/52.86 |
| b2_s16k | PASS | 679.6 | 84.130 | 985.0 | 0.690 | 779.5 | 1050.3 | 0.742 | 43.82/48.44/49.79 | 46.05/52.27/52.75 |

Geomean over the 6 PASS shapes: Step 738.59 (aiter 979.16, ratio 0.7543); Bwd 645.45 (aiter 915.90,
ratio 0.7047); Fwd 1182.18. b1_s64k is excluded from the geomean until its dQ passes (user decision:
keep the absolute > 40 dB gate and fix dQ accuracy first).

Kernel breakdown (torch.profiler, `artifacts/kernel_breakdown.txt`): main kernel 5221.5 us vs aiter
3870.2 us at b2_s4k (0.741), 39256.1 vs 28618.6 us at b1_s16k (0.729). Side kernels 980 us vs 487 us
at b2_s4k; the largest is the q_split-2 dK/dV slot sum (2 x torch reduce, 579.1 us, 9.3 %).

## Measurement notes
- aiter reference = a16 (`PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32=0`, kernel `fmha_bwd_hd192_128_bf16_causal_br_a16_pssk`),
  which Primus training uses (`runner/helpers/envs/base_env.sh` defaults it to 0). Turbo's own default
  (1) selects the a32 kernel: 7.41 ms vs 4.25 ms at b2_s4k (`dev/aiter_dout_layout.log`). A first
  quick run without the env (`logs/baseline_quick.log`) compared against a32 and is void.
- dQ SNR falls about 1.2-2 dB per doubling of S on both sides (bf16 atomic running-sum rounding);
  ours is 1.95-2.67 dB below aiter a16 at every shape.
- q_split 1 (b1_s64k) gives dk/dv 49.60/51.16 vs 48.5/49.7 at q_split 2: the bf16 slot round-trip
  costs about 1.1-1.5 dB of dK/dV SNR.

## Decision
BASELINE.

## Real-training transfer check
- Bucket: K1 (kernel-internal; host flow allocates per call)
- Cache key (if any): compiled-launcher cache keyed on shape/config scalars only; no tensor ids
- id(...) audit: none used
- 4-step pen-and-paper trace hit rate: n/a (no data cache)
- Benchmark gain on this round: n/a (baseline)
- Estimated real-training gain: n/a
- Decision: ACCEPT-as-real (baseline)

## Next direction
Round 2: fix dQ accuracy for long sequences (band-interleaved a16 images summed in fp32), as the
user asked before any performance round. Then q_split 1 (probe: -10.4 % kernel time at b2_s4k,
also +1.1 dB dK/dV), then the main kernel (0.73-0.74 of aiter's).
