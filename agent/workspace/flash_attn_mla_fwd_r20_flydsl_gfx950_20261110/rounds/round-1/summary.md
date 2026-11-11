# Round 1 — BASELINE — 2026-11-10 15:06 (UTC+8)

Kernel: `dev/xiaoming/flydsl_mla_fwd_gfx950` @ 33beb4f8, `primus_turbo/flydsl/attention/gfx950_mla/`
(forward round 20), copied to `src/branch_mla/`. Run with `buffer_ops._get_buffer_flags` patched to the
gfx9 V# word 0x27000 (missing on this tree). aiter 0.1.24.post1 (opus `gqa_d192_v128_kernel`), training
forward with requires_grad leaves. smci355-ccs-aus-n14-17, HIP device 0 = amd-smi GPU 3.

## Results (full set, 3 rotations, `artifacts/benchmark_full.csv`)

| shape | R20 TFLOPS | aiter TFLOPS | ratio | O rel | LSE max |
|---|---|---|---|---|---|
| dsv3 B2 S4K H128 | 1014.1 | 960.4 | 1.056 | 2.01e-3 | 4.75e-3 |
| dsv2lite B8 S4K H16 | 981.4 | 942.9 | 1.041 | 1.99e-3 | 3.17e-3 |
| kimik2 B4 S4K H64 | 1017.5 | 967.6 | 1.052 | 1.97e-3 | 4.01e-3 |
| kimik3 B2 S7168 H96 | 1124.7 | 989.2 | 1.137 | 1.83e-3 | 4.73e-3 |
| B1 S4K | 1004.8 | 948.4 | 1.060 | 1.95e-3 | 4.60e-3 |
| B1 S8K | 1135.2 | 1018.3 | 1.115 | 1.80e-3 | 4.75e-3 |
| B1 S16K | 1189.8 | 1055.1 | 1.128 | 1.75e-3 | 4.93e-3 |
| B1 S32K | 1218.5 | 1072.9 | 1.136 | 1.61e-3 | 4.82e-3 |
| B1 S64K | 1232.6 | 1084.2 | 1.137 | 1.58e-3 | 5.14e-3 |
| B2 S4K | 1013.1 | 974.8 | 1.039 | 2.01e-3 | 4.75e-3 |
| B2 S16K | 1192.0 | 1057.6 | 1.127 | 1.76e-3 | 4.82e-3 |

Geomean: R20 1098.25, aiter 1005.24, ratio 1.0925. All 11 PASS. Target 1.40x = 1407 TFLOPS.

## Sustained loop (`artifacts/power_probe.log`, 10 s each)

| run | TFLOPS | clock | socket power | PPT throttle | TFLOPS/MHz |
|---|---|---|---|---|---|
| R20 B1 S16K | 1177.0 | 1899 MHz | 1278 W | 0.05 | 0.620 |
| aiter B1 S16K | 1046.7 | 1266 MHz | 1332 W | 0.30 | 0.827 |
| R20 dsv3 | 1013.1 | 1990 MHz | 1318 W | 0.14 | 0.509 |
| aiter dsv3 | 962.7 | 1412 MHz | 1400 W | 0.97 | 0.682 |

## ATT (dsv3, one CU, `profiles/att_r1/`), main loop per 1e8 FLOP (wave cycles summed over waves)

R20 278K vs aiter 214K. MFMA-issue time 99K (stall 75K) vs 29K; barrier 49K vs 36K; exp 36K vs 24K;
LDS 35K vs 50K; VALU 19K vs 46K. MFMA pipe busy in the loop ~71 % (R20) vs ~92 % (aiter). The two waves
of a SIMD (w, w + 4) are barrier-locked in the same phase, so their MFMA clusters contend and the pipe
idles in the exp / LDS phases. Fixed region (prologue + drain + epilogue) equal per q-block.

## Decision
BASELINE.

## Next direction
Round 2: QK / PV back on 32x32x16 MFMA (config), testing whether the cycle gain shows up here.
