# Round 6 — ACCEPTED (rowsum moved to s5, P packs pinned in s3)

- Time: 2026-11-09 11:40 (UTC+8)
- Base: round 4 (round 5 rolled back).
- Hypothesis: round 5's idea (rowsum MFMAs out of the MFMA-bound s3 into s5, which waits on
  the trailing group's s4) holds once the exp/cast chain is prevented from sinking into s4.
- Single change: rowsum at the end of s5 as in round 5, plus `Swp8Context.pin_p`: an empty
  tied inline asm (`"=v,0"`, side effects) per P pack at the end of s3. The P packs' MFMA
  consumers are a stage later, so the VALU-def wait states the asm hides are satisfied.
  SWP8_REV 9.
- Resources: 256 VGPR, spill 32 (private 132 B), 176 MFMA, 37 barriers.

## Correctness

- `swp8_check.py`: bitwise equal to round 7 on 5 shapes x 3 reps.
- All 7 PASS vs fp32: O rel 0.0016–0.0020, LSE max abs 0.0046–0.0051.

## Performance (full set, GPU 0)

| shape | round 6 | round 4 | delta |
|---|---|---|---|
| b1_s4k | 877.0 | 875.3 | +0.19 % |
| b1_s8k | 982.6 | 985.5 | -0.29 % |
| b1_s16k | 1040.0 | 1034.8 | +0.50 % |
| b1_s32k | 1077.5 | 1068.1 | +0.88 % |
| b1_s64k | 1091.5 | 1083.1 | +0.78 % |
| b2_s4k | 870.1 | 865.7 | +0.51 % |
| b2_s16k | 1033.1 | 1026.7 | +0.62 % |

Geomean 992.33 vs 987.84 (+0.45 %); aiter 1136.42; ratio 0.8732. No shape regresses >= 5 %.

Decision: ACCEPT.

## ATT (B1 S16K, per visit, round 4 -> round 6)

s3 526/532 -> 446/454; s2 499/452 -> 459/412; s4 399/416 -> 389/374; s5 296/395 -> 329/442;
v_exp is back in s1 (32 per visit). Loop pair 6173 -> 5975 (-3.2 %); whole-trace latency on
the traced CU 121.68 M -> 118.41 M (-2.7 %).

## Why the wall-time gain is smaller than the cycle gain (measured)

Sustained 12 s runs of B1 S16K on the benchmark GPU (amd-smi GPU 3 = HIP device 0), sampled
4x each, two alternations:

| kernel | TFLOPS | GFX clock (MHz) | socket power (W) |
|---|---|---|---|
| round 4 | 1036.9 / 1031.4 | 1585–1632 | 1378–1400 |
| round 6 | 1040.1 / 1040.0 | 1594–1609 | 1392–1398 |
| aiter | 1188.4 / 1192.6 | 1373–1381 | 1383–1398 |

All three sit at ~1.39–1.40 kW (idle 239 W). Round 6 runs ~1 % lower clock than round 4, so
about a third of the cycle gain is returned as clock. aiter runs ~13 % lower clock than
swp8 yet is ~14.5 % faster: per FLOP it uses ~25 % fewer cycles and ~13 % less energy.

## Real-training transfer check

- Bucket: K1. The power cap applies to training steps too, so the transferable gain is the
  sustained-run +0.4–0.8 %, not the ATT cycle delta. Output bitwise unchanged.
