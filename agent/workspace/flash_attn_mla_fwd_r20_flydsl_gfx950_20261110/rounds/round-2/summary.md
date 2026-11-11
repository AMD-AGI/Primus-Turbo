# Round 2 — ROLLED BACK — 2026-11-10 15:20 (UTC+8)

## Hypothesis
R16 / R17 of the source campaign moved QK and P*V to 16x16x32 MFMA, accepting +3-5 % and +4 % cycles for
a higher clock under the n02-25 power cap. ATT here showed the loop losing cycles to MFMA issue stalls,
so 32x32x16 (half the MFMA instructions, 32-cycle ops) was expected to cut cycles and gain on this node.

## Single change
`ops.py`: pass `qk_mfma16=False` to the builder (P*V follows). `CODEGEN_REV` 10 -> 1002.

## Results (full set, 2 rotations, `artifacts/r2.csv`)

| shape | cand | base (R20) | cand / base |
|---|---|---|---|
| dsv3 | 981.7 | 1012.4 | 0.9697 |
| dsv2lite | 949.7 | 980.9 | 0.9682 |
| kimik2 | 982.5 | 1013.4 | 0.9695 |
| kimik3 | 1038.8 | 1121.6 | 0.9262 |
| B1 S4K | 973.2 | 1003.9 | 0.9694 |
| B1 S8K | 1048.6 | 1132.6 | 0.9258 |
| B1 S16K | 1071.8 | 1187.9 | 0.9023 |
| B1 S32K | 1090.6 | 1218.2 | 0.8953 |
| B1 S64K | 1102.1 | 1231.5 | 0.8949 |
| B2 S4K | 989.8 | 1009.0 | 0.9810 |
| B2 S16K | 1081.5 | 1191.2 | 0.9079 |

Geomean cand 1026.90 vs base 1096.29 (-6.33 %); aiter 1004.70 in the same run. All PASS (O rel and LSE
identical to R20 to 4 digits; not bitwise).

## Decision
ROLLED BACK (geomean down; B1 S32K / S64K / S16K regress > 5 %).

## Rollback Analysis
- Change made: QK and P*V on 32x32x16 MFMA instead of 16x16x32.
- Expected effect: fewer cycles, higher throughput on a cycle-bound node.
- Actual effect: B1 S16K sustained 1062.8 TFLOPS at 1559 MHz, 1278 W, PPT 0.07, 0.682 TFLOPS/MHz
  (R20: 1177.0 at 1899 MHz, 1278 W, PPT 0.05, 0.620). Cycles per FLOP fell 9 %, clock fell 18 %.
- Root cause: the node is not cycle-bound. Both builds settle at the same 1278 W socket power with
  PPT barely active, so a non-PPT power / current limit sets the clock, and 32x32x16 costs more energy
  per FLOP. Round 1's "cycle-bound" reading of the PPT counter was wrong.
- Lesson learned: on this node, as on n02-25, throughput = power limit / energy per FLOP. A change that
  saves cycles without saving energy buys nothing; judge candidates by energy per FLOP (TFLOPS at the
  sustained power), not by ATT cycles.

## Real-training transfer check
- Bucket: K1 (MFMA variant). Cache key: no cache. Decision: REJECT (performance), not over-fit.

## Next direction
Energy per FLOP. Measured bounds (source campaign, n02-25): no main-loop K/V DMA +11 %, exp -> fma
+6.7 %, LDS reads halved +9 %. Next: skip exp / softmax work on dead diagonal halves of the drain.
