# Round 4 — ACCEPTED (marginal)

- Time: 2026-11-09 09:10 (UTC+8)
- Hypothesis: continue round 3's trend; k_head 1 frees another 8 VGPRs of head K packs.
- Change (single variable): `ops.py` default `k_head` 2 -> 1. Snapshot: `kernel_snapshot/`.
- Resources: `.vgpr_spill_count` 90 (unchanged), scratch 364 B (unchanged). The remaining spill
  set does not depend on the head K packs (round-3 ISA: 0 scratch in the main loop; 89 in the
  prologue block, 64 at loop exit, ~156 in the epilogue).

## Correctness

All 7 PASS: O rel vs fp32 0.0016–0.0020, LSE max abs 0.0046–0.0051.

## Performance (full set, GPU 0)

| shape | round-4 | round-3 | delta | aiter | ratio |
|---|---|---|---|---|---|
| b1_s4k | 841.1 | 839.4 | +0.20 % | 1022.9 | 0.822 |
| b1_s8k | 952.4 | 949.0 | +0.36 % | 1139.2 | 0.836 |
| b1_s16k | 1010.3 | 1002.0 | +0.83 % | 1189.1 | 0.850 |
| b1_s32k | 1025.3 | 1020.8 | +0.44 % | 1212.3 | 0.846 |
| b1_s64k | 1043.6 | 1044.4 | -0.08 % | 1222.8 | 0.853 |
| b2_s4k | 826.3 | 822.8 | +0.43 % | 1013.6 | 0.815 |
| b2_s16k | 986.3 | 979.4 | +0.70 % | 1194.9 | 0.825 |

Geomean 951.43 vs 947.53 (+0.41 %); aiter 1139.05; ratio 0.8353.
Quick-set subset: 925.21 / 1108.07, ratio 0.8350.

Decision: ACCEPT per rule (aggregate up, no shape down >= 5 %). The gain is near the per-shape
run-to-run noise (0.3–0.4 %), though 6 of 7 shapes moved up. k_head is now at its minimum (1).

## Real-training transfer check

Bucket K1: compile-time kernel parameter, no cache, no tensor-identity dependence.
