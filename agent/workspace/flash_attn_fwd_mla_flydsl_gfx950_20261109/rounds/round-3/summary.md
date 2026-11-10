# Round 3 — ACCEPTED

- Time: 2026-11-09 09:06 (UTC+8)
- Hypothesis: the kernel is register-bound at the 256-VGPR cap (round 2). Reading fewer K
  k-steps before the first QK MFMA frees 16 VGPRs (2 k-steps x 8), which should cut spills.
- Change (single variable): `ops.py` default `k_head` 4 -> 2. Snapshot: `kernel_snapshot/`.
- Resources (ISA dump, B1 S4K build): `.vgpr_count 256`, `.vgpr_spill_count` 105 -> 90,
  `.private_segment_fixed_size` 424 -> 364.

## Correctness

All 7 PASS: O rel vs fp32 0.0016–0.0020, LSE max abs 0.0046–0.0051 (same as baseline).

## Performance (full set, GPU 0, aiter same GPU)

| shape | round-3 | baseline | delta | aiter | ratio |
|---|---|---|---|---|---|
| b1_s4k | 839.4 | 800.8 | +4.8 % | 1022.9 | 0.821 |
| b1_s8k | 949.0 | 911.4 | +4.1 % | 1146.0 | 0.828 |
| b1_s16k | 1002.0 | 987.3 | +1.5 % | 1188.7 | 0.843 |
| b1_s32k | 1020.8 | 1002.6 | +1.8 % | 1215.1 | 0.840 |
| b1_s64k | 1044.4 | 1020.7 | +2.3 % | 1222.5 | 0.854 |
| b2_s4k | 822.8 | 780.9 | +5.4 % | 1012.1 | 0.813 |
| b2_s16k | 979.4 | 961.5 | +1.9 % | 1195.0 | 0.820 |

Geomean 947.53 vs 919.04 (+3.10 %); aiter 1140.09; ratio 0.8311.
Quick-set subset: 922.04 / 1107.50, ratio 0.8325.

Decision: ACCEPT (aggregate up, no shape regressed).

Note: the earlier k_head sweep (prototype note, S 16K, other GPU) only covered 4/6/8/10, where
fewer head k-steps was already monotonically better; 2 continues that trend.

## Real-training transfer check

Bucket K1: compile-time kernel parameter, no wrapper cache, no dependence on tensor identity.
The gain applies to every forward call with this shape class. Short sequences gain most
(+4.8 / +5.4 % at 4K), consistent with spills concentrated in prologue / epilogue.
