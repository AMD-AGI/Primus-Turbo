# Round 1 — BASELINE

- Time: 2026-11-09 08:49 (UTC+8)
- Kernel: `flydsl_mla/` as of the prototype note 2026-11-09_0840 (loop_masks fix, block_m 256,
  waves_per_eu 2, stagger off, k_head 4). Snapshot: `kernel_snapshot/`.
- GPU 0 pinned (smci355-ccs-aus-n04-25), aiter re-measured back to back on the same GPU.
- Harness: `quick_test_bench.py` (5 repeats, ~3 s budget per kernel per shape; mean ± pstdev).

## Correctness (full set, 7 shapes)

All PASS. O rel vs fp32 rows 0.0016–0.0020 (gate 0.003); LSE max abs 0.0046–0.0051 (gate 0.01);
O rel vs aiter 0.0031–0.0033.

## Performance (full set)

| shape | proto TFLOPS | ± | aiter TFLOPS | ratio |
|---|---|---|---|---|
| b1_s4k | 800.8 | 3.4 | 1022.9 | 0.783 |
| b1_s8k | 911.4 | 0.7 | 1146.5 | 0.795 |
| b1_s16k | 987.3 | 1.2 | 1188.9 | 0.830 |
| b1_s32k | 1002.6 | 2.3 | 1212.3 | 0.827 |
| b1_s64k | 1020.7 | 3.9 | 1224.7 | 0.833 |
| b2_s4k | 780.9 | 1.4 | 1014.1 | 0.770 |
| b2_s16k | 961.5 | 1.8 | 1196.7 | 0.803 |

Geomean: proto 919.04, aiter 1140.64, ratio 0.8057.

Quick set (b1_s4k / b1_s16k / b1_s64k / b2_s4k, `quick_baseline.log`): proto geomean 891.07,
aiter 1107.78, ratio 0.8044. Same-shape drift full vs quick run: 0.3–0.4 %.

## Real-training transfer check

Bucket K1 (pure kernel-side, no wrapper cache). Baseline only, nothing to transfer yet.
