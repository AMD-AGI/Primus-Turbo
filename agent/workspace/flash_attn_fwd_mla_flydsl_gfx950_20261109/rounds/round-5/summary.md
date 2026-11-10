# Round 5 — ACCEPTED

- Time: 2026-11-09 09:15 (UTC+8)
- Hypothesis: ATT (round 2 analysis) put 21.9 % of wave latency in `s_barrier`. The kernel already
  has `fold_mem_barriers` (memory-cluster barrier -> sched fence, and with late DMA also the PV
  barrier), gated on `Q_HEADS_PER_WG > 1` (GQA-merged 8-wave WGs). The hazard argument only needs
  all waves of the WG to share the cooperative K/V DMA, which holds for block_m 256 too.
- Change (single variable): `fold_mem_barriers = sink_drains and traits.NUM_WAVES > 4`
  (was `Q_HEADS_PER_WG > 1`). D128 / block_m 128 (4 waves) path unchanged.
  Snapshot: `kernel_snapshot/`.

## Hazard check (steady-state loop, fold on, vm_drain_sync = 0)

Each wave waits lgkmcnt(0) + vmcnt(0) before the barrier closing C1 and C5 (one per tile).
RAW: K buf1 (DMA C3) read at next C0 after the C5 barrier; V buf1 (DMA C3) read at C6/C7 after
the C5 barrier; V buf0 / K buf0 (DMA C7) read at next C2 / C4 after the next C1 barrier.
WAR: K buf1 / V buf1 overwritten in C3 after the C1 barrier, which follows their last reads
(C0/C1, previous C6/C7); K buf0 / V buf0 overwritten in C7 after the C5 barrier, which follows
their reads in C2–C5. The epilogue re-inserts a barrier at loop exit when fold is on.

## Resources

`.vgpr_count 256`, spill 90, scratch 364 B (unchanged). `s_barrier` count in ISA 25 -> 20.

## Correctness

- Harness: all 7 PASS; O rel vs fp32 0.0016–0.0020, LSE max 0.0046–0.0051; full-tensor O rel vs
  aiter identical to round 1/4 to 5 decimals.
- Race check (`profiles/bitwise_check.py round-4 5`, log `artifacts/bitwise_vs_r4.log`): O and LSE
  bitwise equal to round 4 in 5/5 repeats at B1 S4K, B1 S16K, B2 S4K, B1 S64K. The two builds are
  distinct kernels (same process, S16K: round-4 10.80 / 10.83 ms, round-5 10.35 / 10.45 ms).

## Performance (full set, GPU 0)

| shape | round-5 | round-4 | delta | aiter | ratio |
|---|---|---|---|---|---|
| b1_s4k | 882.5 | 841.1 | +4.9 % | 1024.4 | 0.861 |
| b1_s8k | 986.9 | 952.4 | +3.6 % | 1144.6 | 0.862 |
| b1_s16k | 1050.1 | 1010.3 | +3.9 % | 1188.8 | 0.883 |
| b1_s32k | 1082.5 | 1025.3 | +5.6 % | 1211.8 | 0.893 |
| b1_s64k | 1100.7 | 1043.6 | +5.5 % | 1222.4 | 0.900 |
| b2_s4k | 861.3 | 826.3 | +4.2 % | 1012.5 | 0.851 |
| b2_s16k | 1041.9 | 986.3 | +5.6 % | 1195.7 | 0.871 |

Geomean 996.81 vs 951.43 (+4.77 %); aiter 1139.84; ratio 0.8745.
Quick-set subset: 968.15 / 1108.02, ratio 0.8738.

Decision: ACCEPT.

## Real-training transfer check

Bucket K1: synchronization change inside the kernel, no cache, arithmetic bitwise unchanged.
Applies to every forward call of this config.
