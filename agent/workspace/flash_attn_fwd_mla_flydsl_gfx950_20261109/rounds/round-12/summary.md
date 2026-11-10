# Round 12 — ACCEPTED (trade-off)

- Time: 2026-11-09 11:13 (UTC+8). Base: round 7 (1058.67).
- Hypothesis (from the post-round-7 PMC, `logs/optimize.md` "Resume after swp8 pause"): round 7
  and aiter issue the same number of L2 requests at B1 S16K, but round 7 misses 1.77x as often
  (hit rate 0.387 vs 0.653), and both kernels sit at the ~1.4 kW socket cap. The grid is
  (head, q_block desc, batch) with head fastest; workgroups go to the 8 XCDs round-robin in linear
  order, so each XCD holds 16 heads x 2 q-blocks at a time and every K/V tile has ~2 co-resident
  readers. Hkv = Hq for MLA, so the HipKittens GQA head relabel has nothing to share across heads;
  the sharers are the q-blocks of one head. Keeping XCD_Q_GROUP = 8 q-blocks of one head adjacent
  in each XCD's dispatch order should raise L2 hits, cut fabric / MALL energy and, under the cap,
  buy clock.
- Change (single variable): CTA -> (head, q rank) mapping in `attn_helper_mla.py`. With
  `lin = y * NH + x`, XCD `lin % 8` owns heads `h = 8 * h_local + lin % 8`; slot `lin // 8` walks
  groups of 8 q ranks, head-major inside a group (a short last group when NQ % 8 != 0). Causal
  still maps q rank r to q block NQ - 1 - r. Applied only when NH % 8 == 0 (else the round-7
  mapping). `CODEGEN_REV` 1 -> 2 so the warm cache recompiles. Snapshot: `kernel_snapshot/`.

## Correctness

Bitwise equal to round 7 (10 repeats x B1 S4K / B1 S16K / B2 S4K / B1 S64K,
`artifacts/bitwise_vs_r7.log`): each CTA computes the same (head, q-block) tile as before.
Full set: all 7 PASS (O rel 0.0016-0.0020, LSE max 0.0046-0.0051).

## Result (`artifacts/benchmark_full.csv`)

| shape | round-12 | round-7 | delta | aiter | ratio |
|---|---|---|---|---|---|
| b1_s4k | 965.3 | 1004.8 | -3.93 % | 1020.0 | 0.946 |
| b1_s8k | 1088.1 | 1056.4 | +3.00 % | 1140.7 | 0.954 |
| b1_s16k | 1162.5 | 1088.5 | +6.80 % | 1196.3 | 0.972 |
| b1_s32k | 1195.2 | 1095.2 | +9.13 % | 1218.7 | 0.981 |
| b1_s64k | 1216.5 | 1110.1 | +9.58 % | 1227.5 | 0.991 |
| b2_s4k | 969.7 | 987.9 | -1.84 % | 1013.1 | 0.957 |
| b2_s16k | 1165.1 | 1074.1 | +8.47 % | 1194.8 | 0.975 |

Geomean 1104.55 vs 1058.67 (+4.334 %); aiter 1141.17; ratio 0.9679. Quick-set geomean (from the
full-run values of b1_s4k / b1_s16k / b1_s64k / b2_s4k) 1072.63, ratio 0.9664.

Decision: ACCEPT WITH NOTE. Aggregate up; b1_s4k regressed 3.93 % and b2_s4k 1.84 % (< 5 %).

## Attribution (measured, B1, `profiles/pmc_r12`, sustained 12 s loop, amd-smi GPU 3)

| | r7 S16K | r12 S16K | r7 S4K | r12 S4K |
|---|---|---|---|---|
| TCC hit rate | 0.372 | 0.802 | 0.367 | 0.746 |
| TCC_MISS_sum / dispatch | 2.302e8 | 7.31e7 | 1.82e7 | 7.42e6 |
| TCC_REQ_sum / dispatch | 3.664e8 | 3.685e8 | 2.873e7 | 2.924e7 |
| GRBM_GUI_ACTIVE / dispatch | 1.223e8 | 1.291e8 | 9.515e6 | 1.194e7 |
| sustained TFLOPS | 1086.6 | 1171.2 | 999.8 | 958.0 |
| GFX clock (MHz) | 1578-1594 | 1647-1673 | 1639-1643 | 1919-1935 |
| socket power (W) | 1398-1399 | 1400-1401 | 1399-1401 | 1399-1400 |

- L2 misses drop 3.15x at S16K and 2.45x at S4K with the same request count.
- S16K: the kernel takes slightly more cycles (+5.6 % GUI_ACTIVE under the profiler) but runs at a
  ~4-5 % higher clock at the same 1.4 kW, and wall time drops. So the gain is energy, not cycles:
  fewer L2 misses spend less of the power budget off the CU.
- S4K: hit rate improves just as much, yet cycles grow 25 % and the clock rises 17 %, i.e. the GPU
  idles more. Hypothesis (not isolated): load-balance tail. A group spans 8 of the 16 q ranks at
  S4K, so the last ~32 slots of each XCD hold q ranks spread over 8 lengths instead of 2, and the
  longest of them (q block 7, half the longest tile count) is dispatched near the end. At S16K+
  a group spans <= 1/8 of the q range and the tail is small.

## Real-training transfer check
- Bucket: K1 (kernel-internal workgroup mapping; no launch-config change, no wrapper cache)
- Cache key (if any): no cache
- id(...) audit: n/a
- 4-step pen-and-paper trace hit rate: n/a
- Workload distribution robustness (GroupGemm / MoE only): n/a
- Benchmark gain on this round: +4.334 % forward geomean
- Estimated real-training gain: same per shape as the benchmark (every call to `mla_fwd_sbhd`
  gets the mapping). Caveat: a 4K-sequence training step (DSV3 pre-training length) sees the
  -3.9 % (B1) / -1.8 % (B2) side of the trade-off; 8K and longer gain +3 % to +9.6 %.
- Decision: ACCEPT-as-real (with the 4K regression recorded)

## Next direction

Round 13: keep the grouped order for long q ranks but switch the last q ranks (the short tail
jobs) back to round-7 LPT order, so the final dispatches per XCD again span ~2 q lengths. Expected
to recover b1_s4k / b2_s4k while keeping most of the hit-rate gain (the tail ranks carry little
of the total work at S16K+).
