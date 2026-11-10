# Round 19 — ROLLED BACK

- Time: 2026-11-09 18:28-18:33 (UTC+8). Base: round 18. Machine: smci355-ccs-aus-n02-25,
  HIP device 0 (amd-smi GPU 3).
- Hypothesis: round 18's S4K / S8K regression vs its own (16, 8) build (~1.3 %) came from the
  loop that threaded `_lo` through progressive selects, which re-scheduled the main loop even
  when the CTA order was identical. Rewriting the same (32, 16, 8) order as independent section
  maps plus a final select cascade (round 17's two-map form) should recover that code-generation
  cost without changing dispatch order.
- Change (single variable: how the hierarchical map is written; order unchanged):
  - Replaced the `range_constexpr` loop over `XCD_Q_GROUPS` with three `_xcd_sec(lo, g)` calls
    and nested `ArithValue(...).select` (plus the rank-major tail), bounds `_lo0.._lo3`
    precomputed from `_nq_head`.
  - `XCD_Q_GROUPS = (32, 16, 8)` and `XCD_TAIL_Q = 8` kept. Host replica identical to round 18
    for nq 1-400 (`profiles/xcd_map_r19_check.py`). `CODEGEN_REV` 9 -> 10.
  - Snapshot of the attempt: `kernel_snapshot/`. Working tree restored to round 18.

## Correctness

- Mapping identical to round 18 (0 mismatches, bijective).
- O / LSE bitwise equal to round 18 (5 / 5 at B1 S4K, B1 S16K, B2 S4K, B1 S64K;
  `artifacts/bitwise_vs_r18.log`).
- Full set 7 / 7 PASS in every run (O rel 0.0016-0.0020, LSE max 0.0046-0.0051); quick PASS
  in all 9 rotation runs.

## Result (full set with aiter, same session; order r18, r19, g16 then g16, r19, r18)

`artifacts/ab_full_{r18,r19,g16}_{1,2}.csv`. g16 = round 18's (16, 8) package (codegen reference
at S4K / S8K). TFLOPS are 2-run means.

| shape | round 18 | r19 | r19 vs r18 | g16 | g16 vs r18 |
|---|---|---|---|---|---|
| b1_s4k | 1086.8 | 1082.2 | -0.43 % | 1098.7 | +1.10 % |
| b1_s8k | 1235.2 | 1231.5 | -0.30 % | 1238.2 | +0.24 % |
| b1_s16k | 1322.7 | 1323.1 | +0.03 % | 1305.3 | -1.31 % |
| b1_s32k | 1368.1 | 1366.3 | -0.13 % | 1345.3 | -1.67 % |
| b1_s64k | 1404.3 | 1401.5 | -0.21 % | 1382.9 | -1.53 % |
| b2_s4k | 1074.7 | 1072.5 | -0.21 % | 1084.9 | +0.95 % |
| b2_s16k | 1302.5 | 1299.8 | -0.21 % | 1298.5 | -0.31 % |

Geomean: r18 1253.22 / 1247.30 (mean 1250.26), r19 1249.21 / 1246.12 (mean 1247.67, -0.21 %),
g16 1246.12 / 1245.22 (mean 1245.67). r19 below r18 in both rotations. Ratio vs aiter (2-run
mean): r18 1.0364, r19 1.0353.

Quick set, 3 rotations, no aiter (`artifacts/ab_quick.txt`):

| shape | round 18 | r19 | g16 |
|---|---|---|---|
| b1_s4k | 1085.2 | 1084.4 (-0.08 %) | 1095.4 (+0.93 %) |
| b1_s16k | 1326.8 | 1326.6 (-0.02 %) | 1305.2 (-1.63 %) |
| b1_s64k | 1407.8 | 1406.8 (-0.07 %) | 1387.0 (-1.48 %) |
| b2_s4k | 1078.5 | 1075.3 (-0.29 %) | 1086.0 (+0.70 %) |
| geomean | 1216.0 | 1214.6 (-0.12 %) | 1211.4 (-0.38 %) |

Decision: ROLLBACK (aggregate down in both full rotations; S4K / S8K still ~1 % behind g16).

## Rollback Analysis

- Change made: rewrite hierarchical XCD map as independent section maps + select cascade.
- Expected effect: same CTA order, main-loop schedule closer to g16 / round 17, recover ~1.3 %
  at S4K / S8K.
- Actual effect: full geomean -0.21 %; b1_s4k -0.43 % (g16 still +1.10 % over r18 there).
- Root cause (measured): the rewrite barely changed the ISA. VGPR spill stayed 11; main-loop
  instruction count 791 in both, opcode sequence identical (0 diffs); whole kernel 3696 -> 3699
  (+2 `s_and_b64`, +1 `s_mov_b32` in the prologue). So the loop vs independent-maps form was not
  what separated g16 from the (32, 16, 8) builds. The ~1 % S4K gap tracks having three group
  levels (and their bounds / selects) in the setup, not how those levels are written.
- Lesson learned: when two builds share CTA order at a shape, diff the main-loop opcode sequence
  before rewriting setup for "codegen". If the hot loop already matches, a setup rewrite will not
  close a gap that lives in a different package's constants / level count.

## Real-training transfer check
- Bucket: K1 (kernel-internal: CTA-setup arithmetic form)
- Cache key (if any): no cache
- id(...) audit: n/a
- 4-step pen-and-paper trace hit rate: n/a
- Workload distribution robustness (GroupGemm / MoE only): n/a
- Benchmark gain on this round: -0.21 % forward geomean (full set)
- Estimated real-training gain: n/a (rolled back)
- Decision: REJECT (rollback; no transfer)

## Next direction

- The S4K / S8K code-generation cost of (32, 16, 8) vs (16, 8) is real and stable, but it is not
  the loop form. Try a two-level (32, 8) map (drop the middle 16): same G32 at long S, simpler
  setup like round 17, measure whether S4K recovers while keeping most of the long-S gain.
- Or leave the map and take a different lever from round 17's backlog (PV sched hints; rebase
  row-max sharing) / the kv-head0 locality ceiling (~3.5 % at B1 S16K).
