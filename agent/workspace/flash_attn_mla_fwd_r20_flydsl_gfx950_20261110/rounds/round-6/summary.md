# Round 6 — ACCEPTED (trade-off) — 2026-11-10 20:05 (UTC+8)

## Hypothesis
ATT bubble map on round 4 (`wip/att_bubbles.py`, `profiles/att_r3/stats_r4.csv`): the loop loses
1214 cycles per iteration to MFMA pipe bubbles (399 issue overflow + 815 stall) while under-full slots
hold 1355 spare issue cycles. The largest issue overflows are the P*V-site V reads: each substep's 8
`ds_read_b64_tr_b16` sit in one MFMA shadow (50-80 cycles over, 5 such slots per iteration). Reading
one d chunk (2 reads) per slot, in slots that carry no exp2, should remove most of that overflow.

## Single change
`_pv_spec_site` slot plan: the next substep's V is read one d chunk per slot (slots 3, 5, 7, 9 of
each substep), each V slot also carries one max-tree piece (two max3), exp2 pieces fill the other
slots first, the remaining max-tree pieces go last. Every non-empty slot is fenced by
`sched_barrier(0)` on both sides, because otherwise the scheduler hoists the next slot's exp2 above
its own asm MFMA into the V slot's shadow (first version, rev 1039: same timing). `CODEGEN_REV` 1040.
ISA: VGPR 512, spill slots 4 -> 8 (scratch 36 B).

## Results (full set, 2 rotations, `artifacts/r6.csv`, base = round-4 snapshot, same session)

| shape | cand TFLOPS | cand ms | base (R4) TFLOPS | base ms | cand / base | O rel | LSE max | bitwise |
|---|---|---|---|---|---|---|---|---|
| dsv3 B2 S4K H128 | 1029.1 | 1.3358 | 1024.9 | 1.3413 | 1.0041 | 2.01e-03 | 4.75e-03 | yes |
| dsv2lite B8 S4K H16 | 1004.7 | 0.6842 | 1006.2 | 0.6831 | 0.9985 | 1.99e-03 | 3.17e-03 | yes |
| kimik2 B4 S4K H64 | 1025.5 | 1.3405 | 1019.8 | 1.348 | 1.0056 | 1.97e-03 | 4.01e-03 | yes |
| kimik3 B2 S7168 H96 | 1158.0 | 2.7264 | 1153.5 | 2.7371 | 1.0039 | 1.83e-03 | 4.73e-03 | yes |
| B1 S4K | 1016.5 | 0.6762 | 1012.5 | 0.6789 | 1.0040 | 1.95e-03 | 4.60e-03 | yes |
| B1 S8K | 1171.2 | 2.3472 | 1165.8 | 2.3582 | 1.0046 | 1.80e-03 | 4.75e-03 | yes |
| B1 S16K | 1254.4 | 8.7658 | 1259.7 | 8.7288 | 0.9958 | 1.75e-03 | 4.93e-03 | yes |
| B1 S32K | 1297.4 | 33.9 | 1306.6 | 33.6619 | 0.9930 | 1.61e-03 | 4.82e-03 | yes |
| B1 S64K | 1319.8 | 133.2928 | 1323.9 | 132.8804 | 0.9969 | 1.58e-03 | 5.14e-03 | yes |
| B2 S4K | 1028.6 | 1.3366 | 1025.1 | 1.3411 | 1.0034 | 2.01e-03 | 4.75e-03 | yes |
| B2 S16K | 1263.0 | 17.4117 | 1261.6 | 17.4313 | 1.0011 | 1.76e-03 | 4.82e-03 | yes |

Geomean cand 1136.33 vs base 1135.22 (+0.10 %), 11/11 PASS, bitwise identical. S <= 8K shapes +0.3
to +0.6 %; B1 S16K / S32K / S64K -0.4 / -0.7 / -0.3 %.

B1 S16K sustained (8 s): R6 1245.7 TFLOPS at 1944 MHz / 1274 W / 0.6408 TFLOPS/MHz vs R4 1249.8 at
1960 MHz / 1273 W / 0.6377: +0.5 % work per cycle, -0.8 % clock at the same power.

## Decision
ACCEPTED WITH NOTE (trade-off): geomean +0.10 % (noise level), long-S shapes -0.3 to -0.7 %. Kept
because the two-sided slot fences make the P*V-site placement exact, which the next rounds build on.

## Attribution
- ATT dsv3: loop 6705 vs 6734 cycles per iteration; both P*V sites -103 cycles, both QK sites +74
  (more `v_accvgpr_read` / waits from the extra spill slots). Bubble 1165 vs 1214.
- V slots still overflow ~4-8 cycles (2 reads + 2 max3 = ~20 cycles vs 16), because the site has
  67 work items for 64 slots, so V slots must also carry max-tree work.
- At the B1 S16K power cap, fewer cycles at slightly higher energy per FLOP lose the clock back.

## Real-training transfer check
- Bucket: K1 (instruction schedule). Cache key: no cache. id(...) audit: n/a.
- 4-step trace hit rate: n/a. Workload distribution: n/a (dense causal attention).
- Benchmark gain: +0.10 %. Estimated real-training gain: +0.10 % (bitwise identical, stateless; the
  same shape mix). Decision: ACCEPT-as-real.

## Next direction
Give V slots room: move part of the speculative exp2 (lo half) from the P*V site to the QK's
`finish_p`, where every k-step slot ends with ~4 empty MFMA shadows, so V slots carry no max-tree work.
Then the stall bubbles: the K-head LDS wait after the P*V site (~48 cycles), the cluster-sync vmcnt
wait (~40), and the first DMA's issue stall (~15).
