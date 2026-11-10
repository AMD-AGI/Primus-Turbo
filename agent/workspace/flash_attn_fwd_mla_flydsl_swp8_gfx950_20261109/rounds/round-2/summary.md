# Round 2 — ACCEPTED (trade-off): i32 while-loop counter (scalar loop control)

- Time: 2026-11-09 10:35 (UTC+8)
- Hypothesis (from round-1 rollback): the main loop's exit test and the K-tile clamp compare
  index (i64) values; gfx9 has no 64-bit s_cmp_lt, so each costs VALU (`v_mov_b64` +
  `v_cmp_lt_u64` / `v_cmp_lt_i64`) inside mem stages, where it can stall behind the partner
  wave (211 units per visit in round 1, ~20-40 in round 0). An i32 counter makes both SALU.
- Single change: the ping/pong `scf.for` over index becomes a `while j < t_end - 1` loop on an
  i32 counter (FlyDSL `scf_range` always converts bounds to index, so an i32 `range` is not
  enough); the K(t+2) clamp is computed in i32. Arithmetic unchanged. SWP8_REV 4.
- ISA: loop head `s_cmp_lt_u32` + `s_cbranch_scc0`; no 64-bit compare left in the loop
  (remaining ones are in the prologue / epilogue). 256 VGPR, spill 33 -> 31, 176 MFMA,
  37 barriers.

## Correctness

- `swp8_check.py`: bitwise equal to round 7 on all 5 check shapes, 3/3 reps.
- All 7 PASS vs fp32: O rel 0.0016–0.0020, LSE max abs 0.0046–0.0051.

## Performance (full set, GPU 0)

| shape | round 2 | round 0 | delta | aiter | ratio |
|---|---|---|---|---|---|
| b1_s4k | 871.3 | 846.6 | +2.92 % | 1023.2 | 0.852 |
| b1_s8k | 980.7 | 963.7 | +1.76 % | 1145.3 | 0.856 |
| b1_s16k | 1028.1 | 1013.4 | +1.45 % | 1189.8 | 0.864 |
| b1_s32k | 1053.5 | 1047.8 | +0.54 % | 1215.1 | 0.867 |
| b1_s64k | 1053.0 | 1062.2 | -0.87 % | 1224.7 | 0.860 |
| b2_s4k | 863.1 | 846.5 | +1.96 % | 1014.0 | 0.851 |
| b2_s16k | 1020.5 | 1005.2 | +1.52 % | 1195.8 | 0.853 |

Geomean 978.41 vs 965.66 (+1.32 %); aiter 1140.88; ratio 0.8576.

Decision: ACCEPT WITH NOTE — aggregate up, b1_s64k -0.87 % (< 5 %).

## Attribution

Loop-control VALU left the trailing group's stage 0 and the K clamp left stage 4. The gain is
largest at short sequences; the b1_s64k dip is not explained yet (not profiled this round).

## Real-training transfer check

- Bucket: K1 (kernel-internal control flow); no cache; no id() keys.
- 4-step trace hit rate: n/a (no cache).
- Benchmark gain: +1.32 %; estimated real-training gain: +1.32 % of this kernel's forward time
  (same kernel, same shapes; no benchmark-loop dependence).
- Decision: ACCEPT-as-real.

## Next direction

Round 3: re-try the zero QK seed (round 1) on top of this, with the S shift kept packed
(v_pk_add_f32) so stage 5 does not gain 32 scalar adds.
