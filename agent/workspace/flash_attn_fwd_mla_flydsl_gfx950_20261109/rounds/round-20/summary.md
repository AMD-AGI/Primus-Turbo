# Round 20 — ACCEPTED

- Time: 2026-11-09 19:16 (UTC+8). Base: round 18 (round 19 rolled back). Machine:
  smci355-ccs-aus-n02-25, HIP device 0 (amd-smi GPU 3).
- Hypothesis: with BLOCK_M 256 the diagonal band spans four KV tiles (max_m4 .. max_m1), and
  wave w (rows 32w .. 32w + 31 of the block) finds the 32-col half starting at block column c dead
  when c > 32w + 31. That gives 28 dead halves out of 64 per diagonal block. The dense causal drain
  skipped only the halves of max_m2 HI and max_m1 (18 for QK, 18 for P*V). Its comment, "max_m2 LO
  is always live", holds for BLOCK_M 128 only. The other halves ran their MFMAs and V LDS reads on
  P = 0. Skipping them removes MFMA work from waves 0-3, which share a SIMD with live waves.
  Expected: output bitwise unchanged; gain largest at S4K and shrinking with S, since the diagonal
  is a fixed amount of work per q block.
- Change (`flash_attn_fwd_mla.py`, dense causal drain only):
  - `_diag_live(k)` derives at trace time which halves of the k-th last tile any wave can find dead.
    A half is dead-able when its start column `BLOCK_M - k * BLOCK_N` (+ 32 for HI) is at least
    `ROWS_PER_WAVE`. Dead-able halves get `ctx.chunk_live`'s wave-uniform predicate; the others get
    None.
  - `_qk_diag` / `_pv_diag` dispatch to the existing `_qk(live=...)`, `_pv_hi_live` and
    `_pv_both_live` helpers. They cover QK of max_m3 (C1), max_m2 (C5) and max_m1 (C9), and P*V of
    max_m4 (C3), max_m3 (C7), max_m2 (C11) and max_m1 (C13).
  - Newly skipped per diagonal block: QK on max_m3 LO / HI and max_m2 LO (9 halves); P*V on those
    plus max_m4 HI (10 halves). With BLOCK_M 128 every predicate matches round 18's: `(None, None)`
    for max_m3, `(None, hi)` for max_m2, and `(lo, hi)` for max_m1.
  - `CODEGEN_REV` 9 -> 10. `attn_helper_mla.py` is otherwise unchanged, and so are `ops.py` and
    `__init__.py`. Diff: `artifacts/diff_vs_r18.txt`.

## Correctness

- O / LSE bitwise equal to round 18 (5 / 5 calls each at B1 S4K, B1 S16K, B2 S4K, B1 S64K;
  `artifacts/bitwise_vs_r18.log`).
- Full set 7 / 7 PASS in both full runs (O rel vs fp32 0.0016-0.0020, LSE max 0.0046-0.0051, the
  same as round 18). Quick set PASS in all 6 rotation runs.

## Result (full set with aiter, same session; order r18, r20 then r20, r18)

`artifacts/ab_full_{r18,r20}_{1,2}.csv`, table in `artifacts/ab_full_table.txt`. TFLOPS are 2-run
means.

| shape | round 18 | round 20 | r20 vs r18 | aiter (r20 runs) | r20 / aiter |
|---|---|---|---|---|---|
| b1_s4k | 1081.1 | 1111.1 | +2.78 % | 1070.7 | 1.0377 |
| b1_s8k | 1232.7 | 1248.2 | +1.26 % | 1205.2 | 1.0357 |
| b1_s16k | 1316.3 | 1333.7 | +1.32 % | 1268.9 | 1.0510 |
| b1_s32k | 1366.6 | 1370.0 | +0.25 % | 1297.4 | 1.0560 |
| b1_s64k | 1402.7 | 1406.0 | +0.24 % | 1308.1 | 1.0748 |
| b2_s4k | 1077.3 | 1097.1 | +1.84 % | 1070.3 | 1.0250 |
| b2_s16k | 1306.2 | 1312.0 | +0.45 % | 1266.9 | 1.0357 |

Geomean: r18 1248.77 / 1248.45 (mean 1248.61), r20 1261.52 / 1264.62 (mean 1263.08, +1.16 %).
Ratio vs aiter: r18 1.0362, r20 1.0450.

Quick set, 3 rotations, no aiter (`artifacts/ab_quick.txt`): r18 1217.62, r20 1235.21 (+1.44 %).
r20 is above r18 in every run: b1_s4k 1086.9 -> 1113.9, b1_s16k 1325.6 -> 1337.9, b1_s64k 1412.0 ->
1415.8, b2_s4k 1080.4 -> 1103.3.

Decision: ACCEPT. Every shape is up in both suites; geomean +1.16 % (full) and +1.44 % (quick).

## Attribution

- Sustained B1 S4K (`artifacts/clock_power_raw.txt`, 14 s each, order r18, r20, r18, r20):

  | | ms / call | clock (MHz) | socket W | PPT active |
  |---|---|---|---|---|
  | round 18 | 0.6336 / 0.6355 | 2254 | 1366 | 0.62-0.64 |
  | round 20 | 0.6195 / 0.6196 (-2.4 %) | 2235 | 1370 | 0.65 |

  Clock x time falls 3.2 %, so the gain is fewer cycles at the same power, not a higher clock.
- ISA (`profiles/r20_isa/r20` vs `profiles/r19_isa/r18`): VGPR spills 11 -> 10; `s_cbranch` 24 -> 34
  (the 10 new wave-uniform skips); whole-kernel MFMA, `ds_read` and `v_exp` counts unchanged
  (504 / 336 / 197). The main loop has no source change, yet its schedule moved: 791 -> 796
  instructions with 85 lines of opcode order differing (`artifacts/loop_r18.s`,
  `artifacts/loop_r20.s`).
- Reading of the per-shape pattern: S32K / S64K, where the drain is the smallest share of time, gain
  only +0.25 %. That bounds the effect of the main-loop schedule change to about that much. The
  rest of the gain at S <= 16K tracks the share of the drain.

Hypothesis, not measured: the drain gains more than its MFMA count suggests because it sits on
each CTA's critical path. Its stages are separated by CTA-wide barriers, so time saved in waves 0-3
gives their SIMD's MFMA pipe to the partner waves 4-7, which carry the full diagonal work.

## Real-training transfer check
- Bucket: K1 (kernel-internal: skipped MFMA / LDS work on causally dead diagonal halves)
- Cache key (if any): no cache
- id(...) audit: n/a
- 4-step pen-and-paper trace hit rate: n/a
- Workload distribution robustness (GroupGemm / MoE only): n/a
- Benchmark gain on this round: +1.16 % forward geomean (full set, same-session rotation), +1.44 %
  quick set
- Estimated real-training gain: the per-shape numbers above for the same (B, S). DeepSeek-V3-style
  pretraining at seq 4096 maps to b1_s4k / b2_s4k: +2.8 % / +1.8 % on the forward kernel. The skip
  depends only on causal masking and BLOCK_M, which a training step has too. The forward's share of
  a training step was not measured here.
- Decision: ACCEPT-as-real

## Next direction

- Pre-round probes (`logs/optimize.md`, "Pre-round-20 probe") rank the remaining levers. Halving the
  LDS -> VGPR bytes per FLOP bounds at +9.1 % (B1 S16K) / +9.0 % (B1 S4K) at the same clock. That
  needs 64 q rows per wave (one wave per SIMD at BLOCK_M 256), a restructure of the dual-wave
  ping-pong. The rowsum MFMAs bound at about 0 %, so moving the row sum to VALU is dropped.
- Remaining dead diagonal work: QK of max_m4 HI for wave 0 (main loop), and exp / softmax VALU on
  all 28 dead halves.
