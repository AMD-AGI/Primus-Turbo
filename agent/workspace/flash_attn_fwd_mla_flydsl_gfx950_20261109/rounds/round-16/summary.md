# Round 16 — ACCEPTED

- Time: 2026-11-09 14:55 (UTC+8). Base: round 15. Machine: smci355-ccs-aus-n02-25, HIP device 0.
- Hypothesis: the pre-round probe (`logs/optimize.md`, "Pre-round-16 probe") measured
  v_mfma_f32_16x16x32_bf16 sustaining ~11 % higher clock than v_mfma_f32_32x32x16_bf16 for the same
  FLOPs. The forward runs at the ~1.4 kW socket cap at S >= 16K (round 12), so moving the QK MFMAs
  (60 % of the MFMA FLOPs at d_qk 192 / d_v 128) to the 16x16x32 atom should arrive as clock.
- Change (single variable: the QK MFMA variant, plus the operand / result layout it needs):
  - Trait `QK_MFMA16`, auto-on for head_dim 192 without a sliding window (the campaign's causal
    shapes). Each 32-kv half is four 16x16 tiles (kv tile x q tile), 6 k-steps of 32.
  - Result layout: the raw 16x16 accumulators are converted to the 32x32x16 score layout with 8
    `v_permlane16_swap` per half, in `ctx.qk_scores` at each consumer (next cluster), never inside
    the QK region. The QK seed is built with the same swap (`_seed_vec`).
  - K operand: kv row order [0, 8, 4, 12] + r % 4 within a 16-row tile; d chunks of k-steps 0-3 are
    paired across d blocks 0 and 2 (d = 16 s + 128 (c % 2) + 8 (c // 2)); the middle block
    (d 64..127) is stored by the K DMA with 16-byte chunk m at m ^ 4 for the rows lane row groups 0
    and 3 read (`_k_mid_swizzled`), and read at d = 64 + 16 (s - 4) + 32 (c % 2) + 8 (c // 2).
    Q is loaded in the same d order.
  - `CODEGEN_REV` 6 -> 7. Snapshot: `kernel_snapshot/`. The swp8 kernel is untouched (trait off).

## Correctness

Full set 7/7 PASS (causal, both full runs): O rel vs fp32 0.0016-0.0020, LSE max 0.0046-0.0051,
same as round 15 to 4 digits. Quick set PASS on every rotation run.

## Result (full set, same session, r15 / r16 alternated twice; `artifacts/ab_full_*.csv`)

Back-to-back single runs drifted ~1.5 % on an unchanged package this session, so every comparison
in this round alternates the packages in one session (`profiles/ab_rotate.sh`).

| shape | round-16 (2-run mean) | round-15 same session | delta | aiter (r16 runs) |
|---|---|---|---|---|
| b1_s4k | 1079.3 | 1080.8 | -0.14 % | 1058.5 |
| b1_s8k | 1187.8 | 1188.1 | -0.02 % | 1202.5 |
| b1_s16k | 1284.3 | 1278.4 | +0.46 % | 1268.2 |
| b1_s32k | 1320.3 | 1307.9 | +0.95 % | 1292.7 |
| b1_s64k | 1353.7 | 1326.0 | +2.08 % | 1300.7 |
| b2_s4k | 1063.3 | 1063.6 | -0.03 % | 1055.3 |
| b2_s16k | 1264.7 | 1245.5 | +1.55 % | 1265.3 |

Geomean 1219.22 / 1215.08 (mean 1217.15) vs round 15 1210.15 / 1207.46 (mean 1208.81): +0.690 %.
Ratio vs aiter 1.0115 / 1.0138 (round 15 in the same session: 1.0015 / 1.0017). Against round 15's
recorded 1212.71 (earlier session): +0.366 %. Quick set, 3 rotations: 1192.0 vs 1183.7 (+0.70 %).
`benchmark_full.csv` / `full.log` are the first round-16 run.

Decision: ACCEPT (aggregate up in both rotations; worst shape -0.14 %).

## Attribution (B1 S16K)

| | round 15 | round 16 |
|---|---|---|
| sustained GFX clock (MHz, 2 x 14 s) | 1927-1933 | 2025-2040 |
| socket power (W) | 1371-1386 | 1349-1367 |
| ms / call (sustained) | 8.5845 / 8.6278 | 8.5392 / 8.5593 |
| GRBM_GUI_ACTIVE (rocprofv3) | 1.288e8 | 1.325e8 (+2.9 %) |
| SQ_LDS_BANK_CONFLICT | 0 | 0 |
| SQ_INSTS_LDS | 4.741e8 | 4.741e8 |

Measured: the switch costs cycles and wins them back as clock at lower power. Clock x time gives
+4.5 % cycles per call (GRBM: +2.9 %) against +5 % clock. The main-loop instruction diff against
round 15 is only the intended one per 2 tiles: +96 16x16x32 MFMA, -48 32x32x16 MFMA, +34
v_permlane16_swap (SQ_INSTS_VALU counts MFMAs, hence its +35 %). VGPR spills 9, all outside the main
loop (round 15: 0); SQ_VALU_MFMA_BUSY_CYCLES 1.136e10 for both round 15 and the natural-order
round-16 build (not re-collected for the final build, whose MFMA count is the same).

Steps inside the round (B1 S16K unless noted):

| K read scheme | spills | bank conflicts | GRBM | quick geomean (rotated vs r15) |
|---|---|---|---|---|
| swap inside the QK region | 51 | - | - | ~690 (single run) |
| swap at consumers, natural K order | 9 | 8.139e8 | 1.423e8 | - |
| + paired d chunks (blocks 0 / 2) | 9 | 2.712e8 | 1.356e8 | 1183.9 vs 1183.7 |
| + middle-block DMA swizzle (final) | 9 | 0 | 1.325e8 | 1192.0 vs 1183.7 |

Rejected inside the round: splitting the middle-block reads into ds_read_b64 (LLVM merged them into
ds_read2_b64, which conflicts like a 16-byte read: conflicts 1.626e9, +25 spills); QK sched hints
rescaled to 16x16 MFMA time (1180.56 vs 1183.87, 3 rotations).

The gain grows with S (+2.08 % at B1 S64K, about 0 at S <= 8K). Hypothesis, not measured: short
launches spend more of their time in the prologue / epilogue, where the extra swaps (C2 / C6 / C10)
are not hidden and the clock gain does not cover them.

## Real-training transfer check
- Bucket: K1 (kernel-internal: MFMA variant, register layout conversion, LDS swizzle)
- Cache key (if any): no cache
- id(...) audit: n/a
- 4-step pen-and-paper trace hit rate: n/a
- Workload distribution robustness (GroupGemm / MoE only): n/a
- Benchmark gain on this round: +0.690 % forward geomean (same-session rotation)
- Estimated real-training gain: the per-shape numbers above for the same (B, S): +2.1 % at B1 S64K,
  +1.6 % at B2 S16K, ~0 at S <= 8K. The gain is clock under the power cap; in a training step the
  forward shares that cap with the surrounding kernels, so the clock it gets there was not measured
  (sustained-loop numbers only). Calls are 8.5 ms (B1 S16K) to 130 ms (B1 S64K) long.
- Decision: ACCEPT-as-real

## Next direction

- Conversion cost: a wrong-result variant without the 16 swaps per tile measured 1217.7 vs 1184.5
  quick geomean (+2.8 %, `artifacts/ab_quick_upper_bounds.txt`). PV on 16x16x32 consuming the raw
  QK layout (P and the row reductions in the 16x16 layout) would drop the swaps and move the other
  40 % of the MFMA FLOPs to the cooler atom. Large change.
- The same file's "middle block reads conflict-free" variant (+2.7 %) overstated the gain (the real
  swizzle gave +0.7 %): it re-read the paired steps' addresses and the compiler merged those
  reads (ds_read_b128 144 -> 108 in its ISA).
- Still below aiter: b1_s8k (0.988) and b2_s16k (0.999).
