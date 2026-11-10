# flash_attn_fwd_mla (192/128) FlyDSL gfx950 — optimize log

Campaign: forward-only tuning of the DSV3 MLA FlyDSL prototype (`megaattention_overlap_bench/flydsl_mla/`).
Target: same-GPU forward TFLOPS geomean ratio vs Turbo AITER >= 1.0. GPU 0 pinned.

## Baseline (round-1) — 2026-11-09 08:49

- Config: block_m 256, waves_per_eu 2, stagger off, k_head 4, loop_masks fix.
- Full set (7 shapes) all PASS. Geomean proto 919.04 / aiter 1140.64 TFLOPS, ratio 0.8057.
- Per shape ratio 0.770 (b2_s4k) to 0.833 (b1_s64k).
- Quick set geomean proto 891.07 / aiter 1107.78, ratio 0.8044.
- Known resources from the prototype note (bm256 kh4): 256 VGPR cap, 105 spill VGPRs; the main
  loop has 0 scratch instructions, spills sit in the prologue (101), loop exit (36), epilogue (267).
- Historical tips: none for gfx950 attention / FlyDSL.

## Round 2 — 2026-11-09 09:02 — ROLLBACK

- Analysis: counters (B1 S16K) show MFMA busy cycles only 1.04x aiter but MFMA_BUSY/BUSY_CU 2.40
  vs 3.44 and more wave wait (0.345 vs 0.245); ATT (B1 S4K) puts 21.9 % of wave latency in
  s_barrier and 20.8 % in s_waitcnt. ISA: QK MFMAs each wait on a just-issued tail K read.
- Change: QK-cluster sched_group hints for K_STEPS_QK > 8 (one MFMA group per QK MFMA, tail
  DS reads paced 2*K_HEAD MFMAs ahead).
- Result: ISA unchanged in substance (spill 105, reads still just-in-time; tail reads rotate
  through 3 register quads -> post-RA false deps). Full geomean 913.58 vs 919.04 (-0.59 %),
  all PASS. Rolled back to round-1.
- Takeaway: register-bound at the 256-VGPR cap; free VGPRs first.

## Round 3 — 2026-11-09 09:06 — ACCEPT

- Change: ops.py default k_head 4 -> 2 (16 VGPRs of head K packs freed).
- Resources: spill 105 -> 90, scratch 424 -> 364 B.
- Result: all PASS; full geomean 947.53 vs 919.04 (+3.10 %), every shape up (+1.5 to +5.4 %,
  largest at 4K). Ratio vs aiter 0.8311. New best.

## Round 4 — 2026-11-09 09:10 — ACCEPT (marginal)

- Change: ops.py default k_head 2 -> 1 (minimum).
- Resources: spill 90 / scratch 364 B unchanged.
- Result: all PASS; full geomean 951.43 vs 947.53 (+0.41 %, 6 of 7 shapes up, worst -0.08 %).
  Ratio 0.8353. Near noise; accepted per rule.
- Reference read (aiter opus d192 kernel source, `profiles/aiter_ref/`, read-only): 8 stages per
  tile, GEMM0 split along KV into two super units with that half's K fully in registers before
  its MFMAs, two wave groups offset by one stage (ping-pong), head/tail causal merge, Q staged
  through LDS only in the prologue. Load-balance simulation: LPT order is already balanced for
  128 heads, so head/tail merge is predicted ~0 here.

## Round 5 — 2026-11-09 09:15 — ACCEPT

- Change: enable the existing fold_mem_barriers / fold_pv_barriers path for 8-wave single-head
  WGs (`NUM_WAVES > 4` instead of `Q_HEADS_PER_WG > 1`). One barrier per tile instead of four.
- Hazard argument written in the round summary; race check bitwise equal to round 4 (5 repeats x
  4 shapes), builds confirmed distinct by timing.
- Result: all PASS; full geomean 996.81 vs 951.43 (+4.77 %), every shape +3.6 to +5.6 %.
  Ratio vs aiter 0.8745. New best.

## Round 6 — 2026-11-09 09:22 — ROLLBACK

- Analysis (ATT on round 5, B1 S4K): hottest instruction is the barrier after epilogue C11
  (9.6 % of wave latency). The late waves (last causal tile live) rebuild Q from scratch before
  every QK MFMA: 116 scratch dword loads, each pair + `vmcnt(0)` + `v_perm`. ISA shows Q is
  converted per element (`v_cvt_pk_bf16_f32 vD, vF32, s0`) and packed with `v_perm`, so the
  spill holds 96 unpacked dwords.
- Change: reload + rescale Q from global at the loop exit (K_STEPS_QK > 8) for the epilogue QKs.
- Resources: spill 90 -> 141, scratch 364 -> 460 B; Q re-spilled inside the epilogue.
- Result: all PASS, bitwise equal to round 5; full geomean 913.70 vs 996.81 (-8.34 %), 4K shapes
  -15 to -18 %. Rolled back to round 5.
- Takeaway: the epilogue alone exceeds the VGPR budget; fix the unpacked Q representation first.

## Round 7 — 2026-11-09 09:25 — ACCEPT

- Change: `scale_all` truncates Q per MFMA pack (8 elements) instead of one 96-wide fptrunc.
  The 96-wide fptrunc scalarized (non-power-of-two width), leaving one bf16 per VGPR repacked
  with `v_perm` at every use and spilled unpacked.
- Resources: spill 90 -> 0, scratch 364 -> 0 B, `v_perm` 284 -> 0, instructions 4026 -> 3236.
- Result: all PASS, bitwise equal to round 5; full geomean 1058.67 vs 996.81 (+6.21 %), every
  shape up (+0.9 % at 64K to +14.7 % at B2 4K). Ratio vs aiter 0.9291. New best.

## Round 8 — 2026-11-09 09:29 — ROLLBACK

- Analysis (ATT round 7, B1 S16K): main loop ~93 %; MFMA 30.5 %, waitcnt 14.5 %, barrier 13.9 %
  (two loop barriers 6.0 / 6.4 %, drains before them 0.7 % each -> skew, not DMA), ds_read 12.3 %.
  QK MFMAs wait on just-in-time tail K reads; the 4-MFMA row-sum chain before QK stalls.
- Change: k_head 1 -> 2 (largest spill-free value; 3 -> spill 1, 4 -> spill 11).
- Result: all PASS; full geomean 1054.93 vs 1058.67 (-0.35 %), all shapes -0.1 to -0.9 %.
  Rolled back to round 7.

## Round 9 — 2026-11-09 09:32 — ROLLBACK (neutral)

- Change: drop the fence between cast_p_and_sum and the QK MFMAs in loop C1/C5 (row-sum chain
  may interleave with QK). Scheduler moved the chain to the end of QK.
- Result: all PASS, bitwise equal to round 7; full geomean 1058.40 vs 1058.67 (-0.03 %),
  per shape -0.7 to +0.6 %. Rolled back to round 7.

## Round 10 — 2026-11-09 09:37 — ROLLBACK

- Analysis: ATT launch order puts wave w and w + 4 of a WG on one SIMD; both are barrier-locked
  into the same cluster, so the uniform P*V `s_setprio(1)` ranks nothing at block_m 256.
- Change: loop C3/C7 `s_setprio(1)` only for the later-dispatched half (wave_id >= 4).
- Result: all PASS, bitwise equal to round 7; full geomean 1034.43 vs 1058.67 (-2.29 %), every
  shape -1.5 to -2.7 %. Rolled back to round 7.

## Round 11 — 2026-11-09 09:40 — ROLLBACK

- Change: loop C3/C7 `s_setprio(1)` only for the first-dispatched half (wave_id < 4).
- Result: all PASS, bitwise equal to round 7; full geomean 1044.56 vs 1058.67 (-1.33 %), every
  shape -1.1 to -1.6 %. Rolled back to round 7.
- Stall: rounds 8-11 gave no gain. Reference read of aiter opus d192: same tiling, but an
  8-stage-per-tile pipeline (barrier per stage, alternating memory / compute stages, GEMM0
  split along KV with each half's K fully in registers) and a one-stage wave-group offset.
  Closing the remaining gap needs that structure, not single-variable tuning.

### Stagnation review — 2026-11-09 09:41

- Benefit curve: accepted 919.04 (r1) -> 947.53 (r3) -> 951.43 (r4) -> 996.81 (r5) -> 1058.67 (r7).
  Rounds 8-11 (k_head 2, row-sum/QK fence, two priority splits) all rolled back.
- Profile of the best kernel (ATT round 7, B1 S16K): spill 0; main loop ~93 % of latency;
  MFMA 30.5 %, waitcnt 14.5 %, barrier 13.9 %, ds_read 12.3 %. The two loop barriers cost wave
  skew (drains before them 0.7 % each). Both waves of a SIMD (w, w + 4) are barrier-locked in
  the same cluster.
- Proven ineffective here: QK read-pacing hints (r2), more head K (r8), row-sum placement (r9),
  per-SIMD priority asymmetry (r10, r11), moving the Q definition (r6).
- Fundamentally different directions considered:
  1. aiter-style pipeline: 8 stages per tile with a barrier each, memory / compute stages
     alternating, GEMM0 split along KV with each half's K (48 VGPRs) read in the preceding
     memory stage, GEMM1 split along D_V, one-stage wave-group offset (extra barrier for one
     group in the prologue, the other in the epilogue). aiter does this with the same tiling
     and the same double buffers. Large: new loop, prologue, epilogue and causal drain.
  2. Larger KV tile (BLOCK_N 128) to halve barriers and softmax overhead per FLOP: needs 64
     score VGPRs, over budget at 8 waves (k_head 3 already spills).
  3. Head/tail causal merge: LPT simulation predicts ~0 at 128 heads.
- Only direction 1 has a credible path to parity, and it is a kernel rewrite rather than a
  single-variable round.

### Termination Check
- T1 performance_target (ratio >= 1.0): no (0.9291)
- T2 hardware efficiency: no (not claimed)
- T3 max_iterations reached: no (null)
- T4 user requested stop: yes (campaign scope set by the user: "直到追平 aiter 或停滞"; single-variable
  tuning has stalled for four rounds, the remaining direction is a rewrite and is handed back
  to the user)
-> Satisfied condition(s): T4

## Final Report

- Baseline aggregate score (round 1): 919.04 TFLOPS forward geomean, ratio vs aiter 0.8057.
- Final best aggregate score (round 7): 1058.67 TFLOPS (+15.19 %), ratio vs aiter 0.9291
  (aiter 1139.49 in the same run).
- Total iterations: 11 (baseline 1, accepted 4: r3 r4 r5 r7; rollback 6: r2 r6 r8 r9 r10 r11).
- Per shape, baseline -> final (ratio vs aiter): b1_s4k 800.8 -> 1004.8 (+25.5 %, 0.982);
  b1_s8k 911.4 -> 1056.4 (+15.9 %, 0.929); b1_s16k 987.3 -> 1088.5 (+10.3 %, 0.916);
  b1_s32k 1002.6 -> 1095.2 (+9.2 %, 0.902); b1_s64k 1020.7 -> 1110.1 (+8.8 %, 0.906);
  b2_s4k 780.9 -> 987.9 (+26.5 %, 0.975); b2_s16k 961.5 -> 1074.1 (+11.7 %, 0.898).
- Correctness of the final kernel: all 7 shapes PASS vs fp32 rows (O rel <= 0.0020, LSE max
  <= 0.0051); output bitwise equal to round 5 (r7 summary); D 128/128 copy still bitwise equal to
  the installed Turbo FlyDSL forward (12/12, `rounds/round-7/artifacts/d128_regress.log`).
- Key effective optimizations:
  1. r7: truncate scaled Q per MFMA pack instead of one 96-wide fptrunc (scalarized at the
     non-power-of-two width): spill 90 -> 0, +6.21 %.
  2. r5: fold the memory / P*V cluster barriers for 8-wave WGs (one barrier per tile): +4.77 %.
  3. r3 + r4: k_head 4 -> 1 under register pressure: +3.10 % and +0.41 %.
- Verified ineffective directions: r2 QK read-pacing hints (-0.59 %); r6 reload Q from global for
  the epilogue (-8.34 %); r8 k_head 2 once spill-free (-0.35 %); r9 row-sum chain behind QK
  (-0.03 %); r10 / r11 per-SIMD P*V priority split (-2.29 % / -1.33 %).
- Real-training applicability audit:

  | bucket | rounds | gain |
  |---|---|---|
  | S_real (K1-K4) | r3, r4, r5, r7 | +15.19 % forward geomean |
  | W_real (W1) | none | 0 |
  | R_real | none | 0 |

  Headline benchmark gain +15.19 % = S_real + W_real; inflation gap 0. All gains are inside the
  kernel or its launch default and reach any caller of `mla_fwd_sbhd` unchanged; they are largest
  at 4K-8K sequence length and ~9 % at 32K-64K.
- If continuing, recommended next three steps:
  1. Port the aiter d192 pipeline structure (8 stages per tile, KV-split GEMM0 super units with K
     fully in registers, D_V-split GEMM1, one-stage wave-group offset) into a new FlyDSL loop;
     keep the round-7 Q packing. This targets the 0.90 ratio at 16K-64K.
  2. Before that, prototype only the one-stage offset on a 4-stage-per-tile variant to measure
     how much of the barrier skew it recovers, using `bitwise_check.py` as the race detector.
  3. Re-profile after (1) with ATT; check that the scheduler keeps the KV-split K packs resident
     (spill 0) before tuning per-stage MFMA / VALU pairs.
- Data: `rounds/round-{1..11}/summary.md`, `logs/performance_trend.md`, profiles `att_r5`, `att_r7`,
  `isa_r5`..`isa_r10`.

## Resume after swp8 pause (2026-11-09 11:10 UTC+8): round-7 vs aiter, HipKittens ideas

The swp8 aiter-pipeline port peaked at 992.33 (0.8732), below round 7; the campaign resumes from
round 7 (working files byte-identical to `rounds/round-7/kernel_snapshot`). Measured on HIP device 0
(amd-smi GPU 3), B1 S16K H128 causal; PMC from `../flash_attn_fwd_mla_flydsl_swp8_gfx950_20261109/profiles/pmc2`.

| metric | round 7 | aiter | r7 / aiter |
|---|---|---|---|
| sustained TFLOPS (12 s loop) | 1089.1 | 1189.2 | 0.916 |
| GFX clock under load (MHz) | 1542-1589 | 1374-1378 | ~1.13 |
| socket power (W) | 1398-1400 | 1383-1387 | ~1.01 |
| GRBM_GUI_ACTIVE / dispatch | 1.216e8 | 1.075e8 | 1.131 |
| SQ_INSTS_MFMA | 3.719e8 | 3.408e8 | 1.091 |
| SQ_VALU_MFMA_BUSY_CYCLES | 1.136e10 | 1.091e10 | 1.041 |
| SQ_INSTS_VALU | 9.934e8 | 1.564e9 | 0.635 |
| SQ_INSTS_LDS | 4.741e8 | 4.779e8 | 0.992 |
| SQ_WAIT_INST_LDS | 4.352e8 | 9.073e8 | 0.480 |
| TCC_REQ_sum | 3.664e8 | 3.667e8 | 0.999 |
| TCC hit rate HIT/(HIT+MISS) | 0.387 | 0.653 | - |
| SQ_WAVES | 65536 | 32768 | 2.0 |

- Both kernels sit at the ~1.4 kW socket cap; round 7 runs ~13 % faster clocks yet needs ~13 % more
  cycles, i.e. ~10 % more energy per FLOP (0.778 vs 0.859 TFLOPS/W).
- Same L2 request count, but round 7 misses 1.77x as often (2.245e8 vs 1.272e8 TCC misses).
  Grid is (head, q_block desc, batch) with head fastest, so each XCD holds 16 heads x 2 q-blocks
  at a time: every K/V tile has ~2 co-resident readers.
- SQ_LDS_BANK_CONFLICT reads 0 for both kernels: unusable as a bank-conflict signal here.
- ISA (`profiles/isa_r7`, main loop = 2 tiles): 749 instrs, 88 MFMA, 66 v_exp, 344 VALU, 112 DS,
  44 s_nop wait states (30 between mask v_cndmask / v_cmp, 10 at M0 -> DMA, none MFMA -> exp).
  HipKittens' QK / exp lead-depth pipelining has no s_nop to recover here.
- Hypothesis (unverified): the extra L2 misses cost fabric / MALL energy under the power cap, so an
  XCD-aware CTA order (same-head q-blocks co-resident on one XCD) may raise clocks or cut stalls.

## Round 12 — ACCEPTED (trade-off) — 2026-11-09 11:40 (UTC+8)

- Change: XCD-aware CTA -> (head, q rank) mapping. Each XCD owns heads h = 8 * h_local + lin % 8 and
  walks groups of 8 q ranks head-major; causal q rank -> q block NQ - 1 - r is unchanged.
- Correctness: bitwise equal to round 7 (10 x 4 shapes); full set 7/7 PASS.
- Result: geomean 1104.55 (+4.334 % vs round 7), ratio 0.9679. b1_s16k..s64k +6.8 % to +9.6 %,
  b2_s16k +8.5 %, b1_s8k +3.0 %, b1_s4k -3.93 %, b2_s4k -1.84 %.
- Measured mechanism: L2 hit 0.372 -> 0.802 (S16K), same requests; cycles +5.6 % but the clock at
  the 1.4 kW cap rises ~4-5 %, so the gain is energy. At S4K cycles +25 % and the clock +17 %:
  idle tail, hypothesised to be the 8-rank group spanning half of the 16 q ranks.
- Next: round 13 restores LPT order for the tail q ranks.

## Round 13 — ACCEPTED — 2026-11-09 11:55 (UTC+8)

- Change: in the round-12 XCD mapping, the last 8 q ranks (q_split = floor(max(NQ - 8, 0) / 8) * 8
  onward) go rank-major / head-minor (round-7 longest-first order); earlier ranks keep full
  8-rank groups.
- Correctness: bitwise equal to round 12 (10 x 4 shapes); full set 7/7 PASS.
- Result: geomean 1121.20 (+1.507 % vs round 12, +5.906 % vs round 7), ratio 0.9825; every shape up
  vs round 12 (b1_s4k +3.47 %, b2_s4k +3.00 %); b1_s4k -0.60 % vs round 7.
- Measured: S4K cycles -15.7 % vs round 12, sustained S4K 1000.5 (round 7 999.8); S16K hit rate
  0.792 (round 12 0.802), throughput within noise.

## Machine change — 2026-11-09 13:40 (UTC+8)

From round 14 the campaign runs on smci355-ccs-aus-n02-25 (rounds 1-13: n04-25). HIP device 0 =
amd-smi GPU 3 on both; PWR_CAP 1400 W. aiter's opus module had to be JIT-built in the new container.
Re-measured on n02-25 in one session: round 7 1088.92 (ratio 0.9041), round 13 1179.59 (0.9768).
A sustained B1 S16K round-13 loop drew 1322 W at 1927 MHz here (n04-25: 1400 W at ~1630 MHz), so this
GPU is not pinned at the cap at that shape; the XCD order still gains +8.3 % over round 7.

## Round 14 — ACCEPTED (trade-off) — 2026-11-09 14:00 (UTC+8)

- Config sweep G in {4, 8, 16} x T in {4, 8, 16} on both machines: G16 best (+0.42 % n04-25,
  +1.64 % n02-25 over G8 T8), T within noise at G16. Change: XCD_Q_GROUP 8 -> 16.
- Correctness: bitwise equal to round 13; full set 7/7 PASS.
- Result (n02-25): geomean 1203.93 (+2.063 % vs round 13 on n02-25), aiter 1204.14, ratio 0.9998;
  b1_s16k / s32k / s64k beat aiter (1.013 / 1.014 / 1.019). b1_s4k -1.48 %, b2_s4k -2.75 %:
  with G16 T8 the whole S4K range is tail, i.e. the round-7 order.
- Next: round 15 chooses G per launch (16, or 8 when 16 leaves no grouped section).

## Round 15 — ACCEPTED — 2026-11-09 14:15 (UTC+8)

- Change: group size per launch, 16 when at least 16 q ranks precede the 8-rank tail, else 8
  (S4K maps as round 13; S >= 8K as round 14).
- Correctness: bitwise equal to round 14; full set 7/7 PASS.
- Result (n02-25): geomean 1212.71 (+0.729 % vs round 14, +11.368 % vs round 7 on n02-25),
  aiter 1208.92, ratio 1.0031: campaign target met. b1_s4k +1.96 %, b2_s4k +3.01 %, S >= 8K noise.
- Below parity: b1_s8k 0.986, b2_s16k 0.986, b2_s4k 0.996.

## Status after round 15 (2026-11-09 14:15 UTC+8)

- Lineage since the resume: r7 -> r12 (XCD order) -> r13 (LPT tail) -> r14 (group 16) -> r15
  (group per launch). All K1, all bitwise identical to round 7 outputs.
- n04-25: r7 1058.67 -> r13 1121.20 (ratio 0.9291 -> 0.9825). n02-25: r7 1088.92 -> r15 1212.71
  (ratio 0.9041 -> 1.0031).
- Measured mechanism (n04-25, B1 S16K): TCC hit 0.37 -> 0.80 at equal requests; at the 1.4 kW cap
  the gain arrives as clock. HipKittens ideas not used yet: 16x16x32 MFMA atom, three K/V buffers,
  counted lgkmcnt waits; its QK / exp lead-depth hints have no s_nop to recover in round 7's loop.

Correction (timestamps): the round 12 / 13 / 14 / 15 and "Status after round 15" headers above give
11:40 / 11:55 / 14:00 / 14:15 / 14:15 (UTC+8). The benchmark logs finished at 11:13 / 11:19 / 13:51 /
13:53 (UTC+8); those are the correct round times. Results are unaffected.

## Pre-round-16 probe: MFMA atom power (2026-11-09 14:10 UTC+8, n02-25)

`profiles/mfma_power/mfma_power.hip`: 1024 WGs x 8 waves, same FLOPs per iteration with
v_mfma_f32_32x32x16_bf16 (4 accumulators) or v_mfma_f32_16x16x32_bf16 (8 accumulators), operands
rotating over 8 random bf16 register packs; 12 s sustained, amd-smi sampled from 6 s.

| atom | TFLOPS (2 runs) | GFX clock (MHz) | socket power (W) |
|---|---|---|---|
| 32x32x16 | 1774.7 / 1776.6 | 1868-1873 | 1242-1258 |
| 16x16x32 | 2013.9 / 2020.9 | 2074-2077 | 1278-1286 |

16x16x32 sustains ~11 % higher clock and +13.6 % throughput (also ~2 % more per MHz). Neither run
reaches the 1400 W cap, so another limiter sets the clock here (unidentified). Supports switching
the attention MFMAs (round 16 candidate, user-selected).

## Round 16 — ACCEPTED — 2026-11-09 14:55 (UTC+8)

- Change: QK MFMAs 32x32x16 -> 16x16x32 (trait `QK_MFMA16`, auto for head_dim 192 without a
  window). Raw 16x16 accumulators go to the 32x32 score layout with 8 `v_permlane16_swap` per 32-kv
  half in `ctx.qk_scores` at each consumer; K is read with paired d chunks (blocks 0 / 2) and a
  middle-block DMA chunk swizzle (m ^ 4 for lane row groups 0 / 3), conflict-free.
- Full set, same session, alternated twice: 1217.15 vs round 15 1208.81 (+0.690 %), ratio vs aiter
  1.0127 (round 15: 1.0016). B1 S64K +2.08 %, B2 S16K +1.55 %, S <= 8K -0.14 % to -0.02 %.
- Mechanism (B1 S16K, measured): clock 1927-1933 -> 2025-2040 MHz at 1371-1386 -> 1349-1367 W;
  cycles +2.9 % (GRBM) / +4.5 % (clock x time); bank conflicts 0 (natural 16x16 read order: 8.1e8).
- In-round findings: the swap inside the QK region made the scheduler keep all K packs live (51
  spills); b64-split middle reads were merged into ds_read2_b64 (conflicts doubled); rescaled QK
  sched hints lost 0.3 %. Single back-to-back runs drifted ~1.5 %, so all comparisons alternate.
- Remaining ceiling (wrong-result variant without the swaps): +2.8 % quick. Next lever: PV on
  16x16x32 consuming the raw layout.

## Round 17 — ACCEPTED — 2026-11-09 15:25 (UTC+8)

- Change: PV on v_mfma_f32_16x16x32_bf16 (trait `PV_MFMA16`, on with `QK_MFMA16`), consuming the raw
  16x16 QK accumulators as the B operand (O^T = V^T P^T, 32 PV MFMAs per tile); `qk_scores` is the
  identity, so the 16 score-layout `v_permlane16_swap` per tile are gone. Row scalars stay one q
  per lane (16-value max tree per q tile + one swap + permlane32 pair reduce; ones-MFMA row sum
  into an 8-wide l). V read with `ds_read_b64_tr_b16`, odd LDS rows chunk-swizzled (m ^ 2) by the
  DMA; masks and O store rewritten for the raw layout. `CODEGEN_REV` 8.
- Full set, same session, alternated twice: 1237.68 vs round 16 1222.74 (+1.22 %), every shape up
  (+0.46 % b1_s4k to +1.73 % b2_s16k), ratio vs aiter 1.0260 (round 16 same session: 1.0116).
  7/7 PASS, error metrics unchanged. Quick, 3 rotations: 1212.4 vs 1196.5 (+1.33 %).
- Mechanism (B1 S16K, measured): clock 2003-2048 -> 2127-2141 MHz at the same 1.33-1.36 kW;
  ms/call 8.53 -> 8.38; cycles up (GRBM +4.2 %, clock x time +2.7 %); MFMA busy cycles and LDS
  instruction count identical, bank conflicts 0. Main loop per 2 tiles: swaps 34 -> 8 (rebase
  branch only), 835 -> 892 instructions; spills 9 -> 13, outside the loop.
- Next: PV sched hints retune for 32 shorter MFMAs; share the row-max trees between the trigger
  and the rebase branch; b1_s8k still 0.998 of aiter.

## Post-round-17 headroom analysis vs aiter — 2026-11-09 16:20 (UTC+8), n02-25, B1 S16K

Artifacts: `profiles/headroom_r17/` (scripts `profiles/clock_power2.sh`, `profiles/pmc_any.sh`,
`profiles/mfma_power/run_sampled.sh`, `profiles/mfma_power/nop_power.hip`, fakes in
`profiles/variants_edits/fake_{exp_fma,no_kv_loop,noexp_nokv}.py`).

Measured (sustained 14 s, amd-smi GPU 3, throttle = growth of PPT_ACCUMULATED / ACCUMULATION_COUNTER):

| run | TFLOPS | GFX MHz | socket W | PPT throttle | TFLOPS / MHz |
|---|---|---|---|---|---|
| round 17 | 1317-1322 | 2140-2154 | 1357-1371 | 0.68 | 0.615 |
| aiter (opus gqa_d192, 32x32x16) | 1270.5 | 1633-1637 | 1361-1367 | 0.54-0.59 | 0.778 |
| MFMA probe 16x16x32 | 2015 | 2080-2085 | 1275-1286 | 0.03-0.06 | 0.967 |
| MFMA probe 32x32x16 | 1784 | 1879-1882 | 1243-1254 | 0.00 | 0.949 |
| s_nop on all SIMDs | - | 2396 | 346 | 0 | - |
| s_sleep on all SIMDs | - | 2374 | 334 | 0 | - |
| idle | - | 156 | 231 | - | - |

- Correction to rounds 16 / 17: the forward's clock limiter is the socket power limit (PPT active
  68 % of the time for round 17) although sampled SOCKET_POWER reads ~1365 W (< 1400 W PWR_CAP).
  The MFMA probe is not PPT-limited (<= 6 %) yet stays at ~2083 / ~1880 MHz; no reported
  accumulator (PPT, socket / VR / HBM thermal, per-XCD) explains that cap. Max GFX clock 2400 MHz.
- Same-session counters (rocprofv3): GRBM_GUI_ACTIVE r17 1.379e8 vs aiter 1.044e8 (+32 %);
  MFMA busy / GRBM r17 82.4, aiter 104.5, probe 120.8-122.6, i.e. MFMA pipe busy ~67 % (r17) vs
  ~86.5 % (aiter). r17 executes less non-MFMA work (VALU w/o MFMA 0.69e9 vs 1.22e9, SALU 1.06e8 vs
  1.51e8, same exp count 2.73e8, L2 hit 0.79 vs 0.65) but waits more (SQ_WAIT_INST_ANY 3.88e9 vs
  2.11e9). aiter runs 32 q rows per wave, KV tile 64, 8 waves, 32x32x16 (same per-wave tile as
  r17); its 512-row WG comes from the causal head/tail merge.
- ATT (one CU) of the r17 main loop, share of wave latency: MFMA 36 %, ds_read 13 %, v_exp 12 %,
  s_barrier 12 % (two loop barriers ~5 % each), s_waitcnt 8 %, buffer_load issue 6 %.
- Wrong-result upper bounds (alternated twice with r17 in one session):
  exp2 -> one fma: 1402.6-1407.0 TFLOPS (+6.7 %), 2317-2330 MHz, 0.605 TFLOPS/MHz;
  no K/V DMA in the main loop: 1462.1-1463.2 (+11.1 %), 2209-2217 MHz, 0.661 TFLOPS/MHz;
  both: 1533.3-1533.6 (+16.7 %), 2388-2393 MHz (at the 2400 MHz max), 1322-1330 W, PPT 0.23-0.25,
  0.641 TFLOPS/MHz.

Interpretation (estimates, not measured directly):
- An occupied-but-idle GPU costs only ~115 W over idle at 2400 MHz, so at the power limit cutting
  cycles without cutting work saves little; throughput there is set by work energy per FLOP.
  With work power = socket - 231 W - 115 W x f / 2396: r17 ~0.78 pJ/FLOP, aiter ~0.83, MFMA probe
  16x16x32 ~0.47, 32x32x16 ~0.52. Non-MFMA work in r17 ~0.29 pJ/FLOP (after 5 % rowsum MFMAs).
- 1.4x aiter at B1 S16K = 1779 TFLOPS needs >= 0.741 TFLOPS/MHz (2400 MHz cap) and, at ~1364 W,
  work energy <= ~0.58 pJ/FLOP, i.e. non-MFMA energy cut by ~70 %. Removing exp and all main-loop
  K/V traffic outright reaches only 1.21x aiter (clock-capped). Not reachable by the levers known;
  the pure-MFMA ceiling is 2015 TFLOPS (1.59x aiter).
- Levers by measured bound: K/V traffic (<= +11 %; the L2 vs HBM split not yet measured), exp
  energy (<= +6.7 %), then LDS reads / VALU (the remainder; halving LDS bytes per FLOP needs 64 q
  rows per wave, i.e. 1 wave per SIMD and no cross-wave ping-pong).

## Review of branch optimize/mla192_flydsl_gfx950_20261109 (46daaaec) — 2026-11-09 16:13 (UTC+8)

(The previous entry's 16:20 stamp was written ahead of the clock; it precedes this one.)

Claim in the commit message: "Forward 1.72-1.85x aiter on DeepSeek-V3/V2-Lite/Kimi-K2/K3 training
shapes". The commit carries no benchmark or numbers; no workspace or transcript for it on this node.

Method: its `flash_attn_mla_fwd.py` copied unmodified into `profiles/pkg_br/flydsl_mla/` behind a
shim launched exactly like its `_mla_sbhd_forward_op` (LSE always on, as its dispatch does). The
branch's `flydsl/utils/attn_helper.py` and Turbo's aiter path are byte-identical to the installed
checkout. GPU 0, same process, rotated order, median of 3.

Measured (TFLOPS, causal-counted FLOPs; CSVs `profiles/br_full_1.csv`, `br_probe_1.csv`,
`br_probe_noopus_1.csv`):

| shape (B, S, H) | branch | r17 | aiter opus (train) | aiter opus (inference_mode) | aiter v3 asm |
|---|---|---|---|---|---|
| DSV3 1,4K,128 | 842.7 | 1097.9 | 1082.2 | 1083.0 | 956.0 |
| DSV3 4,4K,128 | 845.0 | 1086.6 | 1053.9 | 1051.1 | 950.4 |
| V2-Lite 1,4K,16 | 596.4 | 737.0 | 766.8 | 767.7 | 922.4 |
| V2-Lite 4,4K,16 | 850.8 | 1055.1 | 1082.9 | 1080.4 | 958.3 |
| Kimi-K2 1,4K,64 | 836.5 | 1083.3 | 1088.5 | 1081.6 | 959.3 |
| Kimi-K2 4,4K,64 | 886.8 | 1099.6 | 1075.5 | 1075.5 | 950.8 |
| DSV3 1,16K,128 | 1016.7 | 1314.6 | 1274.1 | 1267.4 | 1031.6 |

- Correctness: PASS on the campaign gate (o_rel vs fp32 rows 0.0016-0.0020, LSE <= 0.0051).
- Branch / aiter opus = 0.77-0.83 on every shape (full set geomean 0.803); r17 / branch = 1.24-1.30.
- aiter runs `gqa_d192_v128_kernel<opus_gqa_d192_traits<32,64,8,...>>` in both training and
  `inference_mode` (bench_attention_turbo --forward-only); same speed.
- With `AITER_DISABLE_FMHA_OPUS=1` aiter falls back to the v3 asm `fmha_fwd_hd192_hd128_bf16_causal`;
  the branch is 0.65-0.99x of that too. The CK `mha_fwd` fallback is not prebuilt here; not measured.
- Not reproduced: no aiter forward path measured here is slower than the branch kernel.
  Where 1.72-1.85x came from is unknown (hypotheses only: a slower aiter build on another node,
  e.g. the CK fallback; a different FLOP count or timing scope).
- Dispatch impact (code reading): `resolve_flash_attn_backend` defaults to FLYDSL and the branch's
  `_flydsl_mla_ok` admits bf16 causal MLA in contiguous sbhd, so auto dispatch would move those
  forwards from aiter opus to a kernel ~20 % slower on this node.

## Branch optimize/mla192 step table check — 2026-11-09 16:25 (UTC+8)

The branch author's table (shared by the user): step = fwd + bwd, aiter vs FlyDSL, DeepSeek-V3
11.14 / 9.69 ms (1.15x), V2-Lite 5.62 / 4.94 (1.14x), Kimi-K2 11.33 / 9.74 (1.16x), Kimi-K3
23.84 / 20.47 (1.16x); its aiter column "taken from the earlier full run" (not same session).

Re-run with the branch's own code: a detached worktree of 46daaaec (csrc identical; `_C` and
`libprimus_turbo_kernels.so` symlinked from the installed build), its
`bench_attention_turbo.profile_attention(..., layout="sbhd")`, FLYDSL / AITER pinned, rotated
F A A F in one process, GPU 0 (`profiles/br_step.py`, `profiles/br_step_1.csv`). All correctness PASS.

| model (B, S4K, H) | AITER fwd / bwd / step ms | FLYDSL fwd / bwd / step ms | step aiter/flydsl |
|---|---|---|---|
| DeepSeek-V3 (2, 128) | 1.273 / 4.824 / 6.097 | 1.635 / 4.746 / 6.381 | 0.955 |
| DeepSeek-V2-Lite (12, 16) | 0.959 / 3.797 / 4.756 | 1.157 / 3.739 / 4.897 | 0.971 |
| Kimi-K2 (4, 64) | 1.276 / 4.800 / 6.075 | 1.572 / 4.733 / 6.305 | 0.964 |

- FlyDSL step is 3-5 % slower than aiter. Fwd 0.78-0.83x aiter (aiter 1074-1080 TFLOPS; the
  branch 834-895). The aiter bwd is ~1.5 % faster on the FLYDSL path (same aiter bwd; called with
  sbhd grads; cause not examined). Bwd here 678-726 TFLOPS (benchmark's 2.5x fwd FLOPs).
- Back-solving the author's DeepSeek-V3 row with this node's branch fwd (1.64 ms): its bwd was
  ~8.05 ms (~427 TFLOPS, matching its "bwd ~430 TFLOPS") and its aiter fwd ~3.1 ms (~440 TFLOPS).
  Its V2-Lite FlyDSL step (4.94) matches this node (4.90); its aiter V2-Lite step (5.62) implies an
  aiter fwd ~1.8 ms (~560 TFLOPS) vs 0.96 ms here. So its aiter column is far slower than the
  current aiter (opus d192 fwd) on this node; source unknown (hypothesis: an older aiter build).
- Estimate, not measured in this harness: r17 fwd (1095 TFLOPS at B2 S4K H128) + this bwd gives a
  DeepSeek-V3 step of ~6.00 ms, ~1.6 % faster than aiter.

## What the optimize/mla192 branch offers us — 2026-11-09 16:35 (UTC+8)

Kernel side (code reading, grep-verified): nothing r17 lacks. Its kernel is Turbo's D128 dual-wave
kernel with the QK side widened on 32x32x16 (12 k-steps, three 64-col K strips) — the same base as
our early proto. r17 already has its wave-uniform dead-half skip on the drain tiles (`chunk_live` /
`qk_live` / `_pv_branch`, plus max_m3 LO), the same compile hints (max-memory-clause, post-misched,
lsr-drop-solution, fast/unsafe fp math, preserve-sign denormals), the same compiled-launcher cache,
and a superset of the shared attn_helper (Turbo main's last attn_helper change is #581, 10-28).

Integration side, measured (`profiles/br_bwd_kern.py`, DSV3 B2 S4K H128, sbhd, 10 bwd calls x2):
- Both paths run the same aiter bwd kernels: `fmha_bwd_hd192_128_bf16_causal_br_a16_pssk`
  (~3.78 ms), `fmha_bwd_hd128_odo_bf16` (~0.15 ms), `fmha_bwd_hd192_dq_shuffle` (~0.14 ms).
- The AITER path has one extra bf16 fill per bwd (fills 121 us vs 53 us): `flash_attn_interface.py`
  allocates dq with `torch.ones(...)` (all three layouts; was `zeros_like` before #275, 2026-04-17,
  reason not stated); the branch's FLYDSL path uses `torch.empty`. ~68 us = the ~1.5 % bwd gap seen
  in `br_step_1.csv`. Not FlyDSL-specific: any aiter dense bwd pays it.
- Benchmark artifact: each timed bwd also carries 3 `.grad` accumulate adds (~0.57 ms, ~12 % of the
  measured bwd) because profile_attention re-runs backward on leaf q/k/v.

## Backward survey (MLA 192/128 and D128, gfx950) — 2026-11-09 16:50 (UTC+8)

B2 S4K H128, sbhd, causal; `torch.autograd.grad` timed with CUDA events (no leaf `.grad` adds),
rotated, median of 4 (`profiles/bwd_survey.py`). FLOPs B*S*(S+1)/2*H*2*(3*dqk+2*dv).

| path | a16 (ATOMIC_FP32=0) | a32 (ATOMIC_FP32=1, Turbo default) |
|---|---|---|
| MLA aiter | 4.249 ms, 841 TFLOPS (main `..._causal_br_a16_pssk` 3.93 ms) | 7.427 ms, 481 TFLOPS (main a32 6.81 ms) |
| D128 aiter | 3.058 ms, 899 TFLOPS | 3.964 ms, 694 TFLOPS |
| D128 FlyDSL (Turbo `flash_attn_bwd.py`) | 3.581 ms, 768 | 3.568 ms, 771 (ignores the switch) |

Accuracy vs fp32 math SDPA, SNR dB (`profiles/bwd_a16_acc.py`; bf16 rounding floor 55.6):
- MLA aiter a16 dq 49.1 (S4K) / 46.1 (B1 S16K H16); a32 dq 52.2-52.3; dk 52.3, dv 52.7 in both.
- D128 aiter a16 dq 49.6, a32 52.2; dk 52.3, dv 52.7. D128 FlyDSL dq 48.0 / 45.9 (S16K),
  dk 48.5, dv 49.7 in both modes. dq is not bitwise repeatable on any path; dk/dv are.
- Which baseline counts: Primus `runner/helpers/envs/base_env.sh` exports
  PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32=${...:-0}, so Primus runs a16. Primus DeepSeek-V3 MI355X BF16
  config has `use_turbo_attention: false` (attention goes through TE there today).
- Reading: the FlyDSL D128 bwd template is 0.85x aiter a16 at D128 with lower dk/dv accuracy;
  a FlyDSL MLA bwd built on it would start behind aiter a16 (hypothesis, not measured).

## Round 18 — ACCEPTED (trade-off) — run 2026-11-09 16:43-17:01 (UTC+8), logged 18:24

- Correction to the section above: "Backward survey ... 16:50 (UTC+8)" was written at about 16:35
  (file mtime 16:35:12); its stamp was ahead of the clock.
- Pre-round probe (wrong results, `profiles/variants/fake_kv_head0`): every CTA reads kv head 0, so
  K/V almost always hit L2 / MALL. B1 S16K sustained 2 x 14 s: 8.034 / 8.052 ms/call vs 8.322 /
  8.356 (-3.5 %), clock ~2146 -> ~2169 MHz at ~1360 W, PPT active 0.67 -> 0.70-0.72
  (`profiles/r18_analysis/clock_power_kvh0.txt`). That bounds what K/V locality can give there.
- Change: `XCD_Q_GROUP` / `XCD_Q_GROUP_SHORT` -> `XCD_Q_GROUPS` (largest first): the ranks ahead of
  the 8-rank tail are cut into full groups of each size in turn, the remainder going to the next
  size. Sweep (16, 8) vs (32, 16, 8) on the same code; kept (32, 16, 8). `CODEGEN_REV` 9. Host
  replica bijective for nq 1-400; O / LSE bitwise equal to round 17.
- Full set, same session, 2 rotations: 1247.73 vs round 17 1239.81 (+0.64 %); (16, 8) 1245.26
  (+0.44 %). Per shape (32, 16, 8): +0.48 % to +1.67 % at S 8K-64K, b1_s4k -1.06 %, b2_s4k
  -0.53 %. Quick, 3 rotations: +0.32 %. Ratio vs aiter 1.0374 (round 17 same session 1.0272).
- Measured: S4K runs the same CTA order in all three packages, yet the (32, 16, 8) build is
  0.5-1.1 % slower there and 1.25 % slower than (16, 8) at S8K (same order again): about 1.3 % is
  code generation (main-loop opcode order differs in 13-36 lines, spills 13 / 11 / 13). L2 misses
  do not track speed: at S16K (16, 8) has 16 % fewer misses than round 17 but the lowest clock and
  the slowest sustained time; at S64K (32, 16, 8) has 3.6 % more misses and is the fastest, with
  the highest clock. Details: `rounds/round-18/summary.md`.
- Next: recover the ~1.3 % code-generation cost at S4K / S8K while keeping the (32, 16, 8) order.

## Round 19 — ROLLED BACK — 2026-11-09 18:33 (UTC+8)

- Change: rewrite the (32, 16, 8) XCD map as independent `_xcd_sec` maps + select cascade (round
  17's two-map form), hoping to recover the ~1.3 % S4K / S8K code-generation cost vs the (16, 8)
  build. CTA order unchanged (host replica identical). `CODEGEN_REV` 10.
- Full set, same session, 2 rotations: 1247.67 vs round 18 1250.26 (-0.21 %); below in both
  rotations. b1_s4k -0.43 % (g16 still +1.10 % over r18). Quick, 3 rotations: -0.12 %.
- ISA: main-loop opcode sequence identical to round 18 (791 insts, 0 diffs); whole kernel
  3696 -> 3699; spill 11 unchanged. The loop form was not the schedule lever.
- Working tree restored to round 18. Next: try (32, 8) two-level map, or leave the map and take
  PV sched / rebase / locality levers.

## Pre-round-20 probe: rowsum and LDS-read bounds — 2026-11-09 19:00 (UTC+8), n02-25

Wrong-result variants of round 18 (`profiles/variants_edits/fake_{no_rowsum,lds_half,lds_half_k,
lds_half_v}.py`, built by `profiles/isa_variant.sh`, distinct `CODEGEN_REV` 901-904), sustained 14 s
each, alternated twice with round 18 in one session (`profiles/clock_power_r20_{fakes,kv}.txt`):

| variant | main-loop change (2 tiles) | B1 S16K TFLOPS | B1 S4K TFLOPS | clock S16K / S4K (MHz) |
|---|---|---|---|---|
| round 18 | 791 insts, 168 MFMA, 48 `ds_read_b128`, 64 `ds_read_b64_tr_b16` | 1321-1330 | 1078-1083 | 2157 / 2244 |
| no rowsum MFMA | 160 MFMA (804 insts) | 1323-1332 (+0.1 %) | 1080-1083 (+0.1 %) | 2176 / 2257 |
| K and V LDS reads halved | 60 `ds_read` (711 insts) | 1448-1450 (+9.1 %) | 1179-1181 (+9.0 %) | 2136 / 2234 |
| K reads only | 28 `ds_read_b128` | 1401-1403 (+5.8 %) | - | - |
| V reads only | 32 `ds_read_b64_tr_b16` | 1381-1388 (+4.4 %) | - | - |

- The halved-read variants gain at the same clock and power (1360-1375 W, PPT 0.64-0.70): fewer
  cycles, unlike the earlier exp / K-V-DMA fakes, which gained by clock. Per byte removed, K
  (~20 KB per wave per 2 tiles, +5.8 %) and V (16 KB, +4.4 %) cost the same (~0.28 % per KB). V's
  transposed reads cost twice per instruction, so the cost follows bytes, not instructions.
- PMC, B1 S16K, one dispatch (`profiles/pmc_r20/`). `SQ_INSTS_LDS_LOAD_BANDWIDTH` (64 B units) is
  5.42e9 = 347 GB per call (the 40 KB per wave per tile model gives 344 GB), ~79 B/clk per CU
  averaged over the kernel. Halving the reads: GRBM_GUI_ACTIVE -9.1 %, `SQ_WAIT_INST_LDS` -46 %,
  MFMA busy cycles unchanged (1.136e10), `SQ_LDS_BANK_CONFLICT` 0 in both.
- Interpretation: LDS -> VGPR bytes per FLOP are a cycle-level bottleneck worth ~9 %. They depend
  only on q rows per wave (2 bytes per MAC divided by rows per wave, for both K and V). 64 rows per
  wave needs Q 96 + O 128 + S 64 VGPRs, which does not fit the 256 per wave of two waves per SIMD;
  it needs one wave per SIMD (HipKittens ladder Level 11) and the loss of the dual-wave ping-pong.
  The rowsum-on-VALU idea is dropped: bound ~0.

## Round 20 — ACCEPTED — 2026-11-09 19:16 (UTC+8)

- Change: the dense causal drain predicates every diagonal 32-col half that some wave can find dead
  at trace time (`_diag_live`: start column >= ROWS_PER_WAVE). At BLOCK_M 256 that adds QK on
  max_m3 LO / HI and max_m2 LO (9 halves per block) and P*V on those plus max_m4 HI (10 halves).
  BLOCK_M 128 predicates are unchanged. `CODEGEN_REV` 10.
- Correctness: O / LSE bitwise equal to round 18 (4 shapes x 5); full set 7/7 PASS.
- Full set, same session, 2 rotations: 1263.08 vs round 18 1248.61 (+1.16 %), ratio vs aiter
  1.0450 (r18 1.0362). Every shape up: b1_s4k +2.78 %, b2_s4k +1.84 %, b1_s16k +1.32 %, b1_s8k
  +1.26 %, b2_s16k +0.45 %, b1_s32k / s64k +0.25 %. Quick, 3 rotations: +1.44 %.
- Measured: sustained B1 S4K 0.6346 -> 0.6196 ms (-2.4 %) at 2254 -> 2235 MHz and ~1368 W, so
  clock x time falls 3.2 % (cycles, not energy). Main-loop schedule moved without a source change
  (791 -> 796 insts); S32K / S64K +0.25 % bounds that effect.
- Next: the LDS-bytes lever (+9 % bound) needs a one-wave-per-SIMD restructure (user decision);
  smaller: skip exp on dead halves, QK of max_m4 HI in the main loop.
