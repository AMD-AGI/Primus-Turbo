# Round 6 — ROLLED BACK

- Time: 2026-11-09 09:22 (UTC+8)
- Hypothesis: the hot epilogue barrier (after epilogue C11, 9.6 % of wave latency at B1 S4K on
  round 5) is waited on by the one or two waves that still compute the last causal tile. Those
  waves rebuild Q from scratch before every QK MFMA in the live branches, so other waves idle.
  Reloading Q from global memory at the loop exit should end Q's live range inside the loop and
  remove the prologue spill stores and the per-MFMA scratch reloads.
- Change (single variable): `q_epi = ctx.scale_all(ctx.load_all())` at the loop exit when
  `K_STEPS_QK > 8`, passed to the five epilogue `_qk` calls through a new `q=` argument.
  D128 path unchanged. Snapshot: `kernel_snapshot/`.

## Evidence before the change (measured, ATT on round 5, B1 S4K, one CU)

- Wave latency split: s_barrier 22.2 %, MFMA 21.6 %, s_waitcnt 15.9 %, ds_read 9.4 %,
  vmem load 8.4 %.
- Hottest single instruction: the `_s_barrier()` after epilogue C11 (9.6 % of wave latency).
  Most waves of a WG wait 13–21K cycles there.
- The late waves run 391 instructions (~13.7K cycles) before that barrier, including 116
  `scratch_load_dword`. Each pair is followed by `s_waitcnt vmcnt(0)` and a `v_perm_b32` that
  packs two bf16 halves into one MFMA B-operand dword (Q pack).
- Root cause seen in the ISA: `scale_all` converts Q one element at a time
  (`v_cvt_pk_bf16_f32 vD, vF32, s0`, 96 per wave) and the packs are built by `v_perm_b32`.
  The allocator spills the 96 unpacked dwords in the prologue (`.LBB0_5`: 88 stores) and
  rebuilds each pack with two dword reloads plus a `v_perm` at every epilogue use.

## Result

ISA (B1 S4K dump `profiles/isa_r6`): `.vgpr_spill_count` 90 -> 141, scratch 364 -> 460 B,
scratch instructions 309 -> 277. Prologue stores 88 -> 2 and loop-exit reloads 69 -> 1 as intended,
but the freshly loaded Q is spilled again inside the epilogue (47 + 25 + 13 stores) and the
late-wave live-branch blocks still rebuild each Q dword from two scratch dwords (54 -> 38 / 35
loads per block).

Correctness: all 7 PASS (O rel 0.0016–0.0020, LSE max 0.0046–0.0051); bitwise equal to round 5
(5 repeats x 4 shapes, `artifacts/bitwise_vs_r5.log`).

| shape | round-6 | round-5 | delta |
|---|---|---|---|
| b1_s4k | 726.9 | 882.5 | -17.6 % |
| b1_s8k | 891.4 | 986.9 | -9.7 % |
| b1_s16k | 994.5 | 1050.1 | -5.3 % |
| b1_s32k | 1056.2 | 1082.5 | -2.4 % |
| b1_s64k | 1088.2 | 1100.7 | -1.1 % |
| b2_s4k | 731.6 | 861.3 | -15.1 % |
| b2_s16k | 981.2 | 1041.9 | -5.8 % |

Geomean 913.70 vs 996.81 (-8.34 %); aiter 1139.10; ratio 0.8021.

Decision: ROLLBACK (aggregate down, four shapes regress >= 5 %). Kernel restored from round-5
snapshot (verified with `cmp`).

## Real-training transfer check

Bucket K1 (kernel-body change, no wrapper cache). Rolled back, nothing to transfer.

## Lesson

The epilogue is over the 256-VGPR budget by itself, so moving Q's definition does not remove
the spill; it only moves the stores into the epilogue, where they cost more. The structural
problem is the unpacked Q representation: 96 spilled dwords plus a `v_perm` per use instead of
48 packed dwords. Next: make `scale_all` emit packed `v_cvt_pk_bf16_f32` pairs so any Q spill
is half the size and needs no repacking.
