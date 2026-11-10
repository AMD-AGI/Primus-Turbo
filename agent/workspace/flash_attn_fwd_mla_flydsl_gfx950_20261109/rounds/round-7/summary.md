# Round 7 — ACCEPTED

- Time: 2026-11-09 09:25 (UTC+8)
- Hypothesis (from the round-6 ISA reading): the whole VGPR spill comes from Q being held as
  96 unpacked bf16 dwords. `scale_all` truncates Q with one `<96 x f32> -> <96 x bf16>` fptrunc;
  96 is not a power of two, so type legalization halves it down to an odd width and scalarizes
  (`v_cvt_pk_bf16_f32 vD, vF32, s0`, one element per VGPR). The MFMA packs are then built with
  `v_perm_b32`, and the allocator spills / rebuilds the unpacked elements (two scratch dword
  reloads + `vmcnt(0)` + `v_perm` per Q dword in the epilogue). At D128 the width is 64, which
  splits cleanly into pairs, so the D128 kernel never showed this.
- Change (single variable): `scale_all` truncates one MFMA pack (8 elements) at a time and
  concatenates the packs (same tree as `load_all`). Same RNE rounding, so results should be
  bitwise identical. Snapshot: `kernel_snapshot/`.

## Resources (ISA dump B1 S4K, `profiles/isa_r7`)

| | round 5 | round 7 |
|---|---|---|
| `.vgpr_count` | 256 | 256 |
| `.vgpr_spill_count` | 90 | 0 |
| `.private_segment_fixed_size` | 364 B | 0 B |
| scratch load / store | 221 / 88 | 0 / 0 |
| `v_perm_b32` | 284 | 0 |
| `v_cvt_pk_bf16_f32` | 272 | 176 |
| total instructions | 4026 | 3236 |

## Result

Correctness: all 7 PASS (O rel 0.0016–0.0020, LSE max 0.0046–0.0051); bitwise equal to round 5
(5 repeats x 4 shapes, `artifacts/bitwise_vs_r5.log`).

| shape | round-7 | round-5 | delta | aiter | ratio |
|---|---|---|---|---|---|
| b1_s4k | 1004.8 | 882.5 | +13.9 % | 1023.6 | 0.982 |
| b1_s8k | 1056.4 | 986.9 | +7.0 % | 1137.1 | 0.929 |
| b1_s16k | 1088.5 | 1050.1 | +3.7 % | 1188.6 | 0.916 |
| b1_s32k | 1095.2 | 1082.5 | +1.2 % | 1214.3 | 0.902 |
| b1_s64k | 1110.1 | 1100.7 | +0.9 % | 1224.9 | 0.906 |
| b2_s4k | 987.9 | 861.3 | +14.7 % | 1013.5 | 0.975 |
| b2_s16k | 1074.1 | 1041.9 | +3.1 % | 1196.2 | 0.898 |

Geomean 1058.67 vs 996.81 (+6.21 %); aiter 1139.49; ratio 0.9291. Quick-set geomean 1046.52
(ratio 0.944).

Decision: ACCEPT (aggregate up, every shape up, outputs bitwise unchanged).

## Real-training transfer check

Bucket K1 (kernel-body change, no wrapper cache, no launch-config change). The gain is in the
kernel itself: the same `mla_fwd_sbhd` call path used by the DSV3 attention forward gets it with
no caller change. The largest gains are at 4K–8K sequence lengths (prologue/epilogue share is
largest there); at 32K–64K the gain is ~1 %, so a long-context training step sees little.

## Lesson

Check vector widths of element-wise ops at a new head dim. A non-power-of-two width
(96 = 3 x 32) can scalarize a conversion that is pairwise at D128, and the unpacked values
double the register footprint at exactly the point where the allocator spills.
