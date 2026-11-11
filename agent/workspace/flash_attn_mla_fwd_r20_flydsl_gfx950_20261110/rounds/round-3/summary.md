# Round 3 — ACCEPTED — 2026-11-10 18:50 (UTC+8)

## Hypothesis
R20 runs 8 waves of 32 q rows (two waves per SIMD). Every K / V pack read from LDS feeds two 16-row
q-tiles. With 64 q rows per wave and one wave per SIMD (4 waves x 64 rows, same 256-row WG), each
pack feeds four q-tiles, so the LDS read bytes per FLOP halve. The round-2 lesson is that this node is
power-limited (throughput = power / energy per FLOP), and the `ldshalf` wrong-result probe (odd K
slots / V steps reuse the previous packs) bounded the LDS-read energy at several percent. Expected:
higher sustained clock at the same ~1280 W, mostly on long-S shapes where the loop dominates.

## Single change
Wave layout rewrite "QT4": `rows_per_wave` 32 -> 64, 4 waves per WG, `waves_per_eu` 1 (`SOLO_WAVE`
trait). One technique, but the pipeline had to be re-scheduled to run it; the sub-steps below are all
part of that and do not change the math (output bitwise identical to R20 at every step):

| rev | sub-step | B1 S16K TFLOPS |
|---|---|---|
| 1016 | QT4 layout: O (128 AGPR) written by asm P*V MFMAs, Q pinned in AGPRs | 1096 |
| 1017 / 1021 | K / V DMA split into per-copy thunks, interleaved one per 6 P*V slots (`PV_DMA_EVERY`) | 1185.8 |
| 1023 | QK as single-k-step slots fenced by `sched_barrier(0)`; softmax finish (hi exps + cvt / row sum) spread over the slots | 1221.9 |
| 1025 | IGroupLP interleave hints per QK slot; running-max fold in the speculative softmax | 1239.0 |
| 1027 | cluster sync `QK_SYNC_TAIL` = 2 slots before the QK end; V head reads issued inside the QK | 1245.6 |

Files: `src/branch_mla/{attn_helper_mla.py, flash_attn_fwd_mla.py, ops.py}`, `CODEGEN_REV` 1035
(= rev 1027 code). Snapshot: `kernel_snapshot/branch_mla/`.

## Results (full set, 2 rotations, `artifacts/r3.csv`, base = round-1 snapshot, same session)

| shape | cand TFLOPS | cand ms | base TFLOPS | base ms | cand / base | O rel | LSE max | bitwise vs R20 |
|---|---|---|---|---|---|---|---|---|
| dsv3 B2 S4K H128 | 1019.4 | 1.3485 | 1014.0 | 1.3558 | 1.0053 | 2.01e-3 | 4.75e-3 | yes |
| dsv2lite B8 S4K H16 | 1000.0 | 0.6874 | 979.3 | 0.7019 | 1.0211 | 1.99e-3 | 3.17e-3 | yes |
| kimik2 B4 S4K H64 | 1016.0 | 1.3530 | 1014.8 | 1.3546 | 1.0012 | 1.97e-3 | 4.01e-3 | yes |
| kimik3 B2 S7168 H96 | 1145.7 | 2.7558 | 1121.3 | 2.8156 | 1.0218 | 1.83e-3 | 4.73e-3 | yes |
| B1 S4K | 1005.9 | 0.6833 | 1003.4 | 0.6850 | 1.0025 | 1.95e-3 | 4.60e-3 | yes |
| B1 S8K | 1156.9 | 2.3764 | 1131.0 | 2.4306 | 1.0229 | 1.80e-3 | 4.75e-3 | yes |
| B1 S16K | 1253.8 | 8.7697 | 1185.0 | 9.2794 | 1.0581 | 1.75e-3 | 4.93e-3 | yes |
| B1 S32K | 1300.2 | 33.8278 | 1215.6 | 36.1816 | 1.0696 | 1.61e-3 | 4.82e-3 | yes |
| B1 S64K | 1319.8 | 133.2984 | 1229.9 | 143.0441 | 1.0731 | 1.58e-3 | 5.14e-3 | yes |
| B2 S4K | 1018.8 | 1.3493 | 1013.9 | 1.3559 | 1.0048 | 2.01e-3 | 4.75e-3 | yes |
| B2 S16K | 1258.1 | 17.4793 | 1188.1 | 18.5092 | 1.0589 | 1.76e-3 | 4.82e-3 | yes |

Geomean cand 1129.25 vs base 1095.84 (+3.05 %), 11/11 PASS, 11/11 bitwise identical, no shape slower.
vs aiter (round-1 measurement 1005.24): 1.123x.

### Sustained (`artifacts/power_probe.log`, 10 s)

| run | TFLOPS | clock | power | PPT | TFLOPS/MHz |
|---|---|---|---|---|---|
| R3 B1 S16K | 1249.8 | 1972 MHz | 1281 W | 0.01 | 0.634 |
| R20 B1 S16K | 1183.0 | 1911 MHz | 1291 W | 0.09 | 0.619 |
| R3 dsv3 | 1016.3 | 2053 MHz | 1277 W | 0.03 | 0.495 |
| R20 dsv3 | 1010.1 | 2000 MHz | 1323 W | 0.17 | 0.505 |

## Decision
ACCEPTED (+3.05 % geomean, no regression, bitwise identical output).

## Attribution
- B1 S16K: +2.4 % work per cycle (0.634 vs 0.619 TFLOPS/MHz) and +3.2 % clock at the same power, i.e.
  both fewer cycles and less energy per FLOP. The long-S shapes (+5.8 to +7.3 %) are loop-dominated.
- dsv3 / 4K shapes (+0.1 to +2.1 %): R3 draws 46 W less and clocks 2.7 % higher, but does 2 % less work
  per cycle. ATT (dsv3, one CU, `profiles/att_r3/stats_r3_qkil.csv`): prologue 11.6 %, loop 72.3 %,
  drain + epilogue 16.1 % of the WG time. Prologue ~7.3K cycles per WG (Q global-load issue stalls
  and waits); drain ~2.7K cycles of barrier stall (causal diagonal imbalance across the 4 waves) and
  ~2K of epilogue store stalls. With one wave per SIMD nothing hides these, so the per-WG fixed cost
  eats the loop gain at S = 4K.
- Loop (ATT): 6855 cycles per KV tile vs 5376 cycles of MFMA (78 % pipe busy; R20 ~71 %). The
  non-MFMA instructions need ~3978 issue cycles per tile against ~4032 free issue cycles in the MFMA
  shadows (336 MFMAs x 12), so the schedule is at the issue-slot limit: exp 1409, s_waitcnt 411,
  cvt 372, ds_read_b64_tr 370, max3 356, ds_read_b128 263, buffer_load 220, s_nop 136, barrier 94,
  accvgpr_read 80, s_setprio 49 cycles per tile.

### Rejected sub-attempts (all rolled back inside the round)
- Seed dedupe for the running max: VGPRs 512 with spills, B1 S16K 1169.6.
- Spreading V head reads over the QK slots: -0.3 %. Q global loads before the K0 `waitcnt`: -0.4 %.
- Row sum on `v_dot2c_f32_bf16` (instead of the ones-MFMA): -0.5 to -2.4 %. dot2c shares the MFMA pipe
  (28 cycles after an MFMA, 4 after another dot2c), so it does not offload anything. First version
  produced inf in qt0 rows: cvts sank next to the asm P*V MFMAs, and the hazard recognizer does not see
  inline asm, so no VALU -> MFMA wait states were inserted (fixed by pinning through side-effect asm).
- Row sum on `v_pk_add_f32` partial sums: -1.5 % (B1 S16K 1224.4 vs 1242.6, dsv3 1005.6 vs 1023.9),
  VGPRs 512 with 6 spills (f32 P live longer), not bitwise.
- IGroupLP `sched_group_barrier` cannot see inline asm, so the asm P*V site is hand-placed with
  `sched_barrier(0)` after each slot; `llvm_options` scheduler knobs were inert.

## Real-training transfer check
- Bucket: K1 (kernel-internal wave layout and instruction schedule).
- Cache key (if any): no cache.
- id(...) audit: no tensor identity used; the kernel reads fresh Q / K / V each call.
- 4-step pen-and-paper trace hit rate: n/a (no cache).
- Workload distribution robustness (GroupGemm / MoE only): n/a (dense causal attention).
- Benchmark gain on this round: +3.05 % geomean forward TFLOPS (forward-only campaign).
- Estimated real-training gain: +3.05 % of MLA forward kernel time. Output is bitwise identical, there
  is no state across calls, and the launch shape set is the training shape set, so the gain is 1:1.
- Decision: ACCEPT-as-real.

## Next direction
The loop is at the issue-slot limit, so cut non-MFMA issue time: `s_setprio` (meaningless with one
wave per SIMD, 49 cycles per tile), hazard `s_nop`s, AGPR read-backs of spilled LDS addresses, merged
`s_waitcnt`s. Then the per-WG fixed cost that keeps the 4K shapes flat: Q-load prologue and the causal
drain imbalance.
