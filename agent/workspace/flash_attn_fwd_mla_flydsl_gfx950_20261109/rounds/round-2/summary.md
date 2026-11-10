# Round 2 — ROLLED BACK

- Time: 2026-11-09 09:02 (UTC+8)
- Hypothesis: in the QK clusters (C1/C5, epilogue C1/C5/C9) the D128 hints anchor only 16 of the
  24 QK MFMAs, and the 16 tail K reads (k_head 4) are issued just ahead of their MFMAs, so each
  MFMA waits on LDS latency (ISA: `s_waitcnt lgkmcnt(2|0)` before nearly every QK MFMA).
- Change (single variable): for `split_k_reads and K_STEPS_QK > 8`, replace the
  `(6,3,EXP)+(10,5,VALU)` pairs with one MFMA group per QK MFMA and one DS-read group after each
  of the first 16, so tail read r leads its consumer by 2*K_HEAD MFMAs. D128 path untouched.
  Snapshot: `kernel_snapshot/`.

## Evidence before the change (measured)

- Counters, B1 S16K: proto 11.02 ms vs aiter 9.52 ms; MFMA busy cycles proto/aiter 1.04;
  MFMA_BUSY/BUSY_CU 2.40 vs 3.44; SQ_WAIT_ANY/WAVE_CYCLES 0.345 vs 0.245; LDS bank conflicts 0 / 0;
  scratch 420 vs 0.
- ATT, B1 S4K, one CU: s_barrier 21.9 %, MFMA 21.4 %, s_waitcnt 20.8 % of wave latency;
  main loop ~60 %, prologue ~12 %, epilogue ~27 %.

## Result

ISA after the change: `.vgpr_count 256`, `.vgpr_spill_count 105` (unchanged); the tail reads stay
just-in-time. Root cause seen in the ISA: the tail K reads rotate through only three register quads
(`v[184:187]`, `v[200:203]`, `v[204:207]`), so post-RA false dependencies pin each read behind the
MFMA that frees its register. The hints cannot move them.

Correctness: all 7 PASS (O rel 0.0016–0.0020, LSE max 0.0046–0.0051).

| shape | round-2 | baseline | delta |
|---|---|---|---|
| b1_s4k | 794.1 | 800.8 | -0.8 % |
| b1_s8k | 901.1 | 911.4 | -1.1 % |
| b1_s16k | 978.9 | 987.3 | -0.9 % |
| b1_s32k | 990.6 | 1002.6 | -1.2 % |
| b1_s64k | 1010.0 | 1020.7 | -1.0 % |
| b2_s4k | 786.4 | 780.9 | +0.7 % |
| b2_s16k | 963.8 | 961.5 | +0.2 % |

Geomean 913.58 vs 919.04 (-0.59 %); aiter 1138.82; ratio 0.8022.

Decision: ROLLBACK (aggregate did not improve). Kernel restored from round-1 snapshot.

## Real-training transfer check

Bucket K1 (pure scheduling hints, no wrapper cache). Rolled back, nothing to transfer.

## Lesson

At the 256-VGPR cap (8 waves / CU) the QK tail is register-bound, not hint-bound. Next rounds must
free VGPRs (k_head 2, carried P as bf16, fewer live V packs) before read pacing can matter.
