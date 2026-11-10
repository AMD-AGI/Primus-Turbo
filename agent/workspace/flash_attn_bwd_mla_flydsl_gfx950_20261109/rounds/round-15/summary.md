# Round 15 — ACCEPTED — 2026-11-10 08:36 — descending q walk without the runtime modulo (qdesc_csub)

## Hypothesis
The hot loop's tail (after the last MFMA of an unmasked q-block) runs two modulos by the runtime
unmasked trip count `_u_nb`: `_desc_q(t) = _u_top - ((t + rot) % _u_nb) * step`, once for the
current q-block and once for the next one (the PF_QB prefetch). Each lowers to a 64-bit
fast/slow-path check plus a 32-bit reciprocal-based division. ATT of the r9 code (B1 S8K, CU 1;
artifacts/att_r11_loop_tail.txt): the two fast paths take 131 + 100 cycles and the 64-bit checks
and q arithmetic 73, out of a 5249-cycle iteration, with no MFMA in flight. Replacing the
modulo with a conditional subtract should recover most of that.

## Single change
`flash_attn_bwd_mla.py`: builder knob `qdesc_csub` (default False = the modulo, ISA byte-identical
to r13). On: `_rot % _u_nb` is hoisted to the band (computed once), and every `_desc_q(trip)`
becomes `i = trip + rot_m; if i >= nb: i -= nb` in 32 bits. Every caller passes trip <= nb, so
one subtract is exact, and the q order is unchanged. `ops_bwd._dkdv_kwargs` passes
`qdesc_csub=True`. Snapshot: kernel_snapshot/ (the cleaned source); the source that was timed,
with all variants below, is in artifacts/tested_source/. The cleaned source compiles to the
same ISA as the timed variant 6 (md5 9aff03ccb5) and, off, to the same ISA as r13 (67d9a2343c);
artifacts/isa_r15d_hashes.txt.

## Variants (ISA: artifacts/isa_r15.txt, isa_r15b.txt, isa_r15_detail.txt)
The first form was a plain select. It lost 43 %, so the same arithmetic was written five more
ways before timing again:

| mode | form | spill | scratch B | masked loop | hot loop | s_cbranch | gm (sweep) |
|---|---|---|---|---|---|---|---|
| 0 | modulo (r13) | 98 | 396 | 17 / 2 | 3 / 2 | 13 | 0 |
| 1 | 32-bit conditional subtract as a select | 132 | 484 | 22 / 6 | 35 / 27 | 9 | -43.26 % |
| 2 | as 1, result through an opaque scalar asm | 140 | 476 | 33 / 16 | 34 / 26 | 9 | not timed |
| 3 | as 1, loop invariants through opaque asm | 139 | 476 | 32 / 16 | 34 / 26 | 9 | not timed |
| 4 | two stride loops (rotation..end, 0..rotation) | 268 | 752 | 27 / 11 | 29 / 19 | 10 | not timed |
| 5 | 32-bit modulo (division kept, no 64-bit branch) | 157 | 520 | 27 / 6 | 28 / 16 | 9 | not timed |
| 6 | as 1, wrap as a real branch (scf.if; asm in the arm so SimplifyCFG cannot fold it) | 94 | 380 | 16 / 1 | 3 / 2 | 13 | +3.26 % |

(The 64-bit select of the first attempt: 134 spills, 492 B, hot loop 36 / 28.)
Mode 5 keeps the division and still jumps to 157 spills, and mode 6 keeps the branch count and
falls to 94: the r13 spill layout depends on the loop's block structure (the division's
fast/slow-path blocks), not on the division's arithmetic. Mode 6 hot loop: 877 -> 600
instructions, SALU 256 -> 25, no v_rcp / s_mul_hi.

## Correctness
- Full set 7 / 7 PASS in all three candidate runs; dK / dV SNR identical to the r13 reference,
  dQ within 0.007 dB (dQ 44.68-47.02, dK 49.54-49.66, dV 51.08-51.38 dB); dK/dV bitwise repeat
  True.
- artifacts/bitwise_r15.py: dK and dV bitwise equal to the modulo walk at B1 S1K / S2112 / S4K /
  S4224 / S8K and B2 S4K (S2112 and S4224 leave a partial last band). dQ vs the modulo walk
  49.67-61.81 dB, the same as the modulo walk against itself (50.03-62.40 dB,
  artifacts/dq_repeat_r15.txt): bf16 atomic order.
- D128 untouched: builder default off; only the MLA `ops_bwd` turns it on.

## Results
Sweep (dev/sweep_proc.py, 3 rotated rounds; artifacts/sweep_r15c.raw): B2 S4K +1.66 %, B1 S8K
+4.06 %, B1 S16K +4.09 %, gm +3.26 %.

Full set (n02-29 GPU 0, rotated vs aiter a16). Order: candidate run1, r13 reference
(`dev/bench_over.py '{"qdesc_csub": 0}'`), run2, run3.

| shape | bwd ms run1 | run2 | run3 | r13 ref (same session) | R13 run3 | Step run2 | d vs r13 ref | step ratio vs aiter |
|---|---|---|---|---|---|---|---|---|
| b1_s4k | 2.628 | 2.511 | 2.506 | 2.676 | 2.700 | 781.80 | +5.164 % | 0.905 |
| b1_s8k | 8.517 | 8.539 | 8.534 | 8.896 | 8.928 | 910.40 | +3.221 % | 0.932 |
| b1_s16k | 31.321 | 31.504 | 31.471 | 32.817 | 32.782 | 983.84 | +3.249 % | 0.952 |
| b1_s32k | 121.094 | 121.634 | 121.461 | 125.912 | 126.530 | 1021.00 | +2.719 % | 0.942 |
| b1_s64k | 482.622 | 483.034 | 483.850 | 503.964 | 504.441 | 1031.55 | +3.389 % | 0.935 |
| b2_s4k | 5.006 | 5.002 | 5.005 | 5.123 | 5.124 | 784.85 | +1.891 % | 0.885 |
| b2_s16k | 63.275 | 63.390 | 63.473 | 64.838 | 64.872 | 979.48 | +1.769 % | 0.934 |

Step geomean: run1 918.97 (ratio 0.9214), run2 922.13 (0.9263), run3 922.41 (0.9269); median
922.132 (+3.302 % vs R13 892.658; +3.052 % vs the same-session r13 reference 894.82, ratio
0.8984). Trend row uses run2 (the median): Fwd 1204.057 (kernel unchanged), Bwd 846.004 (ratio
0.9091), Step 922.132. Run1's b1_s4k (2.628 ms) is 5 % slower than runs 2-3 (2.51 ms); the other
shapes agree within 0.5 %.

## Decision
ACCEPTED: +3.05 % combined-step against the same-session reference, every shape +1.8..+5.2 %.

## Real-training transfer check
- Bucket: K1 (codegen: scalar index arithmetic in the q loop)
- Cache key (if any): compile-time knob, part of the launcher cache key via the builder kwargs
- id(...) audit: n/a
- 4-step pen-and-paper trace hit rate: n/a
- Benchmark gain on this round: +3.05 % combined-step vs same-session reference (median of 3)
- Estimated real-training gain: +1.8..+5.2 % on the MLA attention backward at these shapes. The
  q order and dK/dV bits are unchanged, so nothing about the data path differs in training; the
  spill count drops (98 -> 94) and the scratch stays 25 KB per wave (380 B / lane), so the
  round-8 16 / 32 KB scratch-layout hazard is not touched.
- Decision: ACCEPT-as-real

## Next direction
Re-run the ATT on this code: the loop tail no longer has the division, so the 798-cycle window
before the first MFMA (Q/dO commit, barriers, prefetch issue, GEMM3 operand reads) is the next
MFMA-idle item to measure.
