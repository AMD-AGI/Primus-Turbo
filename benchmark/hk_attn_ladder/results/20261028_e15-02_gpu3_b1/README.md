# HK MLA backward, b1 campaign on e15-02 GPU3 (from 2026-10-28; GPU1 was retired after the 10-27 hang)

GPU3 runs r7 at 2.366 ms / 1300 MHz (GPU1: 2.472 ms / 1251 MHz); compare steps within one GPU only.

## Phase 1 screening (all bitwise == r7, toy -> s640 / b1_s4096 / prod tests first)

| experiment | file | result |
|---|---|---|
| cache policy: dS^T stores th NT / WB / RT_NT, Q / dO TDM th HT, dq dS TDM th NT / LU | cachepol.* | all within +-0.4% (null control 0.15%): no lever |
| WMMA operand-reuse hints: none in dK/dV (ablation), reuse_b on dq's 11-of-12 shared dS operands | reuse.* | -0.2% / 0.0%: register-file reads are not an energy lever |
| micro/wmma_toggle.cpp: pure-register WMMA loop, 1 wave / SIMD, randn | wmma_toggle.log | power-capped at ~1457 MHz (zeros 1978 MHz, 1.357x); sharing A or B between consecutive WMMAs: 1.015x only |

Consequences: randn WMMA alone hits the cap at ~2.9-3.0 PF/s, so dK/dV's 2.81 TFLOP have a power floor of ~0.95 ms
(measured 1.86 ms); the remaining ~49% of its energy is softmax VALU, LDS, TDM / HBM traffic and the static power
of stall cycles. Operand order and cache hints do not move it; removing stall cycles (ABC) and work does.

## r8a: dq rect32 fix (test_r8a.log)

The unrolled steady loop (r2d on) bursts row 2u + 2 with all pairs assumed stored; for the last steady unit that row
is the first masked unit's, whose stored pairs start mid-row when (Skv - Sq) % 64 == 32. r8a runs that unit as a
generic unit. rect32: dq 17.68 -> 52.51 dB (bitwise == hk_v0); every other shape bitwise == hk_v0 / r7.
