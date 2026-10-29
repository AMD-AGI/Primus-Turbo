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

## Wave-specialized dK/dV (r9w, r9x, r9y): correct, but 0.80x of r8a -- closed

4-wave workgroups (one per CU, one wave per SIMD): producer waves run S/dP + softmax, consumer waves dV/dK + TDM + the
dS^T export; P/dS cross through a double-buffered LDS exchange. 665 VGPR, bitwise == hk_v0 at every shape.

| rung | change | step | Mcycles (r8a 3.05) |
|---|---|---:|---:|
| r9w | lockstep v1, 4-stage ring | 0.80x | 4.73 |
| r9x | producers pipeline B(k-1) with A(k) (mask as S C-init), consumers lag 2 and export, split barriers, 5 stages | 1.007x | 4.67 |
| r9y | TDM prefetch distance 3 (6 stages) | 1.003x | 4.69 |
| ws_nop / ws_noc | producers / consumers skip their math | | 4.04 / 4.66 |

Untraced per-step shader-cycle stamps (micro/ws_stamps.py, HKL_WS_STAMP, workgroup (0,0,0), stamps_r9y.log): one step
is ~1350 cycles. Consumer: reads issued +230, dV (24 WMMA) +335, tensor wait +77, dK (36 WMMA) + export +563; producer:
readback issued +181, B + A (60 WMMA) +705, then ~250 waiting. Each role spends 500-800 cycles per step outside its 60
WMMAs (memory-op issue, LDS latency, sync), where the 2-wave kernel hides the same costs among 120 WMMAs per wave
(r8a: ~837 cycles per 96-row step per CU). Splitting the chain across SIMDs at one wave per SIMD adds per-step fixed
cost faster than it removes the chain's stalls; even with the export moved back and reads prefetched, each role stays
above ~900 cycles. ATT is useless for this question (traced WMMAs take 11-18 cycles); the stamp probe is the tool.

## 2026-10-29: where r7/r8a's dK/dV iteration goes, and four schedule experiments (GPU3; r7 2.43 ms / 1265 MHz today)

Untraced per-phase probe (micro/k3_probe.py, HKL_PROBE: SALU-only s_get_shader_cycles sums of workgroup (0,0,0), the
production asm patch regenerated for the probe's .s; probe_r8a_pr.log, probe8.log): ~1566 cycles per full iteration.

| phase | cycles | WMMA floor (10.06 cyc each) | over |
|---|---:|---:|---:|
| loop top + hh0 S/dP + softmax (NL/ND loads, hh1 readback burst, TDM, 30 WMMA) | 455 | 302 | ~150 |
| hh1 S/dP + softmax (30 WMMA) | 342 | 302 | ~40 |
| pack / dO^T issue / dS^T export | 28 / 38 / 89 | | |
| dV + Q^T tr16 (24 WMMA) | 230-250 | 241 | ~0 |
| dK + split ring barrier (36 WMMA) | 340 | 362 | ~0 |
| readback hh0 + loop back | 89 | | |

micro/wmma_chain.cpp (wmma_chain.log): one wave per SIMD issues a WMMA every 10.06 shader cycles when its C was
written >= 2 WMMAs earlier, 15.9 when written by the WMMA right before it. The dV / dK runs are at that rate; the loss
is in the S/dP + softmax half (loop top, readback / NL/ND waits, softmax tail) and the export.

| rung | change | time step vs r8a | cycles | |
|---|---|---:|---:|---|
| r8b | asm patch v2 (msrc: Q^T units re-using the export stores' data registers re-targeted, vm_vsrc recomputed) | 1.003x / 1.000x (two runs) | 1.005x / 1.000x | noise: the store-data waits were cheap |
| r8c | S as two half-chains (no back-to-back dependent S/dP WMMA left in source order; the compiler still issues 20/trip) | 0.995x (0/12) | 0.998x | slower |
| r8d | every barrier waits tensorcnt 0 + hh0 readback of the next stage before the barrier (under dK) | 0.984x (0/12) | 0.964x | slower |
| r8d_tw0 | only the tensorcnt 0 waits (diagnostic) | 0.993x | 0.988x | TDM tiles issued one iteration earlier have not always landed |

All bitwise == hk_v0 (r8c: SNR unchanged to 0.01 dB). The LLVM-scheduled loop sits at a local optimum: moving its
waits / chains / readbacks around costs more than it saves; the remaining ~190 cycles per iteration in the S/dP half
need a hand-scheduled loop (the a08 ABC design). Production branch dev/lhz/dsv3-attn-hk-b1 a8fefb40 (rect32 dQ fix):
tests/pytorch/ops/test_attention_flydsl_gfx1250_mla.py 142 passed on GPU3 (new rect32 test included).
