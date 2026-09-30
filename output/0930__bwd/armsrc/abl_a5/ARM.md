# abl_a5 -- ABLATION A5: +16 dQ-shaped WMMA per k_dkdv body (resident operands)

Question: what does 25% more matrix work (64 -> 80 WMMA/iter) cost inside the current
1-wave k_dkdv when the operands are already in VGPRs? This is stage A of w4_fused_dq with
no barrier, no w4, no atomics and no new loads.

## Change (kernels.py, vs arms/r29/kernels.py)
- L50-56: `A5_XWMMA = True` (switch; False reproduces r29), `NXA = 4`, `NSTX = NST + NXA`.
- L489: `new = [None] * NSTX`.
- L510-523: after each kh's dK GEMM, 2 chains (kb = 0,1) x 4 deep over dt:
  `x = wmma(a_ds[kh], kf[kb][dt], x)`, reuse hints off. 2 kh x 2 kb x 4 dt = 16 WMMA.
  A = the existing a_ds tr16 fragment, B = the loop-invariant kf fragments. No loads.
- L562, 587, 589, 608: carried-state slices `NST` -> `NSTX` (4 extra v8f accumulators).
- L621: init has NSTX zero accumulators.
- L656-668: sink after the loops: `s = X0[0]+X1[0]+X2[0]+X3[0]`,
  `out[0][0] += select(B_ == -7, s, -0.0)`.

Deviation from spec, deliberate: each chain is seeded from its CARRIED accumulator
(zero at loop entry) instead of from an inline zero every iteration. A from-zero chain
would need in-loop VALU adds to stay live (and under KV_U2 the first body's result would be
DCE'd), which would contaminate a pure-matrix-work ablation. The chains are still 4 deep
per body and use exactly the operands the fused dQ would use; the only difference is
C = register instead of C = inline 0 on the first link.

No index, address or predicate expression was changed (the new select is on B_ and picks
a value, not an address). No bounds proof needed; none written.

## ISA evidence (tools/compile.sh, dkdv; .dump/dkdv/k_dkdv_0/21_final_isa.s)
| | r29 | abl_a5 |
|---|---|---|
| vgpr_count | 729 | 761 |
| spill / scratch | 0 / 0 | 0 / 0 |
| LDS | 70656 | 70656 |
| WMMA total | 128 | 160 |
| .LBB0_4 (qloop_mask) WMMA / instrs | 64 / 653 | 80 / 702 |
| .LBB0_11 (qloop_full) WMMA / instrs | 64 / 660 | 80 / 699 |

Opcode diff per loop body is ONLY: +16 v_wmma, s_set_vgpr_msb +34 (mask) / +22 (full),
s_wait_dscnt -1 / +1. No v_mov back-edge copies, no VALU, no memory ops added.
Epilogue gets 3 v_add_f32 + s_cmp_eq_u32 s65,-7 + v_cndmask(0x80000000) + 1 v_add_f32.

Schedule note (full loop): LLVM clustered the extra WMMAs into the output-GEMM tail --
the tr16 reads were split (4 then 36), the tail WMMA run after `s_wait_dscnt 0x0` grew,
and the in-body s_wait_loadcnt 0x7/0x6/0x4/0x0 moved from slots 506/513/556/599 to
522/530/536/677 (the 0x0 now sits one WMMA before the back edge instead of five).
So the result measures the extra WMMAs as a serial tail, not interleaved with VALU.

## Bitwise vs r29
Expected bitwise IDENTICAL: B_ is never -7, so the sink adds -0.0, and x + (-0.0) == x
for every x. The dK/dV accumulation order is unchanged.
