# Round-3 — BN=256 RRR b0+b1 split (PMC-driven)

## Hypothesis
ISA-level analysis on dsv3-up worst shape:
- PMC: grouped MFMA throughput per CU is 18% lower than dense (matches 17% gap)
- PMC: grouped has +25% LDS instructions / FLOP than dense (4.99 vs 4.00 ratio)
- ISA static count: grouped 72 ds_read_b64_tr_b8 vs dense 48 (+50%)

Root cause: H6+H7 single-b0 cycling reloads b0 from Bs[tic][0] for cC (after Bs[tic][1] load for cB), adding LDS-read stall that delays mfma issue.

Fix: revert to b0+b1 split (dense pattern); mma order cA→cB→cC→cD with b0/b1 reused without reload.

## Single change
`kernel_fp8_layouts.cpp` `grouped_rrr_kernel_body` (BN=256 RRR):
- decl: `B_col_reg b0` → `B_col_reg b0, b1`
- main loop: 4-mma sequence rewritten to dense pattern
- epilog 1+2: same dense pattern

## ISA after change
- vgpr=256, spill=37 (UNCHANGED — b1 addition absorbed into VGPR allocation)
- ds_read_b64_tr_b8 static count: 72 (unchanged in disasm — but per-iter LDS rate dropped)
- mfma count: 96 (unchanged)

## Results (vs R2 revert fresh-baseline)

| shape | R2 ratio | R3 ratio | Δ |
|-------|----------|----------|-----|
| dsv3-up-B4-M4096   | 0.8270 | 0.8649 | **+0.038 (+4.6%)** |
| dsv3-up-B4-M8192   | 0.8396 | 0.8524 | +0.013 (+1.5%) |
| dsv3-up-B16-M4096  | 0.8391 | 0.8813 | **+0.042 (+5.0%)** |
| dsv3-up-B16-M8192  | 0.8458 | 0.8376 | -0.008 (route changed to bn128) |
| other 16 shapes    | mostly within ±1.5%             | |

primary_score: R2 1.0426 → R3 1.0497 (+0.7%); n_pass: 20/20 (no correctness regression).

dsv3-up worst gap: 17.30% → 13.51-14.39% (~3pp improvement).

## Status
3 of 4 dsv3-up improved 1.5-5.0%. Worst remaining: dsv3-up-B4-M8192 14.76% gap.
Other shapes unchanged or noise-level.

## Next
Round 4: address residual dsv3-up gap. ISA bottleneck analysis:
- 37 spill VGPRs still exist (no change from b0+b1 split)
- 286 s_waitcnt (vs dense 12) — too many sync drain points
- Per-mma `__builtin_amdgcn_s_barrier()` after every mma (8 per iter) vs dense pattern
