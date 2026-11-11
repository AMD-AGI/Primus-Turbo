# Round 4 — ACCEPTED — 2026-11-10 19:10 (UTC+8)

## Hypothesis
With one wave per SIMD (round 3) `s_setprio` has no partner wave to arbitrate against, but the 4
per KV tile still cost ~49 issue cycles (ATT: 12.3 cycles each, all stall). The loop is bound by the
issue slots in the MFMA shadows, so dropping them should give back ~0.7 % of loop cycles.

## Single change
`attn_helper_mla.py`: `DUALWAVE_SWP_SETPRIO = dualwave_swp_setprio and rows_per_wave != 64`.
`CODEGEN_REV` 1036. ISA: `s_setprio` 8 -> 0; VGPR 512 / 4 spill slots unchanged.

## Results (full set, 2 rotations, `artifacts/r4.csv`, base = round-3 snapshot, same session)

| shape | cand TFLOPS | cand ms | base (R3) TFLOPS | base ms | cand / base | O rel | LSE max | bitwise |
|---|---|---|---|---|---|---|---|---|
| dsv3 B2 S4K H128 | 1021.6 | 1.3457 | 1019.0 | 1.3491 | 1.0026 | 2.01e-03 | 4.75e-03 | yes |
| dsv2lite B8 S4K H16 | 1004.0 | 0.6846 | 1001.8 | 0.6861 | 1.0022 | 1.99e-03 | 3.17e-03 | yes |
| kimik2 B4 S4K H64 | 1016.2 | 1.3528 | 1016.4 | 1.3525 | 0.9998 | 1.97e-03 | 4.01e-03 | yes |
| kimik3 B2 S7168 H96 | 1150.8 | 2.7434 | 1148.1 | 2.75 | 1.0024 | 1.83e-03 | 4.73e-03 | yes |
| B1 S4K | 1011.0 | 0.6799 | 1007.8 | 0.682 | 1.0032 | 1.95e-03 | 4.60e-03 | yes |
| B1 S8K | 1163.5 | 2.3628 | 1159.2 | 2.3716 | 1.0037 | 1.80e-03 | 4.75e-03 | yes |
| B1 S16K | 1263.2 | 8.705 | 1257.5 | 8.7444 | 1.0045 | 1.75e-03 | 4.93e-03 | yes |
| B1 S32K | 1305.4 | 33.6931 | 1301.4 | 33.797 | 1.0031 | 1.61e-03 | 4.82e-03 | yes |
| B1 S64K | 1322.9 | 132.9822 | 1320.5 | 133.2232 | 1.0018 | 1.58e-03 | 5.14e-03 | yes |
| B2 S4K | 1022.7 | 1.3442 | 1020.7 | 1.3468 | 1.0020 | 2.01e-03 | 4.75e-03 | yes |
| B2 S16K | 1263.1 | 17.4112 | 1259.2 | 17.4652 | 1.0031 | 1.76e-03 | 4.82e-03 | yes |

Geomean cand 1133.69 vs base 1130.79 (+0.26 %), 11/11 PASS, 11/11 bitwise identical. Largest gain
B1 S16K +0.45 %; kimik2 -0.02 % (within the 0.1 % spread).

## Decision
ACCEPTED (+0.26 %, no regression).

## Attribution
About half of the ATT estimate (49 of 6855 cycles per tile = 0.7 %): the shapes are partly power-bound
(B1 S16K) or prologue / drain bound (4K), so loop cycles convert to time below 1:1.

## Real-training transfer check
- Bucket: K1 (instruction schedule). Cache key: no cache. id(...) audit: n/a.
- 4-step trace hit rate: n/a. Workload distribution: n/a (dense causal attention).
- Benchmark gain: +0.26 %. Estimated real-training gain: +0.26 % of MLA forward time (bitwise
  identical output, stateless). Decision: ACCEPT-as-real.

## Next direction
Other issue-slot waste in the loop: hazard `s_nop`s (136 cycles per tile), AGPR read-backs of spilled
LDS addresses (80), and the 76 `s_waitcnt`s (411).
