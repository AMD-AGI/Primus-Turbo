# Performance trend — flash_attn_mla_fwd_r20_flydsl_gfx950_20261110

Forward-only campaign: Bwd columns are `-`, Combined Step TFLOPS = Fwd Avg (geomean over the 11-shape
full set). aiter 0.1.24.post1 opus geomean measured once in round 1: 1005.24 (round 2 run: 1004.70).
Target: 1.40x aiter = 1407 TFLOPS. Later rounds compare against the last accepted build in the same
session (no aiter re-run).

| Round | Status | Description | Fwd Avg TFLOPS | Fwd Peak TFLOPS | Bwd Avg TFLOPS | Bwd Peak TFLOPS | Combined Step TFLOPS | vs Baseline | Key Finding |
|-------|--------|-------------|----------------|-----------------|----------------|-----------------|----------------------|-------------|-------------|
| 1 | BASELINE | branch R20 (33beb4f8) on n14-17 | 1098.250 | 1232.600 | - | - | 1098.250 | — | 1.0925x aiter; forward-only; power-limited at ~1278 W (B1 S16K), loop MFMA busy ~71 % vs aiter ~92 % |
| 2 | ROLLBACK | QK / PV on 32x32x16 MFMA (from R1) | 1026.900 | 1102.100 | - | - | 1026.900 | step -6.497%, fwd -6.497%, bwd — | -9 % cycles/FLOP but -18 % clock at the same 1278 W: energy per FLOP decides |
| 3 | ACCEPTED | QT4 rewrite: 64 q rows / wave, 1 wave / SIMD, hand-scheduled QK + P*V sites (from R1) | 1129.250 | 1319.800 | - | - | 1129.250 | step +2.823%, fwd +2.823%, bwd — | same-session +3.05 % vs R20, bitwise identical; long S +5.8-7.3 % (clock +3 %, work/cycle +2.4 %), 4K +0.1-2.1 % (per-WG prologue / drain not hidden with 1 wave / SIMD); loop at the issue-slot limit (78 % MFMA busy) |
| 4 | ACCEPTED | no `s_setprio` with one wave per SIMD (from R3) | 1133.690 | 1322.900 | - | - | 1133.690 | step +3.227%, fwd +3.227%, bwd — | same-session +0.26 % vs R3, bitwise; ~half the 0.7 % ATT loop estimate converts (power / fixed-cost bound) |
| 5 | ROLLBACK | P*V DMA slot also runs a softmax piece, to fill the M0 hazard nop (from R4) | 1112.690 | 1307.200 | - | - | 1112.690 | step +1.315%, fwd +1.315%, bwd — | same-session -1.73 % vs R4: the DMA slot was already full; the added piece overflows the 12-cycle MFMA shadow |
| 6 | ACCEPTED (trade-off) | P*V site: V read one d chunk per slot, exp-free V slots, two-sided slot fences (from R4) | 1136.330 | 1323.900 | - | - | 1136.330 | step +3.468%, fwd +3.468%, bwd — | same-session +0.10 % vs R4 (noise level): 4K +0.3-0.6 %, long S -0.3-0.7 % (fewer cycles, -0.8 % clock at the power cap); spill slots 4 -> 8 |
| 7 | ACCEPTED | C5 QK's K head read in the C3 P*V site tail (from R6) | 1138.090 | 1321.900 | - | - | 1138.090 | step +3.628%, fwd +3.628%, bwd — | same-session +0.34 % vs R6, every shape up; one exposed LDS round trip per iteration gone |
| 8 | ROLLBACK | C1 QK's K head read in the C7 P*V site, carried over the back edge (from R7) | 1135.500 | 1321.500 | - | - | 1135.500 | step +3.392%, fwd +3.392%, bwd — | same-session -0.25 % vs R7: C1's wait was only ~20 cycles; +8 VGPR over the back edge adds AGPR spill traffic |
