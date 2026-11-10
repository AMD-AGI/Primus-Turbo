# Round 4 — ACCEPTED (trade-off) — 2026-11-09 23:20 — config sweep, g3_st_at 0 -> 1

## Hypothesis
Round-3 ATT at B1 S4K put 15 % of the main kernel in the Q/dO commit, mostly `s_waitcnt vmcnt(7)`
before each `ds_write_b128`. One possible cause: vmcnt retires in order, so the dQ atomics still
queued from the deferred GEMM3 sit ahead of the Q/dO loads. The existing knobs that move those
stores (g3_st_at / g3_st_n) and the other schedule knobs had not been re-tuned since the MLA
generalization, so this round is a one-knob-at-a-time sweep of them (config only, one round).

## Sweep (dev/sweep.py, B2 S4K + B1 S16K, rotated, 5 repeats; artifacts/sweep_r4.raw)
All configs PASS the 40 dB gate at B2 S4K heads 0 / 127 except where noted. gm = geomean of the two.

| config | B2 S4K d% | B1 S16K d% | gm d% |
|---|---|---|---|
| g2d 1 / 3 / 4 | +0.28 / +0.35 / +0.39 | -0.04 / -0.12 / -0.00 | +0.12 / +0.12 / +0.19 |
| g3d 1 / 2 / 3 | +0.32 / +0.08 / +0.20 | -0.11 / +0.15 / +0.03 | +0.10 / +0.11 / +0.11 |
| mfma_tie 1 / 2 | +0.59 / +0.67 | +0.09 / +0.05 | +0.34 / +0.36 |
| g3_dbat 6 | +0.22 | -0.09 | +0.06 |
| g3_st_at -1 (burst) | -1.98 | -5.07 | -3.54 |
| g3_st_at 1 | -0.12 | +1.25 | +0.56 |
| g3_st_at 2 | -1.54 | +0.48 | -0.54 |
| g3_st_n 2 / 3 | +0.14 / -1.37 | -1.26 / -3.48 | -0.56 / -2.43 |
| g3_sb 0 | -0.54 | +0.54 | 0.00 |
| iglp 0 / 3 | -0.36 / +0.10 | +0.67 / +0.03 | +0.15 / +0.06 |
| block_q 64 | -28.15 | -26.89 | -27.52 |
| block_kv 64 | -47.17 | -49.42 | -48.31 (dq 45.07 dB) |

- iglp 1: the process aborts in LLVM `AMDGPUIGroupLP.cpp:2220` (DSWCounters assertion, pre-RA);
  artifacts/sweep_r4_iglp1_abort.raw.
- block_q 16: builder assert (BLOCK_Q must cover one GEMM2 K-step).

Confirmation (11 repeats, artifacts/sweep_r4b.raw): g3_st_at 1 -0.21 / +1.34 (gm +0.56),
mfma_tie 2 +0.48 / -0.06 (gm +0.21), both together -0.08 / +1.24 (gm +0.58). Only g3_st_at 1
reproduces above the repeat spread; mfma_tie adds nothing on top.

## Single change
`ops_bwd._dkdv_kwargs` default `g3_st_at` 0 -> 1 (the deferred dQ stores start spreading at phase
boundary 1 instead of 0, one per site). Kernel source unchanged (snapshot diff: ops_bwd.py line 38).

## Results (n02-29 GPU 0, full set, rotated vs aiter a16) — 7 / 7 PASS

| shape | Step TFLOPS | delta vs R3 | Bwd TFLOPS | Bwd ms | step ratio vs aiter | SNR dq/dk/dv |
|---|---|---|---|---|---|---|
| b1_s4k | 711.35 | -0.523 % | 632.78 | 2.825 | 0.821 | 46.43/49.57/51.21 |
| b1_s8k | 780.78 | +0.811 % | 690.73 | 10.348 | 0.799 | 45.18/49.54/51.27 |
| b1_s16k | 816.11 | +0.982 % | 718.69 | 39.779 | 0.789 | 43.99/49.66/51.08 |
| b1_s32k | 831.94 | +0.059 % | 728.65 | 156.938 | 0.766 | 44.07/49.62/51.16 |
| b1_s64k | 842.02 | +0.304 % | 736.09 | 621.398 | 0.763 | 44.39/49.60/51.16 |
| b2_s4k | 718.28 | -0.394 % | 639.72 | 5.587 | 0.808 | 46.39/49.66/51.38 |
| b2_s16k | 815.17 | +1.035 % | 717.54 | 79.687 | 0.778 | 43.83/49.54/51.31 |

Geomean: Step 786.352 (+0.323 % vs R3; aiter 996.94, ratio 0.7888), Bwd 693.741 (ratio 0.7439),
Fwd 1205.115 (unchanged kernel, +0.218 % noise). dK/dV bitwise repeat True on every shape; SNR
unchanged to 0.01 dB.

## Decision
ACCEPTED (trade-off): aggregate +0.323 %; b1_s4k -0.523 % and b2_s4k -0.394 % (both < 5 %, same
sign as in the sweep), 8K / 16K shapes +0.8-1.0 %.

## Attribution
Measured: the gain is at 8K-16K and absent at 32K-64K; the 4K shapes lose ~0.5 %. Not measured:
whether the Q/dO commit's vmcnt waits actually shrank (no ATT this round). The small aggregate
means the store placement is not the main source of the commit-wait time; this is an inference.

## Real-training transfer check
- Bucket: K1 (launch configuration)
- Cache key (if any): no cache
- id(...) audit: n/a
- 4-step pen-and-paper trace hit rate: n/a
- Benchmark gain on this round: +0.323 % combined-step
- Estimated real-training gain: +0.323 % (no dependence on tensor reuse)
- Decision: ACCEPT-as-real

## Next direction
The schedule knobs are exhausted (all within +-0.6 % except the ones that lose). The remaining gap
(bwd 0.744 of aiter) needs code changes in the main kernel. Candidates from the round-3 profile:
LDS bank conflicts (8.4 % of LDS-active cycles, aiter 0), the Q/dO commit (15 % at S4K: in-order
vmcnt waits and per-load address VALU), SALU (2.3x aiter's instruction count).
