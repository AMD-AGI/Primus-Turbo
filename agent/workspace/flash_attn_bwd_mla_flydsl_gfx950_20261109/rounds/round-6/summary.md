# Round 6 — ROLLED BACK — 2026-11-10 00:05 — descending q-walk rotation period sweep (QDESC_R)

## Hypothesis
Under the XCD-major decode each XCD runs 32 consecutive bands of one head, and the descending q
walk rotates the start by band % 2, so ~16 work-groups add into the same dQ rows at the same trip.
If same-address atomic contention is what makes the dQ atomics slow to retire, a longer rotation
period spreads them over more q-blocks (at some cost in Q/dO L2 sharing).

## Single change (config sweep, one round)
New builder knob `qdesc_r` (None = the existing 2); swept 1 / 2 / 4 / 8 / 16 / 32.

## Results (dev/sweep.py, 7 repeats; artifacts/sweep_r6.raw)

| qdesc_r | B2 S4K d% | B1 S16K d% | gm d% |
|---|---|---|---|
| 2 (default) | 0.00 | 0.00 | 0.00 |
| 1 | -0.04 | -0.07 | -0.06 |
| 4 | -0.12 | +0.13 | +0.01 |
| 8 | +0.05 | +0.12 | +0.09 |
| 16 | +0.09 | +0.05 | +0.07 |
| 32 | +0.11 | -0.02 | +0.05 |

All PASS the 40 dB gate; dQ SNR moves by 0.01 dB (summation order changes, so the knob took effect).

## Decision
ROLLED BACK: no setting above the repeat spread, so there is no candidate to validate on the full
set (the full set was not run; the kernel would be round 4's). Knob removed, round-4 code restored.

## Attribution
Measured: spreading the start over up to 32 q-blocks does not change the time. Same-address dQ
contention between the bands of one XCD is not what limits the atomics.

## Real-training transfer check
- Bucket: K1 (launch configuration)
- Cache key (if any): no cache
- id(...) audit: n/a
- 4-step pen-and-paper trace hit rate: n/a
- Benchmark gain on this round: none
- Estimated real-training gain: n/a (rolled back)
- Decision: REJECT (no gain)

## Re-profile after two consecutive rollbacks (round-4 code)
ATT, B1 S8K, target CU 1 (profiles/att_r6_s8k ours, profiles/att_r6_aiter_s8k aiter a16):

| | ours | aiter |
|---|---|---|
| total instruction latency | 67.2M | 46.4M |
| of which stall | 30.7M (46 %) | 11.1M (24 %) |
| issue (latency - stall) | 36.5M | 35.3M |
| waitcnt stall | 12.0M | 1.54M |
| ds_read stall | 6.66M | 2.32M |
| mfma stall | 3.61M | 1.71M |
| barrier stall | 2.41M | 1.14M |

The issue cost is the same as aiter's; the gap is stall. Ours: the Q/dO commit is 22 % of the
time, its vmcnt waits retire the previous head-step's dQ atomics (6.4 % at the wait that retires
the last atomic group, issued at the end of the body). 70 `v_accvgpr_read` per head-step (7.5 %)
read loop-invariant LDS addresses and data blocks parked in AGPRs.

aiter's loop (unrolled x3): q tile 16, 6 atomics per wave per iteration spread over the compute,
one `vmcnt(6)` per iteration (the current iteration's atomics stay in flight), Q/dO fetched with
`buffer_load ... lds` straight into a 160 KB LDS. Code object: 256 threads, 512 VGPR (arch +
acc), 163840 B LDS, so one wave per SIMD like ours (432 VGPR, 106752 B LDS).

PMC, B1 S16K (profiles/pmc_r6/summary.txt):

| counter | ours | aiter |
|---|---|---|
| TA_BUFFER_ATOMIC_WAVEFRONTS | 2.037e8 | 1.361e8 |
| TCC_ATOMIC (= TCC_EA0_ATOMIC: every atomic goes to EA) | 8.116e8 | 5.442e8 |
| TCC_EA0_WRREQ_STALL | 3.72e7 | 3.9e5 |
| TCC_EA0_RDREQ | 8.9e7 | 4.0e8 |
| TCP_TCC_READ_REQ / mean latency | 6.98e8 / 367 cyc | 4.69e8 / 420 cyc |

The atomic wavefront count is exactly 2/3 of ours for the same dQ volume. aiter's own config
table (`hsa/gfx950/fmha_v3_bwd/fmha_bwd_dqdkdv.csv`) lists the kernel as `ts_qo=16, ts=192`:
a 192-row kv tile (128 here), so one third fewer (band, q-row) partials and dQ atomics. Every
atomic is executed past L2 (EA) on both kernels; ours also stalls the EA write path 100x more.
Reads go the other way: our descending walk makes Q/dO L2 hits (EA reads 4.5x fewer than aiter).

## Next direction
A 192-row kv tile: one third fewer dQ atomics and Q/dO reads per unit of MFMA work, and the
per-head-step fixed costs (commit, barriers) amortized over 1.5x the work. Risk: +80 accumulator
registers per lane (dK/dV for 192 x 320 fp32 over 256 lanes = 240) on top of 432.
