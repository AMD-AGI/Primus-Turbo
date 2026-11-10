# Round 5 — ROLLED BACK — 2026-11-09 23:48 — drain VMEM once before the unmasked q-loop

## Hypothesis
Round-3 ATT (B1 S4K) put 15 % of the main kernel in the Q/dO commit, mostly `s_waitcnt vmcnt(7)`
before each `ds_write_b128` plus one `vmcnt(8)` before the (-delta, lse) `ds_write2_b32`. Reading
the ISA: the unmasked q-loop has two entry edges (prologue / masked-loop exit) where the first
Q/dO + (-delta, lse) fetch is the newest VMEM traffic, so the compiler sizes those waits for that
edge. On the latch path the fetch has already been retired by the body's `vmcnt(12)`, and the
waits instead retire the previous head-step's 12 dQ atomics (gfx950 vmcnt is in-order). Draining
once on entry (`s_waitcnt vmcnt(0)`) should let the compiler drop the commit's waits, so the
atomics stay in flight across the commit. Expected: most of the commit's wait time recovered.

## Single change
New builder flag `u_entry_drain` (U_ENTRY_DRAIN: Q_PREF and QDESC only) emitting
`rocdl.s_waitcnt(0x0F70)` (vmcnt(0) alone) between the masked and unmasked QDESC loops;
`ops_bwd._dkdv_kwargs` sets it True.

## ISA check (dev/isa_r5)
Exactly as predicted: the unmasked commit's six `vmcnt(7)` and one `vmcnt(8)` are gone, one
`vmcnt(0)` added at loop entry, nothing else in the loop's VMEM/wait sequence changed. 432 VGPR,
0 spill in both.

## Results
Same-session A/B (dev/sweep.py, 11 repeats; artifacts/sweep_r5.raw): B2 S4K 5.579 -> 5.673 ms
(-1.65 %), B1 S16K 39.676 -> 41.491 ms (-4.37 %); SNR identical.

ATT B1 S4K, target CU 1 (profiles/att_r5_base vs att_r5_drain): total instruction latency
19.89M -> 18.37M cycles (-7.6 %); commit segment 22.0 % -> 10.5 %; the waits move to the
atomics (`buffer_atomic` issue stall 0.63M -> 1.12M) and the body's `vmcnt(12)` (~0 -> 1.2 %).

Clock check (artifacts/clock_ab.txt; GRBM_GUI_ACTIVE + kernel trace, 2 repeats each): the clock
is unchanged (B1 S16K 2.349 / 2.344 GHz base vs 2.341 / 2.342 GHz drain), so the power-cap
explanation is ruled out. At B1 S16K the dkdv kernel takes +4.2 % GRBM cycles (7.31e8 -> 7.61e8)
and 38.9 -> 40.6 ms. At B1 S4K the two are equal within noise (2.735-2.785 vs 2.734-2.743 ms).

Full set (n02-29 GPU 0, rotated vs aiter a16), 7 / 7 PASS:

| shape | Step TFLOPS | delta vs R4 | Bwd TFLOPS | Bwd ms | SNR dq/dk/dv |
|---|---|---|---|---|---|
| b1_s4k | 707.4 | -0.555 % | 628.9 | 2.842 | 46.43/49.57/51.21 |
| b1_s8k | 758.0 | -2.915 % | 666.5 | 10.724 | 45.18/49.54/51.27 |
| b1_s16k | 789.0 | -3.321 % | 690.1 | 41.425 | 43.99/49.66/51.08 |
| b1_s32k | 807.2 | -2.976 % | 703.1 | 162.645 | 44.07/49.62/51.16 |
| b1_s64k | 816.91 | -2.983 % | 710.37 | 643.894 | 44.39/49.60/51.16 |
| b2_s4k | 711.10 | -1.000 % | 632.44 | 5.652 | 46.39/49.66/51.38 |
| b2_s16k | 780.04 | -4.309 % | 680.38 | 84.039 | 43.83/49.54/51.31 |

Geomean: Step 766.00 (-2.588 % vs R4; ratio vs aiter 0.7705), Bwd 672.44 (ratio 0.7232).

## Decision
ROLLED BACK: aggregate -2.588 %. Kernel and host restored from round-4's snapshot.

## Attribution
Measured: the commit's waits were not dead weight. Removing them moves the stall onto the atomic
issue and the body's fetch wait, and on the whole device the kernel needs more cycles at long S
at an unchanged clock. A single-CU ATT at S4K showed the opposite sign, so it is not
representative of the long shapes. Inference, not measured: retiring the previous step's
atomics before the next Q/dO fetch keeps those loads from queuing behind atomics in the same
in-order memory path; with the waits gone the fetch issued at the commit sits behind up to 12
atomics that the 16 same-rotation work-groups per XCD are adding into the same dQ rows.

## Real-training transfer check
- Bucket: K1 (instruction schedule)
- Cache key (if any): no cache
- id(...) audit: n/a
- 4-step pen-and-paper trace hit rate: n/a
- Benchmark gain on this round: -2.588 % combined-step
- Estimated real-training gain: n/a (rolled back)
- Decision: REJECT (regression)

## Next direction
The atomic latency itself is the lever, not the waits on it. Under the XCD-major decode each XCD
runs 32 consecutive bands of one head, and the descending q walk (QDESC, rotation band % 2) puts
~16 of them on the same q-block, i.e. the same dQ rows, at the same trip. Next: widen the
rotation (QDESC_R 2 -> 4 / 8) to spread the atomics over more q-blocks, at the cost of some
Q/dO L2 sharing. Profile at B1 S8K-S16K, not S4K.
