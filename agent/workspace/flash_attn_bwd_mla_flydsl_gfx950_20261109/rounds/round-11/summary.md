# Round 11 — ROLLED BACK — 2026-11-10 02:21 — Q/dO prefetch issue point sweep (qpf_at)

## Hypothesis
Round 9's two hot-loop vmcnt(0) are waits for three spilled MFMA B packs reloaded inside the
previous q-block's GEMM3, which runs right after the Q/dO prefetch for the next q-block
(QPF_AT 0). gfx950's vmcnt is in-order, so those waits also retire the prefetch about 200
instructions after it was issued. Issuing the prefetch later should leave it in flight.

## Single change (config sweep = one round)
`flash_attn_bwd_mla.py`: builder knob `qpf_at` (None keeps `0 if HEAD_DIM >= 128 else 1`).
Swept 0 (r9), 1 (after the previous q-block's GEMM3 at hook 0), 2 (before the last GEMM2 half),
3 (inside the last GEMM2 at d-tile 1). Snapshot: kernel_snapshot/.

## Resources (artifacts/isa_loop_counts.txt; dev/isa_r11, S 1024)

| config | spill VGPRs | scratch B/lane | masked loop scratch_load / vmcnt(0) | hot loop scratch_load / vmcnt(0) |
|---|---|---|---|---|
| qpf_at 0 (r9) | 98 | 396 | 17 / 4 | 3 / 2 |
| qpf_at 1 | 125 | 504 (32 KB per wave) | 32 / 16 | 16 / 11 |
| qpf_at 2 | 75 | 304 | 15 / 5 | 6 / 1 |
| qpf_at 3 | compile error: "operand #61 does not dominate this use" (artifacts/qpf3_compile_error.log) | | | |

qpf_at 3 issues the fetch inside the masked path's per-wave-class branches, so the loaded values
cannot leave the branch. In qpf_at 2 the six in-loop reloads are packs parked in a[56:79] (the
AGPRs the prefetch then stages into), issued right after the commit and waited about 240
instructions later.

## Correctness
SNR gate passed for 1 and 2 (dQ 47.01 / dK 49.74 / dV 51.28 dB; r9 47.00 / 49.74 / 51.28).

## Results (dev/sweep_proc.py, one process per config, 3 rotated rounds; artifacts/sweep_r11.raw)

| config | B2 S4K ms | d% | B1 S16K ms | d% | gm d% |
|---|---|---|---|---|---|
| qpf_at 0 (r9) | 5.210 | +0.00 | 33.308 | +0.00 | +0.00 |
| qpf_at 1 | 5.254 | -0.84 | 52.861 | -36.99 | -20.96 |
| qpf_at 2 | 5.328 | -2.22 | 33.668 | -1.07 | -1.65 |

Full set not run: no candidate.

## Decision
ROLLED BACK. Sources restored to rounds/round-9/kernel_snapshot (cmp-identical). Rollback count:
2 consecutive (rounds 10, 11) -> full re-profile before the next round.

## Finding
Removing one of round 9's two hot-loop vmcnt(0) (qpf_at 2) does not pay for the shorter
prefetch cover: the fetch issued before the last GEMM2 half loses 1-2 %. So the two remaining
waits in r9 are not the main cost of the steady-state loop. qpf_at 1's -37 % at S16K vs -0.8 % at
S4K is larger than its in-loop wait count suggests; its scratch (504 B) is the 32 KB per wave
layout that round 8 measured as slow (hypothesis, not checked here).

## Real-training transfer check
- Bucket: K1 (codegen: VMEM issue order)
- Cache key (if any): no cache
- id(...) audit: n/a
- 4-step pen-and-paper trace hit rate: n/a
- Benchmark gain on this round: -1.65 % (best variant, sweep, 2 shapes)
- Estimated real-training gain: 0 % (rolled back)
- Decision: ROLLBACK

## Next direction
Full re-profile of r9 (ATT stall by instruction class against aiter at B1 S8K, as after round 6),
since the code has moved through rounds 7-9 since the last one.
