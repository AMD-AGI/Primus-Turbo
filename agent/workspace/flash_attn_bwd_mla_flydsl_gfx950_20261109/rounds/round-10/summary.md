# Round 10 — ROLLED BACK — 2026-11-10 02:13 — GEMM1a reads part or all of K from LDS (k_lds_ks)

## Hypothesis
Round 9 spills 98 VGPRs. In the q-loop a spill reload is a scratch_load, and on gfx950 its wait
is a vmcnt that also retires the dQ atomics and the Q/dO prefetch issued before it. Taking K's
band-resident register packs (72 VGPRs at NT 3) partly or wholly out of the register file should
remove the spills and with them those waits.

## Single change
`flash_attn_bwd_mla.py`: new builder knob `k_lds_ks` (MLA, K_REG only). GEMM1a's last k_lds_ks
D-steps read K from the LDS tile through one opaque per-lane base (`_opaque_idx`, ks as XOR,
nt as ds_read offset) instead of the register packs, and the dropped packs are never kept. The
LDS K tile then has to stay raw (no sm*log2e prescale), so `ops_bwd` passes dq_scale = sm to the
dQ reduce. The round-9 assert that MLA needs `k_reg=True` was lifted so that `k_reg=False` (all of
GEMM1a from LDS, the builder's existing per-(nt, ks) addresses) could be timed too.
Snapshot: kernel_snapshot/ (the tested sources).

## Resources (artifacts/isa_loop_counts.txt; dev/isa_r10, S 1024)

| config | spill VGPRs | scratch B/lane | masked loop scratch_load / vmcnt(0) | hot loop scratch_load / vmcnt(0) |
|---|---|---|---|---|
| r9 default | 98 | 396 | 17 / 4 | 3 / 2 |
| k_reg=False | 85 | 276 | 30 / 20 | 31 / 17 |
| k_reg=False, g1_ks_outer | 88 | 272 | 29 / 20 | 35 / 25 |
| k_lds_ks=1 | 76 | 276 | 31 / 24 | 19 / 15 |
| k_lds_ks=2 (=3, =4 identical counts) | 68 | 244 | 23 / 20 | 19 / 15 |
| k_lds_ks=6 (all of K) | 55 | 208 | 14 / 10 | 14 / 10 |
| k_lds_ks=1 / 2, g1_ks_outer | 70 / 71 | 248 / 252 | 24 / 19, 25 / 19 | 19 / 14, 19 / 15 |
| k_lds_ks=6, g1_ks_outer | 63 | 220 | 13 / 9 | 23 / 18 |

Fewer spilled VGPRs in total, but more reloads inside the loops. Round 9's in-loop reloads are
MFMA B packs (dwordx4); the k_lds_ks variants reload per-lane addresses instead (Q/dO prefetch
(row, col) pairs, ds_read / ds_write bases). With 72 VGPRs freed (k_lds_ks=6) the hot loop still
uses all 256 VGPRs and 256 AGPRs.

## Correctness
Every variant passes the sweep's SNR gate (> 40 dB, B2 S4K, heads 0 and 127). k_lds_ks variants:
dQ 47.69 / dK 49.74 / dV 51.28 dB (r9 default 47.00 / 49.74 / 51.28; the raw-K path rounds dQ
differently). k_reg=False: passed the gate; the partial sweep never reached its table, so the
values were not printed.

## Results (dev/sweep_proc.py, one process per config, 2 rotated rounds, n02-29 GPU 0)
artifacts/sweep_r10b.raw:

| config | B2 S4K ms | d% | B1 S16K ms | d% | gm d% |
|---|---|---|---|---|---|
| r9 default | 5.198 | +0.00 | 33.263 | +0.00 | +0.00 |
| k_lds_ks=1 | 8.007 | -35.08 | 54.356 | -38.81 | -36.97 |
| k_lds_ks=2 | 8.104 | -35.85 | 54.391 | -38.85 | -37.37 |
| k_lds_ks=6 | 7.244 | -28.24 | 50.297 | -33.87 | -31.11 |
| k_lds_ks=6, g1_ks_outer | 8.031 | -35.27 | 55.407 | -39.97 | -37.66 |

artifacts/sweep_r10_partial.raw (round 0 only; stopped during round 1 when the k_reg=False child
spent > 6 min of CPU recompiling after a source edit): r9 default 5.189 / 33.264 ms; k_reg=False
8.250 / 55.611; with g1_ks_outer 8.665 / 63.477; with g2d=2 9.281 / 62.306.
Full set not run: no candidate.

## Decision
ROLLED BACK. Every variant is 28-40 % slower. Sources restored to rounds/round-9/kernel_snapshot
(cmp-identical). Rollback count: 1.

## Finding
In this kernel the hot loop's vmcnt(0) count orders performance; the spill total does not.
k_lds_ks=6 has the fewest spills of the variants and the fewest hot-loop vmcnt(0) (10), and is the
least slow (-31 %); the 14-25 vmcnt(0) variants are -35..-40 %. Round 9's own 2 hot-loop vmcnt(0)
come from 3 pack reloads inside the previous head's GEMM3, about 200 instructions after the 8 Q/dO
prefetch loads for the next head-step (QPF_AT 0), so those waits also retire the prefetch.

## Real-training transfer check
- Bucket: K1 (register allocation / spill placement)
- Cache key (if any): no cache
- id(...) audit: n/a
- 4-step pen-and-paper trace hit rate: n/a
- Benchmark gain on this round: -31..-40 % (sweep, 2 shapes)
- Estimated real-training gain: 0 % (rolled back)
- Decision: ROLLBACK

## Next direction
Move the round-9 Q/dO prefetch issue point (QPF_AT, hard-coded 0 for D >= 128) past the GEMM3
reloads, so their waits no longer retire the prefetch: sweep 0 / 1 / 2 / 3.
