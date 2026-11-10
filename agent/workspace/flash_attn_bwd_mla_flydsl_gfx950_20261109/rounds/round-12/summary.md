# Round 12 — ROLLED BACK — 2026-11-10 02:38 — kv tile dispatch order (kv_order)

## Hypothesis
The re-profile after rounds 10-11 (artifacts/att_r11_wg_gaps.txt) shows 643K idle cycles between
work-groups on the traced CU (3.4 % of its window) for r9, against 0.2 % for aiter, whose WGs
all last the same (paired kv tiles). Our causal WGs last 0.23-1.56M cycles and each XCD receives
them long-to-short per kv head. If in-order dispatch stalls on a full XCD, a different tile order
should cut the gaps.

## Single change (config sweep = one round)
`flash_attn_bwd_mla.py`: builder knob `kv_order` in the XCD-major block-id decode (a bijective
remap of the kv tile index; the kernel body is unchanged).
- 0: tiles in order within a kv head (r9).
- 1: zigzag within the head (0, N-1, 1, N-2, ...).
- 2: tile axis outermost (all heads' tile t before any tile t+1: longest first over the grid).
Snapshot: kernel_snapshot/.

## Correctness
SNR gate passed: dQ 47.01 (zigzag) / 46.92 (tile-major) dB vs 47.00; dK 49.74 and dV 51.28 dB
identical (dQ atomics only change order).

## Results (dev/sweep_proc.py, one process per config, 3 rotated rounds; artifacts/sweep_r12.raw)

| config | B2 S4K ms | d% | B1 S8K ms | d% | B1 S16K ms | d% | gm d% |
|---|---|---|---|---|---|---|---|
| kv_order 0 (r9) | 5.206 | +0.00 | 9.042 | +0.00 | 33.330 | +0.00 | +0.00 |
| kv_order 1 zigzag | 6.441 | -19.18 | 11.135 | -18.80 | 44.286 | -24.74 | -20.95 |
| kv_order 2 tile-major | 5.477 | -4.95 | 9.161 | -1.30 | 34.111 | -2.29 | -2.86 |

Full set not run: no candidate.

## Decision
ROLLED BACK. Sources restored to rounds/round-9/kernel_snapshot (cmp-identical). The re-profile
was taken after round 11, so this is rollback 1 of a new run.

## Finding
The r9 order is the best of the three. Tile-major is the longest-first order over the whole grid
and still loses 1-5 %, but it also spreads each XCD's concurrent WGs over different kv heads
(no shared Q/dO rows in L2), so the dispatch-gap hypothesis is neither confirmed nor excluded.
Zigzag keeps the head locality and loses 19-25 %; not diagnosed (concurrent tiles j and N-1-j
share no q rows, and their a16 image index is kv_tile % n_img).

## Real-training transfer check
- Bucket: K1 (grid decode / dispatch order)
- Cache key (if any): no cache
- id(...) audit: n/a
- 4-step pen-and-paper trace hit rate: n/a
- Benchmark gain on this round: -2.86 % (best variant, sweep, 3 shapes)
- Estimated real-training gain: 0 % (rolled back)
- Decision: ROLLBACK

## Next direction
The steady-state tile: 5101 cycles per wave per 192x32 tile against aiter's 4012-4352. Largest
concentrated item: the two GEMM1a vmcnt(0) (349 cycles per tile) waiting on spill reloads of
three K packs issued just before use.
