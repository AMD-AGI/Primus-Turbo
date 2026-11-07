# Round 9 — both wgrads inside the fused kernels, no side stream (ACCEPTED)

1. Hypothesis: the bwd QKV wgrad runs serially after the A2AGemm (85 / 156 us), the bwd o wgrad on
   a side stream. Both can be tiles of the same launch (MegaMoE-style block roles, no extra stream):
   the QKV wgrad tiles walk M in row order and wait per chunk for the m-tiles they read (one source's
   flags each), the o wgrad tiles need nothing. R5/R6 failed with the mega `tn` tile and a side
   stream; the production FlyDSL variable-K wgrad tile (`grouped_gemm_bf16_variable_k_tile`) runs
   96 / 170 us standalone for this shape vs ~350-410 us per tile for the unrolled mega `tn` tile
   inside the kernel.
2. Change:
   - `grouped_gemm_bf16_variable_k_tile`: optional `a_base` (A by address), `k_hook(c)` before the A
     loads of each K chunk (and `k_hook(-1)` for the prologue), `persistent="staggered"` (keeps the
     wave-half barrier offset, realigns at tile end). Production calls unchanged.
   - A2AGemm (`nn` only): `wgrad=(x, dw)`; `K / 256 * N / 256` variable-K tiles on the tile claim
     counter at position `wgrad_at` (default 0); each chunk waits for its m-tiles' flags of the tile's
     source; dqkv is no longer copied out. BM auto counts the wgrad tiles (-> 128, in step).
   - GemmA2A: `wgrad=(dy, o_gathered, dw)`; one variable-K tile per block, dispatched after GEMM tile
     `wgrad_at` (default `num_lead + gemm_tiles / 8`). Sweep (us, 16K / 32K): side stream 245 / 433,
     after all GEMM tiles 257 / 503, at 288-352 / 384-640 flat 228-230 / 419-423; 128-row wgrad tiles
     worse.
   - `ulysses_fused_a2a.py`: both backward paths call the fused wgrad (fallback when a dim is not a
     multiple of 256); `_side_stream` removed.
   - Tests: `s_local = 512` cases (QKV wgrad in the A2AGemm where K_r % 256 == 0), also poisoned.
3. Results (fused us, R4 -> R9, `bench_ulysses_overlap.py`):

   | op | 16K | 32K |
   |---|---|---|
   | fwd QKV | 252 -> 255 | 461 -> 462 |
   | fwd o | 235 -> 238 | 405 -> 410 |
   | bwd o | 246 -> 227 | 437 -> 416 |
   | bwd QKV | 374 -> 323 | 662 -> 556 |

   Step sum 3072 -> 2887 (-6.0 %); fwd ops unchanged code (noise). Correctness: full suite n = 4,
   quick n = 2 / 8 passed.
4. Decision: ACCEPTED.
5. Attribution: the QKV wgrad now overlaps the exchange (A2AGemm dgrad-only is 252 / 504, with wgrad
   323 / 550, vs serial 373 / 660); the o wgrad no longer competes from another stream.

## Real-training transfer check
- Bucket: K3 (single-launch fusion); no cache.
- Cache key (if any): no cache. id(...) audit: n/a. Trace hit rate: n/a.
- Benchmark gain on this round: step sum -6.0 %.
- Layer (`bench_ulysses_a2a.py --fused`, one layer attn + MLP, max rank, A/B back to back):
  16K fwd+bwd 1386 + 2891 = 4277 -> 1382 + 2803 = 4185 us (-2.2 %); 32K 3666 + 7954 = 11620 ->
  3634 + 7801 = 11435 us (-1.6 %). Critical-rank bwd QKV 643 -> 510, bwd o 483 -> 436 us (32K).
- Estimated real-training gain: per layer -92 / -185 us x 48 layers x 8 micro-batches per rank
  (GBS 16, DP 2) = -35 / -71 ms of the e2e fused 2897 / 6103 ms/iter, ~-1.2 % at 16K and 32K;
  the e2e run is the confirmation.
- Decision: ACCEPT-as-real.
