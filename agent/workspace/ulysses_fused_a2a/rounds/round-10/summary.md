# Round 10 — bwd QKV A2AGemm with wgrad: 16 push blocks instead of 32 (ACCEPTED)

1. Hypothesis: with the wgrad tiles in the launch (R9), bwd QKV is compute-bound, not link-bound.
   Trace (16K / 32K, R9 config): the push ends at 193 / 385 us (link roofline ~217 / 435), the
   kernel at 318 / 544; 160 wgrad tiles hold their CUs for the whole exchange (paced by arrival,
   ~45 % busy), the dgrad tiles get ~64 CUs, fall behind and finish 80-90 us after the last push.
   16 pushers should still fill the links and give 16 CUs to the tiles.
2. Change: `ulysses_a2a_gemm(num_push=None)`: 16 when `wgrad` is given, else 32 (fwd o unchanged).
3. Results:
   - Sweep (`e9_wgrad.py`, bwd QKV only, us 16K / 32K): push 4: 522 / 1039, 8: 349 / 635,
     12: 309 / 540, 16: 303 / 513, 20: 300 / 512, 24: 308 / 529, 32: 322 / 549. With 16 the push
     ends at 201-211 / 396-420 us (still link-bound); the tail after it is ~90 us.
   - Also tried (no change): wgrad claim position 32-256 (worse from 128 on), BM 256 (324 / 532),
     BM 64 (not supported by `gemm_bf16_tile`).
   - `bench_ulysses_overlap.py` fused us, A/B in the same session (R9 code -> R10, two R10 runs):

     | op | 16K | 32K |
     |---|---|---|
     | fwd QKV | 254 -> 256 / 256 | 467 -> 471 / 469 |
     | fwd o | 236 -> 236 / 237 | 411 -> 410 / 409 |
     | bwd o | 230 -> 240 / 238 | 425 -> 412 / 425 |
     | bwd QKV | 324 -> 302 / 301 | 559 -> 514 / 516 |

     Step sum 2906 -> 2841 / 2851 (-2.0 %). bwd o is unchanged code (GemmA2A); its 16K 230 vs 239
     is run-to-run noise (32K spans 412-425 on the same code). bwd QKV hidden 59 % -> 71 % (16K),
     72 % -> 86 % (32K).
   - Correctness: full suite n = 4, quick n = 2 / 4 / 8 passed.
4. Decision: ACCEPTED.
5. Attribution: freeing 16 CUs shortens the dgrad backlog; the push rate is unchanged. The rest of
   the gap is structural: in step (tiles + wgrad tiles > CUs, no staggered sources) the last m-tile's
   dgrad tile only starts after its last source lands and takes ~80 us (128 x 256 x 5120 on one CU).
6. Next direction: split the late dgrad tiles' K by source (fp32 partials, fixed-order reduction by
   the last finisher) so the tail after the last push is ~20 us; target bwd QKV <= 269 / 502 us
   (90 % hidden).

## Real-training transfer check
- Bucket: K1 (launch config inside the single fused kernel); no cache.
- Cache key (if any): no cache. id(...) audit: n/a. Trace hit rate: n/a.
- Benchmark gain on this round: step sum -2.0 % (bwd QKV -7 % / -8 %).
- Primus e2e (Qwen3-30B-A3B, 1 x 8 MI355X, CP 4 Ulysses, EP 8, GBS 16, mock data, ms/iter avg of
  iters 3-10, `e2e.txt`), R9 + R10 code: 16K base 3152.7 / fused 2804.4 (-11.0 %), 32K base
  6584.8 / fused 5974.0 (-9.3 %); R4 was 2897 / 6103 vs base 3147 / 6583 (-7.9 % / -7.3 %). Last
  loss fused / base 11.6505 / 11.6495 (16K), 11.6343 / 11.6347 (32K).
- Estimated real-training gain of this round alone: bwd QKV -22 / -44 us x 48 layers x 8 micro-batches
  = -8 / -17 ms per iter (~-0.3 %); R9 + R10 together measured -93 / -129 ms vs R4 e2e.
- Decision: ACCEPT-as-real.
