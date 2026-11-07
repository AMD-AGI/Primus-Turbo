# Ulysses fused GEMM + all-to-all — performance trend

Validation set: Qwen3-30B-A3B, Ulysses n=4, mbs=1, seq 16K and 32K, all four ops
(`benchmark/ops/training/bench_ulysses_overlap.py`, fused column, us, max over ranks).
Primary accept metric: **step sum** = sum of the 8 fused times (one fwd+bwd of the attention
projections at both lengths; lower is better). Roofline = max(FLOPs / 1241 TFLOP/s, off-rank bytes /
145 GB/s); target fused/roof <= 1.11 (90 % of roofline).

| Round | Status | Description | fwd QKV 16K/32K | fwd o 16K/32K | bwd o 16K/32K | bwd QKV 16K/32K | Step sum | vs Baseline | Key Finding |
|-------|--------|-------------|-----------------|---------------|---------------|-----------------|----------|-------------|-------------|
| 1 | BASELINE | pull GemmA2A + push A2AGemm (commit 63c6c077) | 271 / 479 | 269 / 435 | 277 / 430 | 418 / 691 | 3270 | — | exchange-bound; A2AGemm tail = one full 256x256 tile after the last arrival |
| 2 | ROLLBACK | split-K items for leftover A2AGemm tiles | — | — | — | — | — | — | single-segment items ~40 us + slow partial combine (not benchmarked on the full set) |
| 3 | ACCEPTED (trade-off) | A2AGemm: staggered sub-tile pushes + progressive K walk in arrival order | 270 / 481 | 235 / 437 | 275 / 431 | 372 / 722 | 3223 | step -1.4% time | 16K fwd o -13%, bwd QKV -11%; 32K bwd QKV +4.5% (<5%) |
| 4 | ACCEPTED | config: A2AGemm BM 128 when tiles outnumber CUs; GemmA2A BM 128 (+ fwd QKV 24 pull / 8 tail CUs) | 252 / 461 | 235 / 405 | 246 / 437 | 374 / 662 | 3072 | step -6.1% time | half-height tiles: earlier first pull, half the last-tile tail |
| 5 | ROLLBACK | bwd o wgrad tiles inside the GemmA2A grid (tn tiles) | — | — | 269-300 / 509-544 | — | — | — | the side-stream hipBLASLt wgrad was already hidden (dgrad alone 240 us) |
| 6 | ROLLBACK | bwd QKV wgrad by M-chunks on a side stream gated by hipStreamWaitValue64 | — | — | — | 457-662 / 746-933 | — | — | concurrent wgrad slows the exchange more than it hides |
| 7 | ROLLBACK | A2AGemm push units = one head piece, last segment consumed piece by piece | — | 271-287 / 415 | — | 381-445 / 655 | — | — | 256 B row runs push slower; per-piece flag waits drain the MMA pipeline |
| 8 | ROLLBACK | GemmA2A push mode: GEMM epilogue stores straight into the peer's bank | 286-326 / 518-569 | — | 293-349 / 525-567 | — | — | — | 64 B epilogue store runs over XGMI are far slower than 16 B/lane row copies |
| 9 | ACCEPTED | both wgrads inside the fused kernels (variable-K tiles; QKV: per-chunk m-tile waits; o: dispatched after GEMM tile lead + 1/8), no side stream | 255 / 462 | 238 / 410 | 227 / 416 | 323 / 556 | 2887 | step -11.7% time | the production variable-K wgrad tile is ~3x faster in-kernel than the unrolled mega tn tile that sank R5; layer fwd+bwd -2.2% / -1.6% |
