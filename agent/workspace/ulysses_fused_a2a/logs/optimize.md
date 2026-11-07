# Ulysses fused GEMM + all-to-all — optimization log

## Setup
- Kernels: `primus_turbo/flydsl/ulysses/{gemm_a2a_kernel,a2a_gemm_kernel,common,workspace}.py`,
  shared tile pipeline `primus_turbo/flydsl/mega/bf16/gemm_bf16_kernel.py`, op wrappers
  `primus_turbo/pytorch/ops/attention/ulysses_fused_a2a.py`.
- Correctness: `torchrun --nproc_per_node 4 tests/pytorch/ops/test_ulysses_fused_a2a.py --quick`.
- Performance: `bench_ulysses_overlap.py --model qwen3-30b-a3b --seq 16384 32768` (no CUDA graphs).
- Fast loop: `~/.ulysses_dev/quick_a2a.py` (bench-style timing of selected ops),
  `trace_a2a.py` / `trace_ga.py` (s_memrealtime per tile / per unit timelines).

## Rounds 1-4 (accepted lineage: 1 -> 3 -> 4)
- R1 baseline: commit 63c6c077.
- R2 (rolled back): split-K items for the A2AGemm tiles that miss a CU; items too short, combine slow.
- R3: each source pushes `BM/8`-row units starting at m-tile `rank * D` (D = m_tiles / n), so an
  m-tile's sources land at n different times; a tile walks its K segments in arrival order, waiting
  per segment, with a sys acquire fence only on the first wait. Pipeline `persistent="staggered"`
  keeps the wave-half barrier offset across tiles. Falls back to wait-all when tiles > CUs - pushers.
  Commit 94d83cca.
- R4: config only, disjoint ops. A2AGemm: BM = 256 while every tile has a CU (progressive mode),
  else 128 (fallback: halves the last m-tile's compute tail; 32K fwd o 437->405, bwd QKV 722->662).
  GemmA2A: BM = 128 (first tiles ready after ~30 us instead of ~55 us), fwd QKV with 24 pull /
  8 tail CUs, bwd o keeps 16 / 16 (24 / 8 regressed its 32K by 10 %).

## Rounds 5-8 (rolled back, all on the R4 code)
- R5 bwd o wgrad inside the GemmA2A grid: GemmA2A dgrad alone is 240 us and the whole op with the
  side-stream hipBLASLt wgrad is 243 us, so the wgrad was already hidden; extra tiles only added
  contention (269 / 509).
- R6 bwd QKV wgrad by M chunks on a side stream, each gated by `hipStreamWaitValue64` on a counter the
  kernel bumps after copying the chunk's gathered rows: correct (rel err 4e-7 in fp32), but the
  concurrent GEMM slows the exchange: 16K 457 vs 403 serial with fp32 out. CU budget: dgrad + wgrad
  = 173 us of whole-GPU compute vs ~200 us of exchange, with 32 CUs pushing - no slack to hide it.
- R7 A2AGemm push unit = one head piece (BM rows x head_dim): the last segment could be consumed
  piece by piece (tail 30 -> 10 us) but tiles reached it 30-60 us later; per-piece `spin_flags`
  drain the prefetch (s_waitcnt 0) and 256 B row runs copy slower. Even one wait per segment with
  piece units was 274 vs 237.
- R8 GemmA2A push mode (epilogue stores into the peer's bank, local unpack): 286-326 vs 252 at 16K.

## Where the remaining gap is (R4)
- Exchange alone (GemmA2A with K = 128, GEMM ~free): 16K 31.5 MB in 228 us; A2AGemm with N = 256:
  25 MB in 217 us; RCCL a2a 25 MB 214 us. The roofline uses RCCL's 1 GiB rate (145 GB/s); at these
  sizes the fixed launch / first-tile / exit cost is 25-40 us, i.e. 1.05-1.25x roofline before any
  GEMM tail.
- fwd QKV 32K is at 1.06x. 16K ops pay first-tile latency (~30 us) + tail (~30 us) on a ~200 us
  exchange.
- bwd QKV: dgrad + wgrad compute (173 us at 16K) is comparable to the exchange; the wgrad needs the
  whole gathered dqkv and runs after the kernel (85 / 156 us serial).

## 2026-11-07 Round 9 ACCEPTED — both wgrads in the fused kernels, no side stream
- User constraint: no multi-stream overlap on MI355X; fusions follow MegaMoE (block roles in one launch).
- Wgrad tile = `grouped_gemm_bf16_variable_k_tile` (production FlyDSL MoE wgrad) with new `a_base`,
  `k_hook`, `persistent="staggered"`. The first try used the mega `tn` tile (fully unrolled K loop):
  350-410 us per 256x256x4096 tile in-kernel, op 423 vs 374. Variable-K tile: 180-260 us in-kernel
  (96 / 170 us for the whole wgrad standalone, hipBLASLt 84 / 149).
- A2AGemm: wgrad tiles at claim position 0 (sweep 16K: at 0 323, 64 324, 128 337; 32K: 0 550,
  128 565, 256 608); 128-row tiles worse (383 / 685). dgrad-only 252 / 504.
- GemmA2A: wgrad blocks after GEMM tile `num_lead + tiles / 8`; queued after all GEMM tiles they
  hold the tail pulls' CUs (257 / 503).
- Step sum 3072 -> 2887 (-6.0 %); layer fwd+bwd 16K 4277 -> 4185, 32K 11620 -> 11435.
- The R5 / R6 rollbacks were partly the tile, not the idea: R5 measured the mega `tn` tile.

## 2026-11-07 Round 10 ACCEPTED — bwd QKV: 16 push blocks with the wgrad tiles
- Trace of R9 bwd QKV: push ends 193 / 385 us, kernel 318 / 544. The 160 wgrad tiles hold their CUs
  through the whole exchange (paced by arrival, ~45 % busy); dgrad gets ~64 CUs, backs up, and the
  last m-tile's tile (in step: starts after its last source) takes ~80 us after the last push.
- Push-count sweep: 16-20 best (303 / 513, 300 / 512); push still link-bound (ends 201-211 / 396-420).
- Same-session A/B R9 -> R10: bwd QKV 324 -> 302, 559 -> 514; step sum 2906 -> 2841.
- Primus e2e (R9 + R10): 16K 3152.7 -> 2804.4 ms/iter (-11.0 %), 32K 6584.8 -> 5974.0 (-9.3 %).
- Remaining bwd QKV gap (~85 us at 16K): the in-step dgrad tail. Next: split late dgrad tiles' K
  by source with a deterministic fixed-order reduction.

## 2026-11-07 Rounds 11-12 ROLLED BACK — the in-step bwd QKV tail
- R11 split-K (one part per source, fp32 partials in scratch, never-reset counters in `state`,
  last part reduces in source order): correct but 330-411 / 570-637 us. Partials through memory
  move at ~10-15 GB/s per CU under load whatever the cache bits / fences (reduce 512 KB ~40 us,
  stores-only reduce 1-3 us).
- R12 late wgrad tiles: dgrad ends at push end + ~80 us (no backlog) but late wgrad tiles take
  115-134 us -> 352-357 / 661-688 us.
- Floor in step: push end + one 128 x 256 x 5120 dgrad tile = ~280 / ~480 us (82 % / ~97 % hidden).
  Below it: staggered sources (K walk by arrival) for the last m-tiles + rotated wgrad M walk, with
  ~95 CUs left to dgrad through the last ~80 us.

## 2026-11-07 Round 13 ROLLED BACK — staggered sources for bwd QKV
- Block-local stagger (blocks of n * D m-tiles) + K walk by arrival: correct; best 304 / 518 us
  (D=1, push_split 1) vs 302 / 515. After its last segment a dgrad tile needs 17-28 us, but ~10 %
  of the dgrad tiles are claimed at the last push and run whole. BM 256 tiles take ~155 us (no
  CU-us gain); push_split 8 at BM 128 stretches the push to ~286 us.
- Third rollback in a row: the bwd QKV tail is CU-us-bound (160 wgrad tiles pinned at ~45 % busy,
  dgrad needs ~96 CUs, gets ~80). Further gains need a cheaper dgrad tile under load or wgrad
  tiles that do not hold an idle CU.
