# Ulysses fused GEMM + all-to-all on MI355X (FlyDSL) — tips

- Measure the pure exchange first (GemmA2A with K = 128, A2AGemm with N = 256): at 16K (25-31 MB
  off-rank per rank, n = 4) it is 217-228 us, already 1.05-1.25x of a 145 GB/s roofline because of
  ~25-40 us fixed launch / first-data / exit cost. Do not expect GEMM tuning to beat that.
- Remote writes as 16 B/lane contiguous row copies reach ~160 GB/s per rank (n = 4). Storing a GEMM
  tile straight to a peer from the epilogue (64 B row runs) is much slower (+15-30 %), and so are
  256 B row runs (one head piece per row): keep push/pull units as full rank-block rows.
- Pull (remote read) blocks are dispatched after the first GEMM wave; with 128-row tiles the first
  data is ready after ~30 us instead of ~55 us (256x256x2048 at the per-CU rate).
- A2AGemm: staggering each source's m-tile walk (rank r starts at m-tile r * m_tiles / n) lets a tile
  walk K in arrival order so only the last source's segment is left after the last arrival. Only
  worth it when every tile has a CU from the start; otherwise wait-all with 128-row tiles.
- Each extra flag wait inside the MMA loop costs a pipeline drain (`spin_flags` starts with
  s_waitcnt 0): poll one segment's flags together, never per K chunk.
- One sys-scope acquire fence per tile (first wait) is enough; an entry fence at kernel start costs
  10-30 us; wavefront-scope acquire orders the later segments.
- `persistent=True` (every wave barriers each tile) costs ~20 us vs the default wave-half offset;
  use `persistent="staggered"` (low half re-aligns at tile end).
- Before fusing an exchange-free GEMM (wgrad), time the op without it: bwd o's side-stream hipBLASLt
  wgrad was already hidden; in-grid wgrad tiles only added contention.
- `hipStreamWaitValue64` on a counter the kernel bumps (sys release) gates side-stream work with no CU
  spent waiting; it works, but concurrent hipBLASLt slows a CU-driven exchange by more than it hides
  when compute ~ exchange time.
- FlyDSL disk cache: `rm -rf /root/.flydsl` after kernel edits; env-var globals are not in the key.
- Sequence split as a row mapping: keep one helper (`common.seq_row`) shared by the producer (pull)
  and consumer (push) sides; a striped split then costs nothing (layer time within 0.2 %). Test it as
  the contiguous reference with the full sequence permuted (`index_select`), autograd included.
- Cold JIT of the full fused-a2a suite at n = 8 can exceed 30 min; give torchrun 1 h or warm the cache.
- Moving a FlyDSL kernel body into a helper function: decorate the helper with `@ASTRewriter.transform`,
  or its dynamic `if` / `while` are evaluated in Python at trace time (it compiles, runs fast, and is
  wrong). One `SharedAllocator` per kernel: callers that own it pass the view in (`_setup(lds=...)`);
  roles that need different LDS layouts go in an `fx.union`.
- A persistent attention (grid = #CUs, counter-claimed tiles in the tiled launch's order) is bitwise
  equal to the tiled launch and no slower; under 4-rank exchange load it was faster on the max rank.
