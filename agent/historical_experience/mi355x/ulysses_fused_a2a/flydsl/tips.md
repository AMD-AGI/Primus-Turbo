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
- Wgrad inside a fused kernel: use the production variable-K tile (`grouped_gemm_bf16_variable_k_tile`,
  runtime chunked K loop, LDS chunk stride 1152) with `persistent="staggered"`, not the mega `tn`
  tile (fully unrolled K loop): ~3x slower per tile in-kernel. A `k_hook` per 256-row chunk gates
  the M walk on arrival flags (QKV: only the tile's source's flags).
- An exchange-free wgrad in a pull GemmA2A: dispatch its blocks shortly after the first GEMM wave
  (`num_lead + tiles / 8`); after all GEMM tiles they sit in front of the tail pulls.
- No side streams on MI355X: an in-kernel wgrad beat both the side-stream hipBLASLt (bwd o 245 / 433
  -> 228 / 419) and the stream-gated chunks of R6.
