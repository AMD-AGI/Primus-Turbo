# ulysses_a2a_attn_fwd (FlyDSL, gfx950) historical tips

Kernel: `primus_turbo/flydsl/mega_attn/a2a_attn_kernel.py` (fused Ulysses a2a + causal attention + a2a,
forward, symmetric-memory exchange over XGMI on one MI355X node). Campaign
`mega_attn_a2a_fwd_flydsl_gfx950_20261122`: fwd geomean over qwen3-30b-a3b n4 and llama3.1-70b n8 at
32K / 64K went 5111.8 -> 4414.9 us (-13.6%), overlap at 32K 42 / 39% -> 69 / 76%. Times below are UTC.

### 2026-11-22 01:41 round-1 — BASELINE (analysis)
- Context: n = 4 / 8 on one node, bf16, D = 128, seq_chunk 256, 32 pull blocks of a 256-block grid.
- Signal: qkv pull ~410 us at n4 and ~210 us at n8 (32K), ~50 GB/s per XGMI link per direction; with 192
  pull blocks 75% of the data was in flight at once and the KV prefix completed only in bulk.
- Takeaway: the exchange is link bound, so overlap is the lever, not copy speed. A small pull window (32
  blocks) lands the data in sequence order, which is what the causal prefix waits need.
- Applicability: any Ulysses / context-parallel exchange fused with causal attention on a single node.

### 2026-11-22 01:41 round-1 — PROFILING
- Context: rocprofv3 with PMC counters on a multi-rank kernel that spin-waits on peers.
- Signal: PMC serialises each rank's dispatches; the ranks desynchronise and the counters of the waiting
  phases are meaningless.
- Takeaway: use kernel-trace durations plus an in-kernel `s_memrealtime` trace for the timeline; keep PMC
  for per-rank work counters (MFMA busy, EA bytes). rocprofv3 kernel-trace also reports VGPR / AGPR / scratch.
- Applicability: every cross-rank fused kernel with flag waits.

### 2026-11-22 01:55 round-2 — ACCEPTED
- Context: staging the local q / k / v into the symmetric bank was done by the 32 pull blocks before pulling.
- Signal: the 168 MB local copy took ~590 us on 32 blocks (~9 GB/s per CU) and ~50 us on 256 blocks.
- Takeaway: never put a non-waiting bulk copy on a small subset of blocks in front of the critical path; let
  every block do its share before taking its role (-10.7% geomean, -15% at 32K).
- Applicability: any persistent kernel with a prologue copy that other roles wait on.

### 2026-11-22 02:15 round-4 — ACCEPTED
- Context: workers waited on the KV-prefix flags with an acquire load per poll.
- Signal: removing the per-poll acquire made CommOnly 23-26% faster at n8.
- Takeaway: poll flags with relaxed loads and acquire once after the block rendezvous; hundreds of blocks
  polling with acquire slow the concurrent XGMI pulls.
- Applicability: every flag wait on gfx950 that runs while other blocks move remote data.

### 2026-11-22 02:27 round-5 — ROLLED BACK
- Context: 32 instead of 8 loads in flight per thread in the pull copy.
- Signal: the pull time did not move; VGPR pressure rose (the kernel already spills 264 B at 128 VGPR + 128 AGPR).
- Takeaway: the pull is fabric bound, not latency bound; do not trade registers for loads in flight.
- Applicability: XGMI pulls with >= 32 blocks in flight.

### 2026-11-22 02:34 round-6 — ACCEPTED
- Context: the end-of-kernel O gather was split per m-tile (32 units of 2 MB at n8).
- Signal: 224 blocks sat idle while 32 blocks gathered.
- Takeaway: size the final copy to the whole grid; (m-tile, source) units (256) took llama n8 32K -2.1%.
- Applicability: any tail copy in a persistent kernel.

### 2026-11-22 04:20 round-10 — ACCEPTED
- Context: the pull blocks also staged pieces from the shared stage pool until it was empty.
- Signal: the first pull started after the whole stage (49-103 us); without it, at 31-47 us.
- Takeaway: blocks with a latency-critical role must not join a shared work pool that runs ahead of that role.
- Applicability: mixed-role persistent kernels with a common work queue.

### 2026-11-22 04:20 round-10 — PROFILING
- Context: per-rank traces at n8 64K.
- Signal: GPU 4 runs the same attention ~7% slower in every run (at n4, GPU 0 is 2-4% slower); every rank
  waits for it in the gather (210-600 us tail).
- Takeaway: trace all ranks before blaming the kernel's tail; the slowest GPU sets the fused end time.
- Applicability: every multi-rank kernel benchmark on this node type.

### 2026-11-22 04:35 round-11 — ROLLED BACK
- Context: 64 dedicated stager blocks so the other 160 workers start attention earlier.
- Signal: the first item started at ~33 us, but the end time was flat.
- Takeaway: in a persistent kernel whose bulk is compute, the end time follows (summed block-time of all
  non-compute work) / grid + tail. Moving fixed work onto fewer blocks does not shorten the kernel; make the
  work cheaper instead (round-12 confirmed the converse limit: a 25-30% cheaper stage was below noise).
- Applicability: any persistent kernel with copy and compute roles.

### 2026-11-22 05:25 round-13 — ACCEPTED
- Context: after pushing an O tile with SYS_NT stores, the flag was raised with a system-scope release.
- Signal: a system-scope release store is `buffer_wbl2 sc0 sc1` on gfx950, a writeback of every dirty line in
  the XCD's L2; with it removed the comm_only push item went 63.6 -> 51.5 us, every shape improved (-0.8%).
- Takeaway: when the payload stores are already system scope and retired by `s_waitcnt(0)` + barrier, raise
  the flag relaxed; never issue a sys release per work item.
- Applicability: every producer / consumer flag on gfx950 whose payload is written with sc0 sc1 nt stores.

### 2026-11-22 05:40 round-14 — ROLLED BACK
- Context: workgroup-scope instead of agent-scope acquire before re-reading O from L2.
- Signal: the push got cheaper only in comm_only; the fused forward was flat.
- Takeaway: once the release is gone the acquire is not on the critical path.
- Applicability: same kernel family.

### 2026-11-22 06:05 round-15 — ROLLED BACK
- Context: the fraction of q blocks claimed in ascending order (the rest descending), by sequence length.
- Signal: at 32K f = 0.5 is needed (0.375 costs 4-6%: the largest items wait for the whole exchange); at 64K
  a one-run sweep suggested 0.375 (-1%) but two full runs of 0.354 did not reproduce it.
- Takeaway: the ascending part must cover the exchange time; 64K single-process spread is ~1%, so confirm
  any sub-percent gain with two full runs.
- Applicability: causal persistent attention fused with an input exchange.

### 2026-11-22 06:20 round-16 — ROLLED BACK
- Context: push O to its owner straight from the MFMA epilogue registers instead of re-reading the tile.
- Signal: a lane holds 16 B of 32 rows, so the remote SYS_NT stores leave as 32 B row segments; items +28 us,
  the pull slower, qwen n4 32K +3.9%, geomean +1.3%; VGPR / scratch unchanged.
- Takeaway: write remote data in whole rows (512 B here): re-read the tile, or transpose through LDS first.
- Applicability: any remote store over XGMI from an MFMA accumulator layout.

### Next directions (not tried)
- Exchange on SDMA / copy engines to free the 32 pull CUs (41-153 us of block-time each).
- Split-KV for the last wave of items with an LSE merge, for the 64K end tail (~240 us).
- Cross-rank balance for the slower GPU; LDS-transposed epilogue push (see round-16).
