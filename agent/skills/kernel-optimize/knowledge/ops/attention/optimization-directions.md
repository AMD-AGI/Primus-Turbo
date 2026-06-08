# Attention — Optimization Directions

A catalog of directions and designs worth trying on CDNA attention kernels (gfx942 / gfx950), grouped by theme. Read [`overview.md`](overview.md) first for the anatomy, bottleneck model, and lever hierarchy. Each direction states *what to try*, *why it helps*, *when it applies*, and the *failure modes* to watch. Apply them one hypothesis per round.

---

## 1. Online Softmax & Numerics

The softmax fix-up runs every KV block and sits on the critical path between QK and PV — its formulation matters more than any single MFMA tweak.

- **Stream with running `(max, sum)`; never materialize full `S`.** Carry `(m_running, l_running, O_acc)` as loop state. Per block: compute the block row-max, `m_new = max(m_running, row_max)`, correction `corr = exp2((m_running − m_new)·log2e)`, probabilities `p = exp2(fma(s, log2e, −log2e·m_new))`, and `l_new = corr·l_running + sum(p)`.
- **Work in log2 space with the hardware `exp2`.** Prescale scores by `log2e = 1/ln2` and use single-issue `V_EXP` (`exp2`) instead of a software `exp`. ~1-ULP error is fine for attention and removes a transcendental sequence from the hot loop. The same applies to `rcp` for the final normalize.
- **Rescale `O_acc` by `corr` before the PV MFMA each block.** This keeps the accumulator in the new max's frame. It is easy to drop when refactoring the inner loop — the rescale must precede the PV accumulate, not follow it.
- **Guard masked / empty tiles against `-inf − -inf`.** On the first fully-masked block, `exp2(-inf − -inf)` is NaN and poisons `l`/`O` for the rest of the loop. `select`-guard `corr` and the partition rescale against `m == -inf`.
- **Share one reduction tree for max and sum.** A `wave_reduce` (butterfly `shuffle_xor` over `log2(wave_size)` levels) plus an LDS-backed `block_reduce`, parameterized by a mode tag (`max` / `sum`) and a neutral element (`-inf` / `0`), serves both passes with one implementation and one LDS scratch slot per wave. Bracket the second use with a barrier; pad the LDS slot to avoid bank-conflict tails.

## 2. K/V Staging & Operand Transpose

Get K and V into LDS cheaply and into the MFMA's expected lane layout without a manual shuffle.

- **Cooperative DMA K/V → LDS.** Issue direct global→LDS loads per wave/lane so K/V bypass the VGPR detour, keeping VGPRs free for Q packs and `O_acc`. XOR-swizzle the destination LDS address so the subsequent MFMA / transpose-read is bank-conflict free; re-apply the identical swizzle on the consumer side.
- **Double / triple-buffer the K prefetch.** Overlap the next KV block's DMA with the current block's QK MFMA (ping-pong, or a 3-buffer K prefetch when latency is high). Gate each consumer batch with a `vmcnt`/`lgkmcnt` fence + barrier; under double-buffering, flip the loop-carried buffer id by `N_SUBTILES % 2`.
- **Transpose-load V on gfx950.** V is stored `(k_row, d_col)` in LDS but PV consumes it transposed. `DS_READ_*_TR` returns the operand already in MFMA-A lane order, eliminating the register transpose. Two staggered-address reads form one complete operand; the k-stride between them encodes the MFMA K-dimension — re-derive it if the tile geometry changes.
- **Pre-transposed Vt in LDS on gfx942.** gfx942 has no transpose-read, so stage a software-transposed `Vt` region in LDS (or do a cross-lane shuffle). Keep the pipeline shape identical to the gfx950 path and switch only the V-source addressing on the arch flag; the Vt staging cost is justified only when the alternative is a per-MFMA shuffle.

## 3. QK / PV MFMA Layout & Pipelining

- **Compute `K @ Q^T` so `S` lands in PV-aligned register layout.** Computing the score tile in the orientation PV wants avoids a transpose between the two GEMMs.
- **k-pack prefetch software pipeline.** Preload Q once into register packs; per KV subtile, bulk pre-read K from LDS a few k-packs ahead, then issue `mfma(k_pack, q_pack)` while loading the next k-pack — a classic load/MFMA pipeline that keeps the DS and matrix pipes busy together.
- **Keep `S`/`P` in registers between QK and PV.** No LDS round-trip for the score/probability tile; this is the core FlashAttention invariant. Build `P` directly from the softmax outputs via a packed cast (`bf16` truncate-pack, or FP8 pack for an FP8 PV) with no intermediate store.
- **Overlap the V read under PV MFMA.** Flatten the PV schedule so each step issues the PV MFMAs for the current V pair while the next V pair's transpose-read is in flight, keeping `ds_read` latency hidden under matrix work.

## 4. Masking

Make masking branchless and free of per-element index arithmetic.

- **Fold mask thresholds into immediate operands.** Because the MFMA accumulator's intra-lane column offsets are statically known, a causal/banded/sliding-window mask reduces to one per-row scalar `rel = q_pos − k_base` compared against compile-time column constants. A two-instruction `v_cmp` (immediate threshold) + `v_cndmask` (overwrite with `-inf`) per packed pair replaces the per-lane index recompute. Only a single broadcast constant (`-inf`) can be the masked value; data-dependent thresholds force a register comparand instead.
- **Layout-aware register-tile triangular primitives.** Implement `make_causal` / `tril` / `triu` as ops over the register tile: strictly below the diagonal degenerates to a copy, strictly above to a broadcast fill, and only the diagonal base tile needs a real per-lane decision — encode that as a compile-time `uint64` lane-mask table indexed by packed element. No `__shfl`, no LDS, no per-element divide. The mask tables are layout-specific (a 16×16 fragment differs from 32×32); re-derive them per atom shape, and keep them in immediates/SGPRs (verify in the ISA that they did not spill to memory, or the win evaporates).
- **Separate the structural mask from the boundary handler.** The fold-into-immediate trick handles the regular interior; the last KV tile / first Q tile, where `rel` can fall outside the assumed range, still needs an explicit boundary path.

## 5. Cross-Lane & Cross-Warp Reduction

Softmax needs a row reduction that crosses the MFMA lane partners and, when a row is split across warps, the warps too.

- **Peer reduction over MFMA partner lanes.** Reduce the per-lane local max/sum across the MFMA row-partner with a single `ds_bpermute` on `lane ^ 32` (wave64 MFMA32) or a `shuffle_xor` — no LDS round-trip. The XOR mask is tied to the tile geometry; a wrong mask reduces over the wrong lane group and silently corrupts the softmax stats.
- **Branchless cross-warp softmax fix-up.** When several warps each computed a partial softmax over the same query row, write per-warp `(max, sum)` to LDS, then do one fused fix-up: load all per-warp stats with a single vector LDS load, reduce in registers over the warp count, compute per-warp rescale factors in log2 space, and select *this* warp's factor with a chained `select` keyed on `warp_id` (not a divergent `if` ladder). Pre-scale the running `O` accumulators by the merge factor so the next PV just adds in.
- **In-LDS / DPP reductions.** For block-wide reductions, `DS_*` atomics fold partials directly into LDS, and DPP fuses a neighbor-lane fetch into the same VALU issue — both avoid an explicit shuffle+barrier+broadcast stage.

## 6. Low-Precision KV / FP8

For FP8 KV (serving) the probability path and the dequant folding are the levers.

- **Pack probabilities to FP8 for the PV MFMA.** After the softmax fix-up, pack four f32 probabilities per MFMA tile into one packed FP8 lane (chained pack converts) and store to an LDS staging region that is addressable both as the writer's element type and as the MFMA reader's operand granularity (the same bytes viewed two ways). Feed them straight into the FP8 PV MFMA.
- **Fold the V dequant into the prob scale.** Multiply the per-warp rescale × partition-to-new factor (× per-token V norm factor in the per-token-KV path) into the probabilities *before* the FP8 pack, so the probs are already in the right numeric frame and only a single scalar V-correction is needed after the MFMA — no per-token post-MFMA rescale pass. If you skip the V-max normalization, FP8 probs saturate and lose dynamic range.
- **Bank-conflict-free prob staging.** Pad the probability row stride (e.g. 32 data + 8 padding bytes) so the i32 store and the wider MFMA read do not collide on the same bank; shrinking it to the exact data width reintroduces conflicts on both sides.
- **FP8 encoding is a correctness gate.** gfx942 is FNUZ (max 240), gfx950 is OCP (max 448); re-derive KV scales when porting rather than reusing the other generation's.

## 7. Decode & Paged Attention

Decode is memory-bound on KV; the levers are partitioning and avoiding redundant passes.

- **One CTA per KV compute tile.** Map a 3D grid `(batch, kv_heads·groups, context_partition)` with each CTA owning one fixed KV tile (e.g. 256 tokens). This keeps each decode CTA cheap (one tile, no inter-CTA softmax state) and packs waves well.
- **Block-split partitioning + softmax merge.** Each partition produces `(exp_sum, max_logit, unnormalized_out)`; a separate reduce kernel does the cross-partition softmax merge and the final normalized write. Empty / out-of-window partitions **must** still write neutral values `(sum=0, max=-inf, 0)` or the reducer reads uninitialized slots.
- **Fused single-partition fast path.** When there is exactly one partition, skip the partial-output round trip and the reduce kernel entirely — normalize inside the CTA and write the final output directly. Gate this on a compile-time switch; it is only safe when the "single visible tile" precondition holds.
- **Sliding-window: size the grid to the visible region.** Anchor partitions to the tail `seq_start = context_len − query_len − sliding_window` (clamped) so only the tokens that can contribute are launched, instead of the full context. Per-query-token `seq_start` / causal bound must be computed per token, not shared across a grouped CTA, or window-edge tokens are dropped.
- **GQA / MTP Q-reuse fan-out.** Let one CTA amortize the Q load and emit results for all `(query_token, group_lane)` pairs that share a KV head / MTP group, multiplexing the grid-Y axis as `kv_head · groups + group`. The host launcher and per-head pointer math must use the divided head index, not the raw grid id.

## 8. Backward Pass

The backward `dQ` accumulation is the contended hot spot (multiple KV-block warps sum into the same `dQ` rows).

- **Warp-disjoint packed-bf16 atomics for `dQ`.** Accumulate with `buffer_atomic_pk_add_bf16` (two packed bf16 per 32-bit op) and bias each warp's per-lane byte offset by `warpid · stride` so concurrent warps in the CTA write disjoint packed slots and never collide on the same RMW word. The destination must have at least `num_warps · 2 · wave_size` bf16 columns along the disjoint axis (≥512 for 4 warps, ≥256 for 8), or warps alias and contention returns. This removes contention *within* a CTA only; this assumes intra-CTA split-K (one CTA per KV block).
- **Pre-shuffle `dQ` for coalesced atomics.** A dedicated pre-shuffle kernel that reorders `dQ` rows into the lane/tile layout the backward kernel expects makes the subsequent split-K atomic stores land coalesced and bank-conflict free.
- **Recompute vs reload.** Recomputing `S`/`P` in the backward pass (FlashAttention-style) trades extra MFMA for far less HBM traffic than reloading a materialized `S`; on bandwidth-bound shapes recomputation usually wins. Treat the choice as a tunable.
- **Reuse the forward masking primitives.** The same immediate-fold / layout-aware mask (theme 4) applies to the backward score tile; instantiate it for the backward tile shape rather than generalizing the forward one.

## 9. Fused Pre / Post Ops

Fusing the ops around attention removes whole tensor re-loads and keeps descriptors resident.

- **RoPE partner-lane swap via cross-lane shuffle, no LDS.** For NeoX RoPE, lane `tid`'s rotary partner is `tid ^ vecs_per_half`; fetch it with one `ds_bpermute` (byte-addressed: `partner_lane · 4`) instead of staging both halves through LDS. Requires `vecs_per_head ≤ wave_size` so each rotary pair lives in one wave; the partner XOR constant differs for interleaved (non-NeoX) RoPE (`tid ^ 1`).
- **Fuse RoPE + reshape-and-cache + FP8 pack in one kernel.** Load cos/sin and buffer descriptors once before branching on Q vs K, apply the rotation, and write the (optionally FP8-packed, `cvt_pk_fp8` chained for 4 bytes/store) KV cache in the same pass — amortizing address/SGPR setup and avoiding a Q/K/V re-load.
- **Epilogue normalize-and-scatter.** After the KV loop, `O = O_acc · rcp(l_final)`, cast to the output dtype, and scatter using the MFMA accumulator's lane→(row,col) map. The lane map is atom-specific — do not reuse a `32x32` map for a `16x16` accumulator.

## 10. Scheduling & Occupancy

- **Hand-interleave QK / softmax / PV.** Use `sched_*` hints (and `sched_barrier(0)` fences) to interleave the QK MFMAs, the softmax VALU/transcendentals, the V `ds_read`s, and the PV MFMAs so no pipe starves while another runs. Raise `s_setprio` around the MFMA blocks only.
- **Budget AGPR for accumulators.** Keep `O_acc` (and large probability/score register tiles) in the AGPR pool; AGPR pressure caps occupancy independently of arch-VGPR. Decode kernels that pin an MFMA-AGPR split do so to keep probs in VGPR and outputs in AGPR — removing that split regresses occupancy.
- **Watch the loop-carried state width.** The `(m, l, O_acc, prefetch slots, buffer id)` carried through the KV loop is all live registers; growing the PV-iteration count or the O-accumulator width can force spills. Re-verify register pressure after any tile-geometry change.
- **L2 / XCD locality along the sequence.** Attention tiles have spatial locality along the sequence dimension; the chiplet-chunked work-group remap (keep consecutive ids on one XCD) warms the shared L2 for adjacent tiles, the same as for GEMM.

---

## Cross-cutting Failure Modes

- Producer/consumer LDS swizzle masks diverge → read garbage with no warning.
- `O_acc` rescaled after the PV accumulate instead of before → wrong online-softmax result.
- Peer-reduction XOR mask or epilogue lane→row map not matched to the MFMA atom → silent numerical error.
- Masked-tile `-inf − -inf` not guarded → NaN poisons the running sum for the rest of the loop.
- Empty/out-of-window decode partition not writing neutral values → reducer consumes uninitialized stats.
- Any cache keyed on `id(Q/K/V/grad_out)` → ~0 hit rate in real training even if a benchmark makes it look free; reject at analysis time.
