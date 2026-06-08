# Attention — Optimization Orientation

Read this first when `target_op` is an attention-class kernel on gfx942 / gfx950: FlashAttention forward and backward (prefill / training), attention decode and paged attention (serving), and MLA-style decode. The forward and backward passes are separate kernels with different bottlenecks; in training, optimize the **combined fwd+bwd step**, not the forward kernel alone.

This file gives the anatomy, the bottleneck model, and the lever hierarchy. The detailed catalog of techniques lives in [`optimization-directions.md`](optimization-directions.md). Instruction-level behavior (MFMA layout, LDS transpose-load, transcendentals, cross-lane primitives, FP8 encodings) lives in [`../../hardware/gfx942/`](../../hardware/gfx942/overview.md), [`../../hardware/gfx950/`](../../hardware/gfx950/overview.md), and [`../../hardware/overview.md`](../../hardware/overview.md) — consult those rather than re-deriving instruction semantics here.

## Kernel Anatomy

### Forward (FlashAttention)

The forward kernel streams over KV blocks and keeps the running softmax state in registers — it never materializes the full `S = QK^T` matrix. One work-group owns a `BLOCK_M` slice of query rows; the inner loop over KV blocks of `BLOCK_N` carries `(m_running, l_running, O_acc)`:

1. **Load K** for the next KV block (cooperatively, global→LDS).
2. **QK MFMA** → score tile `S` in registers.
3. **Mask** (causal / sliding-window / padding) folded onto `S`.
4. **Online softmax**: row-max over the block, update running max `m`, correction `corr = exp2((m_old − m_new)·log2e)`, probabilities `p = exp2(...)`, running sum `l`.
5. **Rescale `O_acc` by `corr`** before accumulating.
6. **Load V**, **PV MFMA** → accumulate into `O_acc`.
7. After the loop: **normalize** `O = O_acc · rcp(l)`, cast, and scatter to global.

### Backward

Heavier than forward: recomputes `S`/`P` (or reloads), and produces `dQ`, `dK`, `dV`. `dQ` is typically a split-K-style reduction over KV blocks (multiple warps/blocks accumulate into the same `dQ` rows), which makes the reduction strategy a first-class concern.

### Decode / Paged Attention

`BLOCK_M` collapses to a few query tokens (often grouped-query). The work is dominated by streaming KV from HBM (often FP8) across a paged block table, partitioned across CTAs, with a softmax merge across partitions either in a second reduce kernel or fused when there is a single partition.

## Bottleneck Classification

| Regime | Dominant cost | What helps |
|--------|---------------|------------|
| Prefill / long-seq forward | QK+PV MFMA throughput, plus softmax transcendentals, V transpose, and cross-lane reductions sitting on the critical path | wider-K MFMA atom, keep S/P in registers, transpose-load V, overlap ds_read in PV, hardware `exp2`, peer reduction |
| Backward | recompute + three output GEMMs + `dQ` reduction contention | split-K `dQ` with warp-disjoint packed atomics, pre-shuffle for coalesced atomics, recomputation vs reload trade-off |
| Decode / paged | HBM bandwidth on KV (memory-bound) | FP8 KV, one CTA per KV tile, block-split partitioning, fused single-partition path, GQA Q-reuse |

Attention rarely sits at a single clean bottleneck — the MFMA pipe, the softmax VALU/transcendental work, the V-operand transpose, and the cross-lane reduction all contend. The win usually comes from **overlapping** these (software pipeline) rather than speeding one in isolation.

## Optimization Lever Hierarchy

1. **Fuse the whole step in registers** — keep `S`/`P` and `O_acc` resident; never spill the score tile to LDS between QK and PV. This is what makes it FlashAttention rather than a two-pass matmul.
2. **Online-softmax numerics** — `exp2` + log2e prescaling, running max/sum, rescale `O` by the correction; this is on the critical path every block.
3. **K/V staging + V transpose** — cooperative global→LDS with bank-conflict-free swizzle; transpose-load V (gfx950) or pre-transposed Vt LDS (gfx942) so PV needs no per-call shuffle.
4. **QK/PV pipelining** — k-pack prefetch, overlapped V ds_read under PV MFMA, P built directly via packed cast.
5. **Masking** — fold thresholds into immediates / layout-aware register-tile mask primitives so masking is branchless and adds no index VALU.
6. **Cross-lane / cross-warp reduction** — peer reduction over MFMA partner lanes; branchless cross-warp softmax fixup.
7. **Decode-specific** — partitioning, two-kernel vs fused reduce, FP8 prob-pack.
8. **Backward-specific** — `dQ` reduction strategy.
9. **Scheduling / occupancy** — sched hints, AGPR budget, loop-carried state width.

## Healthy Metric Ranges (starting heuristics)

| Metric | Good | Needs attention |
|--------|------|-----------------|
| MFMA-issue ratio (prefill) | > 40% | < 25% |
| HBM BW utilization (decode) | > 60% | < 30% |
| LDS bank conflicts | < 5% | > 20% |
| Occupancy (waves/SIMD) | ≥ 2 | 1 |
| Register / scratch spills | 0 | any |
| `s_barrier` / `s_waitcnt` stall fraction | low | dominates → pipeline not overlapping |

## Default Starting Configs

- `BLOCK_M ∈ {128, 256}`, `BLOCK_N = 64`, `head_dim` a multiple of 32 (ideally ≥ 64) so Q fits in registers, K/V fit in LDS as separately-swizzled regions, and the transpose-load geometry is clean.
- 4 waves / 256 threads is a typical forward CTA; MFMA `32x32x16` (gfx950) / `32x32x8` (gfx942) or `16x16x*` are the usual atoms.
- Decode: one CTA per KV compute tile (e.g. 256 tokens); size the grid to the visible region for sliding-window.
- Wave size is 64 — peer-reduction masks and lane→(row,col) maps are wave64-specific; do not import wave32 (RDNA/WMMA) assumptions.

## Training-Transfer Caveats

- Optimize the **combined fwd+bwd step**. Backward is the larger, more contended kernel; a forward-only win can be swamped by backward `dQ` contention.
- Activations (`Q`, `K`, `V`, `grad_out`) are fresh tensors every training iteration — never cache attention work keyed on their identity; the real-training hit rate is ~0 even when a reuse-the-same-tensor benchmark makes it look free.
- Causal-mask and partition fast-paths must stay correct on the real shape distribution (variable sequence lengths, partial tiles), not only the benchmark's aligned shapes.

## Directions Map

| If the kernel is… | Start with these themes in [`optimization-directions.md`](optimization-directions.md) |
|-------------------|------------------------------------------------------|
| Prefill / training forward | Online softmax & numerics; K/V staging & V transpose; QK/PV pipelining; masking |
| Long-context / causal | Masking (immediate / layout-aware); peer & cross-warp reduction |
| Decode / paged / serving | Decode & paged attention; low-precision KV / FP8; fused pre/post ops |
| Backward / training | Backward pass (dQ reduction); recompute vs reload |
| Stall-bound | K/V staging double-buffer; QK/PV pipelining; scheduling & occupancy |
| With RoPE / KV-cache write | Fused pre/post ops |
