# GEMM — Optimization Directions

A catalog of directions and designs worth trying on CDNA GEMM kernels (gfx942 / gfx950), grouped by theme. Read [`overview.md`](overview.md) first for the anatomy, bottleneck model, and lever hierarchy. Each direction states *what to try*, *why it helps*, *when it applies*, and the *failure modes* to watch. Apply them one hypothesis per round.

---

## 1. Tiling & Thread Decomposition

The tile shape and MFMA atom are the dominant levers — they fix dynamic MFMA count, register layout, LDS footprint, and the occupancy ceiling simultaneously.

- **Three-level decomposition.** Map grid → `(tile_m, tile_n)`, waves → N-slabs, lanes → rows/cols inside each MFMA atom. Derive the repeat counts (`m_repeat = tile_m/16`, `num_acc_n = tile_n / waves / 16`, `k_unroll` from `tile_k · elem_bytes / 64`) so the per-tile MFMA count is explicit and the scheduler counts (next theme) can be derived from it.
- **Pick the widest-K MFMA atom that divides the K-loop.** Fewer, wider-K issues mean fewer dynamic MFMAs for the same work. On gfx942 the widest BF16/FP16 forms are `32x32x8` (K=8) and `16x16x16` (K=16); on gfx950 they widen to `32x32x16` and `16x16x32`, roughly halving the MFMA count of a fixed K-loop. INT8 reaches K=32 (gfx942) / K=64 (gfx950).
- **Choose the atom before everything else.** It pins the A/B/C/D lane layout, hence the LDS tile layout, the swizzle formula, and the epilogue re-tile. Re-choosing it later invalidates all of them.
- **Tile-size trade-off.** Bigger tiles raise arithmetic intensity and reuse but cost LDS and registers; cross an occupancy boundary and you lose a wave. Sweep `tile_m / tile_n / tile_k` as a single config dimension (this counts as one round), and re-check register/LDS pressure at each point.
- **Operand broadcast for small-K / grouped tiles.** The MFMA `CBSZ`/`ABID` fields broadcast one A-block across peer blocks and `BLGP` swizzles B, so a shared operand is not re-loaded from LDS/VGPR. Useful when K is tiny or several output blocks share an operand.

## 2. MFMA Operand Layout & Preshuffle

Make each lane already hold the exact VGPR item the MFMA expects, so no shuffle stands between the load and the issue.

- **Preshuffle B offline.** For inference weights (stable across iterations), reorder B once on the host into a layout like `(N/16, K/kpack, 4, 16, kpack_bytes)` so global loads land directly in MFMA register order. This removes the in-loop transpose/shuffle entirely. Get `kpack` right: `kpack = 64 / elem_bytes` for FP8/INT8, `kpack = 4` for BF16/FP16 — a wrong kpack silently produces garbage because loads no longer align to the atom.
- **Direct global→register, skip LDS, for the preshuffled operand.** When B is preshuffled it can stream straight into the VGPRs the MFMA reads, leaving LDS for A only. Encode the preshuffle as a layout descriptor so address math is a single index transform.
- **Prefilled swizzled offset tables.** Precompute per-iteration swizzled byte offsets into a small constant table so the hot loop issues loads with no per-step address VALU.
- **Transpose-load on gfx950.** Where an operand must be transposed for the atom, prefer the LDS transpose-read (`DS_READ_*_TR`) instead of a manual VGPR transpose pass — it delivers the operand in MFMA lane order for free. gfx942 has no such instruction; there you need a pre-swizzled LDS write or a cross-lane shuffle.
- **Weight identity caching is bounded.** If you memoize a quantized/preshuffled weight keyed on `(id(weight), weight._version)`, its per-step training gain is capped by `quant_time(weight) / step_time`. Never key such a cache on an activation or grad_out tensor — that hit-rate is ~0 in real training even though it looks free in a reuse-the-same-tensor benchmark.

## 3. LDS Pipelining & Prefetch

Turn a stall-bound K-loop compute-bound by overlapping the next tile's memory traffic with the current tile's MFMAs.

- **2-stage LDS ping-pong.** Allocate two LDS buffers; each main-loop body processes two K-tiles, computing on one buffer while the next K-tile streams into the other. One barrier separates the stages. This is the baseline CDNA GEMM pipeline.
- **Deeper multi-buffer pipelines.** On gfx950 the 160 KiB LDS (2.5× gfx942) makes 3+ stage rings realistic; more stages let the producer run further ahead of compute when HBM latency is high. Watch the occupancy trade-off — each stage is more LDS.
- **A0 cross-tile prefetch.** Immediately after the post-store barrier validates a freshly-written LDS buffer, fire the first LDS read of the *next* tile so its ~20–40-cycle latency overlaps the upcoming global loads. Issue it strictly *after* the barrier, never before (you would read stale data).
- **Producer–consumer warp specialization.** Split the CTA into a small producer warp-group that only streams global→LDS and a larger consumer warp-group that only issues MFMA, coupled by an N-stage LDS ring. Matrix math never stalls on HBM. Bracket each consumer MFMA with raised scheduler priority and drop it immediately after; pair each LDS read with an `lgkmcnt` fence and each ring rotation with a `vmcnt` fence + barrier. Request 2 blocks/CU so one block's producer stall hides under the other.
- **Software-pipeline structure.** Prologue primes iteration 0, the steady loop carries the "next" buffer as loop state, an explicit 2-tile drain follows; size the loop `num_tiles − stages` and never let it read past the prefetched window.

## 4. Bank-Conflict-Free LDS Layout

A bank conflict serializes an LDS phase and silently caps effective bandwidth — fix it before chasing scheduling.

- **XOR swizzle.** XOR the column index with bits of the row index so a wavefront hits distinct banks; near-zero cost, no extra LDS. A common 16-byte-granularity form is `col_swz = col ^ ((row % k_blocks16) * 16)`.
- **Apply the identical swizzle on both write and read paths.** Mismatched masks read the wrong rows with no compiler warning — the most common silent corruption in LDS GEMM.
- **Size the formula to the bank count of the target arch:** 32 banks on gfx942, 64 on gfx950. Reusing a 32-bank formula on gfx950 is a classic porting bug.
- **Padding as an alternative.** Padding the row stride to break bank alignment also works but wastes LDS; prefer XOR swizzle unless the layout makes swizzle awkward.
- **Coalesce LDS traffic.** Use paired `DS_READ2`/`DS_WRITE2` (and `_ST64` forms) to halve LDS instruction count on tile load/store.

## 5. Async Global→LDS Staging

Bypass the VGPR detour so registers stay free for accumulators.

- **Direct-to-LDS loads.** `BUFFER_/GLOBAL_/SCRATCH_LOAD_LDS_*` stream device memory straight into LDS without a register round-trip — the backbone of double-buffered prologues, and it reclaims the VGPRs an A-tile would otherwise occupy (often ~32).
- **Mind the per-arch granularity.** gfx942 async copy moves 4 B/op; gfx950 moves 16 B/op. On gfx942 the per-issue cost can exceed the savings for small tiles — reserve async copy for `tile_m ≥ 128`. On gfx950 it is profitable more broadly.
- **Gate the gate on `vmcnt`.** A precise `s_waitcnt(num_b_loads)` (with a compile-time-known load count) gates exactly the loads the next consumer needs, instead of a blanket `s_waitcnt 0`. If the load count is dynamic, this gate cannot be precise — fall back to a conservative fence.

## 6. Hot-Loop Instruction Scheduling

When LDS overlap is in place but the pipes still don't co-saturate, hand-steer the instruction interleave.

- **`sched_*` hints.** Insert scheduler-group hints (`sched_mfma` / `sched_dsrd` / `sched_vmem` / `sched_dswr`, terminated by `sched_barrier(0)`) to pin the interleave. A standard sync-copy pattern: prologue pulls 2 LDS-reads ahead of 2 MFMAs, then each iteration emits `vmem(1) → mfma(N) → dsrd(1) → mfma(N)`, with LDS-writes deferred to a tail window so they land before the boundary barrier without stealing issue slots from compute.
- **Derive counts from tile algebra.** Express the group counts as functions of the tile/wave constants (MFMA cluster count, buffer-load count, ds-read count) so a tile-shape change auto-retunes the interleave instead of needing a hand re-tune.
- **Token-budget scheduler for non-integer ratios.** When MFMA-per-VMEM is not integral, a release/consume token-budget scheduler spreads the MFMA pool evenly across memory issues and drains the remainder back-to-back, avoiding MFMA holes or VMEM starvation.
- **`sched_barrier(0)` is load-bearing.** It fences the compiler from coalescing LDS-reads ahead of MFMAs across iteration boundaries; removing it collapses the manual schedule.
- **Raise priority around MFMA only.** `s_setprio(1)` before the MFMA block and `s_setprio(0)` immediately after keeps MFMAs from being displaced by VMEM/LDS; forgetting to drop it starves co-resident waves and tanks occupancy.
- **Tune the LDS-write tail window.** Too early and writes contend with active MFMAs; too late and they miss the boundary barrier (visible as `s_barrier` stall). The `max(sched_iters − num_a_loads − 2, 0)` heuristic is a good default to perturb from.

## 7. Epilogue

The MFMA-native lane layout is column-strided; a naive store is uncoalesced.

- **CShuffle LDS re-tile.** Write accumulators row-major into LDS, barrier, then re-read with threads remapped to `(MLane, NLane)` (e.g. 8×32 for a 256-thread block) so lanes within a row hold contiguous columns and the global stores collapse into 128-B transactions. Worth it when `tile_n` is large (≥128); for small `tile_n` a direct store wins and the two barriers are pure overhead.
- **Split-LDS CShuffle variant.** When the staged tile would blow the LDS budget, partition waves in half so each group stages half the columns; halves per-CU LDS at the cost of one extra control-flow split.
- **Pre-issue dependent loads.** Hoist any per-row dependent global load (e.g. a sorted-index lookup) ahead of the store loop so they pipeline instead of serializing on `s_waitcnt vmcnt(0)`.
- **Fuse the epilogue.** Fold dequant, bias, and activation into the epilogue using packed VOP3P (`V_PK_*`) ops to halve VALU cost on the elementwise tail. Fused dequant+matmul and fused activation are single-launch fusions that transfer directly to training (no cross-iteration assumption).

## 8. Split-K

When K is large relative to M·N, a single block under-fills the device; split K across blocks and reduce.

- **In-block last-arriver fold.** Each K-split block writes a partial C-tile to a workspace and arrives on a two-word per-tile semaphore (a `signal` to publish, an atomic-add `semaphore` to count). The last arriver folds all partials and recycles the slot — no separate reduction kernel. Bound the live-tile count to the workspace slot count or late tiles spin forever.
- **Packed-bf16 atomic accumulation.** Reduce bf16 partials into HBM with `buffer_atomic_pk_add_bf16` (two packed bf16 per 32-bit op). Bias each warp's per-lane byte offset by `warpid · stride` so concurrent warps in the CTA target disjoint packed slots and never collide on the same RMW word — effectively uncontended. The destination must have at least `num_warps · 2 · wave_size` bf16 columns along the disjoint axis, or warps alias and contention returns. This avoids contention *within* a CTA only; cross-CTA writers to the same tile still contend.
- **Pre-shuffle for coalesced atomics.** A dedicated pre-shuffle kernel that reorders rows into the lane/tile layout the consumer expects makes the subsequent split-K atomic stores land in coalesced, bank-conflict-free addresses.
- **Separate reduction kernel.** For MoE/expert partials, a downstream reduction kernel that merges per-expert accumulators (or atomic-add row accumulation) is acceptable instead of an in-kernel cross-CTA barrier — but it must run after all producer launches complete.

## 9. Skinny / Small-M GEMM (decode, output projection)

When `M` is tiny (≤ ~16–32) and N·K are large, the generic 2D warp decomposition wastes waves on M.

- **Hard-wire a small M tile and map all waves to N.** Fix `tile_m = 16`, one M-warp, and spread every wave across N so the block is "one small M slice, many N workers."
- **N-tile A-reuse.** Reuse one A fragment set across several independent N tiles' worth of B fragments and accumulators, amortizing the A load. Watch the loop-carried register footprint — it multiplies by the N-tile-repeat count and can collapse occupancy; pair it with a `waves_per_eu` knob the autotuner can sweep.
- **Two scheduler variants.** A "wide-N, B-through-LDS" variant interleaves A and B VMEM separately with `mfma(2)`; an "N-tile-repeat" variant uses the token-budget scheduler for the non-integer MFMA/VMEM ratio. They are not interchangeable — mixing them leaves MFMA holes or starves VMEM.
- **Split-K pairs naturally** with skinny GEMM because M is too small to fill the device on its own.

## 10. Low Precision & Block-Scale

- **Fuse block-scale dequant into the MFMA inner loop, not the epilogue.** For per-block FP8 scales, decode the per-block scale slice for the current K and fold it into the accumulator inside the compute tile. Re-decode scales freshly for tail tiles instead of carrying the prologue's scale state (stale scales corrupt the last K block).
- **MX / FP4 scale tiles (gfx950).** The block-scaled MFMA multiplies one E8M0 exponent per 32-element K-block of A and of B before accumulation, and the scale lasts exactly one instruction — every K-block needs its own SCALE op. Precompute per-32-lane E8M0 scales into a compact tile (quantize → pack → MFMA-repack) so the inner loop just streams scales; clamp exponents away from `0xFF` (NaN) before packing.
- **Mixed F8F6F4 (gfx950).** A and B can independently select FP8/BF8/FP6/BF6/FP4 via the alternate operand-field encoding, halving operand bandwidth. The format is selected by how mantissas are packed, not by a different opcode — mis-packing is silently wrong.
- **W4A16 / W4A8 int4 weights.** Preshuffle B so each lane's MFMA-K micro-step reads a contiguous int4 pack, then unpack to bf16 in two latency-hidable phases (raw `buffer_load_dword`, then packed nibble→bf16 conversion). On gfx950 use the scaled-convert path; on gfx942 a shift-based f32→bf16 truncation (exact for scaled ints) avoids a 5-VALU-per-element conversion. Hoist the single shift that covers all nibbles out of the loop, and optionally defer the ×16 correction (and groupwise scale) to the epilogue — but then the epilogue *must* apply it.
- **FP8 encoding is a correctness gate across generations.** gfx942 FP8 is FNUZ (bias 8, max 240, no Inf); gfx950 FP8 is OCP (bias 7, max 448). Re-derive scales when porting; do not reuse the other generation's.

## 11. Grouped / MoE GEMM

MoE GEMM produces per-expert partials that must be combined per token. This is the area most prone to benchmark over-fitting — guard it explicitly.

- **Per-expert tile dispatch.** Dispatch tiles per expert using the routing/sorting metadata; write one row per `(token, slot)` with the output index packed from the sorted layout.
- **2-stage structure.** Stage-1 does the gate/up GEMM with LDS ping-pong and fused block-scale dequant; the activation (e.g. SiLU·up for SwiGLU) and optional routing weight fold into the stage-1 epilogue. Stage-2 / a separate reduction kernel merges per-expert partials (predicated stores or `atomic_add` rows).
- **Predicate out sentinel tokens.** Padding from token sorting emits out-of-range token ids; guard the store with an explicit `token_id < num_tokens` predicate. An unconditional out-of-range `buffer_store` is not safe on all paths.
- **Skew robustness is mandatory (do not over-fit the uniform benchmark).** Real routing produces a skewed, batch-varying `tokens_per_expert` histogram that is generally not a clean multiple of any tile dimension, and some experts get 0 tokens. Therefore:
  - Treat `tokens_per_expert` as a runtime tensor, never a compile-time constant `M_per_group`, and never `assert M_per_group % BLOCK_M == 0`.
  - Handle the empty-group case explicitly (skip launch / branch on zero rows).
  - Do not use static work partitioning that assumes each expert gets a fixed token count.
  - Validate every MoE round on a skewed distribution (e.g. top-1 with capacity factor, and a near-degenerate case where one expert takes ≥ 50% of tokens), not only the uniform layout. A gain that vanishes under skew is benchmark over-fit and must be rolled back.

## 12. Occupancy & Register Budget

- **Accumulators in AccVGPR.** Keep C/D in the AGPR pool across the K-loop and shuttle to arch-VGPR only at the epilogue; AGPR pressure caps occupancy independently of arch-VGPR.
- **Mind the arch-VGPR boundaries.** B-tile regs + A-prefetch + addresses pile into arch-VGPR. Crossing 128 drops 2 waves → 1 on a SIMD; crossing 256 spills. Async-copy (theme 5) reclaims the A-tile VGPRs.
- **Quantum-aware budgeting.** Occupancy is `min` over VGPR/SGPR/LDS pools, each rounded up to its allocation quantum (VGPR 8 Dwords, SGPR 16, LDS 512 B on gfx942 / 1280 B on gfx950). Trimming a register only buys a wave if it crosses a quantum boundary; a 64-bit operand forces even-pair alignment and can silently push you over.
- **Joint co-saturation.** Co-select tile shape, MFMA width, and waves/SIMD so VGPR+AGPR budget, LDS footprint, and matrix-issue rate all saturate together — tuning one in isolation starves another.
- **VGPR liveness remap.** A liveness pass over the compiled assembly can find dead VGPR windows and remap above-boundary registers into them, bumping waves/SIMD at the next occupancy step (64/73/85/102/128/170/256).

## 13. L2 / XCD Locality & Scheduling

The CU array is 8 XCD chiplets, each with its own L2 slice; the dispatcher round-robins work-groups across XCDs, so consecutive `blockIdx` land on different L2s.

- **Chiplet-chunked work-group remap.** Rewrite the logical work-group id so `chunk_size` consecutive ids stay on the same XCD (`chunk_idx · (num_xcds · chunk_size) + xcd · chunk_size + pos_in_chunk`); adjacent tiles then share a warm L2. Worth it when tiles have spatial locality (fixed K stripe) and the grid is comfortably larger than `num_xcds · chunk_size`. Tune `chunk_size` per op; pad the grid to a multiple of `num_xcds · chunk_size` for a clean partition.
- **WGM swizzle sweep.** The L2-locality work-group-mapping parameter (group-N width) is cheap to sweep at fixed problem size; treat it as one config-sweep round.
- **Persistent kernels.** Remap the *persistent work-id*, not `blockIdx.x`, or every persistent block keeps one XCD assignment forever. Compose only one swizzle — stacking two locality remaps usually defeats both.

---

## Cross-cutting Failure Modes

- Swizzle applied on only one of write/read paths → silent wrong rows.
- LDS buffer not rotated (or rotated at the wrong time) in ping-pong → reads stale tile.
- `s_waitcnt` / barrier dropped between an LDS producer and its consumer → silent corruption under double-buffering.
- Wrong preshuffle `kpack` or wrong bank count in the swizzle → loads/atomics misalign to the MFMA atom.
- A benchmark gain larger than the kernel change can structurally produce → re-check for an identity-keyed activation cache or a uniform-MoE assumption before accepting the round.
