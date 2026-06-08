# FlyDSL — Optimization Directions

A catalog of directions and designs worth trying on FlyDSL kernels, grouped by theme. Read [`overview.md`](overview.md) for the abstraction ladder and arch matrix, and [`programming-model.md`](programming-model.md) for the API spine and the correctness rules each of these relies on. The **algorithm-level** "why" for a given op lives in the op catalogs ([`../../ops/gemm/optimization-directions.md`](../../ops/gemm/optimization-directions.md), [`../../ops/attention/optimization-directions.md`](../../ops/attention/optimization-directions.md)); this file is the **FlyDSL expression** of those levers — the mechanism, the knobs, and the FlyDSL-specific failure modes. Apply one hypothesis per round, verify correctness first, then benchmark.

Each direction must improve a real fwd+bwd training step, not just the benchmark loop: prefer kernel-side mechanisms (tiling, staging, swizzle, scheduling, dynamic dispatch) over wrapper caches keyed on activation/grad_out identity, and validate MoE work on a skewed token distribution (see "Training-transfer guardrails" at the end).

---

## 1. Software-Pipelined Prefetch (loop-carried double buffer)

The foundational latency-hiding move: restructure a `load → compute` loop so the next tile's global load is in flight while the current tile's MFMA/WMMA runs. In FlyDSL this is *only* real when expressed through `range(..., init=[...])`.

- **Three-region shape.** Prologue issues iteration 0's loads; the steady `scf.for` (`N−1` trips) reads the previously-prefetched value from loop-carried state, issues the loads for `i+1`, then computes on the current tile; the epilogue consumes the last prefetch. Carry every "next" input the load depends on (offsets, block-table indices, per-block scales, online-softmax `m`/`l`, accumulators) in `init`.
- **Why it works here.** `buffer_load` returns immediately and the compiler places `s_waitcnt` at the first consumer; issuing the load earlier gives the scheduler slack to push that wait past more compute. The technique gives the scheduler slack — it does not itself set `s_waitcnt vmcnt(N)` or remove a barrier.
- **Cross-phase hoist.** When a later phase has a barrier-heavy reduction (e.g. an online-softmax cross-wave reduce), issue that phase's loads *before* the barrier so the barrier dead time overlaps the load.
- **FlyDSL failure modes.** Python-int loop bounds silently unroll and drop `init=` (use `fx.Index`); a Python `data = next_data` swap is not a phi (thread it through `yield`); reset `SmemPtr._view_cache = None` before the epilogue; replicate any conditional load (e.g. a scale load gated by a quant mode) inside the prefetch block or the prefetched value diverges from the consumer.
- **When not to.** Trip count 1, already compute-bound (MFMA util > ~90%), or the kernel already spills — extra "next" buffers grow arch-VGPR (~4 per `buffer_load_dwordx4`) and on CDNA3 compete with AccVGPR in the shared 512-entry pool. Never use `maxnreg` to zero AccVGPR to "make room."

## 2. LDS Ping-Pong & Multi-Buffer Staging

Stage one operand through LDS in two (or more) buffers so the producer writes the next tile while MFMAs consume the current one, separated by a single barrier.

- **Stage toggle.** Express the buffer as a composed layout with an explicit `STAGES` dimension and toggle `write_stage = read_stage ^ 1` each K iteration; unroll the K loop in **pairs** so the alternation lines up, and keep exactly **one** `gpu.barrier()` after the LDS write. Forgetting to advance the read stage (or a partner register buffer) reads stale LDS with no compile-time signal.
- **Asymmetric staging.** A frequent CDNA shape: stage A through LDS (2 stages) while B streams straight to registers (theme 4), or double-buffer B in registers parallel to A's LDS. Match each consumer's buffer index to its producer in lockstep.
- **Interleaved-cluster / decoupled loaders.** For FP8/INT8 GEMM, factor the moves into reusable `G2SLoader` (`buffer_load → ds_write`), `S2RLoader` (`ds_read` with a precomputed XOR table), and an MMA stage, then interleave them per MFMA cluster (`mfma → g2s.load_one → s2r.load_one → mfma`) over a deeper (e.g. 8-buffer) ring so global-load, LDS-read, and matrix latency all hide under one pipeline. Buffer aliases must be rotated after every K iteration; precise `s_waitcnt` counts (a function of the prologue depth and per-stage tile counts) gate exactly the loads the next consumer needs.
- **Deeper rings on gfx950.** 160 KiB LDS (2.5× gfx942) makes 3+ stage rings realistic so the producer runs further ahead when HBM latency is high; each extra stage costs LDS and may demote occupancy — sweep it.
- **When not to.** LDS already saturated (prefer 1-stage with async direct-to-LDS, theme 5), or K too small to amortize the prologue/tail.

## 3. Bank-Conflict-Free LDS

A bank conflict serializes an LDS phase and silently caps effective bandwidth. Fix it before chasing scheduling. CDNA LDS is 32 banks (gfx942) / 64 banks (gfx950) of 4 bytes; a row stride that is a multiple of `NBANKS·4` bytes makes every row's column-`c` element hit the same bank.

- **XOR swizzle (preferred, zero overhead).** Replace `col` by `col ^ ((row & MASK) << SHIFT)`; the XOR is its own inverse, so apply the *identical* transform on every store and every load. In FlyDSL this can live inside a `make_composed_layout(SwizzleType.get(...), base)` so the descriptor carries it, or as explicit address math with a precomputed per-iteration XOR table the hot loop indexes (no per-step swizzle VALU). Size `MASK`/`SHIFT` so each vectorized access covers all `NBANKS` — a mask tuned for 32 banks leaves residual 2-way conflicts on gfx950.
- **Padding (fallback).** Pad the row stride so it is coprime with `NBANKS` in dwords (e.g. RDNA WMMA tiles pad `BLOCK_K=32`→`40`). Costs LDS; a power-of-two pad that keeps `(stride/4) % NBANKS == 0` does nothing.
- **Transpose-load instead of swizzle (gfx950).** Where an operand must be transposed for the atom, prefer the LDS transpose-read (`ds_read_*_tr` / `ds_read_b64_tr_b8`) — it delivers the operand in MFMA lane order for free. gfx942 has no such instruction; there, pre-swizzle the LDS write or pre-transpose the operand into a separate LDS region (e.g. a `Vt` tile for attention's PV).
- **Coalesce LDS traffic.** Use paired `ds_read2` / `ds_write2` to halve LDS instruction count on tile load/store.
- **Diagnose, don't guess.** Apply when ATT shows `ds_read`/`ds_write` per-hit stall > ~100 cycles or LDS stall > ~15% of total *and* the tile stride is a multiple of `NBANKS·4`. A `lgkmcnt(0)` stall from exposed write-to-read latency (no intervening work) is a *different* bug — fix it with interleaving (theme 6), not swizzle.

## 4. Preshuffle Operands as Layout Descriptors

Make each lane already hold the exact VGPR item the MMA expects, so no shuffle stands between the load and the issue — and, for a static weight, skip LDS for that operand entirely.

- **Encode the offline permutation as a nested layout.** Permute the weight once on the host, then view it on-device through a `make_layout((... nested shape ...), (... strides ...))` that reproduces the MMA fragment tiling (e.g. an N-major `(16, N/16)` × K-major `(kpack, 4, K/kpack)` descriptor for a 16×16×16 atom). Because the view presents each thread's fragment as a contiguous ≤128-bit chunk, `copy` streams B straight into the MMA B fragment with no LDS round-trip — freeing LDS and bandwidth for deeper A pipelining.
- **Derive the copy from the MMA.** Build the `TiledMma` first and use `make_tiled_copy_A/B/C(copy_atom, tiled_mma)` + `retile(frag)` so the global/LDS copy partitions *exactly* into the MMA's A/B/C fragments without hand-computing per-thread offsets. Do not tune the MMA replication layout and the copy tiling independently — they must change together.
- **Get `kpack` right.** The K-pack width is dtype-coupled (`64 / elem_bytes` for FP8/INT8, 4 for BF16/FP16); a wrong pack silently produces garbage because loads no longer align to the atom. The host permutation and the device descriptor must agree exactly — a mismatch still type-checks and runs.
- **When not to.** B changes per launch (activations / dynamic weights — the offline cost is not amortized), B genuinely benefits from LDS reuse across many fragments, or the fragment layout needs XOR-permuted addressing that a single affine descriptor can't express. A preshuffled-and-skip-LDS B also removes B's barrier — a second consumer of B in the block re-introduces the need for an LDS copy.
- **Weight-cache caveat.** Memoizing a preshuffled/quantized weight keyed on `(id(weight), weight._version)` is bounded — its per-step training gain is capped by `quant_time(weight) / step_time` (low single digits). Never key such a cache on an activation, grad_out, or activation-scale tensor; that hit rate is ~0 in real training even though a reuse-the-same-tensor benchmark makes it look free.

## 5. Async Global→LDS Staging

Bypass the VGPR detour so registers stay free for accumulators.

- **Direct-to-LDS loads.** `buffer_load_lds` / `raw_ptr_buffer_load_lds` stream device memory straight into LDS without a register round-trip — the backbone of double-buffered prologues; it reclaims the VGPRs an A-tile would otherwise occupy. The cooperative form issues per wave/lane with an XOR-swizzled destination so the subsequent MFMA / transpose-read is conflict-free, gated by an `s_waitcnt(0)` + `gpu.barrier()` per consumer batch.
- **Mind the per-arch granularity.** gfx942 async copy moves **4 B/op**, gfx950 **16 B/op**. On gfx942 the per-issue cost can exceed the savings for small tiles — reserve async copy for large `tile_m` (≥ 128); on gfx950 it is profitable more broadly.
- **Gate precisely.** A compile-time-known load count lets `s_waitcnt(num_loads)` gate exactly the loads the next consumer needs instead of a blanket `s_waitcnt 0`; if the count is dynamic the gate cannot be precise.
- **Port raw `buffer_ops` to the layout API** when you want this cleanly: `make_buffer_tensor` + `logical_divide` + `copy_atom_call` with a `BufferCopy*b` atom replaces manual byte arithmetic / `shrui(...,2)` / i32→dtype bitcasts. Two caveats: `copy_atom_call` has no `mask=`, so re-introduce explicit `is_valid.select(...)` / `if is_valid:` OOB guards; and a wave-uniform row offset may currently fold into voffset (VGPR) rather than soffset (SGPR), so re-check VGPR pressure after porting.

## 6. Hot-Loop Instruction Scheduling

When LDS overlap is in place but the pipes still don't co-saturate, hand-steer the instruction interleave with `sched_*` hints. FlyDSL exposes the AMDGPU sched-group barriers directly.

- **`sched_*` group hints.** Emit a deterministic sequence of `sched_mfma(n)` / `sched_dsrd(n)` (LDS read) / `sched_dswr(n)` (LDS write) / `sched_vmem(n)` (global) at the *end* of the stage body, terminated by `sched_barrier(0)` to lock ordering. A standard sync-copy pattern: prologue ramps a couple of LDS-reads ahead of MFMAs, each iteration emits `vmem(1) → mfma(N) → dsrd(1) → mfma(N)`, and LDS-writes are deferred to a tail window so they land before the boundary barrier without stealing issue slots. **The hint counts must sum to exactly the number of each opcode the body emits** — a mismatch makes the backend silently drop the policy and revert to the default schedule.
- **Token-budget scheduler for non-integer ratios.** When MFMA-per-VMEM is not integral, a small `release`/`consume` token-budget object (pure Python, constexpr-resolved at trace time) spreads a fixed MFMA pool evenly across memory issues and drains the remainder back-to-back — pairing e.g. 2 MFMA per VMEM and flushing leftovers. Keep it local to the schedule function (do not mutate an outer captured object); a miscount silently shortens the interleaved phase.
- **Derive counts from tile algebra.** Express the group counts as functions of the tile/wave constants (MFMA cluster count, buffer-load count, ds-read count) so a tile-shape change auto-retunes instead of needing a hand re-tune; re-derive rather than copying numeric literals when tiles change.
- **Raise priority around MFMA only.** `s_setprio(1)` before the MFMA block and `s_setprio(0)` immediately after keeps MFMAs from being displaced by VMEM/LDS; forgetting to drop it starves co-resident waves and tanks occupancy. Bracket discrete pipeline phases with `sched_barrier(0)` so the scheduler can't float instructions across them.
- **`sched_barrier(0)` is load-bearing and ordering-only.** Removing it collapses the manual schedule (LLVM coalesces across iteration boundaries); placing it *inside* a constexpr loop instead of after pessimizes overlap. It does not change register allocation — occupancy regressions still need tile/stage tuning. The schedule must run after the `gpu.barrier()` that owns the LDS write it reorders.
- **gfx1250 analogue.** Bracket the TDM issue inside the WMMA tile with `sched_barrier(0)` so the load can't hoist above the matrix issue (theme 13).

## 7. Epilogue Staging & Fusion

The MMA-native lane layout is column-strided; a naive store is uncoalesced.

- **CShuffle LDS re-tile.** Write accumulators row-major into an LDS `[tile_m, tile_n]` tile, barrier, then re-read with threads remapped to `(MLane, NLane)` (e.g. 8×32 for a 256-thread block) so lanes within a row hold contiguous columns and the global stores collapse into 128-B transactions. Worth it when `tile_n` is large (≥ 128); both barriers are mandatory, and the shuffle math assumes the default MMA row mapping (a different atom needs a new row iterator + `(MLane, NLane)`).
- **Split-LDS CShuffle.** When the staged tile would blow the LDS budget, partition waves in half so each group stages half the columns and route group B to a second buffer — halves per-CU LDS at the cost of one extra control-flow split.
- **Pre-issue dependent loads.** Call a `precompute_row(...)` hook for *every* shuffle row before the store loop so dependent global loads (e.g. a MoE `sorted_idx` lookup) issue back-to-back and pipeline instead of serializing on `s_waitcnt vmcnt(0)`.
- **Fuse the tail.** Fold dequant, bias, activation (SiLU/SwiGLU), and even per-row quant into the epilogue using packed conversions. These single-launch fusions transfer directly to training (no cross-iteration assumption). For MoE stage-1, the activation + optional routing-weight scale + FP4/INT8 quant + sorted-scale write all fuse into one epilogue.

## 8. Cross-Lane Primitives Without LDS

Reductions, broadcasts, prefix sums, and argmax can stay in registers using cross-lane hardware, removing an LDS round-trip and its barriers.

- **DPP butterfly.** Compose `update_dpp` row-xor/shift immediates into a 16-lane butterfly (`dpp_xor`); chaining offsets 1,2,4,8 over an associative op gives a wave reduction, and a Kogge–Stone pattern gives a prefix sum. The control/mask immediates must be compile-time constants; 32-bit only (bitcast f32↔i32 around the call); bounded to a 16-lane row, so cross-row needs `ds_bpermute` / `permlane`.
- **`ds_bpermute` for cross-row / peer reduction.** Cross the 16-lane row boundary (e.g. `lane ^ 32` for an MFMA32 row partner, or `(lane & 0x30) − 1` to extend a DPP scan to a full wave). Peer-reduction lane width is wave64-specific — the XOR mask must change for other tile sizes or softmax stats are computed over the wrong lane group.
- **`shuffle_xor` sub-warp reductions.** Reduce within a power-of-two lane group (e.g. MoE gating binds one token to a `THREADS_PER_TOKEN` group and reduces max/sum/argmax with `width=TPT` butterflies); the width must equal the group size, not `WARP_SIZE`, or reductions cross token boundaries. Break argmax ties deterministically (`greater | (equal & lower_idx)`).
- **Shared-mesh + DPP for sorting/scan.** For MoE token sorting, an LDS token×expert mesh + an all-wave inclusive prefix sum (intra-wave DPP scan, per-wave LDS partials, cross-wave fixup) + a lane-group DPP scatter does the whole counting-sort in one block; OOB lanes must be steered to a harmless slot (predication alone is not enough under SIMT).
- **Wave-then-block reduction backbone.** For norm/softmax, a `shuffle_xor` wave reduce + per-wave LDS slot + second-stage wave reduce, masked by `select(in_range, v, identity)`, works for any `BLOCK_THREADS/WARP_SIZE` ratio (wave64 and wave32 alike). Fuse two channels (e.g. `sum` and `sum_sq`) through one barrier pair to halve the LDS round-trip — but only when both share the reduction identity (ADD); do not reuse an ADD-fused helper for an ADD+MAX pair.

## 9. Online-Softmax Attention Pipelines

Attention fuses the whole step in registers; FlyDSL expresses it as a `range(..., init=...)` over KV blocks carrying `(m_running, l_running, O_acc, ...)`. See [`../../ops/attention/`](../../ops/attention/overview.md) for the algorithm; the FlyDSL-specific mechanics:

- **Keep S/P in registers; never spill to LDS between QK and PV.** Build P directly from softmax outputs via a packed cast (bf16/f16) with no LDS round-trip.
- **Cooperative K/V DMA + V transpose.** Cooperatively `buffer_load_lds` K/V into separately XOR-swizzled LDS regions; feed PV from `ds_read_*_tr` transpose-loads on gfx950 (V never leaves LDS untransposed) or a pre-transposed `Vt` LDS region on gfx942. Let an `IS_GFX950` switch change *only* the V-source argument and keep the pipeline shape identical.
- **k-pack prefetch + overlapped ds_read.** Bulk pre-read K from LDS for a couple of steps, issue MFMA while the next k-pack loads; in PV, pre-read the next V pair while the current MFMAs run.
- **Online update numerics.** Use `exp2` + log2e prescaling (`corr = exp2((m_old − m_new)·log2e)`, `p = exp2(fma(s, log2e, −log2e·m_new))`), rescale `O_acc` by `corr` *before* the PV MFMAs each step, and reduce the row max/sum across the MFMA peer lane with `ds_bpermute`.
- **Multi-tile decode pipeline.** Double-buffer K/V LDS by tile parity, carry online-softmax state + 2-ahead prefetch offsets through one `scf.for`, and dispatch first/middle/last tiles out of the loop (first has no rescale, last interleaves GEMM2 with the epilogue store). Make boundary checks compile-time constants per branch (a runtime flag regresses prefetch scheduling); on gfx950 bounce the last tile's output through the opposite-parity buffer to avoid colliding with the V reads still issuing from the current buffer.
- **Decode/paged specifics.** Map one CTA per visible KV tile; offer a compile-time-selectable fused single-partition path that writes directly to the final output and a multi-partition path that stores `(exp_sum, max_logit, tmp_out)` for a separate reduce kernel. Cross-warp softmax rescale can fuse with an FP8 prob-pack into LDS for the next PV MFMA.

## 10. Split-K & Multi-Stage Reduction

When K is large relative to M·N (or for MoE per-expert partials), split the contraction and reduce.

- **In-block last-arriver fold via semaphore workspace.** Each K-split block writes a partial tile to a workspace and arrives on a per-tile semaphore (a publish `signal` + an atomic-add counter); the last arriver folds all partials and recycles the slot — no separate reduction kernel. Bound the live-tile count to the workspace slot count or late tiles spin forever. Combine with `buffer_load_*b ... lds` async G2S and an ldmatrix-style streaming inner loop.
- **Packed-bf16 atomic accumulation.** Reduce bf16 partials with a packed atomic-add (two bf16 per 32-bit op). Bias each warp's per-lane byte offset by `warpid·stride` so concurrent warps target disjoint packed slots and never collide on the same RMW word; the destination needs ≥ `num_warps·2·wave_size` columns along the disjoint axis. This removes intra-CTA contention only — cross-CTA writers to the same tile still contend.
- **Arch-gate the atomic.** bf16 global atomics exist on gfx94+/gfx95+/gfx12+; fall back to scaled-f32 atomics on older targets. Gate at trace time.
- **Pre-shuffle for coalesced atomics.** A dedicated pre-shuffle kernel that reorders rows into the consumer's lane/tile layout makes the split-K atomic stores land coalesced and bank-conflict-free.
- **Separate reduction kernel.** For MoE/expert partials, a downstream kernel that merges per-expert accumulators (predicated stores or `atomic_add` rows) is acceptable instead of an in-kernel cross-CTA barrier — but it must run after all producer launches complete.

## 11. Low-Precision Packing & Conversion

FlyDSL reaches the conversion intrinsics and bit tricks needed for FP8/FP4/INT4 with no native single-instruction path.

- **Fuse block-scale dequant into the MFMA inner loop, not the epilogue.** Decode the per-block FP8 scale slice for the current K and fold it into the accumulator inside the compute tile; re-decode scales freshly for tail tiles (stale scales corrupt the last K block).
- **MX / FP4 scale tiles (gfx950).** The block-scaled MFMA multiplies one E8M0 exponent per 32-element K-block and the scale lasts exactly one instruction — every K-block needs its own SCALE op. Precompute per-32-lane E8M0 scales into a compact tile (quantize → pack → MFMA-repack) so the inner loop just streams scales; clamp exponents away from NaN before packing.
- **Branchless f32→E2M1 (MXFP4) pack.** Convert via masked-select on the i32 bit-pattern (isolate sign/abs, build denormal and normal predicates, RNE the normal path by odd-bit injection then shift, saturate to `0x7`); derive the per-32 E8M0 block scale with a `shuffle_xor` butterfly max and the `(254 − e8m0)` reciprocal trick, multiply it in *before* conversion, then bit-pack nibbles into i8/i16/i32 stores. The `_fp_headroom` constant differs FP4 vs FP8 — a wrong value rescales the whole tensor by `2^6`; the routine saturates NaN/Inf to max (not IEEE-faithful).
- **W4A16 / W4A8 int4 weights.** Preshuffle B so each lane's MFMA-K32 micro-step reads a contiguous 8-byte (16-nibble) pack, then unpack in two latency-hidable phases: one `buffer_load_dword`, then packed int4→bf16x4. On gfx950 use `cvt_off_f32_i4` (SDWA `byte_sel` covers all 8 nibbles with **one** shift total) + `cvt_pk_bf16_f32`; on gfx942 use a shift-based f32→bf16 truncation (exact for scaled ints, ~5-VALU cheaper than `truncf`). Hoist the single `>> 4` out of the loop; optionally defer the ×16 correction (and groupwise scale) to the epilogue — but then the epilogue *must* apply it.
- **FP8 encoding is a correctness gate across generations.** gfx942 FP8 is FNUZ (bias 8, max 240, no Inf); gfx950 FP8 is OCP (bias 7, max 448). Re-derive scales when porting; do not reuse the other generation's. The 1–5% mismatch from FP8 requant is expected tolerance, not a bug.

## 12. MoE Dispatch, Sorting & Gating

MoE produces per-expert partials grouped per token; this is the area most prone to benchmark over-fitting — guard it explicitly.

- **Per-expert tile dispatch with predicated sentinels.** Dispatch tiles per expert from the routing/sorting metadata; padding from token sorting emits out-of-range token ids, so guard every store with an explicit `token_id < num_tokens` predicate (an unconditional OOB `buffer_store` is not safe on all paths) and sentinel-fill the padded region before scatter.
- **2-stage structure.** Stage-1 does the gate/up GEMM with LDS ping-pong + fused block-scale dequant; the activation (SiLU·up for SwiGLU) and optional routing weight fold into the stage-1 epilogue. A separate reduction kernel merges per-expert partials.
- **Interleaved half-iteration pipeline.** For a fused mixed-dtype MoE GEMM-1 with a separate scale stream, split each ping/pong half into `sched_barrier(0)`-bounded phases (scale VMEM + first ds_reads + prev-tile MFMAs, then distributed B VMEM + remaining ds_reads + MFMAs), with `s_setprio` around MFMA — enough heterogeneous traffic to fill multiple phases per K-step.
- **Fused sort + gating** as in theme 8 (LDS mesh + DPP prefix scatter for sorting; sub-warp `shuffle_xor` argmax for top-K gating, with a layout chosen so each lane owns a power-of-two slice loaded as one wide atom). Fall back to a multi-phase HBM-staged path when the token×expert mesh exceeds the LDS budget.
- **Skew robustness is mandatory.** Real routing produces a skewed, batch-varying `tokens_per_expert` histogram that is generally not a clean multiple of any tile dimension, and some experts get 0 tokens. Therefore: treat `tokens_per_expert` as a runtime tensor, never a compile-time `M_per_group` and never `assert M_per_group % BLOCK_M == 0`; handle the empty-group case (skip launch / branch on zero rows); avoid static work partitioning that assumes a fixed per-expert count; and validate every MoE round on a skewed distribution (top-1 with capacity factor, and a near-degenerate case where one expert takes ≥ 50% of tokens), not only the uniform layout. A gain that vanishes under skew is benchmark over-fit and must be rolled back.

## 13. RDNA / WMMA Paths & the gfx1250 TDM Pipeline

RDNA is wave32 with WMMA, not MFMA — the lane layout, operand width, and barriers all differ; share one kernel across generations by isolating the difference behind a thin wrapper.

- **Confine the generation-specific cost** to four places: the `_wmma_op` call (RDNA3 `gfx11` uses v16 operands with lanes 16–31 mirroring 0–15; RDNA4 `gfx120x` uses v8 operands), the LDS-read shape (gfx11 synthesizes a v16 operand by concatenating two v8 loads), the accumulator store-back row formula (generation-specific — using the wrong one silently transposes output rows), and the barrier asm (`gfx11` uses `s_waitcnt lgkmcnt(0)` + `s_barrier`; `gfx12+` uses split `s_barrier_signal` / `s_barrier_wait` / `s_wait_dscnt`). Keep the inner WMMA loop ("load all B, then 1 A → `reg_n` WMMAs") identical to keep live registers low; reversing it inflates register pressure and spills. Use 128-bit (v8) LDS loads/stores and K-pad the LDS tile for bank conflicts.
- **Preshuffle B for WMMA too.** Lay FP8/low-bit B out in the WMMA-native byte tiling so each lane `buffer_load`s a contiguous pack straight into the WMMA operand; flatten loop-carried register tiles to a flat list for `scf.for` and unflatten inside the body.
- **gfx1250 Tensor Data Mover.** TDM issues 2-D async copies driven by a two-dgroup descriptor — the RDNA4 analogue of CDNA `buffer_load_lds`. Factor the descriptor so only the LDS address and `addr_lo` are patched per issue (reuse `dgroup1`/`addr_hi`); prime `num_buffers−1` stages in the prologue with `pipeline_fence(outstanding=...)`, then issue the next stage's `tensor_load_2d` from a **mid-compute callback inside the WMMA tile** so loads of stage N+P overlap stage N's matrix work, gated by `pipeline_fence_signal` / `pipeline_fence_wait`. A `sched_barrier(0)` before the compute tile pins the TDM issue ordering; the tail must emit a final `pipeline_fence(outstanding=0)` before the epilogue reads LDS.
- **TDM gather with carry-safe addr64.** For MoE row-gather A, hoist the gather descriptor build out of the K loop (cache row indices, predicate, `addr_lo`/`addr_hi`) and advance only the K-byte offset per iteration via `update_tensor_gather_descriptor_addr64` — the `_addr64` variant propagates the lo-add carry into `addr_hi`. The shorter addr-lo-only update silently wraps a 4 GiB page and manifests as a hard hang (not a wrong result) at scale; always use the carry-safe form. Hoist the B / B-scale descriptors too, or the loop stays SALU-bound.
- **Wave-specialized TDM.** AND a wave-owner test into the issue predicate so only the loader wave(s) drive each chunk; enabling it on a kernel that still expects both descriptors per wave issues half the loads.
- **Do not import CDNA scheduling knobs to RDNA.** `sched_mfma`/`s_setprio` ratios are MFMA/wave64-specific; the right WMMA interleave is different.

## 14. Cross-GPU Collectives

For intra-node multi-GPU collectives in the small/medium-message regime where a launched NCCL kernel is too heavy.

- **Peer-signal start/end barriers over IPC-mapped uncached buffers.** Bracket a unit of cross-GPU work with a `signal_start` / `signal_end` pair: each block increments a per-block generation flag, the first `ngpus` lanes one-sidedly store it into every peer's slot with `SC0|SC1` cache modifiers, then poll the local slot. This coordinates one-sided peer writes without a device-wide fence.
- **L1 invalidation on the END poll.** Invalidate L1 inside the end-poll body so each iteration sees fresh peer-written data despite caching; omitting it lets a rank observe stale end-flags. The signal buffer **must** be allocated uncached or the poll spins forever / observes stale generations.
- **Two-stage variants** (reduce-scatter + all-gather) protect each phase with its own start/end pair — do not collapse to a single barrier; the reduce-scatter result must be visible before the all-gather begins. The cache modifiers, the uncached allocation, the per-block (not global) flag, and the rank-ordered peer base list are all load-bearing.

## 15. Host-Side & Deployment

- **JIT warmup before graph capture.** A JIT kernel cannot be captured directly — the first launch issues non-capturable host work. Warm it up once on an ordinary stream with the *same* shapes/dtypes/constexpr key, `torch.cuda.synchronize()`, then re-launch inside `torch.cuda.graph` on a dedicated capture stream that `wait_stream`s the current stream, passing `stream=capture_stream` to the launcher. Any change to a `Constexpr` key between warmup and capture re-triggers JIT inside the capture region; reset destination buffers if correctness depends on a clean state.
- **Amortize host-side preshuffle.** Weight preshuffle (theme 4) is a one-time host cost — do it once at load/quantize time, not per step.
- **Keep launch facades torch-agnostic.** A small dtype-string table (dtype → MLIR type / vec width / byte size) plus a `TensorView` over swappable load/store impls keeps kernel callsites from branching on `torch.dtype`; cache the compiled function on the kernel object so the first call compiles+runs and later calls fast-dispatch. Extend the dtype table when adding a dtype rather than handling it at the callsite (it silently returns `None` otherwise).

---

## Training-Transfer Guardrails

Every accepted gain must reproduce in a real fwd+bwd training step, not just the reuse-the-same-tensor benchmark loop:

- Prefer **kernel-side** mechanisms (themes 1–13) — they transfer 1:1.
- A **weight-quantization/preshuffle cache** keyed on `(id(weight), weight._version)` is bounded by `quant_time(weight) / step_time`; report that per-step number, not the benchmark headline. Never cache on activation / grad_out / activation-scale identity.
- For **MoE / grouped** work, every gain must hold under a skewed `tokens_per_expert` distribution (theme 12). Treat per-expert counts as runtime values, handle empty groups, and never bake `M_per_group` into a compile-time constant.

## Cross-cutting Failure Modes (FlyDSL-specific)

- Python-int loop bounds → loop unrolls, `init=` dropped, prefetch/double-buffer silently disabled (use `fx.Index`).
- XOR swizzle applied on only one of write/read → silent wrong rows; mask sized for the wrong bank count (32 vs 64) → residual conflicts on gfx950.
- Ping-pong buffer (or parity flag) not rotated → reads stale LDS.
- `s_waitcnt` / `gpu.barrier` dropped between an LDS producer and consumer → silent corruption under double-buffering; a divergent barrier hangs the GPU.
- `sched_*` hint counts not summing to the body's opcode counts → backend reverts to the default schedule with no warning.
- Wrong preshuffle `kpack`, wrong MMA operand order, or `buffer_load` offset in bytes instead of elements → loads misalign / garbage data, not a crash.
- gfx1250 TDM gather without the `addr64` carry-safe update → hard hang at large tensor sizes.
- A benchmark gain larger than the kernel change can structurally produce → re-check for an identity-keyed activation cache or a uniform-MoE assumption before accepting the round.
