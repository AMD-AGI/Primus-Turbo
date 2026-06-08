# FlyDSL — Programming Model & Authoring Rules

This file is the working reference for *editing* FlyDSL kernels: the concrete API spine, how loop-carried state and shared memory actually work under tracing, the correctness rules that silently corrupt results when broken, and the debug / profiling workflow. Read [`overview.md`](overview.md) first for what FlyDSL is and which abstraction path to use; read [`optimization-directions.md`](optimization-directions.md) for what to change. Skeletons below are illustrative FlyDSL, not copies of any specific kernel.

## 1. The Layout-Algebra Authoring Pipeline

A kernel maps a global tensor → per-block tile → per-thread fragment → compute → store, entirely through layout transforms.

```python
# 1) wrap so AMD buffer descriptors are emitted
A = fx.rocdl.make_buffer_tensor(A)                      # default max_size=True

# 2) divide to a per-block tile, then select this block's tile
bA = fx.slice(fx.zipped_divide(A, fx.make_tile(BM, BK)), (None, bid))

# 3) atoms = one hardware instruction each
copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), dtype)
mma_atom  = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.Float32))

# 4) tile atoms across threads (TiledCopy / TiledMma)
tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((M_rep, N_rep, K_rep), strides))
thr_mma   = tiled_mma.thr_slice(tid)

# 5) per-thread partition + 6) fragments + retile bridge
src_A  = thr_copy_A.partition_S(bA)        # this thread's global slice
frag_A = thr_mma.make_fragment_A(bA)       # the register tile the MMA consumes
fx.copy(copy_atom, src_A, thr_copy_A.retile(frag_A))   # load straight into MMA layout

# 7) execute
fx.gemm(mma_atom, frag_C, frag_A, frag_B, frag_C)
```

Key API building blocks:

- **Layout:** `make_layout(shape, stride)`, `make_tile`, `make_ordered_layout`, `make_composed_layout` (compose a `SwizzleType` over a base layout), `logical_divide`, `zipped_divide`, `slice`, `make_view`, `get_iter`.
- **Atoms:** `make_copy_atom(BufferCopy{16,32,64,128}b() | UniversalCopy(bits), dtype)`, `make_mma_atom(MFMA(M,N,K,acc) | WMMA(...))`.
- **Tiled ops:** `make_tiled_copy(atom, layout_tv, tile)`, `make_tiled_copy_A/B/C(atom, tiled_mma)` (derive the copy partition from the MMA so fragments match without manual indexing), `make_tiled_mma(atom, layout, tile)`.
- **Partition/fragments:** `get_slice(tid)` → `partition_S/D`; `thr_slice(tid)` → `partition_A/B/C`; `make_fragment_like`, `make_fragment_A/B/C`; `retile(frag)` (rewrites a view, moves no data) to let one register tile satisfy both copy and MMA layouts.
- **Execute:** `copy(atom, src, dst, pred=...)`, `copy_atom_call`, `gemm(atom, D, A, B, C, traversal_order=...)`, `mma_atom_call_ssa`.

> The MMA atom is the anchor: it fixes the A/B/C/D register layout, hence the LDS tile layout, the swizzle formula, and the epilogue re-tile. Choose it before designing staging — re-choosing it later invalidates all of them. The `make_tiled_mma` replication layout (e.g. `(M_rep, N_rep, K_rep)` strides) decides how many waves cover M/N/K; it and every derived `tiled_copy_*` must change together, never independently.

## 2. Loop-Carried State & Software Pipelining

This is the single most error-prone area, because tracing decides whether a loop becomes a real `scf.for`.

- **`range(start, stop, step, init=[...])`** lowers to `scf.for` with explicit loop-carried operands (true phi nodes). This is the only way to carry a prefetched value, accumulator, or online-softmax state across iterations. **The bounds must be `fx.Index(...)`** — plain Python ints make the trace treat it as a Python `range`, unroll the loop, and *silently drop* `init=`.
- **`range_constexpr(...)`** is a compile-time-unrolled Python loop — used for fixed inner steps (MFMA clusters, tile repeats, `sched_*` emission). Lists of register fragments built inside `range_constexpr` are legal precisely because the loop is unrolled; converting such a loop to a runtime `scf.for` would break that and move data to memory.
- A pure Python **`data = next_data` rebinding does not create a phi** — both names alias the same SSA value and the load is hoisted as loop-invariant. Prefetch only takes effect when the prefetched value is threaded through `init=` and `yield`.
- **Unwrap at the boundary only:** `init=` and `yield` lists need raw IR values (`v.ir_value() if hasattr(v, "ir_value") else v`); keep typed `fx.Int32` / `Vector` / `ArithValue` inside the body.

```python
# Prologue issues iteration 0; steady loop carries the "next" buffer; epilogue drains.
next_A = buffer_ops.buffer_load(rsrc_A, offsets(START), vec_width=4)
init   = [_unwrap(v) for v in (next_A, acc)]
for iv, state in range(fx.Index(0), fx.Index(N - 1), fx.Index(1), init=init):
    A, acc = state[0], state[1]
    next_A = buffer_ops.buffer_load(rsrc_A, offsets(iv + 1), vec_width=4)  # overlaps MFMA below
    acc    = rocdl.mfma_f32_16x16x16_f16(t(A), t(B), acc)
    results = yield [_unwrap(v) for v in (next_A, acc)]
A, acc = results[0], results[1]          # epilogue consumes the last prefetch
acc    = rocdl.mfma_f32_16x16x16_f16(t(A), t(B), acc)
```

## 3. Shared Memory (LDS)

- Allocate inside the kernel module: `SmemAllocator(None, arch, "smem0").allocate_array(T.i8, n_bytes)`; obtain a typed pointer via `buf(allocator.get_base())`; and call `allocator.finalize()` in the GPU module body before launch — forgetting it leaves the LDS symbol unresolved.
- For a swizzled, multi-stage tile, compose the layout: `make_composed_layout(SwizzleType.get(b, m, s), make_ordered_layout((BM, BK, STAGES), order))` then `make_view(get_dyn_shared(dtype), layout)`. The `STAGES` dimension is how a double buffer is expressed; the `SwizzleType` removes bank conflicts inside the same descriptor.
- **`SmemPtr.get()` caches the view it produces inside the loop scope.** Reusing that cached view in an epilogue after an `scf.for` raises an MLIR dominance error. Reset `smem_ptr._view_cache = None` before the epilogue. For the same reason, obtain raw memrefs once at the top of the block so they dominate all child `scf.for` / `scf.if` regions.
- LDS budget is **64 KiB on gfx942 / 160 KiB on gfx950**; allocation is 512 B-granular on gfx942, 1280 B-granular and 1280 B-aligned on gfx950 — a small over-request can round up and demote occupancy.
- `vector.store` to LDS hard-codes 16-byte alignment; the allocation must satisfy it.

## 4. Load-Bearing Correctness Rules

Each of these produces a silent wrong result, a hang, or a dropped optimization — none raise an obvious error at the call site.

| Rule | Why it bites |
|------|--------------|
| Loop bounds must be `fx.Index(...)` for a real `scf.for`. | Python ints unroll the loop and drop `init=`; the prefetch/double-buffer becomes invisible to MLIR. |
| Thread state crosses the loop only through `init=` / `yield`. | Missing an entry breaks `scf.for` result typing; a Python swap is not a phi. |
| Reset `SmemPtr._view_cache = None` before the epilogue. | A cached LDS view from inside the loop violates SSA dominance after the loop. |
| No value defined in only one `if`/`else` arm and used after. | MLIR result typing fails; hoist the definition or yield a single merged value. |
| `copy`/atom width must satisfy `vec_width * sizeof(elem) ≤ atom_bits`, capped at 128 b. | A mismatch silently fails to vectorize or corrupts; there is no `BufferCopy256b`. Pick the atom from `VEC * elem_bits` (`f16×8`→128 b, `i8×8`→64 b, `f32×1`→32 b). |
| `buffer_load` / `buffer_store` offsets are in **elements**, not bytes. | Off-by-`sizeof(elem)` address bugs look like garbage data, not crashes (FP8 addressed as `i32` needs the byte address ÷ 4). |
| `make_buffer_tensor(t)` should keep the default `max_size=True`. | The JIT cache keys on dtype not shape; a `num_records` captured from the first shape silently truncates larger tensors later. |
| `const_expr(lane == 0)` / `const_expr(thread_id ...)` is wrong. | `lane` / `warp_id` / `gpu.thread_id` are runtime SSA values; the compiler knows their *range*, not the executing lane. Use a runtime `if` or `.select`. |
| Never put `gpu.barrier()` under divergent control flow. | A barrier reached by only some threads of a work-group hangs the GPU. |
| `retile` before mixing copy and MMA fragments. | Without it the same register tile cannot satisfy both the copy layout and the MMA layout. |
| Apply XOR swizzle identically on every LDS write and read. | Asymmetric swizzle reads the wrong rows with no compiler signal — the most common LDS corruption. |
| Rotate ping-pong buffers (and any parity flag) every iteration. | Forgetting one swap reads stale LDS the next iteration; invisible at compile time. |
| Out-of-range GPR/LDS access on CDNA3 does **not** fault. | An over-indexed read returns 0 / register 0; a budgeting bug looks like a numerical bug. |

## 5. Debugging Workflow

Order checks cheapest-first; most "fix didn't work" reports are stale cache.

1. **Invalidate caches unconditionally** — `rm -rf ~/.flydsl /tmp/flydsl*` and clear any `lru_cache`. Re-run before believing any result.
2. **Classify by symptom** and dispatch:
   - all-NaN → softmax `-inf − (−inf)` or `1/0`; guard with `.select` (`(qk_max > NEG_INF).select(diff, 0)`, `(sum > 0).select(sum, 1.0)`).
   - all-zeros → wrong output stride, wrong partition slot, or unwritten buffers; sentinel-init with `-999.0` to detect skipped writes.
   - >50% mismatch → missing partitions (`grid_z < total_parts` without a multi-partition loop) or layout/addressing.
   - 1–5% mismatch → FP8 requant tolerance or a scale-factor double-apply, not necessarily a bug.
3. **Isolate with synthetic inputs** — fill Q/K/V with `1.0` so softmax is uniform and PV collapses to a known value; any deviation is a pure layout/addressing bug. Caveat: uniform inputs do **not** expose a swapped V/P MMA operand — cross-check against a reference.
4. **Isolate by topology** — force a single partition (`one_shot`) to bisect main-kernel vs reduce-kernel bugs.
5. **Verify MFMA operand order** (`gemm(LHS, RHS, acc)`; LHS→M, RHS→N) and `range` vs `range_constexpr` usage before suspecting deeper layout problems.

## 6. Profiling & Stall Triage

FlyDSL kernels are tuned against `rocprofv3` Advanced Thread Trace (ATT), mapping per-instruction stalls back to source lines. The detailed `rocprofv3` / `rocprof-compute` mechanics live in the profiling skill ([`../../../../primus-turbo-develop/run_profile/tool-rocprof/SKILL.md`](../../../../primus-turbo-develop/run_profile/tool-rocprof/SKILL.md)); the FlyDSL-specific parts are:

- Collect ATT with `advanced_thread_trace: true`, a single target CU, a warmup-skipping iteration range, and `FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1` so the source-location column is populated. The primary artifact is `code.json` (per-instruction asm, source loc, total/stall/issue cycles).
- Aggregate stall cycles **per source line** and classify each into a small fixed taxonomy by opcode prefix: `VMEM-load` (`buffer_load_*`/`global_load_*`), `VMEM-wait` (`s_waitcnt vmcnt`), `LDS/SMEM-wait` (`s_waitcnt lgkmcnt`), `barrier` (`s_barrier`), `MFMA/FMA` (`v_mfma_*`), `LDS` (`ds_read`/`ds_write`). This is what turns "memory- vs compute- vs stall-bound" into "*which line* and *why*."
- The stall taxonomy maps cleanly to the directions catalog: `LDS` stalls → bank-conflict swizzle; `VMEM-wait` → deeper prefetch / async G2S; `barrier` → ping-pong overlap or tail-window tuning; high `MFMA` with low TFLOPS → barrier/`s_waitcnt` stalls, not a scheduler win.

### Occupancy model (CDNA3 / CDNA4)

Stall fixes that spill registers can regress — always cross-check occupancy. On gfx942/gfx950 arch-VGPR and AccVGPR share **one 512-entry per-SIMD pool**, so occupancy is the min across every limiter:

```
occupancy = min( 512 // (arch_vgpr + accum_vgpr),   # NOT 256 / max(arch, accum)
                 lds_limit_per_simd,
                 800 // sgpr,
                 8 )
```

- Read the authoritative `Accum_VGPR_Count` / `LDS_Block_Size` / `SGPR_Count` from the kernel-trace CSV, and take `arch_vgpr = max(ISA_scan, CSV)` — a single-CU `code.json` disassembly is AGPR-form-blind and under-reports accumulators.
- **Never** use `maxnreg` to force `accum_vgpr = 0` to "make room" — it routes MFMA results through arch-VGPR spills and was measured ~4.5× slower. AccVGPR pressure is paid in occupancy, not avoided.
- A hotspot that collapses onto the `@flyc.kernel` decorator line is a debug-info aggregation artifact for compiler-generated prologue/address math — ignore it and focus on explicit user ops.
- ATT has no cache counters; switch to PMC L2 analysis for L2-hit / 32 B-partial / over-fetch, and split PMC into single-pass jobs (multi-pass with many TCC counters has hung gfx942).
