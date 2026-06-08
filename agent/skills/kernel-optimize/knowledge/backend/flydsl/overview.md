# FlyDSL — Backend Orientation

Read this first when `target_lang` / `target_backend` is **FlyDSL**: a Python-embedded tile-programming DSL that lowers a kernel function through MLIR → ROCDL → LLVM AMDGPU. It is AMD-only and spans CDNA3/CDNA4 (`gfx942` / `gfx950`, wave64, MFMA) and RDNA3/RDNA4 (`gfx11*` / `gfx1200` / `gfx1201` / `gfx1250`, wave32, WMMA, and the gfx1250 Tensor Data Mover).

This file gives the orientation needed before reading or tuning a FlyDSL kernel: the compilation model, the abstraction ladder, the layout-centric authoring spine, and the architecture matrix. The concrete API surface and the load-bearing correctness rules live in [`programming-model.md`](programming-model.md); the catalog of techniques to try lives in [`optimization-directions.md`](optimization-directions.md). For instruction-level hardware behavior (MFMA/LDS/occupancy, FP8 encodings, cross-generation differences) consult [`../../hardware/gfx942/`](../../hardware/gfx942/overview.md), [`../../hardware/gfx950/`](../../hardware/gfx950/overview.md), and [`../../hardware/overview.md`](../../hardware/overview.md); for the algorithm-level levers of a given op, consult [`../../ops/gemm/`](../../ops/gemm/overview.md) and [`../../ops/attention/`](../../ops/attention/overview.md). FlyDSL knowledge is **how those levers are expressed and tuned in this DSL**, not a re-derivation of the algorithm.

## What FlyDSL Is

A FlyDSL kernel is a Python function (`@flyc.kernel`, launched via a `@flyc.jit` host wrapper) whose body is **traced** into MLIR. The DSL exposes hardware instructions as composable Python objects under an `fx`-style namespace: layout algebra, copy/MMA *atoms*, tiled cooperative ops, an LDS allocator, and a large set of ROCDL intrinsics. There is no autotuner-managed scheduling underneath — the author controls tiling, staging, swizzle, pipelining, and even instruction interleave explicitly, then the AMDGPU backend does register allocation and final scheduling.

Two consequences shape everything:

- **It is closer to hand-written HIP/CK than to Triton.** You decide the wave/lane decomposition, the LDS layout, the prefetch depth, and the `sched_*` interleave. The DSL's value is that layout algebra makes those decisions composable and the call sites tiny.
- **Tracing semantics leak into correctness.** Because the Python body is traced, a plain Python `for` loop is *unrolled*, a plain `data = next_data` rebinding does *not* create a loop phi, and a value defined only inside one `if` arm breaks MLIR typing. These are not style preferences — they silently change the emitted IR. They are enumerated in [`programming-model.md`](programming-model.md) and must be internalized before editing a kernel.

## Compilation & JIT Model

- The **first** launch of a given kernel triggers host-side compilation, module load, and caching; later launches with a matching cache key fast-dispatch the cached device kernel.
- Compiled artifacts are cached on disk (under `~/.flydsl` and `/tmp/flydsl*`) plus any in-process `lru_cache` wrappers. The cache key tracks the kernel source / closure and dtype — **not always the tensor shape**. A change to a C++ lowering pass, or a shape that should re-specialize but shares a dtype key, can therefore be masked by a stale cache.
- Practical switches: clear `~/.flydsl /tmp/flydsl*` (and any memoization) whenever "the change did nothing"; set `FLYDSL_RUNTIME_ENABLE_CACHE=0` to bypass the disk cache while iterating on lowering; set `FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1` so profiler traces carry source locations.
- For repeated low-overhead replay, a JIT kernel must be **warmed up and synchronized before** it is recorded into a CUDA/HIP graph (the first launch issues non-capturable host work). See the host-side direction in [`optimization-directions.md`](optimization-directions.md).

## The Authoring Spine: "Layout Is the Glue"

Almost every FlyDSL kernel is the same pipeline from a global tensor down to a per-thread register fragment and back. Divide, partition, copy, and gemm are all defined as layout transforms, so getting the layouts right is most of the work and the operation calls are mechanical:

1. **Wrap** the global tensor — `make_buffer_tensor(T)` so loads/stores emit AMD buffer descriptors.
2. **Divide** with layout algebra to expose a per-block tile — `logical_divide` (1-D / elementwise) or `zipped_divide` (2-D), then `slice` to select this block's tile.
3. **Pick atoms** that map 1:1 to hardware — `make_copy_atom(BufferCopy{16,32,64,128}b / UniversalCopy)` for transfers, `make_mma_atom(MFMA(M,N,K,acc))` or the WMMA equivalent for one matrix issue.
4. **Tile atoms across threads** — `make_tiled_copy` / `make_tiled_mma` (and the MMA-derived `make_tiled_copy_A/B/C`) declare which thread owns which value.
5. **Partition** to this thread's view — `get_slice(tid)` / `thr_slice(tid)` then `partition_S/D` and `partition_A/B/C`.
6. **Allocate fragments** — `make_fragment_like` or `make_fragment_A/B/C`; `retile` when one register tile must satisfy both a copy layout and an MMA layout.
7. **Execute** — `copy(atom, src, dst)` and `gemm(atom, D, A, B, C)`.

Shared memory follows the same shape through `SmemAllocator`. This spine is detailed in [`programming-model.md`](programming-model.md).

## Abstraction Ladder — Pick the Right Path

FlyDSL exposes three levels of control; choose the highest one that still expresses the access pattern, and drop down only where micro-control is genuinely needed.

| Path | What it is | Use when | Cost of using it |
|------|-----------|----------|------------------|
| **Layout API** | `make_buffer_tensor` + layout algebra + atoms + `TiledCopy` / `TiledMma` | The default for new kernels: elementwise, reductions, GEMM/attention, epilogue fusions. Layout encodes the addressing and vectorization. | Some transforms (transpose, preshuffle, XOR swizzle) need a hand-built layout descriptor; a few helpers still fold wave-uniform offsets into VGPR (voffset) instead of SGPR. |
| **Raw `buffer_ops`** | `create_buffer_resource` + manual byte/element-offset `buffer_load` / `buffer_store` | Maximum micro-control: specialized buffer addressing, hand-scheduled `s_waitcnt`, scattered/non-contiguous stores, masked OOB loads. | Addressing and vectorization arithmetic lives in the kernel body; diverges from the layout-API kernels and is harder to reason about. |
| **`TensorView` shim** | A `TensorView` façade over `GTensor` / `STensor` / `TorchTensor` backends with a small dtype table | Lightweight, torch-agnostic ad-hoc launches and host-side unit testing of an indexing/copy body. | Only row-major + per-axis tiling; dtype table covers `{f32,f16,bf16}` and silently returns `None` outside it. Not a substitute for layout algebra. |

A common, supported mix: A staged through the layout API + LDS, while a preshuffled weight B is viewed through a hand-built layout descriptor and streamed straight to registers. Layers are orthogonal — they can coexist in one kernel.

## ROCDL Intrinsic Families Exposed

FlyDSL surfaces the AMDGPU instruction set so optimization can reach the metal. The families that matter for tuning:

- **Matrix:** `mfma_*` (CDNA wave64) and `wmma_*` (RDNA wave32), wrapped by `make_mma_atom` / `gemm` or called directly (`mma_atom_call_ssa`).
- **Global memory:** `BufferCopy*b` atoms, `buffer_load` / `buffer_store`, and the direct-to-LDS `buffer_load_lds` / `raw_ptr_buffer_load_lds` (async global→LDS, bypassing VGPRs).
- **LDS:** `ds_read` / `ds_write`, paired `ds_read2` / `ds_write2`, and the gfx950 transpose-loads `ds_read_*_tr` / `ds_read_b64_tr_b8`.
- **Cross-lane:** `ds_bpermute`, `shuffle_xor`, and the DPP butterfly via `update_dpp` — reductions/prefix-sums/argmax with no LDS round-trip.
- **Scheduling & sync:** `sched_mfma` / `sched_dsrd` / `sched_dswr` / `sched_vmem` / `sched_barrier`, `s_setprio`, `s_waitcnt`, `gpu.barrier`.
- **Conversion:** `cvt_off_f32_i4`, `cvt_pk_bf16_f32`, and bit-pattern tricks for FP4/FP8/int4 pack/unpack.
- **gfx1250 Tensor Data Mover (TDM):** `tensor_load_2d` / `tensor_load_gather`, `make_tensor_gather_descriptor`, `update_tensor_gather_descriptor_addr64`, and the `pipeline_fence` / `pipeline_fence_signal` / `pipeline_fence_wait` handshake — async 2-D descriptor-driven copies, the RDNA4 analogue of CDNA's `buffer_load_lds` + `s_waitcnt`.

## Architecture Coverage Matrix

FlyDSL kernels are written once but specialize per target; the generation-specific cost is usually confined to the MMA op, the LDS transpose path, the barrier sequence, and a few address formulas.

| Target | Family | Wave | Matrix | Async global→LDS | LDS / banks | FP8 | Notes |
|--------|--------|------|--------|------------------|-------------|-----|-------|
| `gfx942` | CDNA3 (MI300X/MI325X) | 64 | MFMA | `buffer_load_lds`, 4 B/op | 64 KiB / 32 banks | FNUZ | no `ds_read_*_tr`; software V-transpose or pre-swizzled LDS |
| `gfx950` | CDNA4 (MI350/MI355X) | 64 | MFMA (+ wider-K, MX/FP4 scale, F8F6F4) | `buffer_load_lds`, 16 B/op | 160 KiB / 64 banks | OCP | `ds_read_*_tr` transpose-load; deeper multi-buffer rings realistic |
| `gfx11*` | RDNA3 | 32 | WMMA (v16 operand) | — | bank-conflict via K-pad | — | lanes 16–31 mirror 0–15; legacy `s_barrier` |
| `gfx1200` / `gfx1201` | RDNA4 | 32 | WMMA (v8 operand) | — | — | OCP | split `s_barrier_signal` / `s_barrier_wait`; FP8 preshuffle WMMA paths |
| `gfx1250` | RDNA4 + TDM | 32 | WMMA | TDM `tensor_load_2d` / gather + `pipeline_fence` | — | — | mid-compute TDM callbacks; `addr64` gather for >2 GiB tensors; cluster barriers |

FP8 encoding (FNUZ on gfx942 vs OCP on gfx950), LDS bank count (32 vs 64), and async granularity (4 B vs 16 B) are **correctness / occupancy gates**, not recompiles — re-derive scales and swizzle masks when porting in either direction.

## Op Families Implemented in FlyDSL

The backend ships kernels across the whole training/serving surface, so most tuning tasks have a same-family reference to read first:

- **GEMM** — dense BF16/FP16/FP8 (preshuffle B, double-buffered LDS, interleaved-cluster G2S/S2R), split-K HGEMM, low-bit W4A16/W4A8.
- **Attention** — FlashAttention prefill, MLA decode, paged / sliding-window decode, FP8 paged attention.
- **MoE** — token sorting, top-K gating, 2-stage block-scale GEMM, fused SwiGLU + FP4-quant epilogue.
- **Norm / reduction / quant** — RMSNorm (+ fused-add + smooth/dynamic quant), LayerNorm, softmax, MXFP4 / E2M1 packing.
- **Fusions & collectives** — RoPE + reshape-and-cache + FP8 pack, cross-GPU all-reduce.

## Directions Map

| If you are… | Start with these themes in [`optimization-directions.md`](optimization-directions.md) |
|-------------|------------------------------------------------------|
| Writing/modifying any FlyDSL kernel | Software-pipelined prefetch; LDS ping-pong staging; bank-conflict-free LDS |
| Tuning a compute-bound MFMA/WMMA loop | Hot-loop instruction scheduling; async global→LDS staging; occupancy & loop-carried state |
| Optimizing a weight GEMM | Preshuffle operands as layout descriptors; low-precision packing & conversion |
| Working on attention / decode | Online-softmax pipelines; cross-lane primitives; bank-conflict-free LDS (V transpose) |
| Working on MoE | MoE dispatch, sorting & gating (with skew robustness); split-K & multi-stage reduction; fused epilogue |
| Targeting RDNA / gfx1250 | RDNA / WMMA paths & the gfx1250 TDM pipeline |
| Store-bound epilogue | Epilogue staging & fusion |
| Multi-GPU | Cross-GPU collectives |
| Deploying (graphs, repeated launch) | Host-side & deployment |

Before changing a kernel, confirm the FlyDSL **correctness rules** in [`programming-model.md`](programming-model.md) — most "silent wrong result" bugs in this backend come from tracing semantics (loop unroll, view-cache dominance, branch-local definitions, element-vs-byte offsets) rather than from the optimization itself.
