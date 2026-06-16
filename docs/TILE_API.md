# Shared GEMM tile-emit API — design + handoff

Status: **design approved, not yet implemented.** This document is the executable
spec for a follow-up agent. Hard constraint from the requester: the abstraction must
be **performance-neutral** (zero emitted-IR change) and **simple to call**.

Goal: one shared per-tile "emit" API that serves BOTH the standalone GEMM emitters
and the mega-MoE fused operators, so the GEMM tile is written once.

---

## 1. Scope

**In scope (v1):** the `_nt_pipeline`-backed uniform-K family — NT and (inlined) NN —
for both sides:
- standalone: dense fp8 NT/NN (`kernels/gemm_fp8_kernel.py`), bf16 grouped NT/NN
  (`kernels/gemm/gemm_bf16_kernel.py`)
- fused: dispatch NT/NN (`kernels/mega_moe/dispatch_grouped_gemm.py`), combine NT/NN
  (`kernels/mega_moe/combine_grouped_gemm.py`)

**Out of scope (do NOT touch in v1):**
- All TN variants: dense-fp8 TN (dual-transpose + inplace asm-MMA + `_LDS_CS=1056`
  bank-spread), CuTe-fragment TN (`kernels/grouped_gemm.py::gemm_tn`).
- The forced-persistent TN+combine kernels: `kernels/mega_moe/grouped_gemm_tn_combine.py`,
  `kernels/mega_grouped_gemm_combine.py::kwc`, `kernels/mega_moe/wgrad_transposed.py`.
- The entire CuTe-fragment family in `kernels/grouped_gemm.py` (tiled_mma/partition_S/
  `fx.gemm` — a different abstraction layer, not byte-view loaders).

**Why TN is excluded:** TN contraction is **runtime-ragged-K** (`n_chunk=(m1-m0)//BK`),
so its K-loop is a runtime `scf.for`. FlyDSL rewrites runtime loops to scf **only
lexically inside the `@flyc.kernel` body**, so a ragged-K loop CANNOT be a shared
module-level function — this is exactly why those kernels carry duplicated
"KEEP BYTE-IDENTICAL" skeletons. `_nt_pipeline` is safe to share only because it is a
`range_constexpr` (compile-time-unrolled) loop. Folding TN in would require a
runtime-loop-capable shared body = framework work, deferred.

---

## 2. Final API

File: `kernels/gemm/tile.py` (NEW).
**This module must NOT contain `from __future__ import annotations`** — it stringifies
the `@fx.struct` field annotations and FlyDSL can no longer compute the LDS layout.
(Every `kernels/gemm/*.py` already omits it for this reason.)

```python
def make_tile_spec(
    *,
    layout: str,            # "nt" | "nn"            -> compile-time: B-loader + b_kstep formula
    dtype: str,             # "fp8" | "bf16"         -> compile-time: buffer/mfma/s2r classes + K_SUB
    contraction_size: int,  # K
    block_m: int,
    block_n: int,
    b_kpad: int = 8,        # nn tr16 only: B-LDS STRIDE_NT pad
) -> "TileSpec": ...        # host-side, @lru_cache(maxsize=256); geometry + dtype/layout policy inside


class TileSpec:
    shared_storage          # the @fx.struct class -> kernel: SharedAllocator().allocate(spec.shared_storage).peek()
    k_unit: int             # group_base unit multiplier: bf16 -> K_BYTES (=2K), fp8 -> K

    def emit(self, *,
        # tensors (host wrapper has flattened them)
        A, B, C, A_scale, B_scale,
        lds,                # the allocated SharedStorage instance
        # resolved tile coords (dynamic; caller computes)
        block_m, block_n, c_m, c_n,
        # optional seams; defaults == standalone-dense behavior
        group_base = 0,     # per-expert B slab base = g_idx * c_n * spec.k_unit; 0 = dense
        row_limit  = None,  # store row clamp (predication, NOT a branch); None = c_m
        nt_vmcnt   = 3,     # NT G2S drain depth (tuning, not correctness)
        drain      = False, # NN tr16 read drain; fused (co-resident comm) passes True
        max_size   = False, # buffer num_records cap; fused pool byte-view >4GB MUST pass True
    ) -> None: ...
```

`lane_id/wave_id/wave_m/wave_n` are NOT params — `emit()` derives them from
`fx.thread_idx.x` internally. **`emit()`'s first statement must be `_ = str(fx.thread_idx.x)`**
(materialize tid before any lazy tr16 S2R use) when `layout=="nn"`.

### Caller usage (the whole public surface)

```python
SPEC = make_tile_spec(layout="nt", dtype="bf16",
                      contraction_size=hidden_size, block_m=BM, block_n=BN)  # module-level / compile-time

# inside @flyc.kernel:
lds = fx.SharedAllocator().allocate(SPEC.shared_storage).peek()
# ... caller resolves block_m/block_n via its OWN scheduler (xcd swizzle OR linear no-sync) ...
SPEC.emit(A=POOL, B=WEIGHTS, C=OUTPUT, A_scale=ones, B_scale=ones, lds=lds,
          block_m=block_m, block_n=block_n, c_m=c_m, c_n=c_n,
          group_base=g_idx * c_n * SPEC.k_unit, row_limit=c_m, max_size=True)   # dispatch NT
# dispatch NN: same + drain=True ; standalone dense: group_base=0, row_limit=None, max_size=False
```

---

## 3. The seams (why each param exists)

| param | reason | evidence |
|---|---|---|
| `block_m/block_n` | tile-id mapping is the scheduler's job; xcd-swizzle (standalone, L2 reuse) vs linear no-sync (fused, front-load) is a HARD divergence — fused MUST be linear or padding tiles starve real work | dispatch_grouped_gemm.py:216-222; scheduler.py:48-63 |
| `group_base` (+`k_unit`) | per-expert B slab; the only thing making a tile "grouped" vs "dense". Units differ: NT bytes, NN elements | dispatch:173 (`g_idx*c_n*K_BYTES`), :383 (`g_idx*c_n*hidden`); gemm_bf16_kernel.py:1242 |
| `row_limit` | clamp stores for over-launched padding rows; uses `arith.select` predication, NOT control flow (a branch would be dynamic control flow, illegal in a shared free fn) | pipeline.py:62,242-248; gemm_bf16_kernel.py:1237 |
| `drain` (NN only) | THE single fused-vs-standalone NN difference: co-resident comm breaks the implicit lgkmcnt ordering the standalone relies on -> MFMA eats undrained (~1e38) B. Exact sites below | see §4 |
| `max_size` | fused pool byte-view exceeds 4GB; `num_records` wraps mod 2^32 and bound-zeros valid rows. standalone uses False | dispatch:389-394; memory `bf16-byteview-buffer-4gb-num-records-overflow` |
| `layout`/`dtype` (compile-time) | selects loader/mfma/store classes, K_SUB, b_kstep formula, STRIDE_NT — the whole fp8/bf16 + NT/NN policy | gemm_fp8_kernel.py vs gemm_bf16_kernel.py loader tables |

### Must stay caller-side (cannot enter the API)
Tile-id mapping; scoreboard spin (`while signal<expected`); padding early-exit (`if
block_m<real_tiles`); the comm role (`dispatch_tile`/combine push); role split
(`block_index<comm_blocks`); grid sizing. All of these only COMPUTE and FEED
`block_m/block_n/group_base/row_limit` into `emit()`. They contain dynamic control
flow, which is only legal inside the `@flyc.kernel` body or a nested-free-fn — never in
a shared module-level function.

---

## 4. The `drain` seam — exact lines (NN)

Fused NN is byte-identical to standalone NN EXCEPT `drain=True` on the two main-loop
b reads + the two epilog b reads:

- standalone `gemm_bf16_kernel.py::_compile_grouped_nn_contiguous_bf16`:
  main `:1296`, `:1303` (default `drain=False`); epilog `:1329`, `:1331`
- fused dispatch `dispatch_grouped_gemm.py::_make_kernel_nn::gemm_tile`:
  main `:439`, `:446` (`drain=True`); epilog `:472`, `:474`
- fused combine `combine_grouped_gemm.py::_make_kernel_nn::gemm_tile`:
  main `:488`, `:495` (`drain=True`); epilog `:520`, `:522`

Everything else (prelude `_db_drain`, 4-MFMA interleave, `b_g2s.issue/commit`, per-iter
`_db_drain()`, swaps, stores) is identical. NT has NO drain knob — it already uses the
shared `_nt_pipeline` and differs only by group_base/row_limit/tile-map.

---

## 5. FlyDSL constraints (bound the API shape)

1. **Trace-time inline emission.** A Python fn called inside `@flyc.kernel` runs at trace
   time and splices ops inline; no device function frame. Emitting the SAME op sequence
   = bit-identical IR. This is WHY the abstraction is perf-neutral by construction.
2. **AST transform is body-only.** Inside a shared free fn, SAFE = straight-line SSA math,
   `arith.select`, op-emitter method calls, `range_constexpr`, static `if` on Python/compile-time
   values. UNSAFE = runtime `if`/`while`/`for` on dynamic values (need lexical scf rewrite).
   So scoreboard spin / early-exit / comm role stay in the kernel body, not in `emit()`.
3. **Lexical scf hazard.** Runtime loops can't be shared free fns -> "KEEP BYTE-IDENTICAL".
   `_nt_pipeline` is exempt because it is `range_constexpr` (unrolled, no scf node). Rule:
   the uniform-K pipeline = ONE shared fn, never copy-and-edit it.
4. **SharedStorage factory is OK.** A factory may build & return a `@fx.struct` class closed
   over the lds sizes; kernel does `.allocate(cls).peek()`. Caveats: no `from __future__`,
   sizes must be compile-time consts, don't mix with `get_dyn_shared`.
5. **Ordering:** `str(fx.thread_idx.x)` first when tr16 loaders are used; if any runtime loop
   is ever added, wrap the body in a Name-called nested fn threading only scalars (lists become
   scf iter_args and break).

---

## 6. Rollout (lowest risk first)

1. **Extract `_nn_pipeline(drain=False)`** into `kernels/gemm/pipeline.py`, sibling to
   `_nt_pipeline`. Move the prelude `_db_drain` + K-loop + epilog + 4 stores; `drain`
   gates the 4 `b_s2r.load` calls. Repoint standalone `_compile_grouped_nn_contiguous_bf16`
   (drain=False) and the two fused NN `gemm_tile`s (drain=True) at it. Removes the
   hand-synced NN copies. **Validate before step 2.**
2. **Add `kernels/gemm/tile.py`** with `make_tile_spec` + `TileSpec.emit` for
   `layout in {nt,nn}`, `dtype="bf16"`. Repoint dispatch NT/NN, combine NT/NN, and the
   bf16 standalone NT/NN at it. emit() internally: derive lane/wave -> compute offsets
   from block_m/block_n/group_base -> build loaders/mfma/store per spec -> call
   `_nt_pipeline` / `_nn_pipeline`.
3. (Optional) `dtype="fp8"`: migrate dense-fp8 NT/NN onto the same path. Op sequences
   were confirmed identical to `_nt_pipeline`. NOTE: this diverges vendored Primus-Turbo
   `gemm_fp8_kernel.py` from upstream — only do it if upstream re-sync is not a concern.
4. TN / TN-combine: untouched.

---

## 7. Perf-neutrality proof (required gate)

1. **IR diff (primary, sufficient):** `FLYDSL_DUMP_IR=1`, dump MLIR before/after for each
   affected kernel, normalize SSA numbering, diff. Empty diff = perf-neutral by construction.
2. **ISA dump (confirm):** diff instruction stream; check barrier / `s_waitcnt` / MFMA counts
   unchanged (these drive the known bottlenecks; the hand-scheduled ladder must not move).
3. **gate-3 (e2e):** `./run.sh overlap` (cos vs torch), `./run.sh overlapperf` (cos + perf vs
   turbo), `./run.sh roofline` (no component regressed). Isolated: `tests/bench_mega_dispatch_gg_nt.py`.

**CRITICAL test-runner gotcha:** `run.sh` sets `ROOT=<main tree>` and tests do
`sys.path.insert(0, ../)`, so running the MAIN test imports MAIN code, not the worktree.
To validate worktree changes, run the **worktree's own** test file
(`/apps/zhuang12/megamoe-uccl-intranode-workspace/.claude/worktrees/bold-fox-u7oz/tests/...`)
so `__file__/..` resolves to the worktree.

---

## 8. Key file:line index (so the next agent need not re-investigate)

- shared NT pipeline: `kernels/gemm/pipeline.py:42-248`; `_barrier_if_eq` scf note `:23-40`;
  `range_constexpr` loop `:113`; `nt_vmcnt` drain `:150-157`; `row_limit` store `:242-248`.
- standalone NN body (to extract): `kernels/gemm/gemm_bf16_kernel.py:1149-1366`
  (loop `1291-1326`, drain sites `1296/1303/1329/1331`, SharedStorage `1192-1201`,
  STRIDE_NT `1189-1190`).
- standalone NT body: `kernels/gemm/gemm_bf16_kernel.py:395-477` (calls `_nt_pipeline:457`).
- bf16 loader/mfma/s2r adapters: `kernels/gemm/gemm_bf16_kernel.py` `_Bf16MfmaAdapter:159`,
  `_Bf16S2RAdapter:149`, `_Bf16G2S:218`, `_Bf16S2RTrAdapter:274`.
- dense fp8 NT (op-identical to `_nt_pipeline`): `kernels/gemm_fp8_kernel.py:103-328`;
  dense NN `:418-639`; dense TN (excluded) `:667-965`.
- fused dispatch NT tile: `dispatch_grouped_gemm.py:165-206`; NN tile `:372-488`;
  linear map + scoreboard + early-exit `:208-235`; comm role `:128-163`; max_size `:389-394`.
- fused combine NT/NN tile: `combine_grouped_gemm.py:112-154` / `:479-536`; producer signal `:201-204`.
- scheduler (must stay caller-side): `kernels/gemm/scheduler.py` (`NormalScheduler.locate:48-63`,
  `GroupedContiguousScheduler`, `KGroupedScheduler:200-211`).
- utils interfaces: `kernels/gemm/utils.py` (`G2SLoader:107`, `StoreCPerTensor:242`,
  `make_fp8_buffer_tensor:38`, `compute_global_swizzle`, `mask_a_tail`, `make_value_attrs`).
- excluded TN-combine: `grouped_gemm_tn_combine.py` (ragged `:127-128`, persistent broadcast
  `:92-100,183-190`); `mega_grouped_gemm_combine.py:415,497` ("KEEP BYTE-IDENTICAL").

Relevant memories: `fused-tr16-nn-needs-explicit-drain`, `bf16-byteview-buffer-4gb-num-records-overflow`,
`mega-moe-dispatch-gg-nt` (no `__future__`, per-iter autotune reset, LINEAR over-launch map),
`pipeline-barrier-stall`, `d-topk-w-fused-into-swiglu-bwd` (worktree-test gotcha).
