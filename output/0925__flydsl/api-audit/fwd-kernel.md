# Stable API usage audit: NOT STABLE-ONLY (gfx1250 fwd kernel)

Scope: `output/0925__flydsl/fwd341/op_clean/flydsl_fwd/fmha_fwd_prefill_a16w16_m32x8.py`
(abbreviated **K** below). All K line numbers are from the **edited** file. Helper modules
are covered only where K reaches them (§ "Reached through helpers").

Policy: FlyDSL `docs/api_stability.md` at tag `v0.3.4.1`. Stable catalog: `scripts/list_stable_apis.py
--repo-root <synthetic root = installed /home/lihuzhan/.local/flydsl0341/flydsl + v0.3.4.1 docs>`,
which gives 493 stable paths (518 with `--include-deprecated`). On main, the catalog adds only
`flydsl.extension.coop.*` and `struct.Empty`, and neither is used here.

Verdict: after cleanup, K has **no DEPRECATED uses** and **no imports of `_`-private FlyDSL
names**. What is left unstable is all gfx1250/cdna5 intrinsics with no stable wrapper in 0.3.4.1,
plus one raw `llvm.load`. The fix is **ISA byte-identical** for the prod shape and two coverage
variants (§ Verification).

---

## Stable uses

| resolved path | K lines |
|---|---|
| `flydsl.compiler.jit` | 1080, 1168, 1242, 1768, 1876 |
| `flydsl.compiler.kernel` (+ returned `launcher.launch(...)`) | 1506, 1630; 1830, 1936 |
| `JitFunction.compile_hints` (non-underscore member of a `flyc.jit` result; returned-object rule) | 1836, 1840, 1942, 1946 |
| `fx.numeric.{Int32, Int64, Float32, Uint32, Index, BFloat16, Float16}` (+ `.ir_value()`, `.ir_type`) | Int32 about 60 sites (e.g. 198, 215, 481, 824); Int64 228, 243-244, 1418, 1546, 1666; Float32 247, 434-436, 463, 497, 567, 943-946, 1364-1368; Uint32 1800, 1907; Index 1245, 1799-1803, 1906-1910; dtype map 141, 292, 707; `.ir_value()` 216, 463, 466, 472-473, 953-955; `Int32.ir_type` 538 |
| `fx.typing.{Pointer, Stream, Vector, T, as_ir_value}` | Pointer 1508-1515, 1632-1637, 1770-1777, 1878-1883; Stream 1795, 1902; Vector 228, 289, 495, 572, 955, 1177, 1323, 1328, 1423 (+ members `.shuffle` 346/639/643, `.broadcast_to` 1179/1328, `.to` 572, `.filled`, `.from_elements`, `.make_type`); T 216, 247, 432; **`as_ir_value` 300-302, 1200 (new)** |
| `fx.arith.{min, max, ceildiv}` | min 506, 813, 815, 829, 838, 1226, 1231, 1240; max 816, 827, 837, 1222, 1226, 1236, 1240; ceildiv 824, 1800, 1907 |
| `fx.arith.{FastMathFlags, fastmath, maxnumf}` | FastMathFlags 433, 447; **`fastmath` context 442, 451, 455, 459 (new)**; **`maxnumf` 439 (new)** |
| `fx.math.{fma, log2}` | fma 567; log2 1368 |
| `fx.primitive.{get_iter, ptr_load, ptrtoint, copy_atom_call}` | get_iter 227, 243; ptr_load 228; ptrtoint 243, 661; copy_atom_call 881, 883, 1086, 1088 |
| `fx.gpu.{SharedAllocator, block_idx, grid_dim, barrier}` | SharedAllocator 660 (+ `.allocate().peek().ptr` 661); **`block_idx` 265, 266, 793, 807, 1409, 1410, 1536, 1661 (new)**; grid_dim 939, 1444; barrier 885, 918, 1074 |
| Numeric/Boolean `.select` | 508, 510, 512, 540, 1325, 1378, 1431, 1449 |
| `range(lo, hi, 1, init=...)` + `yield` loop form (rewriter) | 1245-1253 |

## Private-field writes

- **K: none** (checked assignments, `setattr` and `__dict__` for `obj._x = ...` on FlyDSL objects).
- `[PRIVATE-WRITE]` reached from K through `_run_compiled` (K:2060, K:2235), **in the untouched
  baseline** `op0341/flydsl_fwd/tensor_shim.py:276`: `exe._cf = cf`. This adds a field
  that FlyDSL does not define to a `flyc.jit` `JitFunction`. The object is module-global
  (`_launch_fns`), so the field is shared across calls and across configs. It bypasses nothing, but it
  depends on `JitFunction` accepting new attributes. No public "compiled handle cache" API exists
  (0.3.4.1 does have `flyc.compile` -> `CompiledFunction`). In `op_clean/flydsl_fwd/tensor_shim.py:18-31`
  it is **already fixed by a concurrent helper-cleanup pass (not this change)**: a module dict
  `_COMPILED[exe]`.
- The `self._x = ...` writes in `fmha_b16_buffer_managers.py` (e.g. :263, :380, :1114, :1765) are on
  the kernel's own manager classes, not on FlyDSL objects, so they are not findings.

## Deprecated / unstable uses

No DEPRECATED uses remain. `fx.tdm_ops` (§3 deprecated) is not used; K imports the target path.

| status | resolved path | K lines | reason / why it stays |
|---|---|---|---|
| UNSTABLE (§2.1: not in `rocdl.__all__`) | `flydsl.expr.rocdl.tdm_ops` (`tensor_wait`) | import 47; 884, 1070 | cdna5/TDM is deliberately left out of `rocdl.__all__` (`rocdl/__init__.py:32`). §3 names it the target-specific replacement for `fx.tdm_ops`. No stable wrapper exists. |
| UNSTABLE | `rocdl.wave_id` (FlyDSL wrapper over ODS) | 198 | SGPR wave id. A `thread_idx.x // 32` substitute would move it to VALU. |
| UNSTABLE | `rocdl.mbcnt_lo` | 216 | Stable `fx.gpu.lane_id()` exists, but **it changes ISA** (see Proposed P1). |
| UNSTABLE | `rocdl.wmma_f32_16x16x32_{bf16,f16}` | 291-293 (`_wmma`, annotated at 296) | SSA-returning raw WMMA with `reuseA/B=False`. `fx.rocdl.WMMA` + `fx.gemm` is structural (fragments/tiled MMA), so it is not a cleanup. |
| UNSTABLE | `rocdl.exp2` | 463 | Native `v_exp_f32`. Stable `fx.math.exp2` lowers through `math.exp2` (range reduction), so it is not an ISA-neutral swap (Proposed P3). |
| UNSTABLE | `rocdl.permlanex16` | 470 | Cross-16 lane swap. Stable `fx.gpu.shuffle_xor(v,16,32)` would lower to ds_bpermute/DPP, not `v_permlanex16` (Proposed P2). |
| UNSTABLE | `rocdl.ballot` | 538 | Wave-uniform deferred-rescale gate. No stable ballot. |
| UNSTABLE | `rocdl.s_wait_dscnt` | 620, 1335 | gfx12+/gfx1250 split counters. Stable `fx.rocdl.s_waitcnt` dispatches gfx942/950/11xx/120x only. |
| UNSTABLE | `rocdl.s_wait_asynccnt` | 917, 1072, 1310 | Same reason as `s_wait_dscnt`. V1 loader only; compiled out with `USE_TDM_LOADER=True`. |
| UNSTABLE | `rocdl.sched_barrier(0)` | 1073, 1075, 1113, 1127, 1129 | Full scheduling fence. The stable `fx.rocdl.sched_{mfma,vmem,dsrd,dswr}` are group barriers with different semantics. |
| UNSTABLE (kernel-local helper) | `_async_load_to_lds` -> `rocdl.cluster_load_async_to_lds_b128` / `global_load_async_to_lds_b128` (managers :154-197) | import 107; 915-916, 1091-1092 | Async global->LDS. V1 only. |
| UNSTABLE (vendored helper) | `buffer_ops.create_buffer_resource` / `buffer_store` (raw rocdl/llvm through `flydsl._mlir`, `buffer_ops.py:35-39`) | 1360, 1381 (LSE store, **on the prod path**); 1420, 1432, 1437, 1450 (THD zero-fill) | Stable path is `fx.rocdl.make_buffer_tensor` + `fx.copy`. That swaps the buffer mechanism, which is forbidden here (Proposed P5). |
| UNSTABLE (vendored helper) | `kernels_common.create_llvm_ptr` | 246 | Sink path only. |

### Reached through helpers (not edited here; in the kernel group)
- `fmha_b16_buffer_managers.py`: `fx.rocdl.make_tdm_atom` (cdna5) :981, :1096, :1684, :1710;
  `rocdl.ds_load_tr16_b128` :908, :1323; `rocdl.global_store_async_from_lds_b128` :1851; `tdm_ops.tensor_wait`
  :1125; raw `llvm_dialect` import :37; `_to_raw as _ir` re-export :51-52 (see below).
- `buffer_ops.py`: `flydsl._mlir.ir`, `_mlir.dialects.{llvm, rocdl, arith}`, `llvm.GEPOp` :171, raw `fly`
  dialect `extract_aligned_pointer_as_index` :226-229 (UNSTABLE raw FlyDSL binding, not UPSTREAM),
  `flydsl.expr.meta.dsl_loc_tracing` :38 (undeclared module), `flydsl.runtime.device.is_rdna_arch` :39 (STABLE, §2.4).
- `tensor_shim.py`: `flydsl._mlir.ir`, `ir.Value._CAPICreate` (private C-API) in `_to_raw`.
- **K no longer imports `_ir`.** The re-export at `fmha_b16_buffer_managers.py:51-52` is now dead code
  (its own comment says "Drop once it stops"). I left it in place because another pass was editing that
  file at the same time.

## Upstream MLIR operations

- `[UPSTREAM-MLIR]` K:41 `from flydsl._mlir.dialects import llvm as llvm_dialect`, call K:247
  `llvm_dialect.load(T.f32, gptr)` (sink logit, flat global load). This is a direct upstream builder:
  allowed, but unstable under §2.5, because FlyDSL does not guarantee its name, signature or semantics. A stable
  `fx.ptr_load` exists but **changes ISA** (Proposed P4). Only the `has_sink` variants reach this.
- `[UPSTREAM-MLIR]` the `rocdl.*` ODS builders in the table above (`mbcnt_lo`, `wmma_*`, `exp2`,
  `permlanex16`, `ballot`, `s_wait_dscnt`, `s_wait_asynccnt`, `sched_barrier`) are upstream ROCDL ops
  re-exported by `flydsl.expr.rocdl` through `from ..._mlir.dialects.rocdl import *` and left out of `__all__`.
  §2.5 applies to each.
- **Removed by this cleanup:** `arith.MaxNumFOp` (ODS class), `arith.addf`/`subf`/`mulf`
  (reached through `flydsl.expr.arith`'s `from .._mlir.dialects.arith import *`). They are now stable
  `fx.arith.maxnumf` and operators inside `fx.arith.fastmath(...)`, which gives the same MLIR.
- In helpers: `buffer_ops.py` (llvm GEP/ptr ops, rocdl raw buffer ops), `fmha_b16_buffer_managers.py`
  (`llvm_dialect` :37, rocdl async/ds ops).

## Unresolved

- `rocdl.ballot(fx.Int32.ir_type, need)` K:538: passes a DSL `Boolean` straight to an ODS builder. It
  works because an implicit value caster converts it. Static inspection cannot show that this conversion
  is a contract.
- `_launch.compile_hints["llvm_options"]["amdgpu-expert-scheduling-mode"]` K:1836-1839, K:1942-1945:
  `compile_hints` is a public member, but the **key contents** are LLVM backend options that FlyDSL
  passes through. Their meaning is controlled by LLVM, not by FlyDSL.
- `range(fx.Index(lo), fx.Index(hi), 1, init=state)` + `yield` K:1245-1253 is AST-rewriter syntax. It is
  documented in the cleanup skill, but no path in the catalog covers it.

---

## Applied cleanups (all ISA byte-identical, verified)

| # | change | K lines (edited) |
|---|---|---|
| A1 | removed `from flydsl.expr.utils.arith import _to_raw as _raw` (private name). Every `_raw(x)` became `x.ir_value()` for known Numeric/Vector values and `fx.as_ir_value(...)` for mixed raw/Vector values or lists | 439-473, 953-955, 1200 |
| A2 | K stopped importing the `_ir` (= `tensor_shim._to_raw`) re-export. `_wmma` operands now use `fx.as_ir_value` | 107, 296-305 |
| A3 | redundant re-wraps removed: `fx.Vector(_ir(s[kvt]))` -> `fx.Vector(s[kvt])`; `fx.Vector(_ir(o_acc[qt][dt]))` list -> `list(o_acc[qt])`; `fx.Int32(n_tiles)` x6 -> `n_tiles` (already `Int32` from `fx.ceildiv`) | 495, 1180, 829, 1082, 1214-1215, 1225, 1272 |
| A4 | `fx.Int32(gpu.block_id(ax))` (unstable wrapper) -> `gpu.block_idx.ax` (stable; the same `Int32(gpu.block_id(ax))` underneath) x8 | 265-266, 793, 807, 1409-1410, 1536, 1661 |
| A5 | softmax helpers: `arith.MaxNumFOp(...)` -> `arith.maxnumf(a, b, fastmath=fast)`; `arith.addf/subf/mulf(..., fastmath=F)` -> `with arith.fastmath(F): a op b`. The no-reassoc sum-tree flag set is preserved | 438-460 |
| A6 | `fmath.fma(_raw(..)x3)` -> `fmath.fma(s, log2e, neg_m)` (stable `fx.math.fma` accepts DSL values) | 567 |
| A7 | comments only: `# UNSTABLE(gfx1250): ...` at the imports (41-47, 106) and at the `_wmma` definition (296); stale "empty scaffold" / "EMPTY, unwired" docstrings fixed; get_gfx-fallback comment corrected; managers-block comment moved to the managers import | - |

## Proposed, needs on-card A/B (not applied: compile-only shows the ISA changes)

| # | proposal | measured compile-only effect |
|---|---|---|
| P1 | `_lane_id()`: `rocdl.mbcnt_lo(...)` -> stable `fx.gpu.lane_id()` | prod: **+37 instructions** (4306 -> 4343), same VGPR/SGPR/LDS. The schedule changes across the whole kernel (about 4.6k diff lines). |
| P2 | `peer()`: `rocdl.permlanex16` -> `fx.gpu.shuffle_xor(v, 16, 32)` | not compiled. It would replace `v_permlanex16` with a DS/DPP permute on the softmax critical path. |
| P3 | `exp2`: `rocdl.exp2` -> `fx.math.exp2(x, fastmath=afn)` | not compiled. It is likely to add denorm range-reduction VALU to 264 `v_exp_f32` sites. |
| P4 | `_load_sink_logit`: `llvm.load` + `create_llvm_ptr` -> `fx.ptr_load(fx.get_iter(ptr_sink) + i, result_type=f32)` | win_sink variant, measured together with P1: +64 instructions, **VGPR 455 -> 457**, and `global_load_b32` count changes. |
| P5 | LSE / zero-fill stores: vendored `buffer_ops` -> `fx.rocdl.make_buffer_tensor` + `fx.copy` (BufferCopy32b/128b) | not attempted. It swaps the buffer mechanism and would change the OOB-drop contract (`0x7FFFFFFF` byte-offset masking). |
| P6 | `_named_barrier_pair` K:201-213 is a documented no-op placeholder. Deleting its calls leaves the ISA unchanged, but it holds the named-barrier TODO | a judgment call for the owner, not an API issue |

## Verification (compile-only, no GPU work)

Harness: `api-audit/compile_fwd_isa.py <impl_dir> <dump_dir> [prod|thd|win_sink]`, run in `fa-repro` with
`CUDA/HIP/ROCR_VISIBLE_DEVICES=` (empty), `COMPILE_ONLY=1 ARCH=gfx1250 FLYDSL_GPU_ARCH=gfx1250
FLYDSL_RUNTIME_ENABLE_CACHE=0 FLYDSL_DUMP_IR=1 FLYDSL_RUNTIME_CACHE_DIR=/tmp/flycache_agent_apiaudit`.
The harness builds `_launch_fns[...]` through the kernel's own `_ensure_*_kernel` and calls `flyc.compile`
with tiny CPU placeholder tensors, because only pointers enter the ABI, the same way `_run_compiled`
passes torch tensors. It also passes prod-shape integer strides/lengths and `stream=None`. Dumps go to `api-audit/isa/<tag>/<kernel>/`. All 23
stages were compared at run time; afterwards only 00_origin.mlir, 21_llvm_ir.ll and 22_final_isa.s were
kept, to save space (183 MB -> 20 MB). Stats: `api-audit/isa_stats.py`. Tags `probe_*` hold the ISA for P1/P4.

- prod = `bshd`, b4 s8192 hq32 hkv8 (gqa 4) d128, causal (mask_right), bf16, return_lse (what `impl.py` calls)
- thd = varlen entry + kv_len==0 zero-fill path (covers the A4 edits at 1409/1410/1536)
- win_sink = bshd, window_left=128 + causal + sink (covers mask_left and the sink/`llvm.load` path)

before = untouched `fwd341/op0341`. after-kernel-only = `op0341` helpers + edited K (isolates this change).
opclean-combined = current `op_clean` (edited K plus the concurrent helper edits).

| variant | VGPR | SGPR | VGPR/SGPR spill | LDS (group seg) | scratch (private seg) | instr total | histogram delta | stages 00_origin .. 22_final_isa |
|---|---|---|---|---|---|---|---|---|
| prod before / after | 445 / 445 | 100 / 100 | 0,0 / 0,0 | 327680 / 327680 | 0 / 0 | 4306 / 4306 | 0 ops | **all 23 files byte-identical** |
| thd before / after | 445 / 445 | 104 / 104 | 0,0 / 0,0 | 327680 / 327680 | 0 / 0 | 4902 / 4902 | 0 ops | 00_origin, .ll, .s identical |
| win_sink before / after | 455 / 455 | 107 / 107 | 0,5 / 0,5 | 327680 / 327680 | 0 / 0 | 6299 / 6299 | 0 ops | 00_origin, .ll, .s identical |
| prod, opclean-combined | 445 | 100 | 0,0 | 327680 | 0 | 4306 | 0 ops | .s identical (md5 fcfbd7fc...) |

The top of the prod histogram is the same before and after: v_pk_mul_f32 320, v_exp_f32 264,
v_wmma_f32_16x16x32_bf16 256, ds_load_b128 160, ds_load_tr16_b128 128, v_max3_num_f32 120,
s_wait_dscnt 62.

Because even the traced MLIR (`00_origin.mlir`, before any pass) is byte-identical, the edits are
exact source-level no-ops for codegen. No on-card run is needed for A1-A7.

## Verification (independent adversarial pass)

Scope: `diff -u fwd341/op0341/flydsl_fwd/fmha_fwd_prefill_a16w16_m32x8.py fwd341/op_clean/flydsl_fwd/fmha_fwd_prefill_a16w16_m32x8.py`
(the kernel file is the only file in this change). Every hunk was checked against the installed 0.3.4.1 source to try to
refute equivalence. **Result: no hunk refuted, nothing reverted.**

Per-hunk reasoning:

- `fx.Int32(gpu.block_id(ax))` -> `gpu.block_idx.ax`: `flydsl/expr/gpu.py:133` defines `block_idx = Tuple3D(gpu.block_id)`, and
  `Tuple3D.__getattr__` (`expr/typing.py:1388-1398`) returns `Int32(gpu.block_id(ax))`, so it is the same op with the same index->i32 cast
  and signedness. All 8 sites (`x`, `y`, `z`) use the same dtype as before.
- `fx.Int32(n_tiles)` -> `n_tiles` (6 sites, including `nxt < n_tiles`, the `_run_tiles` bound and the non-causal `clean_hi = n_tiles`):
  `n_tiles = fx.ceildiv(kv_len_wg, fx.Int32(n_block))` (K:824) is already a signed `Int32`, so the wrapper was an identity. The
  comparison is still a signed i32 `cmpi`, and `_run_tiles` still casts it with `fx.Index(...)`.
- Softmax helpers: `arith.maxnumf` (`expr/arith.py:295`) passes `fastmath=` directly to the upstream op, the same as `MaxNumFOp(..., fastmath=fast)`.
  `with arith.fastmath(F): a op b` sets the ambient `tracing_context(fastmath=...)` that the Float32 operators read. The dump
  confirms that the flags survive exactly: `00_origin.mlir` has 328 `fastmath<fast>` and 248 `fastmath<nnan,ninf,nsz,arcp,contract,afn>`
  (the no-reassoc sum tree) in both trees, so reassoc was not reintroduced. Each `with` block wraps a single op, so no flag leaks into
  neighbouring ops.
- `fmath.fma(s, log2e, neg_m)` takes the DSL values directly. The wrapper calls `as_ir_value` itself, so the operand order and the f32 type are unchanged.
- `_raw(x)` -> `x.ir_value()` / `fx.as_ir_value(...)`: each argument is already an `Int32`/`Float32`/`Vector`, so this is the same raw value.
  The `fx.as_ir_value([m, d, *o])` list form keeps shape and order (`typing.py:194`, list -> list), so the loop-carried
  iter_arg order (m, d, O[0..d_tiles)) per q-tile is preserved.
- `fx.Vector(_ir(s[kvt]))` -> `fx.Vector(s[kvt])` and `[fx.Vector(_ir(o)) ...]` -> `list(o_acc[qt])`: `o_acc[qt][dt]` is already
  `fx.Vector(state[...])` (K:1001-1004), so the re-wrap was an identity. The shallow list copy leaves no aliasing that anything mutates.
- Removed imports (`_to_raw`, `_ir`): no remaining references in the kernel file (grep). Nothing live was removed.
- Comments and docstrings: the moved staging-manager comment, the "get_gfx() fallback is dropped" comment (true: `get_lds_capacity_bytes`
  requires `gfx`, and the old "kept only for fidelity" text was the stale one) and "compute stages: QK GEMM, online softmax, PV GEMM"
  (`_qk_gemm`/softmax/`_pv_gemm` exist and are wired) are accurate. `fx.tdm_ops` is indeed the deprecated alias (`expr/__init__.py:25`).
  The `# native v_exp_f32 ... unlike fx.math.exp2` note is an unmeasured claim, because P3 was never compiled. Treat it as a rationale, not a fact.

Independent re-compile (fresh, not reusing the previous dumps): `api-audit/work/verify/`. Baseline = untouched `fwd341/op0341`.
New = an op0341 copy with only the edited kernel file swapped in (`work/verify/tree`). Container `fa-repro`, all `*_VISIBLE_DEVICES`
empty, `COMPILE_ONLY=1`, cache disabled. Nothing ran on the GPU.

| variant | stages compared | result | 22_final_isa md5 |
|---|---|---|---|
| prod (bshd causal lse bf16, b4 s8192 hq32 hkv8 d128) | all 23 | byte-identical | fcfbd7fc27b0 (matches the earlier report) |
| thd (varlen + zero-fill) | all 23 | byte-identical | 7da478be695a |
| win_sink (mask_left + sink) | all 23 | byte-identical | 35e5c124f07d |
| **new:** non-causal, no LSE, fp16 (covers `clean_hi = n_tiles`, the `wmma_f32_16x16x32_f16` path, and no-LSE) | all 23 | byte-identical | (256 v_wmma f16) |

The extra variant uses `work/verify/compile_v.py`, which is the harness with `V_CAUSAL`/`V_LSE`/`V_DT` env overrides. Only 00/21/22 were kept
afterwards. Not covered: `USE_TDM_LOADER=False` (V1 loader). It is a module constant set to True (K:180), and none of the edited hunks sits inside
a V1-only branch.
