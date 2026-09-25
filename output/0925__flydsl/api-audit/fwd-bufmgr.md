# API-stability audit: `fmha_b16_buffer_managers.py` (gfx1250 FA forward, K/V/Q/O staging)

## Stable API usage audit: NOT STABLE-ONLY

- **Scope:** `output/0925__flydsl/fwd341/op_clean/flydsl_fwd/fmha_b16_buffer_managers.py`. This is the edited copy. Line numbers below refer to it.
- **Baseline:** `fwd341/op0341/flydsl_fwd/fmha_b16_buffer_managers.py`, left untouched.
- **Policy:** FlyDSL `docs/api_stability.md` at repo HEAD `89ad52fb`. The stable catalog comes from `scripts/list_stable_apis.py --repo-root <0.3.4.1 install>`, which returned 511 paths. That run points a `python/flydsl` symlink at `/home/lihuzhan/.local/flydsl0341/flydsl`, so the catalog describes the release that actually runs. Repo HEAD has 531 paths, and the extra 20 are all `extension.coop.*` and `struct.Empty`.
- **Prod path:** `USE_TDM_LOADER=True` and `O_VARIANT="v3"`, which means QManager16bV2, KManager16bV2, VManager16bV2 and OManager16bV3. The V1 loaders and OManager16bV1/V2 are only reached when those module flags change.

### Result: ISA is byte-identical

| config (prod shape b4 s8192 hq32 hkv8 d128, causal, bf16, lse) | VGPR | SGPR | LDS B | scratch | spills v/s | instrs / distinct | ISA `.s` | LLVM IR |
|---|---|---|---|---|---|---|---|---|
| prod (TDM loaders + O v3): before | 445 | 100 | 327680 | 0 | 0/0 | 4341 / 137 | — | — |
| prod: after | 445 | 100 | 327680 | 0 | 0/0 | 4341 / 137 | **byte-identical** | identical (metadata excluded) |
| O v2 variant: before → after | 447 → 447 | 101 → 101 | 327680 | 0 | 0/0 | 4075 / 126 both | **byte-identical** | — |
| V1 loaders + O v1 (`ptr_load` patched in both arms, see P1) | 481 → 481 | 50 → 50 | 327680 | 0 | 0/0 | 4505 / 145 both | **byte-identical** | — |
| determinism control: before vs before (2nd compile) | same | | | | | | byte-identical | |

How these were produced:

- **Compile mode:** Compile-only inside `fa-repro` with `COMPILE_ONLY=1 ARCH=gfx1250 FLYDSL_RUNTIME_ENABLE_CACHE=0 FLYDSL_DUMP_IR=1`. The GPU was hidden with `HIP_VISIBLE_DEVICES=ROCR_VISIBLE_DEVICES=-1`, so nothing touched the card.
- **Inputs:** The tensors were on the `meta` device and the stream was `None`. The driver calls `flyc.compile` on the `_launch` built by the host path `_ensure_bshd_kernel(False, True, True, False, 4, 128, "bf16")`.
- **ISA source:** Each ISA comes from `22_final_isa.s`.
- **Isolation:** Each arm is a private copy of `op0341` with only this one file swapped. That way, concurrent edits to other files in `op_clean` cannot leak into the comparison.
- **Scratch files:** Driver, logs and dumps are in `api-audit/.bufmgr_work/`.

## Stable uses

These paths pass §2.1 or §2.2, or are public members of a stable type or returned object (§1):

- `flydsl.compiler.jit` (`flyc.jit`): 1847. This is a local runtime-`if` boundary, the pattern kernel-code-cleanup §5 recommends.
- `fx.BFloat16` (numeric): 231, 513, 730, 1017, 1162, 1250, 1369, 1588, 1748. `elem_dtype.ir_type` appears at 972, 1082, 1666.
- `fx.Int32` / `fx.Int64` (numeric) and `.ir_value()` on them: 194, 1466. Selected `fx.Int64(...)` widening casts: 306, 326, 596, 615, 647, 811, 829, 857, 969, 978, 1076-1078, 1090, 1463, 1674-1676, 1697-1699, 1881, 1906-1912.
- `.select` / `.to` on Numeric: 315, 608, 639, 822, 852, 1548 and 423, 1127, 1501, 1645, 1808.
- `fx.Vector`, `fx.Vector.make_type`, `.shuffle`, `.ir_value()` (typing.Vector):
  - `make_type`: 422, 681, 893, 1126, 1219, 1307, 1471
  - `fx.Vector(...)`: 695, 908, 1146-1147, 1235, 1323, 1525
  - `.shuffle`: 448, 467, 476, 1148
  - `.ir_value()`: 1504, 1652, 1838
- `fx.max` / `fx.min` (arith): 1074, 1664, 1884-1885, 1892.
- Primitive-layer calls:
  - `fx.range_constexpr`: 308, 309, 345, 383, 405, 438, 440, 462, 469, 616, 630, 669, 830, 844, 881
  - `fx.ptrtoint`: 306, 596, 811, 1881
  - `fx.get_iter`: 306, 596, 811, 970, 1079, 1678, 1701, 1881
  - `fx.inttoptr`: 989, 1104, 1670
  - `fx.add_offset`: 978, 1090, 1678, 1701
  - `fx.make_view`: 979, 991, 1092, 1106, 1680, 1688, 1703, 1717
  - `fx.make_layout`: 979, 991, 1093, 1108, 1681, 1690, 1705, 1719
  - `fx.PointerType.get`: 971, 1081, 1665
  - `fx.AddressSpace.Shared`: 973, 1083, 1667
  - `fx.copy_atom_call`: 1113, 1725. The TDM `load_views` tuples are also issued with it from the kernel file.
  - `fx.ptr_load`: 426-427. This call is stable but broken here: see P1.
- `fx.Tensor(...)` (typing.Tensor): 979, 990, 1091, 1105, 1679, 1687, 1702, 1716.

## Private-field writes

- **None in this file.** Every `self._*` assignment targets the manager's own object (`_q_gptrs`, `_warp_region`, `_pending`, `_cfg` and so on), not a FlyDSL object.
- **Out of scope, for the owner of `tensor_shim.py`:**
  - `_run_compiled()` does `exe._cf = cf` (`tensor_shim.py:276`). That writes an underscore attribute onto a FlyDSL `JitFunction` and bypasses the per-call cache-key dispatch.
  - `kernels_common.create_llvm_ptr` reads `ptr._value` (`kernels_common.py:195`), which is a private read.

## Deprecated or unstable uses

- `[UNSTABLE] flydsl.expr.rocdl.tdm_ops` (import at 43; `tensor_wait` at 1125). It is not in `rocdl.__all__`, and §3 names it a "target-specific unstable path". The deprecated `fx.tdm_ops` alias is not used, which is correct. Annotated at the import.
- `[UNSTABLE] fx.rocdl.make_tdm_atom` (980, 1096, 1684, 1710). It reaches `rocdl` through `from .cdna5 import *`. `rocdl/__init__.py:31` comments out `"cdna5"` as "unstable for now", and the name is not in `rocdl.__all__`. Annotated at its first call (979).
- `[UNSTABLE] rocdl.sched_barrier` (409, 418, 1835, 1842). This is a FlyDSL wrapper (`rocdl/__init__.py`, around line 142) but it is not in `rocdl.__all__`. The only stable scheduling helpers in 0.3.4.1 are `sched_mfma/vmem/dsrd/dswr`, and there is no stable `sched_barrier(0)`.
- `[UNSTABLE] rocdl.s_wait_asynccnt` (442, 463, 470; V1 Q only). This is a FlyDSL wrapper (`rocdl/__init__.py:593`) outside `__all__`. A stable alternative exists, `fx.rocdl.asyncmark/wait_asyncmark`, but it is semantically different (see P2).
- `[UNSTABLE] local helpers wrapping raw ops` (import 46, 48, 52):
  - `buffer_ops.get_element_ptr` builds a raw `llvm.getelementptr`: 694, 907, 1143, 1145, 1234, 1322, 1650, 1813.
  - `buffer_ops.create_buffer_resource`: 1465. `buffer_ops.buffer_store`: 1549. Both are V1 O only.
  - `kernels_common.create_llvm_ptr` is built from stable `fx.inttoptr`/`fx.to_llvm_ptr` plus the private `_value` read. It appears 23 times, including 326, 615, 829, 1134, 1213, 1300, 1503, 1524, 1643, 1805, 1911 and 1914.
  - `tensor_shim._to_raw` (as `_ir`) is a raw `ir.Value` conversion that uses `ir.Value._CAPICreate`. It is now **unused in this file** and kept only as a re-export (see "Applied" below).
- No DEPRECATED (§3) API is used.

## Upstream MLIR operations

These are §2.5 reminders: each use is allowed, but its name, signature and semantics are controlled by upstream MLIR, with no FlyDSL compatibility commitment. All sites are gfx1250-specific, and none has a stable FlyDSL wrapper in 0.3.4.1.

- `[UPSTREAM-MLIR] llvm.load` via `flydsl._mlir.dialects.llvm as llvm_dialect` (import 37): 336 (docstring), 695, 1146, 1147, 1235, 1525. These are the natural `ds_load_b128` sites. 695 and 1235 are in the K hot path.
- `[UPSTREAM-MLIR] llvm.store`: 1504, 1652, 1838 (the O `ds_store_b128`).
- `[UPSTREAM-MLIR] rocdl.ds_load_tr16_b128` (an ODS builder re-exported through `from ..._mlir.dialects.rocdl import *`): 908, 1323. This is the V transpose load, in the hot path.
- `[UPSTREAM-MLIR] rocdl.s_wait_dscnt` (ODS): 444, 446, 465, 472, 474, 1496, 1516, 1534, 1653, 1843. The stable `fx.rocdl.s_waitcnt` raises on gfx1250 because it only supports gfx942/950/11xx/120x, so there is no replacement.
- `[UPSTREAM-MLIR] rocdl.global_load_async_to_lds_b128` (ODS): 197.
- `[UPSTREAM-MLIR] rocdl.cluster_load_async_to_lds_b128` (ODS, called directly on purpose): 195. In 0.3.2 through 0.3.4.1, FlyDSL's generic wrapper `cluster_load_async_to_lds` passes `(offset, cpol, mask)` positionally into the ODS `(offset, mask, *, cpol)` order (`rocdl/__init__.py:574`). The wrapper therefore puts the int `cpol` into the `mask` operand slot, and the direct ODS call is the only correct form. This is worth an upstream bug report.
- `[UPSTREAM-MLIR] rocdl.global_store_async_from_lds_b128` (ODS): 1851.
- Indirect, through the helpers above:
  - `llvm.getelementptr` (from `buffer_ops.get_element_ptr`).
  - `llvm.inttoptr`/`fly.to_llvm_ptr` (from `create_llvm_ptr`).
  - `rocdl.raw.ptr.buffer.store` and `make.buffer.rsrc` (from `buffer_ops`, V1 O only).

## Unresolved paths

- `fx.rocdl.<name>` and `rocdl.<name>` resolve through two wildcard imports: `from ..._mlir.dialects.rocdl import *` and `from .cdna5 import *`. The skill treats wildcard imports as UNRESOLVED for static tooling. I resolved each name above by hand against the 0.3.4.1 sources: `rocdl/__init__.py`, `rocdl/cdna5.py` `__all__`, and `_rocdl_ops_gen.py`. A re-run against a different FlyDSL release has to repeat that resolution.
- `elem_dtype` is a constructor parameter. The classification assumes callers pass a `fx.Numeric` class. The kernel's `_DTYPE_MAP` passes `fx.BFloat16` or `fx.Float16`.

## Applied (ISA-verified byte-identical)

1. **`_to_raw` → `.ir_value()`:**
   - 194: `fx.Int32(0).ir_value()` as the mask operand.
   - 1466: `num_records_bytes=o_num_records_bytes.ir_value()`.
   - 1504, 1652, 1838: `bf.ir_value()` into `llvm.store`.
   - 1851: dropped two identity wraps. `create_llvm_ptr` already returns a raw `!llvm.ptr` `ir.Value`, so `_to_raw` returned its argument unchanged.

   Each replaced `_to_raw(x)` either returned `x` unchanged (Vector and llvm pointers are `ir.Value`) or returned `x.ir_value()`, so no op changes. The `_ir` import stays at 52 as a documented re-export. The untouched kernel file (`op0341/.../fmha_fwd_prefill_a16w16_m32x8.py:104`) imports `_ir` from this module and uses it at :294, :480 and :1165. Removing the import broke that kernel, which the first after-compile caught. The `op_clean` kernel file has since dropped the import in a concurrent edit, so the re-export is now only a compatibility shim. After the ISA check I rewrote only that comment, and the AST is identical to the verified copy.
2. **Redundant `fx.Int32(<python int>)` wraps removed (about 45 sites).** Each removed wrap was the right-hand operand of an operator whose left side is a runtime `fx.Int32`. Examples: `lane_idx % fx.Int32(8)` → `lane_idx % 8`, and `warp_idx * fx.Int32(self.rows_per_warp * self.row_bytes)` → `warp_idx * (...)`. Two `fx.Int32(<fx.Int32>)` double wraps were also removed, at 848-849 (`row_idx`, `col_idx`). This is IR-identical by construction: `numeric._make_binop → _try_coerce_rhs` coerces an in-range Python int to `Int32` exactly like `fx.Int32(C)`.

   Kept on purpose:
   - leading wraps such as `fx.Int32(r * _WAVE_LANES) + lane_idx`
   - `.select(x, fx.Int32(0))` and `fx.max/min(.., fx.Int32(..))`
   - `valid_rows > fx.Int32(0)` inside the `@flyc.jit` if
   - every `fx.Int64(...)`, because those are real widening casts
3. **Dead code removed:**
   - `_assert_multiple()`, which had no caller in this file or in the kernel.
   - `OManager16bV3.n_rounds`, which was never read. `_warp_addrs` recomputes it locally from `rows_per_warp`.
4. **Stale comment corrected** at 191-193 (why `cluster_load_async_to_lds_b128` is called directly). It now names the real 0.3.4.1 wrapper bug instead of "0.3.2 argument order".
5. **`# UNSTABLE(gfx1250)` annotations, one each:** the imports at 36, 39, 42, 45, and the first `make_tdm_atom` call at 980. Nothing was added at the other call sites.

I did not touch hot-loop structure, `sched_barrier`, `s_wait_*` placement, load/store mechanisms or the scheduling-mode hint.

## Proposed, not applied (each needs a compile-ISA diff; P3-P5 also need an on-card A/B)

- **P1: pre-existing breakage, non-default path.** `QManager16bV1._read_tile` (426-427) calls stable `fx.ptr_load` on an `!llvm.ptr<3>`. On 0.3.4.1 that fails with `DSLCompileError: 'fly.ptr.load' op operand #0 must be ... but got '!llvm.ptr<3>'`, so `USE_TDM_LOADER=False` does not compile at all, in the baseline or after this cleanup. The fix is `fx.Vector(llvm_dialect.load(v8_ty, ptr))`, the same form K/V V1 use. With that patch applied in scratch only, the V1 config compiles: 481 VGPR, 50 SGPR, no spill. I did not apply it because it changes codegen of that path (from none to some). The V1-variant ISA check above was done with this patch in both arms.
- **P2:** `rocdl.s_wait_asynccnt(n)` (V1 Q, 442/463/470) → stable `fx.rocdl.asyncmark()` + `wait_asyncmark(k)`. This has group semantics rather than a raw counter, so every refill has to be regrouped. It touches wait placement.
- **P3:** `create_llvm_ptr` + `buffer_ops.get_element_ptr` + `llvm.load/store` → stay on `fly.ptr`, then use `fx.add_offset` + `.llvm_ptr`/`fx.to_llvm_ptr`, or `fx.ptr_load/ptr_store` once `fly.ptr.load` accepts LDS. This could change GEP inbounds/nuw flags or alignment, and therefore `ds_load` offset folding in the K/V hot loop (695, 908, 1235, 1323). Risky.
- **P4:** OManager16bV1 `buffer_ops.create_buffer_resource/buffer_store` (1465, 1549) → `fx.rocdl.make_buffer_tensor(..., num_records_bytes=)` + `BufferCopy128b` + `fx.copy`. The `0x7FFFFFFF` mask-redirect OOB trick has to be re-expressed. Non-default path.
- **P5:** `fx.copy_atom_call` → `fx.copy` for the TDM atoms (1113, 1725, and the kernel-file issue sites), per kernel-code-cleanup §7b. It is stable, but I have not verified that `fx.copy` accepts a TDM atom over these single-tile views.
- **P6:** once every kernel revision that is still in use has stopped importing `_ir` (the `op_clean` one already has), delete the re-export at managers:52. That also removes the last `tensor_shim._to_raw` / `ir.Value._CAPICreate` reference from this file.
- **Upstream (FlyDSL) asks that would retire the remaining UNSTABLE sites:**
  - a stable gfx1250 wait API (`s_wait_dscnt/asynccnt/tensorcnt`) and a public `sched_barrier`
  - exporting `cdna5` (TDM atoms) and `tdm_ops.tensor_wait`
  - a wrapper for `ds_load_tr16_b128` on an LDS `fly.ptr`
  - fixing the `cluster_load_async_to_lds` argument order

## Verification (adversarial re-check, 2026-09-25)

**Verdict: I could not refute equivalence. Nothing was reverted and nothing was recompiled.** No GPU work was done.

I diffed `op_clean/flydsl_fwd/fmha_b16_buffer_managers.py` against `op0341/flydsl_fwd/fmha_b16_buffer_managers.py` (396 diff lines) and checked every hunk:

- **Numeric coercion (the ~45 `fx.Int32(C)` removals).** In FlyDSL 0.3.4.1, `numeric._make_binop` runs `_try_coerce_rhs` and then `as_numeric`. `Numeric.from_python_value(int)` returns `Int32(v)` for any v in the int32 range and `Int64` only outside it. Every removed constant is small and non-negative, so `X op C` builds the same Int32 constant as `X op fx.Int32(C)`. It then takes the same `_coerce_operands` promotion path (including the Int64 promotion in QManagerV1 `g_off`) and the same signed floordiv/mod. Dtype and signedness are unchanged.
- **Left operands are always Numeric, never a raw ArithValue or Python int.**
  - `warp_idx`: runtime `fx.Int32`.
  - `lane_idx`: `_lane_id()` returns `fx.Int32`.
  - `block_x` and `kv_head`: `gpu.block_idx.*`, where `Tuple3D` wraps in `Int32`.
  - `lds_base`/`ptr_lds`: `_alloc_lds` base + `fx.Int32`, via `_k_lds_buf`/`_v_lds_buf`.
  - `tile_id`/`row_idx`/`col_idx`: derived from `warp_idx`, so the dropped `fx.Int32(<Int32>)` double wraps were identity (`type(x) is ty -> x.value`).
  - `q_st`/`d_half`/`off_elems`/`c`/`pr`: derived from `lane_idx`.
  - When a sub-product became pure Python (`qtile*_WMMA_M*row_bytes`, `tile*_WMMA_K`, `q_len*gqa` if `q_len` were constexpr), it is always combined with a Numeric right after, so the value and dtype are the same.
- **`_ir(x)` → `x.ir_value()` or `x`.** `tensor_shim._to_raw` returns an `ir.Value` unchanged and otherwise recurses on `.ir_value()`, so it is identity at all six sites. `create_llvm_ptr` already returns the raw `ptr._value`.
- **Dead code.** `_assert_multiple` and `OManager16bV3.n_rounds` have no reader anywhere in `op_clean`/`op0341`. The only other hits are older frozen copies of this same file.
- **Comments.**
  - The cluster-load comment is accurate for 0.3.4.1: `expr/rocdl/__init__.py:575` calls `fn(global_ptr, lds_ptr, offset, cpol, mask)` against the ODS `(global_ptr, lds_ptr, offset, mask, *, cpol)`.
  - The "cdna5 unstable for now" wording matches `rocdl.__all__`.
  - `sched_barrier`/`s_wait_asynccnt` are wrappers outside `__all__`, as claimed.
- **Artifact cross-check.** I recomputed the md5 of each `22_final_isa.s`:
  - prod `77f5ad2c…`: before, before2 and after are identical.
  - O v2 `6c677ff5…`: before and after are identical.
  - V1 + O v1 `50c838fb…`: before and after are identical.
- **Branch coverage.** Together the three variants cover every manager class: Q/K/V V1 and V2, and O V1, V2 and V3.
- **Tree consistency.**
  - `.bufmgr_work/after` differs from `op_clean` only in the `_ir` re-export comment.
  - `managers.orig.py` and `.bufmgr_work/before` are identical to `op0341`.
  - `py_compile` passes.

**Residual risk.** Branches that no compiled shape reaches (V1 `check_oob`, `gqa > rows_per_warp`, O V3 `gqa == 1`, the non-`num_waves == n_wr_tile_rows` K/V write path, `cluster=False`) are covered only by the coercion argument above, not by ISA. That argument holds for any Numeric left operand, so I rate the risk as negligible.
