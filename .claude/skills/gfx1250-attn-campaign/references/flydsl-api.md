# FlyDSL 0.3.4.1 API stability on the gfx1250 attention kernels

Condensed from `PT/output/0925__flydsl/api-audit/{fwd-kernel,fwd-bufmgr,fwd-helpers,bwd-champion}.md`
(2026-09-25; `PT` = `/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo`). Read the source file
for per-line tables. Policy = FlyDSL `docs/api_stability.md` at tag `v0.3.4.1`; stable catalog =
`scripts/list_stable_apis.py` (493 paths at the tag; main adds only `extension.coop.*`, `struct.Empty`).
FlyDSL repo with skills: `/home/lihuzhan/code/2026_0925__flydsl/FlyDSL`.

## 0. Verdict in one table

| tree | verdict | cleaned copy | ISA after cleanup |
|---|---|---|---|
| fwd kernel `fmha_fwd_prefill_a16w16_m32x8.py` | NOT STABLE-ONLY; 0 deprecated, 0 private imports | `fwd341/op_clean/flydsl_fwd/` | byte-identical (prod, thd, win_sink, non-causal fp16; all 23 stages) |
| fwd buffer managers | NOT STABLE-ONLY | same | byte-identical (prod V2+O v3, O v2, V1+O v1) |
| fwd helpers (`tensor_shim`, `kernels_common`, `buffer_ops`) | `kernels_common` stable-only; others use `_mlir` | same (547->52, 243->26, 572->398 lines) | byte-identical |
| bwd champion r20 (7 kernels) | NOT STABLE-ONLY; no longer imports any aiter module | `bwd341/op_clean/kernels.py` | 161 dump files byte-identical |

Nothing left unstable has a codegen-neutral stable replacement in 0.3.4.1. **Cleanups are
source-level no-ops; every "proposed" item changes ISA and needs a card A/B.**

## 1. Stable idioms to use in new code

- `flyc.jit`, `flyc.kernel`, `flyc.compile` (-> `CompiledFunction`); `JitFunction.compile_hints` (public member; its `llvm_options` keys are LLVM's, not FlyDSL's contract).
- Numerics `fx.Int32/Int64/Float32/BFloat16/Index/Uint32`, `.ir_value()`, `.select`, `.to`; Python ints coerce to `Int32` on the RHS of an `Int32` op (so `x % 8` == `x % fx.Int32(8)` in IR).
- `fx.as_ir_value(x)` (replaces aiter `_to_raw` / `_ir`; recurses lists; passes `ir.Value` through).
- `fx.gpu.block_idx.x` (== `Int32(gpu.block_id(x))`), `thread_idx`, `grid_dim`, `SharedAllocator().allocate().peek().ptr`, `barrier`.
- `fx.arith.{min,max,ceildiv,maxnumf}`, `with fx.arith.fastmath(F): a op b` (same MLIR as `arith.addf(..., fastmath=F)`), `fx.math.{fma,log2}`.
- LDS pointer: `fx.PointerType.get(Int32, AddressSpace.Shared, align 4)` + `fx.inttoptr` + `fx.to_llvm_ptr` (the bwd `_lds_ptr` helper; replaces aiter `create_llvm_ptr`, which reads private `ptr._value`).
- `fx.rocdl.make_buffer_tensor` + `BufferCopy*` + `fx.copy`; `range(lo, hi, 1, init=state)` + `yield` loop form (rewriter contract, not a catalog path).
- Cache compiled launchers in a module dict keyed by the `JitFunction` (`_COMPILED[exe]`), never `exe._cf = ...` (private write on a FlyDSL object).

## 2. Unstable but required on gfx1250 (keep; annotate `# UNSTABLE(gfx1250)` once at import)

| op | why no stable swap |
|---|---|
| `rocdl.wmma_f32_16x16x32_{bf16,f16}` | `fx.rocdl.WMMA`+`fx.gemm` is structural (fragment packing, VGPR); bwd sits at 904/960 VGPR |
| `rocdl.ds_load_tr16_b128` | no stable wrapper; `cdna4.LDSReadTrans` is gfx950-only |
| `rocdl.exp2` | `fx.math.exp2` lowers via `math.exp2` (likely adds range-reduction VALU; unmeasured) |
| `rocdl.s_wait_dscnt / s_wait_asynccnt` | stable `fx.rocdl.s_waitcnt` only covers gfx942/950/11xx/120x and **raises** on gfx1250 |
| `rocdl.sched_barrier(0)` | stable `sched_mfma/vmem/dsrd/dswr` are group barriers, different semantics |
| `rocdl.permlanex16`, `ballot`, `mbcnt_lo`, `wave_id` | `shuffle_xor`/`lane_id` exist but change ISA (`lane_id`: +37 instr in fwd prod) |
| `rocdl.tdm_ops.tensor_wait`, `fx.rocdl.make_tdm_atom` (cdna5) | cdna5 deliberately left out of `rocdl.__all__`; `fx.tdm_ops` is the **deprecated** alias -- do not use it |
| `flydsl._mlir.dialects.llvm` load/store/GEP on `!llvm.ptr<3>` | `fx.ptr_load` rejects `!llvm.ptr<3>` in 0.3.4.1 |
| `rocdl.cluster_load_async_to_lds_b128` (direct ODS call) | FlyDSL's wrapper passes `(offset, cpol, mask)` into ODS `(offset, mask, *, cpol)`: **wrapper bug**, direct call is the only correct form |
| vendored `buffer_ops` (`MakeBufferRsrcOp`, `RawPtrBufferStoreOp`) on the LSE/O store | stable path swaps the buffer mechanism and the `0x7FFFFFFF` OOB-drop trick |

## 3. Known breakage / traps found by the audit

- **V1 loader does not compile on 0.3.4.1**: `QManager16bV1._read_tile` calls `fx.ptr_load` on
  `!llvm.ptr<3>` -> `DSLCompileError`. `USE_TDM_LOADER=False` is broken in baseline too. Scratch fix:
  `fx.Vector(llvm_dialect.load(v8_ty, ptr))` (481 VGPR, no spill). Matters for fwd lever L1 (`O_VARIANT` v1).
- `is_rdna_arch("gfx1250")` is **False** in 0.3.4.1 -> V# flags take the CDNA branch (`0x27000`; the
  inherited docstring's `0x20070` is miscomputed). Same in FlyDSL's own `make_buffer_tensor`. gfx1250 V#
  correctness is unverified: use explicit predicates, not descriptor OOB semantics.
- `fmha_b16_buffer_managers.py:51-52` still re-exports `_to_raw as _ir` for old kernel copies; `op0341`'s
  kernel imports it, `op_clean`'s does not. Delete only when no live copy imports it.
- `_G07_CLAMP` in bwd `kernels.py` looks dead but external screens read it via `getattr` -- keep.
- Stale comments in bwd champion: "h8 PROBE (throwaway)" is the shipped g03 layout; "740 VGPR" is now 904.
- `PROVENANCE.md` in `fwd341/op_clean` does not record the helper trim.

## 4. Proposed, not applied (each needs compile-ISA diff + card A/B)

fwd: P1 `mbcnt_lo`->`fx.gpu.lane_id()` (+37 instr), P2 `permlanex16`->`shuffle_xor`, P3 `rocdl.exp2`->`fx.math.exp2`,
P4 sink `llvm.load`->`fx.ptr_load` (VGPR 455->457), P5 LSE/O stores -> `make_buffer_tensor`+`fx.copy`,
P6 delete no-op `_named_barrier_pair` (holds the named-barrier TODO); managers: `s_wait_asynccnt`->`asyncmark`
(group semantics), `copy_atom_call`->`fx.copy` for TDM atoms (unverified). bwd: raw WMMA -> `make_mma_atom`,
LDS `llvm.store` -> layout views + `fx.copy`. Upstream asks: stable gfx1250 wait API + public
`sched_barrier`, export cdna5/`tensor_wait`, `ds_load_tr16_b128` wrapper, fix `cluster_load_async_to_lds` arg order.

## 5. Compile-only verification recipe used by the audit (no GPU)

`api-audit/compile_fwd_isa.py <impl_dir> <dump_dir> [prod|thd|win_sink]` (fwd; `work/verify/compile_v.py`
adds `V_CAUSAL/V_LSE/V_DT` overrides), `.bwd_work/compile_bwd.py` (bwd, stubs the aiter package chain).
Env: `COMPILE_ONLY=1 ARCH=gfx1250 FLYDSL_GPU_ARCH=gfx1250 FLYDSL_RUNTIME_ENABLE_CACHE=0 FLYDSL_DUMP_IR=1
FLYDSL_RUNTIME_CACHE_DIR=/tmp/flycache_agent_<name> HIP_VISIBLE_DEVICES=-1`. Build `_launch_fns[...]` via the
kernel's own `_ensure_bshd_kernel(False, True, True, False, 4, 128, "bf16")`, then `flyc.compile(...)` with
CPU/`meta` tensors and `stream=None`. Compare `00_origin.mlir`, `21_llvm_ir.ll`, `22_final_isa.s` with `cmp`;
compile the control twice to prove dump determinism. `isa_stats.py` gives VGPR/SGPR/LDS/spill/histogram.
Reference prod fwd ISA: VGPR 445, SGPR 100, LDS 327680, spill 0, 4306-4341 instr; top ops
`v_pk_mul_f32` 320, `v_exp_f32` 264, `v_wmma` 256, `ds_load_b128` 160, `ds_load_tr16_b128` 128.
