# Modular GEMM TileSpec — design review & refactor plan

Status: **structural refactor (P0/P1) complete & validated; P2/P3 open.** Companion
to `docs/TILE_API.md`. Goal: evolve the FlyDSL `TileSpec` abstraction toward the
CUTLASS 3.x / ROCm Composable Kernel (CK) modular-GEMM model, **without breaking
perf-neutrality** (zero emitted-IR change; every structural step gated by the IR
diff in `docs/TILE_API.md` §7).

## Progress snapshot

Done (each validated in `dev_primus`, all at baseline cosine):
- ★ P0 **LayoutPolicy** (NT/NN/TN) — `gemm_tile_spec.py`.
- P0 **emit-copy eliminated** — single shared uniform-K template; grouping = a
  `schedule` + a `group_base` scalar seam.
- P1 **Scheduler** policy (`XcdSwizzle` / `LinearNoSync`), held not subclassed.
- P1 **Epilogue** policy (`PerTensorEpilogue`) + **EVT node chain** (`elem_fn`
  hook in `StoreCPerTensor`) + `act` factory plumbing (`relu`); `cache_key`
  folds into `cache_tag`.

The spec now **composes** `{layout_policy, scheduler, epilogue}` as held objects.

Verified non-regressed across ALL paths touching the changed modules: mega
dispatch NT/NN, mega combine NT/NN/TN, standalone dense fp8 NT/NN/TN (cos 1.0 vs
torch), epilogue `act="relu"` (== clamp). (pytorch op test needs the unbuilt
`_C` extension — env limitation, not a regression.)

Open — **higher risk, need steer (perf-bench gated, NOT IR-neutral):**
- P2 persistent / stream-K scheduler (ragged-M load balance) — plugs into the
  new `Scheduler` seam.
- P2 pipeline policy (CK interwave / `Stages>2`).
- P2 epilogue-scatter combine variant (re-scoped from P1 — combine-push is a
  full-N + cross-rank + scoreboard concern, not an `elem_fn` node).
- P3 block-scaled (MXFP) scale seam; P3 `max_size` as a tensor-arg property
  (preventive; not currently triggered — mega outputs are < 4GB).

Files in scope:
- `primus_turbo/flydsl/gemm/gemm_tile_spec.py` — core abstraction (`TileSpec`,
  `DenseFp8TileSpec`, `emit_uniform_k_tile`, `run_uniform_k_pipeline`).
- `primus_turbo/flydsl/mega/group_tile_spec.py` — grouped extension
  (`GroupFp8TileSpec`).
- `primus_turbo/flydsl/mega/dispatch_grouped_gemm_fp8.py`,
  `grouped_gemm_fp8_combine.py` — fused consumers.

---

## 0. Reference model (how CUTLASS / CK modularize GEMM)

| Concern | CUTLASS 3.x | CK / ck_tile | Our design (today) |
|---|---|---|---|
| Tile→CTA map | `TileScheduler` (Persistent / StreamK) | grid map + grouped arg array | `schedule` hook + `schedule_tile` |
| Mainloop | `CollectiveMma` (`Stages`-parameterized pipeline) | `BlockGemmPipeline` (Intra/Interwave policy) | `run_uniform_k_pipeline` + `PipelineGeometry` |
| Epilogue | `CollectiveEpilogue` + **EVT** (visitor tree) | `CElementwiseOperation` | `build_store` → `StoreCPerTensor` |
| Data layout | CuTe `Layout` / `TiledCopy` / `TiledMMA` | tensor descriptor / coord transform | `make_buffers` / `global_swizzle` / loaders |
| Assembly | `GemmUniversal` kernel | `GridwiseGemm::operator()` | `build_launch` |
| Auto-select | `CollectiveBuilder` (Auto schedule) | policy selection | `_autotune_*` + `make_tile_spec` |

**What we already got right** (keep): trace-time inline emission =
zero-overhead-by-construction (CUTLASS template-inlining equivalent); per-stage
hooks = CK policy-based design; `PipelineGeometry` decoupling pipeline scalars
from `self`.

**Core gap**: we compose by **inheritance chain**; CUTLASS/CK compose by
**orthogonal policies**. That drives most items below.

---

## 1. The orthogonal axes (motivates the whole plan)

Today's chain forces a total order on independent axes:

```
DenseFp8TileSpec → GroupFp8TileSpec → DispatchGroupFP8TileSpec → (Combine…)
```

But 5 axes actually vary independently:

1. **Layout** (nt/nn/tn) — compile-time. **Today this is the worst-encapsulated
   axis**: one layout's knowledge is split across 8 sites (see §2 ★). It must
   become a `LayoutPolicy` object first.
2. **Scheduler** (xcd-swizzle / linear-no-sync / future stream-k)
3. **Epilogue** (per-tensor store / combine-scatter / +bias+act)
4. **Prologue/Role** (none / dispatch-push / combine-signal)
5. **Grouping** (dense / per-expert base)

Dispatch is really a *prologue fusion*, not an "is-a grouped gemm". Target shape
= spec **holds (composes)** policies instead of **inheriting** them:

```python
class TileSpec:
    layout_policy   # nt/nn/tn loaders + offsets + swizzle
    scheduler       # .map(block_idx) -> (block_m, block_n)
    epilogue        # visitor chain
    prologue        # None | DispatchPrologue | CombinePrologue
    pipeline        # Pipeline policy (Stages / Intra-vs-Interwave)
```

Dense/Group/Dispatch/Combine become the same `TileSpec` with different
`{scheduler, epilogue, prologue}` — collapsing the duplicated `build_launch`
bodies.

---

## 2. Work items (priority order)

### ★ P0 (most important structurally) — Extract `LayoutPolicy` objects ✅ DONE

**Status (done):** `_LayoutPolicy` + `_NT/_NN/_TNLayoutPolicy` added to
`gemm_tile_spec.py`; `DenseFp8TileSpec` now holds `self.layout_policy` and
forwards `lds_geometry` / `base_offsets` (with the `group_base=0` seam) /
`global_swizzle` / `build_loaders` + the 4 race-fix booleans. All `if self.layout
==` branches removed from the spec body. Validated in `dev_primus`: dispatch
NT cos 0.999624 / NN 0.999398; combine NT 0.999943 / NN 0.999857 / TN 0.999909
— all at baseline (perf within run-to-run variance). Next: P0 emit-copy.

**This is the single largest gap vs CUTLASS/CK style.** Today "everything about
one layout" is sliced into **8 scattered pieces**:

| layout knowledge | site |
|---|---|
| LDS geometry (n_lds_steps / chunk_stride / lds sizes) | `gemm_tile_spec.py:245-260` |
| per-tile base offsets + k-step unit | `:404-425` (`if self.layout ==`) |
| global-load swizzle | `:441-450` |
| G2S/S2R loader selection | `:477-503` |
| `mask_a_in_tail` bool | `:302` |
| `main_b0_no_drain` bool | `:304` |
| `inplace_mma` bool | `:305` |
| `materialize_tid` bool | `:301` |

Four of those are explicit `if self.layout ==` switches; the rest are loose
booleans set in `__init__`. A layout's invariants (esp. the race-fix booleans)
are easy to mis-pair when scattered this way.

**How the references do it.**
- **CUTLASS**: layout is a *tag type* (`cutlass::layout::RowMajor`, or a CuTe
  stride); `GmemTiledCopyA/B` and the offset math are picked by partial
  specialization on that tag — never a `switch` in a function body.
- **CK**: NT/NN/TN are expressed as different *tensor descriptors* via coordinate
  transforms (merge / unmerge / pad / embed); the layout difference is the
  descriptor, and the mainloop is identical.

**Fix.** A `LayoutPolicy` trio — `NTPolicy` / `NNPolicy` / `TNPolicy` — each
class concentrating *all* of that layout's knowledge:

```python
class LayoutPolicy:           # NTPolicy / NNPolicy / TNPolicy
    # geometry + booleans (compile-time, set per layout)
    chunk_stride: int
    mask_a_in_tail: bool
    main_b0_no_drain: bool
    inplace_mma: bool
    materialize_tid: bool
    def lds_geometry(self, *, BLOCK_M, BLOCK_N): ...   # n_lds_steps, lds sizes
    def base_offsets(self, *, block_m, block_n, c_m, c_n, K, group_base=0): ...
    def global_swizzle(self, *, lane_id, wave_id, c_m, c_n): ...
    def build_loaders(self, *, a_div, b_div, gl_off_a, gl_off_b, wave_id, wave_m, wave_n): ...
```

`DenseFp8TileSpec` **holds** `self.layout_policy` and forwards the per-stage
hooks to it (the hook surface in `TileSpec` is unchanged — only the impl moves).
The dtype/scale policy (`build_mfma`'s cbsz/blgp atom, `build_store`) stays on the
spec, since it is orthogonal to layout.

**Payoff.**
- A new layout = **a new class**, not edits in 4 places — directly lowers the
  cost of adding TN variants / other layouts.
- A layout's invariants are **localized** → safer review (the race-fix booleans
  stop drifting apart from the loaders they guard).
- group/dispatch/combine specs **stop seeing layout branches** — they only care
  about grouping; layout is fully encapsulated.

**Note on `group_base`.** Give `LayoutPolicy.base_offsets` the optional
`group_base=0` scalar param up front (see P0-emit below) so grouping is a single
scalar add inside each policy — the grouped spec then overrides *nothing*
layout-related.

**Gate.** Pure mechanical move of existing branches → `FLYDSL_DUMP_IR=1`
empty-diff for nt/nn/tn dense + the grouped paths (`docs/TILE_API.md` §7).

**Risk:** low-medium (large but mechanical). **Touches:** `gemm_tile_spec.py`
(new policy classes + forwarding), `group_tile_spec.py` (drops layout awareness).

---

### P0 — Eliminate the `emit` copy (align with `docs/TILE_API.md` §2) ✅ DONE

**Status (done):** `DenseFp8TileSpec.emit` / `emit_uniform_k_tile` / the `TileSpec`
Protocol now take `group_base=0` + `lds=None`. `GroupFp8TileSpec` **dropped both
its `emit` and `base_offsets` overrides** — the duplicated uniform-K template is
gone; the shared template lives in ONE place. Grouping is now: a custom
`schedule` + a caller-side `group_base = g_idx * K * c_n` scalar (new static
helper `GroupFp8TileSpec.group_base`). Repointed all three callers (standalone
`build_launch`, dispatch GEMM role, combine GEMM role). Validated in `dev_primus`:
dispatch NT 0.999537 / NN 0.999273; combine NT 0.999933 / NN 0.999889 /
TN 0.999901 — all at baseline. Next: P1 (composition).

**Problem.** `GroupFp8TileSpec.emit` (`group_tile_spec.py:186-220`) re-copies
`emit_uniform_k_tile` (`gemm_tile_spec.py:519-595`) verbatim just to thread
`group_res`/`lds` — exactly the drift risk `docs/TILE_API.md:26-30` warned
against. Cause: `base_offsets` does `buffer_load(group_res, block_m)` internally
(`group_tile_spec.py:181`), coupling the table lookup into offset math and
forcing `emit` to grow a `group_res` param.

**Fix (the doc's `group_base` scalar approach).**
- Add optional `group_base=0` and `lds=None` params to the base
  `emit_uniform_k_tile` and `DenseFp8TileSpec.base_offsets`. Dense default 0 →
  bit-identical IR.
- `base_offsets` grouped variation becomes one scalar add: `B0 + group_base`,
  `B1 + group_base` (CUTLASS grouped = "group only shifts the global tile base").
- Caller (standalone launcher / `dispatch_tile` body) computes
  `group_base = g_idx * c_n * k_unit` and passes the scalar; the table lookup
  stays caller-side (it's already in the kernel body for dispatch).
- **Delete** `GroupFp8TileSpec.emit` entirely; grouped spec overrides only
  `base_offsets`. Pipeline template now exists once.

**Gate.** `g_idx` `buffer_load` site moves vs today → IR order may shift. Run the
`FLYDSL_DUMP_IR=1` diff (`docs/TILE_API.md` §7); require empty diff for the dense
path and for the grouped path vs the pre-refactor grouped kernel.

**Risk:** low. **Touches:** `gemm_tile_spec.py`, `group_tile_spec.py`,
`dispatch_grouped_gemm_fp8.py`, `grouped_gemm_fp8_combine.py`.

---

### P1 — Epilogue as a composable visitor chain (CUTLASS EVT)  🔶 IN PROGRESS

**Status (seam done).** The element-wise node chain is in and proven:
- `StoreCPerTensor` (`gemm_helper.py`) gained an IR-neutral `elem_fn=None` hook
  applied **post-scale / pre-cast** in the store loop (`None` -> identical IR).
- `PerTensorEpilogue` is now a composable chain: `nodes` (each a trace-time
  `f32 -> f32` emit callable) folded into `elem_fn`; `nodes=()` -> bit-identical.
- **Design finding (enforced):** epilogue nodes must discriminate the JIT cache —
  two specs with different epilogues but equal `cache_tag` collide and FlyDSL
  returns the first compiled kernel. `PerTensorEpilogue.cache_key` (() when no
  nodes) now folds into `DenseFp8TileSpec.cache_tag`.
- Validated in `dev_primus`: a `v -> v+v` node **doubles** the output (ratio
  2.0000, cos 1.0); empty-chain suite at baseline (dispatch NT 0.999566 /
  NN 0.999412; combine NT 0.999921 / NN 0.999894 / TN 0.999911).

**Factory plumbing done.** `act` (optional activation name) now threads through
`make_tile_spec` / `make_group_tile_spec` / `compile_grouped_gemm` /
`grouped_gemm_fp8_only`; registry `_EPILOGUE_ACTS = {"relu": ...}` (relu via
`maximumf` to keep the numeric wrapper's `.to` cast). Validated end-to-end in
`dev_primus`: `act="relu"` output == `clamp(plain, 0)` and plain cos 1.0 vs torch;
`act=None` suite at baseline (dispatch NT 0.999569 / NN 0.999459; combine
NT 0.999955 / NN 0.999873 / TN 0.999907). So the EVT chain is now a **usable
feature**, not just an internal hook.

**Finding — combine-push is NOT an epilogue node (re-scoped).** Inspected
`grouped_gemm_fp8_combine.py`: the combine role **spins until all `n_blocks`
N-tiles of a pool block finish** (`SB_L2` scoreboard, `:138`) then pushes a
**full `out_features`-wide row** routed per-row cross-rank
(`origin_rank`/`origin_slot`). A single GEMM tile's epilogue only holds BLOCK_N
columns, so it cannot push a complete row -- the L2Y round-trip is required by the
current full-N + cross-rank + scoreboard design. Eliminating it is a **kernel
redesign** (per-tile partial BLOCK_N-wide cross-rank scatter + reworked peer
buffer layout), not an `elem_fn` node. **Re-scoped to a P2-scale item**
("epilogue-scatter combine variant", high risk, perf-bench gated) -- removed from
P1-epilogue.

**Other future nodes (deferred, no current consumer):** bias / SwiGLU. Note
SwiGLU pairs gate*up across N-halves, so it is NOT a pure per-element `elem_fn`
node either (needs cross-column access) -- a different epilogue shape.

**Problem.** `build_store` returns a monolithic `StoreCPerTensor`. MoE needs
per-expert scale, bias, activation (silu/gelu), combine scatter-add + topk
weight, residual — none composable today; each variant becomes a new subclass.

**Fix.**
- Model epilogue as a chain: `acc → [scale] → [bias] → [act] → [scatter/combine]
  → store`. Each node emits an inline fragment; `build_store` assembles them.
- This demotes dispatch/combine differences from "new subclass" to "epilogue
  recipe difference".

**Perf win.** Fuse combine (per-expert scale + topk weight + scatter-add) into
the GEMM epilogue → removes a full `[tokens, hidden]` output read/write
round-trip (currently a separate `grouped_gemm_fp8_combine.py` kernel).

**Risk:** medium. Gate each node's addition with the IR diff against the
hand-written combine kernel.

---

### P1 — Inheritance chain → composition  🔶 IN PROGRESS

**Status (partial).** The spec now **composes 3 of 5 axes as held objects**:
- ✅ **LayoutPolicy** (★ P0) — `self.layout_policy`.
- ✅ **Scheduler** — `_Scheduler` + `XcdSwizzleScheduler` (dense) +
  `LinearNoSyncScheduler` (fused). `DenseFp8TileSpec` holds `self.scheduler` and
  `schedule` forwards; `GroupFp8TileSpec` swaps in the linear map in `__init__`
  (no `schedule` override).
- ✅ **Epilogue** — `PerTensorEpilogue` held as `self.epilogue`; `build_store`
  forwards. Seeds the P1-epilogue visitor chain (bias/act/combine become
  alternative Epilogue objects).

Validated each step in `dev_primus` at baseline (latest: dispatch NT 0.999559 /
NN 0.999380; combine NT 0.999924 / NN 0.999857 / TN 0.999898).

**Remaining — and why partly blocked by FlyDSL (`docs/TILE_API.md` §5):**
- (b) single `build_launch` skeleton: **blocked**. `@flyc.kernel`/`@flyc.jit`
  introspect *concrete named signatures*, and the 4 kernels have genuinely
  different ones (dense 7 params / grouped +2 / dispatch ~18 / combine ~12). No
  `*args` skeleton can wrap them.
- (c) collapse the subclass ladder: the **prologue/role** axis (dispatch push,
  combine signal, scoreboard spin) is *dynamic control flow* that the AST
  rewriter only lowers **lexically inside the `@flyc.kernel` body** — it cannot
  move into a held `prologue` object or a shared free function. So the fused
  `build_launch`es must stay per-kernel. *Feasible* alternative (larger, later):
  move `build_launch` out of the specs into free `build_*_launch(spec, ...)`
  functions so there is ONE spec class (config-only) — the ladder collapses even
  though each launcher keeps its own body. Deferred (touches the public factory
  API + all `spec.build_launch()` call sites).

**Problem.** Linear chain + each subclass re-copies the whole `@flyc.kernel` +
`@flyc.jit` `build_launch` (`dispatch_grouped_gemm_fp8.py:93-224` duplicates grid
sizing / value_attrs / guard / emit call).

**Fix.**
- Extract a `Scheduler` strategy object the spec **holds**
  (`DenseXcdSwizzle` / `LinearNoSync` / future `StreamK`), replacing the
  `schedule` method + baked-in `GroupFp8TileSpec.schedule`.
- Extract a `build_launch(extra_params, prologue_fn, scheduler)` skeleton helper
  so the kernel `operator()` skeleton is written once; only the genuinely-
  divergent comm-tensor signature + prologue closure differ.
- Spec composes `{layout_policy, scheduler, epilogue, prologue}`; drop the
  Dense→Group→Dispatch subclass ladder.

**Risk:** medium. Do **after** P0 (smaller diff surface).

---

### P2 — Persistent / Stream-K scheduler for ragged-M (biggest MoE throughput lever)

**Problem.** MoE per-expert M is **ragged**; a static tile→CTA map (current
linear no-sync, self-bounded by `NUM_TILE_BLOCKS`) idles CUs under expert skew.

**Fix.** Persistent kernel pulling tiles from a global work queue (atomic
counter / work-stealing), à la CUTLASS Stream-K. Plugs in as a `Scheduler`
policy (depends on the P1 scheduler extraction).

**Risk:** high (new sync structure; dynamic loop must live in the `@flyc.kernel`
body per FlyDSL constraint `docs/TILE_API.md` §5).

---

### P2 — Pipeline as a first-class policy (CK Intra/Interwave, CUTLASS `Stages`)

**Problem.** `run_uniform_k_pipeline` hardcodes the 2-buffer (cur/next) structure
+ the exact barrier ladder + 4-MFMA interleave. `PipelineGeometry` carries only
scalars/flags, **not the pipeline structure** → a new pipeline = a new function,
not a parameter.

**Fix.**
- Refactor `run_uniform_k_pipeline` into a `Pipeline` object method; spec holds
  `spec.pipeline`; default `UniformK2StagePipeline` (bit-identical to today).
- Then add as alternative implementations:
  - **`Stages > 2`**: deeper prefetch (3-buffer) for large-K grouped GEMM (MI300/
    MI355 LDS allows it) — better HBM latency hiding.
  - **Interwave scheduling**: two wave-groups stagger MFMA vs load — best for
    memory-bound small-per-expert-M MoE (CK's key knob; we are intrawave today).

**Risk:** high. Each new pipeline is a *new implementation*, gated by perf bench
(not IR diff — IR legitimately changes here).

---

### P3 — Block-scaled (MXFP8/MXFP4) scale seam — design now

**Problem.** Per-tensor scale only; the mainloop hooks have no "per-K-block
scale" seam. The repo already ships mxfp4/mxfp8 dequant kernels (git log), so
block-scaled GEMM is coming, and CUTLASS applies block scale as a **mainloop
fusion**, not an epilogue op.

**Fix.** While generalizing the pipeline (P2), reserve a per-K-block scale-apply
seam in the `Pipeline` policy (scale loaded + applied inside the K-loop). Cheaper
to design in now than to retrofit.

**Risk:** medium.

---

### P3 — `max_size` / 4GB num_records as a tensor-arg property

**Problem.** The byte-view >4GB `num_records` overflow (memory
`bf16-byteview-buffer-4gb-num-records-overflow`) is threaded as a per-`emit`
bool — an easy-to-miss call site.

**Fix.** Wrap inputs in a small descriptor that knows its byte-size policy
(CUTLASS handles this in the layout/tensor-descriptor), so the cap is a property
of the argument, not a per-call flag.

**Risk:** low.

---

## 3. Priority summary

| Pri | Item | Payoff | Risk | Gate |
|---|---|---|---|---|
| **★ P0** ✅ | Extract `LayoutPolicy` (NT/NN/TN) — **DONE** | localizes 8 scattered layout sites; new layout = new class; biggest CUTLASS/CK style gap | low-med | validated (cos at baseline) |
| **P0** ✅ | Eliminate `emit` copy (`group_base` scalar) — **DONE** | kills drift source; matches own doc | low | validated (cos at baseline) |
| **P1** ✅ | Epilogue visitor chain (node seam + `act` factory plumbing) — **DONE**; combine-fusion re-scoped to P2 (not an elem_fn node) | composable bias/act epilogue | med | validated (relu==clamp; empty at baseline) |
| **P2** | Epilogue-scatter combine variant (was "fuse combine") — kernel redesign, removes L2Y round-trip | saves full output round-trip | high | perf bench + vs combine kernel |
| **P1** 🔶 | Inheritance → composition — **Scheduler done**; build_launch skeleton + ladder collapse remain | kills subclass + `build_launch` dup | med | validated (cos at baseline) |
| **P2** | Persistent / stream-k scheduler | ragged-M load balance (top single lever) | high | perf bench |
| **P2** | Pipeline policy (Interwave / Stages>2) | latency hiding for memory-bound MoE | high | perf bench |
| **P3** | Block-scaled scale seam | enables MXFP path | med | perf + numeric |
| **P3** | `max_size` as tensor-arg property | removes error-prone call site | low | IR diff (empty) |

**Sequencing:** ★ P0 (LayoutPolicy) → P0 (emit copy) → P1 (composition) → P1
(epilogue) → P2 → P3. Do LayoutPolicy first: it gives `base_offsets` the
`group_base` seam the emit-copy fix relies on, and it removes the layout branches
that the composition step would otherwise have to carry. Both P0 items are
mechanical and shrink the diff surface for everything after.

---

## 4. Invariants (do not break)

- No `from __future__ import annotations` in any `gemm/*.py` / tile-spec module
  (stringifies `@fx.struct` fields → breaks LDS layout). `docs/TILE_API.md`
  §2/§5.4.
- Dynamic control flow (scoreboard spin, padding early-exit, comm role, runtime
  loops) stays in the `@flyc.kernel` body or a nested Name-called fn — never in a
  shared module-level free function. `docs/TILE_API.md` §5.2/§5.3.
- `str(fx.thread_idx.x)` first when tr16 loaders are used (nn/tn). §5.5.
- Every structural (non-pipeline) change is perf-neutral → validate with the
  `FLYDSL_DUMP_IR=1` empty-diff gate; pipeline-policy changes (P2) validate by
  perf bench instead. `docs/TILE_API.md` §7 (incl. the worktree test-runner
  gotcha).
