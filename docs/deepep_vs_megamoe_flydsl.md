# DeepEP vs MegaMoE vs FlyDSL — coverage and ownership

Working notes from an architecture discussion on whether to reimplement
DeepEP in FlyDSL, and whether DeepEP still needs to exist next to MegaMoE.

## TL;DR

| Question | Answer |
|---|---|
| Can DeepEP be rewritten in FlyDSL? | **Intranode yes** (already exists). **Internode no** today. |
| Should everything unify onto FlyDSL? | **No.** Internode is ~76% of DeepEP source and needs rocshmem. |
| Is MegaMoE faster than DeepEP? | **Yes on the fused MoE step** (1.08–1.12×), but they are not the same product. |
| Does DeepEP still need to exist? | **Yes** for multi-node EP, low-latency decode, and uncoupled dispatch/combine. |

Recommended ownership:

- **MegaMoE (FlyDSL + HIP IPC)** — single-node training main path.
- **DeepEP (HIP + rocshmem)** — multi-node, low-latency, and any caller that needs a library-style dispatcher.
- Optional middle ground: a `BackendType.FLYDSL` **standalone** intranode dispatch/combine in Turbo, so DeepEP's `intranode.cu` can shrink without touching internode.

## 1. FlyDSL is the language; Turbo is the product

Primus-Turbo's import graph is the hard evidence:

- Turbo imports `flydsl.expr` / `flydsl.compiler` / `flydsl._mlir` heavily.
- Turbo imports **zero** symbols from FlyDSL's own `kernels/` package.

So FlyDSL's `kernels/comm/flydsl_dispatch_combine_intranode_*.py` and Turbo's
`primus_turbo/flydsl/mega/*` are independent implementations written in the same
language. FlyDSL kernel samples prove the language can hit hardware; Turbo owns
backend selection, framework glue, fallbacks, and the HIP/CK home.

`BackendType` already has nine members (`CK`, `HIPBLASLT`, `AITER`, `TRITON`,
`DEEP_EP`, `TURBO`, `FLYDSL`, `HIPKITTENS`, `GLUON`). FlyDSL is one backend
among several, not a replacement for Turbo.

Inside Turbo, FlyDSL is already the largest single kernel language by line count
(~47k vs Triton ~14k). The trend is "FlyDSL replaces Triton as the preferred
authoring surface," not "FlyDSL replaces Turbo."

## 2. DeepEP maintenance pain is real — but not "dual .cu/.hip source"

DeepEP lives under `csrc/kernels/deep_ep/`. Important facts:

- `setup.py` only globs `cpp` / `cc` / `cu`.
- `.gitignore` ignores `*.hip`, `*_hip.cuh`, `*_hip.hpp`, etc.
- `tools/build_utils.py` runs vendored `hipify_torch` at build time.

So the `.hip` / `*_hip.cuh` files in the tree are **stale build artifacts**, not
a second source of truth. Counting both doubles the apparent size (~8.5k →
~4.1k real source lines).

Real split of the **source** that matters:

| File | Lines | Role |
|---|---|---|
| `internode.cu` | ~2145 | RDMA / rocshmem path |
| `intranode.cu` | ~990 | NVLink / P2P path |
| helpers (`utils`, `launch`, `buffer`, `layout`, `runtime`) | ~967 | shared |

Internode alone is ~52% of DeepEP; with shared helpers that only internode needs
kept, a FlyDSL rewrite of intranode removes at most ~1k lines and leaves ~3k.

The real maintenance cost is **upstream-fork scar tissue**: CUDA-shaped code
mechanically ported to HIP/rocshmem (`nvshmem_*` names calling `rocshmem_*`,
`NVSHMEM_DISABLE_P2P` env vars in `buffer.py`). Rewriting in FlyDSL does not
remove that cost — it replaces "painful rebase from upstream" with "permanent
fork, every RDMA bugfix is ours."

## 3. What FlyDSL can and cannot do for DeepEP

### Already done (intranode)

FlyDSL already has:

- `kernels/comm/flydsl_dispatch_combine_intranode_kernel.py` (~60 KB)
- matching op / all-reduce companions

It targets MORI-EP semantics (symmetric tensors + spin waits), not the DeepEP
`Buffer` API. Layout algebra barely appears (2 uses, both 1-D LDS views); the
kernel is mostly `buffer_load`/`buffer_store` plus hand-rolled LLVM atomics /
fences in `communication_ops_utils.py` because high-level FlyDSL APIs do not
expose system-scope ordering. FlyDSL's GEMM/layout strengths are unused here —
this is "HIP written in Python."

Turbo MegaMoE further forked that path into `primus_turbo/flydsl/mega/ep_intranode.py`
(see the Apache / FlyDSL adaptation header). So the "rewrite once" fantasy
already failed once: language reuse ≠ code reuse.

### Blocked (internode)

`internode.cu` is deep on device-side rocshmem:

- `rocshmem_ctx_*_put_nbi(_wave)`, `quiet`, atomics, barriers, teams
- `__shared__ rocshmem_ctx_t` contexts

FlyDSL has `extern` / `extern_link` (declare symbol + attach bitcode), but the
FFI type table is scalars only — no pointers/structs — and `@fx.struct` cannot
host an opaque C++ `rocshmem_ctx_t`. That is compiler work, not kernel work.
Grep for `internode|rdma|rocshmem` under FlyDSL is empty.

## 4. MegaMoE is faster — and still does not replace DeepEP

From `mxfp8_e2e_training_note.md` (median iter 3–50, only `use_turbo_mega_moe` flipped):

| Config | ms / iter | vs DeepEP baseline |
|---|---|---|
| bf16, DeepEP MoE | 9540 | — |
| bf16, MegaMoE | 8817 | **1.082×** |
| mxfp8, DeepEP MoE | 8432 | — |
| mxfp8, MegaMoE | 7508 | **1.123×** |

This times the **whole MoE layer** (dispatch + GEMM1 + SwiGLU + GEMM2 + combine
fused), not dispatch/combine alone. Gain is HBM trip elimination and overlap.

Hard coverage gaps vs DeepEP:

1. **Multi-node.** MegaMoE peer memory is HIP IPC
   (`primus_turbo/pytorch/core/symm_mem.py` — `hipIpcOpenMemHandle`). IPC is
   single-node. The file is honestly named `ep_intranode.py`; there is no
   internode twin.
2. **Low-latency decode.** DeepEP exposes `low_latency_dispatch` /
   `low_latency_combine` / clean + next-buffer APIs. MegaMoE has none.
3. **Coupling.** MegaMoE welds expert compute into the same kernels. Any caller
   that needs a library dispatcher with swappable experts still needs DeepEP
   (or a future standalone FlyDSL dispatcher).

Also: `mega_moe_ab_2node_note.md` reported **0.992×** for bf16 MegaMoE vs stock
bf16 MoE. That baseline is Megatron stock, not DeepEP, and the node/depth setup
differs. **Do not mix the two ratios** when making ownership decisions; reconcile
baselines before treating "fusion alone" as settled.

## 5. Recommendation — shrink DeepEP's role, do not unify languages

```
Single-node fused training MoE  →  MegaMoE (FlyDSL)
Standalone single-node EP       →  optional BackendType.FLYDSL dispatch/combine
Multi-node / LL / library EP    →  keep DeepEP (HIP + rocshmem), stay rebaseable
```

Concrete next steps if this note is accepted:

1. **Do not** start an internode FlyDSL rewrite.
2. Keep DeepEP's internode path as the intentional fork of upstream; invest in
   rebase hygiene, not a greenfield rewrite.
3. If standalone single-node EP still matters next to MegaMoE, add
   `BackendType.FLYDSL` to `moe_dispatch_combine_impl.py` (today only
   `TURBO` / `DEEP_EP`), wrapping the existing MORI-parity FlyDSL kernels behind
   Turbo's API — then delete or stop compiling DeepEP `intranode.cu` once parity
   is proven.
4. Align the 1.082× (vs DeepEP) and 0.992× (vs stock) measurements under one
   baseline matrix before claiming fusion ROI independent of mxfp8.

## 6. Non-goals

- Replacing Turbo's backend dispatcher with "everything is FlyDSL."
- Porting rocshmem device APIs into FlyDSL solely to delete DeepEP.
- Treating MegaMoE's step-level win as proof that DeepEP's dispatcher is slower
  in isolation.
