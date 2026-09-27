# op_baseline — vendored gfx1250 FlyDSL forward prefill kernel

Vendored 2026-09-23 from `/home/lihuzhan/code/aiter-src` (the aiter checkout, not an
installed wheel). The job imports THIS COPY; nothing here imports `aiter`.

## Files copied (verbatim except for the import edits below)

| vendored path | source (under `/home/lihuzhan/code/aiter-src/aiter/ops/flydsl/kernels/`) |
|---|---|
| `flydsl_fwd/fmha_fwd_prefill_a16w16_m32x8.py` | `fmha_gfx1250/fmha_fwd_prefill_a16w16_m32x8.py` |
| `flydsl_fwd/fmha_b16_buffer_managers.py`      | `fmha_gfx1250/fmha_b16_buffer_managers.py` |
| `flydsl_fwd/kernels_common.py`                | `kernels_common.py` (unmodified, 0 changed lines) |
| `flydsl_fwd/tensor_shim.py`                   | `tensor_shim.py` |
| `flydsl_fwd/buffer_ops.py`                    | `buffer_ops.py` (unmodified, 0 changed lines) |

The two `fmha_gfx1250/` files were flattened into the same package as
`kernels_common` / `tensor_shim` / `buffer_ops`, so their `..x` imports became `.x`.

## Import rewrites — 11 sites, not 8

Source line numbers are the ORIGINAL file's.

`fmha_fwd_prefill_a16w16_m32x8.py`
- `:51 from aiter.jit.utils.chip_info import get_lds_capacity_bytes` → **deleted**, body
  inlined (below)
- `:52 from aiter.ops.flydsl.kernels import buffer_ops` → `from . import buffer_ops`
- `:54 from ..kernels_common import LOG2E, create_llvm_ptr` → `from .kernels_common ...`
- `:55 from ..tensor_shim import _run_compiled` → `from .tensor_shim import _run_compiled`
- `:61 from .fmha_b16_buffer_managers import (...)` → unchanged (already same-package)

`fmha_b16_buffer_managers.py`
- `:39 from aiter.ops.flydsl.kernels import buffer_ops` → `from . import buffer_ops`
- `:41 from ..kernels_common import create_llvm_ptr` → `from .kernels_common ...`
- `:42 from ..tensor_shim import _to_raw as _ir` → `from .tensor_shim import _to_raw as _ir`

`tensor_shim.py`
- `:21 from aiter.ops.flydsl.kernels.kernels_common import get_warp_size` → `from .kernels_common import get_warp_size`
- `:50` and `:59` — two FUNCTION-LOCAL `from aiter.ops.flydsl.kernels import buffer_ops`
  inside `ptr_rsrc()` and `buf_load_scalar()` → `from . import buffer_ops`. These were
  not in the briefed list of 8; a module-header grep misses them.

## `get_lds_capacity_bytes`

Copied from `aiter/jit/utils/chip_info.py:71-88` into
`flydsl_fwd/fmha_fwd_prefill_a16w16_m32x8.py` (map + function), so the tree has no
aiter dependency. The single call site is
`_alloc_lds()` → `get_lds_capacity_bytes("gfx1250")`, which returns
`320 * 1024 = 327680`. The vendored copy makes `gfx` a required argument, because the
optional path is what called `get_gfx()` back inside aiter.

## Host entry

`flash_attn_batch_m32x8(q, k, v, softmax_scale=None, causal=False,
window_size=(-1,-1), out=None, return_lse=False, sink=None, lse=None)` —
`flydsl_fwd/fmha_fwd_prefill_a16w16_m32x8.py:2093-2104`. With `return_lse=True` it
returns `(out, lse)`; otherwise `out` alone (same file, `:2262-2264`).
`out` is `[B, Sq, Hq, 128]` in q's dtype, `lse` is `[B, Hq, Sq]` fp32.

Causal is BOTTOM-RIGHT: `causal_off = kv_len - q_len`
(`flydsl_fwd/fmha_fwd_prefill_a16w16_m32x8.py:793`), matching this job and the
backward job.

## Module identity

`impl.py` loads `flydsl_fwd/` through `_sibling_pkg()`, a package-shaped version of the
backward job's `_sibling()`. Reason unchanged: a plain `import flydsl_fwd` binds
`sys.modules["flydsl_fwd"]`, so a second implementation in the same process silently
reuses the first one's modules and its JIT-compiled kernels — two arms then differ by
under 0.05% and look like a real result.

## External dependencies after vendoring

flydsl, torch, numpy (`flydsl_fwd/tensor_shim.py:12`) and the stdlib. Nothing else.

---

## Op Setup addendum (2026-09-25, op_setup_v000)

**Branch:** neither corpus nor from-scratch: baseline supplied by op.baseline.root, copied verbatim.

- Source: `/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0925__flydsl/fwd-job/op_baseline/`
  copied with `rsync -a --exclude __pycache__`; `diff -r` against the root was empty
  before the one edit below.
- Upstream: `/home/lihuzhan/code/aiter-src` HEAD `6963ae9d77acb207d8f03fc92dddd8a57faf8fa6`
  (2026-09-17, clean worktree, also the last commit touching `aiter/ops/flydsl/kernels`).
  Re-diffed on 2026-09-25: kernels_common.py and buffer_ops.py 0 changed lines;
  tensor_shim.py 6 and fmha_b16_buffer_managers.py 6 (the import rewrites);
  fmha_fwd_prefill_a16w16_m32x8.py 39 (import rewrites + the inlined `get_lds_capacity_bytes`).
- Corpus checked first (knowledge HEAD 58b2134): `backends/flydsl/attention/recipes/hd128.md`
  (last touched 8ebf3e4) conflicts on heads_q_per_kv ([8,256] pow2 vs 4), format
  (thd-varlen/sbhd vs bshd), direction (fwd+bwd vs fwd), arch (gfx950 vs gfx1250),
  determinism, and its implementation is Primus-Turbo's FlyDSL, which refuses gfx1250.
  `hd64.md`: head dim 64, gfx950. Neither matches; nothing was taken from either.

**One edit, `_env.py`:** `os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "0")` was
replaced by ASSIGNING `TORCH_BLAS_PREFER_HIPBLASLT="1"` and
`HIPBLASLT_TENSILE_LIBPATH=/home/lihuzhan/.local/hipblaslt-gfx1250/gfx1250` (spec
runtime.env; sitecustomize.py:18 makes a setdefault a no-op), plus an `env_line()` helper.
No kernel file was touched.
