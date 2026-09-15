# Vendored fused MHA backward — provenance

Files this document covers:

| file | role |
|---|---|
| `primus_turbo/triton/attention/fused_mha_bwd_kernel.py` | the Triton kernels + the hard-coded config |
| `primus_turbo/pytorch/kernels/attention/attention_fused_bwd_impl.py` | host-side launch wrapper + Primus-Turbo adapter |

Neither file is wired into `attention_impl.py`. They exist and are importable; dispatch is a
separate change.

---

## 1. Where it came from

**Upstream:** [ROCm/aiter](https://github.com/ROCm/aiter)

**Pinned commit:** `ffa945f93115d21d6396e9fb16023e833b660764` (2026-09-13,
*"[FlyDSL] some moe optimization (#5448)"*).

**Source files:**

| upstream path | lines used | vendored into |
|---|---|---|
| `aiter/ops/triton/_triton_kernels/attention/mha_onekernel_bwd.py` | 1–1765 | `fused_mha_bwd_kernel.py` |
| `aiter/ops/triton/attention/mha_onekernel_bwd.py` | the `flash_attn_onekernel_backward` body | `attention_fused_bwd_impl.py` |
| `aiter/ops/triton/utils/_triton/mha_kernel_utils.py` | `_compute_fp8_scaling_factors` (8 lines) | inlined into `fused_mha_bwd_kernel.py` |
| `aiter/ops/triton/utils/types.py` | the `_is_fp8` dtype set | inlined into `attention_fused_bwd_impl.py` |
| `aiter/ops/triton/configs/gfx1250/triton/attention/mha/DEFAULT.json` | the `bkwd_onekernel` block | re-expressed as a Python dict, **with three tuned deltas** |

**This is *not* `aiter/ops/triton/attention/mha_fused_bwd.py`.** That is a different kernel and
it is numerically broken on this arch — it produced `dk` at **−0.22 dB** SQNR. It is imported
only by aiter's public `attention/mha.py`, which is not vendored, so it cannot leak in. The
kernel here is the *one-kernel* backward (`mha_onekernel_bwd`), which measured in the canonical
correct band (§5).

---

## 2. Licence

The task brief said Apache-2.0. That is **wrong**: `/home/lihuzhan/code/aiter-src/LICENSE` is the
**MIT License**, `Copyright © Advanced Micro Devices, Inc. All rights reserved.` Every vendored
source file carries `# SPDX-License-Identifier: MIT` plus
`# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.`
(`mha_kernel_utils.py` says `2024-2025`).

MIT is the same licence that covers Primus-Turbo (root `LICENSE`), so this vendoring needs:

* **no** new licence file,
* **no** `LICENSE-APACHE`-style second licence,
* **no** `3rdparty/` submodule,
* **no** "distributed under … not the MIT license that covers the rest of Primus-Turbo" paragraph
  (the one the Megatron- and FlyDSL-derived files carry).

That is exactly the situation of `primus_turbo/triton/attention/attention_kernel.py`, whose header
says *"MIT License, the same license that covers Primus-Turbo (see LICENSE)"*. The vendored header
copies that house style, preserves the upstream SPDX and AMD copyright lines verbatim, and adds the
Primus-Turbo modification notice. `tools/check_license.py` passes on both new files.

---

## 3. What was changed, and why

### 3.1 `fused_mha_bwd_kernel.py`

The kernel bodies are **byte-identical to upstream lines 1–1765** except for the three edits below.
This was verified mechanically with a `difflib` comparison of the vendored body against the
programmatic excision of the upstream file; the only diff was one leading and one trailing blank
line.

| # | change | why |
|---|---|---|
| 1 | Dropped `from aiter.ops.triton...` imports (3 lines) and `import functools`. | **The single most important change.** Any surviving `from aiter…` import — of *any* aiter module, there is no partial import — re-executes `aiter/__init__.py`, which unconditionally (deliberately not in a `try/except`) pulls in `aiter.jit.core` and ~45 `from .ops.* import *` lines: the CK/HIP JIT build machinery, ~7 s and ~130 lines of stderr per process, and it does not build on gfx1250. `AITER_TRITON_ONLY=1` is upstream's escape hatch; the vendored copy must not depend on an env var being set. `functools` was used only by the dropped `_get_config`. |
| 2 | `make_kernel_repr` dropped: the three `_*_repr = make_kernel_repr(...)` module statements deleted and `@triton.jit(repr=…)` → `@triton.jit`. | Purely cosmetic — it names compiled artifacts for debuggability. Removing it avoids vendoring a 57-line module with zero numeric or performance effect. |
| 3 | `_compute_fp8_scaling_factors` inlined verbatim (with its own SPDX/copyright attribution comment). | It is a `@triton.jit` **device** function called from four `if IS_FP8:` branches inside the kernel body. It cannot be stubbed out even though the production path is bf16 — copying it is what keeps those branches unedited. |
| 4 | `_get_config()` (JSON discovery) replaced by `get_fused_bwd_config()` over a hard-coded table. | See §4. |

### 3.2 `attention_fused_bwd_impl.py`

| # | change | why |
|---|---|---|
| 1 | `AiterTritonLogger` dropped along with its single `_LOGGER.info(f"…")` call. | One call site. The f-string was built unconditionally on every backward call — pure per-call overhead — and the logger reads `AITER_TRITON_LOG_LEVEL`, which means nothing here. |
| 2 | `_is_fp8` reimplemented as a local dtype-set predicate. | Upstream's version additionally asks `arch_info.is_fp8_avail()`, and `arch_info` evaluates `triton.runtime.driver.active.get_current_target()` **at module import time**, with a fallback into `jax._src.lib`. That is an import-time device touch we should not inherit. The predicate itself is a four-element dtype set. |
| 3 | The `if causal: bwd_kernel_causal[grid](...) else: bwd_kernel_noncausal[grid](...)` duplication collapsed to one `kernel = … ; kernel[grid](…)`. | The two upstream launch blocks were argument-for-argument identical. Purely a de-duplication; the argument list is unchanged. |
| 4 | The dead `stride_descale_q_z = … = None` chain in the non-fp8 branch replaced by `descale_strides = (None, None, None, None)`. | Same value, four fewer dead locals. |
| 5 | Added `dense_fused_backward`, which is not upstream at all. | The Primus-Turbo-shaped entry point. See §6. |

`arch_info`, `config_utils`, `logger`, `device_info`, `kernel_repr`, `pid_preprocessing`,
`_gluon_kernels/`, and every aiter `__init__.py` are **not** vendored. In particular
`utils/device_info.py:get_num_sms()` does `from aiter.jit.utils.chip_info import get_cu_num`
inside a bare `except Exception` — a direct reach into the JIT tree. It is forward-wrapper-only
and is not on this path.

**Verification (no GPU, no torch, no triton required):**

```bash
python3 -c "import ast; ast.parse(open('primus_turbo/triton/attention/fused_mha_bwd_kernel.py').read())"
python3 -c "import ast; ast.parse(open('primus_turbo/pytorch/kernels/attention/attention_fused_bwd_impl.py').read())"
grep -n '\baiter\b' primus_turbo/triton/attention/fused_mha_bwd_kernel.py \
                    primus_turbo/pytorch/kernels/attention/attention_fused_bwd_impl.py
```

The `grep` must return only comment and docstring lines. It currently does. The full import list is
`os`, `triton`, `triton.language` (kernel file) and `typing`, `torch`, `triton`, plus two
first-party `primus_turbo.triton.attention` imports (adapter). Nothing imports at module scope that
touches a device.

---

## 4. The config

Upstream resolves the config at call time:

```python
@functools.lru_cache(maxsize=1024)
def _get_config():
    cfg_dir = resolve_config_dir("attention", "MHA", backend="triton")
    config = load_config_json(f"{cfg_dir}/DEFAULT.json")
    return config["bkwd_onekernel"]
```

That whole mechanism is dropped:

* `resolve_config_dir` exists to pick a file per GPU arch. A single-arch vendored kernel has
  nothing to discover, and the path it takes reaches `arch_info`'s import-time device probe.
* A vendored `.json` **would not ship**. `setup.py` sets
  `package_data={"primus_turbo": ["lib/*.so"]}` and `MANIFEST.in` lists only
  `README.md` / `LICENSE` / `LICENSE-APACHE`. Hard-coding the table in Python is not merely
  simpler, it is the only form that survives a wheel build without also editing packaging.

**The shipped values are not upstream's.** aiter's gfx1250 `DEFAULT.json` `bkwd_onekernel` block is
`BLOCK_M1=32, BLOCK_N1=128, BLOCK_M2=128, BLOCK_N2=32, BLK_SLICE_FACTOR=2`. Three deltas:

| key | upstream gfx1250 | shipped here |
|---|---|---|
| `BLOCK_N1` | 128 | **256** |
| `BLOCK_M2` | 128 | **256** |
| `BLK_SLICE_FACTOR` | 2 | **1** |

`BLOCK_M1=32`, `BLOCK_N2=32`, `PRE_BLOCK=128`, `waves_per_eu=1`, `matrix_instr_nonkdim=16`,
`num_warps=4`, `num_ctas=1`, `num_stages=1` are unchanged.

### Tuning surface

Following the `PRIMUS_TURBO_ATTN_TRITON_TUNE` convention in `attention_kernel.py`:

```
PRIMUS_TURBO_FUSED_MHA_BWD_TUNE=BLOCK_N1=128,BLOCK_M2=128,BLK_SLICE_FACTOR=2
```

overrides individual keys (that particular spec recovers upstream's shipped gfx1250 config).
Unset, empty, `0` or `off` means the shipped Primus-Turbo values. **Anything malformed raises**
— an unknown key, a non-integer value, a non-positive value, a missing `=`, or a set-but-empty
spec. It never silently falls back, because a typo that quietly measured the default config would
read as "this knob does nothing".

### The block invariant

`get_fused_bwd_config()` asserts `BLOCK_N1 == BLOCK_M2` and `BLOCK_M1 == BLOCK_N2`. The kernel
walks the score matrix twice — `BLOCK_M1 × BLOCK_N1` for dK/dV, `BLOCK_M2 × BLOCK_N2` for dQ — and
the launch grid is sized off `BLOCK_N1` only. A config where the two passes disagree does not
error; it silently computes the wrong thing. The tuning harness
(`tools/gfx1250/tune_attention.py`) enforces the same pair; carrying it into the library means a
hand-written env override cannot get it wrong either. The shipped champion satisfies it (256/256,
32/32).

---

## 5. The measured numbers that justify this

gfx1250, `b=4 s=8192 hq=32 hkv=8 d=128`, bf16, causal, on a VR-throttled card.

| variant | fwd | bwd | total | TFLOP/s |
|---|---|---|---|---|
| Primus-Turbo shipped (non-Triton path) | — | — | 59.642 ms | 129.0 |
| torch flex attention | — | — | 31.337 ms | 245.6 |
| Primus-Turbo in-tree Triton backward | — | 32.25 ms | — | — |
| **this kernel (champion config)** | **3.26 ms** | **18.42 ms** | **21.67 ms** | **355.1** |

* **1.75×** faster backward than the in-tree Triton backward (32.25 → 18.42 ms).
* **1.45×** the torch-flex total, **2.75×** the shipped starting point.

Accuracy, same shape: **SQNR out / dq / dk / dv = 53.67 / 52.24 / 52.31 / 52.71 dB** — the
canonical correct band. (Compare the *other* aiter kernel, `mha_fused_bwd.py`, at dk = −0.22 dB.)

### Caveat on the forward number

The `fwd 3.26 ms` in that row came from **aiter's own forward**
(`aiter/ops/triton/attention/mha.py:flash_attn_func`, with `fwd.default` from the same
`DEFAULT.json` and `num_stages` overridden 1 → 2:
`BLOCK_M=128, BLOCK_N=64, PRELOAD_V=true, waves_per_eu=2, num_warps=4, num_ctas=1, num_stages=2`).
**That forward is not vendored.** If Primus-Turbo keeps its own forward
(`primus_turbo/triton/attention/attention_kernel.py`), the 3.26 ms figure does **not** transfer and
the total must be re-measured. The 18.42 ms backward win is independent of the forward and does
transfer — subject to §6.

---

## 6. LSE / delta convention — read this before trusting the gradients

**Getting this wrong produces smoothly wrong gradients, not a crash.** It is stated here rather
than left to be inferred.

### What the vendored kernel expects

* `softmax_lse` shaped `[batch, num_q_heads, seqlen_q]`, **row-major and densely packed** — one
  float per query row, addressed with plain strides taken from `delta.stride()`.
* `delta` is a **separate** tensor of the same shape, allocated inside
  `flash_attn_onekernel_backward` as `torch.zeros_like(softmax_lse)` and returned to the caller.
* LSE in **natural-log units**. The kernel loads `m` and multiplies by `1/ln2` itself
  (`USE_EXP2=True` is passed unconditionally; upstream `mha_onekernel_bwd.py:259, 478`).
* Tensor layout **bshd**: `q`/`k` are `(batch, seqlen, nheads, d)`. Strides are reordered to
  `(b, h, s, d)` at the launch site, so non-contiguous inputs are fine as long as the *shape* is
  bshd. This is **not** bhsd.
* GQA is handled inside the kernel; grid axis 0 is `num_k_heads`.
* `dq`, `dk`, `dv` must be **pre-allocated by the caller**; the function returns only `delta`.

### What Primus-Turbo has

Primus-Turbo's forward (`attention_kernel.py:attn_fwd`) does **not** write a plain `[B, Hq, Sq]`
LSE. It allocates a `[B, Hq, 2*Sq]` scratch and writes the LSE for block `m` at element offset
`m * BLOCK_M * 2`; the in-tree backward preprocess later writes `delta` at `+ BLOCK_M` of that same
block. **LSE and delta interleave every `FIXED_BLOCK_M` (= 64) rows, not every row.** The host-side
reconstruction lives in `attention_triton_impl._lse_delta_views`.

Units agree: `attention_kernel.py:1093-1099` computes `m_i/ln2 + log2(l_i)` then multiplies by
`ln2`, i.e. natural log. The `USE_EXP2=False` branch writes `m_i + log(l_i)`, also natural log. So
**no unit conversion is required in either branch** — only a layout conversion.

### What the adapter assumes

`dense_fused_backward` inspects `softmax_lse.shape[2]`:

* `== 2 * seqlen_q` → treated as the **packed Primus-Turbo scratch**. The LSE rows are gathered
  out with `(row // 64) * 128 + (row % 64)` into a dense `[B, Hq, Sq]` float32 tensor for the
  kernel, and after the launch the returned `delta` is **scattered back** into the delta half at
  `+ 64`, so a following `dense_sink_grad(softmax_lse, …)` still finds what it expects.
* `== seqlen_q` → treated as a plain natural-log LSE. Nothing is written back.
* anything else → **raises**. It does not guess.

The index arithmetic is repeated in the adapter rather than imported from
`attention_triton_impl._lse_delta_views`, because that helper is private to a module this file
must not depend on. `FIXED_BLOCK_M` **is** imported from `attention_kernel`, since it is the
documented cross-kernel ABI constant and the two must not be allowed to drift.

Two consequences worth stating:

1. The gather/scatter is an extra pair of small passes over a `[B, Hq, Sq]` fp32 tensor per
   backward (at the production shape, 4 × 32 × 8192 × 4 B = 4 MiB each way). It is not free, and it
   is not in the 18.42 ms figure, which was measured against aiter's own flat LSE. If the fused
   kernel becomes the default, the right fix is to teach the forward to emit a flat LSE and delete
   the conversion — not to keep paying it.
2. Because the adapter *reads* LSE and *writes* delta, calling it twice on the same scratch is
   fine, but calling it on a scratch whose delta half has not yet been produced is also fine — the
   adapter never reads the delta half.

### Other fixed assumptions of `dense_fused_backward`

* `dbias=None` (upstream raises otherwise), `alibi_slopes=None`, `dropout_p=0.0`,
  `sink=dsink=None`, `cu_seqlens_*=None` (dense path only — varlen is supported by the wrapper but
  not by this adapter).
* `window_size` is Primus-Turbo's `(left, right)`. The fused kernel has only a **left**
  (`SLIDING_WINDOW`) window, so `right` must be `-1` or `0` and a non-negative `left` requires
  `causal=True`; both are checked and raise.
* `sink` is deliberately **not** plumbed through, even though the kernel supports it via
  `Sink`/`DSink` and `ENABLE_SINK`. Wiring `dsink` (which the kernel produces with
  `tl.atomic_add`) is a separate change.

---

## 7. Known rough edges

* The vendored kernel file carries upstream's black-at-88-columns formatting. The repo's
  `pre-commit` runs `ruff-format` at `line-length = 110`, which will reflow it on first commit;
  four lines currently exceed 110 columns. `ruff` was not installed in the environment where this
  was vendored, so the reflow has not been applied. The excision recipe in §1/§3.1 is exact, so the
  upstream diff can be regenerated at any time regardless of formatting.
* Nothing here has been executed. The constraint was strictly no GPU, so only `ast.parse`, the
  import-list audit, a CPU-only exercise of `get_fused_bwd_config()` (defaults, overrides, and all
  five raise paths), and the upstream `difflib` comparison were run.
