# Attention Backends

`turbo.ops.flash_attn_func` (dense `[b, s, h, d]` tensors) and `turbo.ops.flash_attn_varlen_func`
(packed THD) pick one kernel backend per call. The forward and the backward of a call always run
on the same backend.

## Table of Contents

- [1. Selecting a backend](#1-selecting-a-backend)
- [2. FlyDSL on gfx1250 (MI455X)](#2-flydsl-on-gfx1250-mi455x)
  - [2.1 DeepSeek-V3 MLA support matrix](#21-deepseek-v3-mla-support-matrix)
  - [2.2 FlyDSL version](#22-flydsl-version)
  - [2.3 Backward launch knobs](#23-backward-launch-knobs)

## 1. Selecting a backend

Dense attention registers the backends `FLYDSL`, `AITER`, `HIPKITTENS`, `GLUON` and `TRITON`;
varlen attention registers `FLYDSL` and `AITER`. Each backend checks whether it supports the call
(arch, dtype, head dims, mask, layout, ...). Priority, high to low:

1. Code: `GlobalBackendManager.set_attn_backend(BackendType.<NAME>)`.
2. Environment: `PRIMUS_TURBO_ATTN_BACKEND=<NAME>` (case-insensitive; a backend name, not
   `autotune`).
3. AutoTune (`PRIMUS_TURBO_AUTO_TUNE=1`): time every eligible backend once per shape key.
4. The default backend, `FLYDSL`, when it supports the call.
5. Fallback: the first registered backend that supports the call, with a one-time warning.

A backend named in code or in the environment is used as is: if it does not support the call,
the call raises `ValueError` instead of falling back. Use that to make sure a run really uses
the kernels you expect.

```bash
# Require the FlyDSL kernels (raise on any call outside their support matrix).
export PRIMUS_TURBO_ATTN_BACKEND=FLYDSL

# Force the portable Triton kernels (enabled on gfx1250 only).
export PRIMUS_TURBO_ATTN_BACKEND=TRITON

# Default: FlyDSL where it is supported, another backend otherwise.
unset PRIMUS_TURBO_ATTN_BACKEND
```

```python
from primus_turbo.pytorch.core.backend import BackendType, GlobalBackendManager

GlobalBackendManager.set_attn_backend(BackendType.FLYDSL)
# ... flash_attn_func calls ...
GlobalBackendManager.set_attn_backend(None)  # back to the environment / default
```

## 2. FlyDSL on gfx1250 (MI455X)

On gfx1250 the `FLYDSL` backend has two kernel families, chosen by the head dims of the call:

| head dims (q/k, v) | kernels | scope |
|---|---|---|
| 128, 128 | `primus_turbo/flydsl/attention/gfx1250/` | bf16, GQA, bottom-right causal or full attention |
| 192, 128 | `primus_turbo/flydsl/attention/gfx1250_mla_fwd/`, `gfx1250_mla_bwd/` | DeepSeek-V3 MLA, see below |

Any other head dims are not handled by FlyDSL on gfx1250.

### 2.1 DeepSeek-V3 MLA support matrix

The MLA kernels cover DeepSeek-V3's attention as Megatron trains it: q/k head dim 192
(128 nope + 64 rope), v head dim 128, `causal=True`, bf16.

| | supported | not supported |
|---|---|---|
| dtype | bf16 (q, k, v) | fp16, fp32, fp8 |
| head dims | q/k 192, v 128 | any other pair |
| heads | MHA (`heads_q == heads_kv`) | GQA, MQA |
| mask | bottom-right causal: query `i` sees keys `j <= i + seqlen_kv - seqlen_q` (`window_size` `(-1, -1)` or `(-1, 0)`) | non-causal (full) attention, sliding window |
| sequence lengths | `seqlen_q % 64 == 0`, `seqlen_kv % 64 == 0`, `0 < seqlen_q <= seqlen_kv` | other lengths, `seqlen_q > seqlen_kv` |
| layout | dense `[b, s, h, d]` tensors in any byte order; SBHD storage (Megatron) is read with no copy, other orders are copied | varlen / THD (`flash_attn_varlen_func`) |
| softmax scale | any value (applied at run time in fp32); default `1/sqrt(192)` | |
| extras | | dropout, bias, ALiBi, attention sink, `return_attn_probs` |
| size | q / dO up to 1 GiB, k / v below 2 GiB, lse up to 256 MiB | larger tensors |

- Deterministic: every element of dq / dk / dv is written once (no atomics), and repeated calls
  give bitwise identical results.
- `torch.compile`: the kernels are custom ops with fake implementations; `fullgraph=True` works.
- Calls outside the matrix go to another backend (or raise, when `FLYDSL` was named).

### 2.2 FlyDSL version

The MLA kernels use gfx1250 FlyDSL operations whose API changes between releases (the Tensor
Data Mover, `s_wait_dscnt` / tensorcnt waits, split barriers, `ds_load_tr16`). They need
**flydsl >= 0.3.4, < 0.3.5** (`gfx1250_mla_version.FLYDSL_REQUIREMENT`):

- Under any other flydsl, or without one, the MLA gate reports why and the call falls back to
  another backend; no MLA kernel module is imported.
- Importing `gfx1250_mla_fwd` or `gfx1250_mla_bwd` directly raises `ImportError` naming the
  requirement and the flydsl that was found.

Primus-Turbo's install requirements pin `flydsl==0.2.4`, which the MLA kernels do not accept;
install a 0.3.4.x flydsl to use them.

### 2.3 Backward launch knobs

Read once at import. They only choose launches: dq / dk / dv are bitwise identical under every
setting.

| variable | default | effect |
|---|---|---|
| `FLY_BWD_HEAD_GROUP` | `64` | launch order: a launch with more heads per batch than this runs them in groups of at most this many heads; `0` keeps all heads of a batch in one group |
| `FLY_BWD_SMALL_GRID` | `1` | `0` disables the small-grid fallback (one-wave kernels when a launch would not fill the GPU) |
| `FLY_BWD_SIDE_STREAM` | `0` | `1` runs the dQ kernels on a side stream concurrently with dK/dV (slower at DeepSeek-V3 training shapes) |
