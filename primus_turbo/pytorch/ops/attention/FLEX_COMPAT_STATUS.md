# Primus-Turbo `flex_attention` compatibility layer status

This document describes the current capability boundary of
`primus_turbo.pytorch.ops.attention.flex_attention`. It answers: "which torch flex variants can
dispatch straight to Turbo's high-performance kernels today, which ones raise, and how do we close
the remaining gaps?"

Entry points:

```python
from primus_turbo.pytorch.ops.attention import (
    flex_attention,        # torch-compatible: bhsd [B,H,S,D] in and out
    flex_attention_bshd,   # layout-native:   bshd [B,S,H,D] in and out
    flex_attention_varlen, # THD packed + cu_seqlens
    create_block_mask,
)
```

The signature matches `torch.nn.attention.flex_attention.flex_attention` and appends a few
**optional Turbo extension parameters** (a superset; all disabled by default via `None`/`0.0`, so
torch-style calls need zero changes and remain a drop-in replacement for torch):
`flex_attention(query, key, value, score_mod=None, block_mask=None, scale=None,
enable_gqa=False, return_lse=False, kernel_options=None, alibi_slopes=None, softcap=None,
dropout_p=0.0, sink=None, bias=None)`.
Internally it converts between `bhsd ([B,H,S,D])` and `bshd ([B,S,H,D])` layouts and maps every
recognised variant onto `flash_attn_func` (which auto-selects FlyDSL/AITER on gfx950).

**`flex_attention_bshd` -- the layout-native entry (no transposes).** torch flex speaks bhsd; the
Turbo kernel speaks bshd. `flex_attention` therefore does `transpose(1,2).contiguous()` on q, k and
v and transposes the output back -- **4 full tensor copies per forward**, plus their mirror images
on the gradients in backward. That is the residual wrapper overhead left after the classification
cache (measured at ~0.03-0.27 ms, i.e. ~0.5-1.2% of an E2E step). Callers that already hold bshd --
or sbhd, which is one *free* permute away, as in Megatron / Primus-LM attention layers -- were
paying all of it for nothing. `flex_attention_bshd` takes exactly the same arguments and has exactly
the same semantics, validation and error messages, but takes `[B,S,H,D]` in and returns `[B,S,H,D]`
out, so a bshd-contiguous q/k/v reaches the kernel as the *same buffer* (verified by `data_ptr()` in
the unit tests) and the kernel's output is returned untouched: **zero layout copies**. It is
deliberately *not* a torch drop-in (its layout differs) -- that contract stays with
`flex_attention`, which is unchanged. `score_mod` / `block_mask` index semantics are identical:
`q_idx` / `kv_idx` are sequence positions in either layout.

### Turbo extension parameters (superset, disabled by default)

- **`alibi_slopes: Optional[torch.Tensor] = None` (dispatches to Turbo today)**: explicit per-head
  ALiBi slopes. Must be 1D, `length == Hq` (query head count) and fp32 (otherwise a clear
  `ValueError`); the device is aligned to q automatically. Passing it **skips the
  `_detect_alibi_slopes` auto-detection** and forwards the slopes directly to
  `flash_attn_func(alibi_slopes=...)` -- so it **bypasses the detector's conservative limits and
  works right now**, and can be combined with causal / sliding-window masks (ALiBi is usually paired
  with causal). **Conflict handling**: passing both an explicit `alibi_slopes` and a
  **non-trivial (non-identity)** `score_mod` is treated as ambiguous and raises `ValueError` (pick
  one); a `None`/identity `score_mod` may coexist with explicit slopes.
- **`softcap: Optional[float] = None` (interface in place, currently gated with a raise)**: logits
  soft cap (`cap*tanh(score/cap)`, Gemma2/Grok). `None` or `0`/`0.0` = disabled (no-op, existing
  paths unaffected); a positive value currently **raises `NotImplementedError`** (the interface is
  in place but is blocked by this build's aiter dense fwd/bwd kernels, which lack a softcap
  parameter -- see "softcap status" below; the parameter takes effect as soon as the upstream kernel
  supports it). An explicit `softcap` and a soft-cap detected from `score_mod` **funnel into the
  same single raise site** (no duplication, no conflict) and the cap is **never silently dropped**.
- **`dropout_p: float = 0.0` (dispatches to Turbo today)**: attention dropout probability, validated
  as `0 <= p < 1` (`0` = disabled, the drop-in default), forwarded directly to
  `flash_attn_func(dropout_p=...)`. Consistent with flash-attn / torch
  `scaled_dot_product_attention`: `p>0` takes effect (training semantics -- pass `0` for eval). It
  can coexist with `return_lse`; the compat layer always dispatches with `deterministic=False`, so
  there is no dropout/determinism conflict to reject. Measured on GPU: `p=0` is byte-for-byte
  identical to not passing it (zero regression), and `p=0.1` runs forward+backward correctly
  (finite outputs/gradients, correct shapes).
- **`sink: Optional[torch.Tensor] = None` (dispatches to Turbo today)**: attention sink (one
  learnable logit per query head). Must be 1D, `length == Hq` and fp32 (otherwise a clear
  `ValueError`); the device is aligned to q automatically. The sink kernel path additionally
  requires `head_dim_qk == head_dim_v` and a power-of-two head_dim (backend constraint, see
  `attention_aiter_impl.AttnFwdAiterBackend`). Forwarded directly to `flash_attn_func(sink=...)`.
  Measured on GPU: the forwarded result is **byte-for-byte identical** to calling
  `flash_attn_func(sink=...)` directly (identity), and `sink=None` is a zero regression.
- **`bias: Optional[torch.Tensor] = None` (dispatches to Turbo today)**: additive bias on the
  pre-softmax logits, forwarded directly to `flash_attn_func(bias=...)`. **Key constraint (verified
  empirically, see "bias status" below)**: the aiter dense kernel only accepts a **single
  `[Sq, Skv]`** bias (shared across batch/head) whose dtype must match q (fp16/bf16;
  **fp32 produces NaN**, and 4D / per-head bias is rejected by the kernel with
  `bias shape should be [sq, sk]`). The entry point accepts `[Sq,Skv]` or broadcastable shapes with
  leading singletons (`[1,Sq,Skv]`/`[1,1,Sq,Skv]`), casts to q's dtype and aligns the device; a true
  per-head/per-sample bias raises `ValueError`. Forward and backward were verified numerically on
  GPU.

## Supported (dispatches directly to Turbo)

- **full attention** (`block_mask is None` or all-True) -> `causal=False`
- **causal** (`q >= kv`) -> `causal=True`
- **sliding-window causal** (`(q >= kv) & (q-kv <= W)`) -> `causal=True, window_size=(W,0)`; **windows
  larger than the probe grid (W>512, e.g. W=1024/2048/4096) are also supported on long sequences
  S>512**: the True->False flip point of W is located by binary search on the last query row, then
  several sampled rows are verified bit-exactly as standard left-window causal before classifying
  (see "large-window probing" below)
- **varlen / document packing (explicit entry `flex_attention_varlen`)**: THD `[total,H,D]` packing +
  `cu_seqlens` dispatched straight to `flash_attn_varlen_func` (see "varlen / document packing
  status" below)
- **document-causal (auto-detected in the dense entry)**: when a `block_mask` is **verified by exact
  reconstruction** to be `same_doc(q,kv) & (q>=kv)`, the dense `flex_attention` automatically routes
  through varlen (block-diagonal `cu_seqlens`) instead of a dense causal that would attend across
  documents (details below)
- **GQA/MQA**: relies on Turbo's native support; requires `Hq % Hkv == 0`, and when `Hq != Hkv` the
  caller must pass `enable_gqa=True` explicitly (matching torch flex semantics)
- **score_mod=None**
- **ALiBi (auto-detected)**: only when `score_mod` is strictly verified to be
  `score + slope[h] * (kv-q)` (additive on score with coefficient 1, linear in `(kv-q)`,
  translation-invariant, batch-independent) is it mapped to `alibi_slopes`; otherwise it is not
  recognised
- **ALiBi (explicit parameter)**: pass per-head slopes via `alibi_slopes=` (1D/fp32/len==Hq), which
  **skips auto-detection and dispatches straight to `flash_attn_func`**, bypassing the detector's
  conservative limits; equivalent to auto-detection for the same slopes
- **return_lse**: forwards Turbo's `softmax_lse` (returns `(out, lse)`)
- **scale**: defaults to `1/sqrt(D)`, matching torch flex; an explicit scale is forwarded to the
  backend
- **dropout (explicit parameter)**: `dropout_p` forwarded to `flash_attn_func(dropout_p=...)`, `0`
  disables it (the drop-in default); measured on GPU, `p=0` is a zero regression and `p=0.1` passes
  forward+backward
- **attention sink (explicit parameter)**: `sink` forwarded to `flash_attn_func(sink=...)`
  (1D/fp32/len==Hq, head_dim constraints above); measured on GPU as byte-for-byte identical to a
  direct backend call, with `sink=None` a zero regression
- **additive bias (explicit parameter)**: `bias` forwarded to `flash_attn_func(bias=...)`, requires
  `[Sq,Skv]` / q's dtype (see above); forward and backward verified numerically on GPU (bf16/fp16,
  full/causal, all rel-L2 < 2e-2)

### Performance routing layer (`choose_backend`, everything routes to Turbo by default)

Once a variant is recognised (mask classification + score_mod mapping), dispatch passes through a
thin performance routing layer:
`choose_backend(mask_cfg, *, shape, dtype, has_alibi, has_softcap=False, has_dropout=False,
has_sink=False, has_bias=False) -> {"turbo","custom"}`:

- **Returns `"turbo"` by default**, i.e. every supported variant still dispatches straight to
  `flash_attn_func`, byte-for-byte identical to previous behaviour.
- Provides a registry API: `register_backend_override(matcher, backend)` /
  `clear_backend_overrides()`. `matcher(ctx)->bool` can read the routing context
  (`kind/causal/window_size/shape/dtype/has_alibi/has_softcap/has_dropout/has_sink/has_bias/mask_cfg`);
  a match forces that backend (registration order, first match wins). This lets a tuner steer
  specific shapes/kinds to the `_dispatch_custom` hook without touching the classifier.
- The `custom` branch shares `_dispatch_custom(...)` (still a stub that raises
  `NotImplementedError`). It is both the entry point for "arbitrary score_mod" and for "a supported
  variant explicitly routed to custom".

### Key assumptions / known constraints

- **ALiBi sign convention (build-dependent)**: this compat layer assumes Turbo's `alibi_slopes`
  (positive slopes) is equivalent to flex's `+slope*(kv-q)`. That sign was measured as
  `alibi_sign=+1` on `rocm/primus:v26.5` (primus_turbo 0.3.2.dev48, commit 6ccf00ff) -- see
  `bench/bench_results_ext2.md`: `plus_err=1.6e-3` matches, `minus_err=1.32` does not.
  **The sign must be re-validated when changing builds**, otherwise results may be silently wrong.
  A self-check is now shipped for exactly that: `check_alibi_sign_convention()` runs one small
  attention through the real kernel and scores it against both `+slope*(kv-q)` and
  `-slope*(kv-q)` fp32 references, returning the measured `sign`, both relative-L2 errors and
  `matches_assumption`; it reports `sign=None` (rather than guessing) when neither hypothesis
  matches decisively. `assert_alibi_sign_convention()` is the same check as a hard build gate --
  it raises `RuntimeError` unless the build matches, so a flipped-sign container fails loudly
  instead of quietly producing wrong ALiBi outputs. Run it once per new build/container.
- **Mask probe limit (including large-window location)**: the classifier probes
  `block_mask.mask_mod` on a `min(S,512)` grid. When the probed corner looks full-causal but the far
  corner is invisible (`mask_mod(S-1,0)=False`), it binary-searches the window boundary on the
  **last query row** (`W = the largest d such that mask_mod(S-1, S-1-d) is visible`), then verifies
  several sampled rows (the boundary row plus ~16 evenly spaced rows) bit-exactly as standard
  left-window causal `(q>=kv)&(q-kv<=W)`. **Only when that verification passes completely** is it
  classified as `sliding_window_causal, (W,0)`; otherwise it still raises `NotImplementedError`
  (never a misclassification). Full-causal (far corner still visible) remains causal.
- **Document packing beyond the probe grid**: packed sequences used to be capped at the same
  512 probe grid, because the boundaries simply were not visible in the probed corner. They no
  longer are. When the probe is truncated, `_locate_document_segments` reads the boundaries off
  `mask_mod` itself over the full sequence -- the diagonal and the sub-diagonal, two vectorised
  calls, `O(S)` elements -- and then **verifies the reconstruction exactly, not by sampling**:
  `same_doc(q,kv) & (q>=kv)` is compared against the real mask row-block by row-block (256 rows
  per vectorised call, so peak memory is a few MB rather than `S^2`). Any deviation -- a window,
  a hole, a non-causal block -- returns `None` and the caller raises exactly as before. Because
  the full comparison costs `O(S^2)`, it is bounded by `_DOC_EXACT_VERIFY_LIMIT` (16384): past
  that length we decline to classify rather than downgrade to sampled verification. Both
  truncation shapes are handled: documents shorter than the probe (the corner looks
  block-diagonal) and a first document longer than it (the corner looks exactly causal, and only
  the invisible far position distinguishes it from plain causal / a large window).
- **Classification/detection cache (performance, behaviour unchanged)**: `block_mask` classification
  and `score_mod` ALiBi/soft-cap detection results are cached by **object identity**
  (`weakref.WeakKeyDictionary`) -- when the same `block_mask`/`score_mod` is reused across layers or
  steps, re-probing (up to a 512x512 grid plus the detectors) is skipped, removing a fixed
  ~1.6-4.1ms probe cost per call (the GPU end-to-end wrapper forward overhead drops from ~1.5-2.3ms
  to ~0.03-0.27ms, the remainder being the retained bhsd<->bshd transposes). **Pure speedup**:
  different objects are re-classified, results are bit-identical to the uncached path, objects that
  cannot be weakly referenced skip the cache automatically, and `clear_classification_cache()`
  resets it.
- **Unrecognised means raise**: any `score_mod`/`mask_mod` that cannot be mapped onto the fixed
  kernels above raises `NotImplementedError` (the custom fast path `_dispatch_custom` is currently a
  stub). It never silently degrades, which keeps correctness first.

## Unsupported / to do (by priority)

### P0/P1 (path A: pattern-recognition mapping, small change, high value)

| Feature | Reason | Path | Prerequisite | Difficulty |
|---|---|---|---|---|
| ALiBi (equivalent formulations beyond the conservative detector) | auto-detection currently accepts only the strictly linear form, so complex equivalent formulations are treated as unrecognised | Provided: an **explicit `alibi_slopes` parameter** that bypasses the detector; strengthening the auto-detector is tracked separately | add equivalent-form and sign regression unit tests | mitigated |
| softcap (`logits_soft_cap`) | **blocked at the kernel layer**: this build's aiter dense fwd/bwd kernels have no softcap parameter (see "softcap status" below) | A (needs upstream aiter support) | upstream aiter's dense `mha_fwd`/`fmha_v3_fwd`/`mha_bwd` must expose and implement softcap (fwd+bwd) | blocked |
| attention sink | **supported**: the entry point gained an explicit `sink` parameter (1D/fp32/len==Hq, head_dim constraints), forwarded to `flash_attn_func(sink=...)`; measured on GPU as byte-for-byte identical to a direct backend call | A | - | done |
| dropout | **supported**: the entry point gained an explicit `dropout_p` (`0<=p<1`) forwarded to the backend; measured on GPU, `p=0` is a zero regression and `p=0.1` passes forward+backward | A | - | done |
| additive bias / relative position bias | **supported**: via AITER dense, requires **shape `[Sq,Skv]` and q's dtype (bf16/fp16)** (fp32 -> NaN, 4D per-head rejected by the kernel); the entry point adapts shape/precision automatically before forwarding (see "bias status" below) | A | - | done |

### softcap status (P0, investigated: detected but blocked at the kernel layer)

softcap (logits soft cap, Gemma2/Grok): `score = cap * tanh(score / cap)` (cap>0; 0/None = disabled).

**The Python side is in place**:

- Auto-detection: `_detect_softcap(score_mod)` strictly recognises pure softcap (depends only on
  score, independent of b/h/q/kv, `f(0)=0`, tail saturates to cap, fits `cap*tanh(s/cap)` across the
  whole grid, and is odd-symmetric).
- Explicit parameter: the entry point gained `softcap: Optional[float] = None`; `None`/`0` disables
  it, a positive value requests softcap.
- **Single enable point**: an explicit `softcap>0` and a detected soft-cap both funnel into
  `effective_softcap`, which also sets `has_softcap=True` for `choose_backend`, and then hit the
  **same single** `if effective_softcap > 0.0:` inside `flex_attention` that raises
  `NotImplementedError` (**the cap is never silently dropped**). That site is marked
  `# TODO(softcap)`: once upstream aiter dense fwd+bwd supports it, deleting the guard and threading
  `effective_softcap` into `flash_attn_func(softcap=...)` is a **one-line switch to enable**.

**Why explicit detection is required (fixes a silent-wrongness risk)**: the ALiBi detector
`_detect_alibi_slopes` only probes at `score=0`, and `cap*tanh(0)=0`, so it would misclassify a pure
softcap as "zero-slope ALiBi (no-op)" and fall through to `alibi_slopes=None` -> dispatch straight to
Turbo **ignoring the cap**, producing silently wrong results. For typical caps (20-50) this
misclassification does occur (within the `|f(1)-1| < 5e-3` tolerance). `_detect_softcap` intercepts
first, turning a silent error into an explicit one.
Measured: in `bench/softcap_flex_validation.py`, an active cap=1.0 gives
`rel_l2(no-cap vs cap)=0.576` (dropping the cap would be badly wrong); cap=30 on ~N(0,1) logits gives
gap=0.0048 (a large cap is nearly a no-op, but must still be honoured).

**Blocking point (kernel layer, measured aiter signatures, rocm/primus:v26.5)**:

- The dense forward `aiter.ops.mha._flash_attn_forward` (which the compat layer reaches via
  `attention_aiter_forward_impl` -> `AttnFwdAiterBackend`) has **no** `logits_soft_cap`/`softcap`
  parameter:
  `(q,k,v,dropout_p,softmax_scale,causal,window_size_left,window_size_right,sink_size,bias,
  alibi_slopes,q_descale,k_descale,v_descale,return_lse,return_softmax,how_v3_bf16_cvt=1,
  cu_seqlens_q=None,cu_seqlens_kv=None,sink_ptr=None,out=None)`. The underlying
  `mha_fwd`/`fmha_v3_fwd` runtime type hints likewise have no softcap parameter.
- The dense backward `aiter.ops.mha._flash_attn_backward` is a `torch_compile_guard` wrapper
  (`(*args, **kwargs)` forwarded to `torch.ops.aiter.<name>`) and the current call does not pass
  softcap; the underlying `mha_bwd`/`fmha_v3_bwd` are likewise wrappers without a softcap parameter.
  **softcap changes the gradient, so a backward without the parameter means it cannot be trained
  correctly.**
- The varlen forward `_flash_attn_varlen_forward` **does have** `logits_soft_cap: float = 0.0` (with
  a `ret = ret and logits_soft_cap == 0.0` gate), but the varlen backward
  `_flash_attn_varlen_backward` has **no** softcap parameter -> even the varlen route cannot train;
  and the compat layer's entry point is dense-only for now.
- FlyDSL: this build's installed package **does not include** `attention_flydsl_impl`
  (`ModuleNotFoundError`), so it is unavailable.
- The aiter Triton dense forward `aiter.ops.triton.attention.mha._flash_attn_forward`
  **hardcodes `softcap=0.0`** internally and its Python wrapper does not expose the parameter; the
  Triton backward `flash_attn_onekernel_backward` has no softcap parameter.

**Conclusion**: without modifying the C extension or recompiling aiter (a constraint of this task),
softcap cannot be wired through for both dense forward and backward. softcap is therefore marked
"**detected but blocked at the kernel layer**" in the compat layer.

**Possible ways forward (require upstream changes, out of scope here)**:

1. Upstream aiter adds and implements `logits_soft_cap` in the dense
   `mha_fwd`/`fmha_v3_fwd`/`mha_bwd` (and the corresponding CK/assembly kernels), consistently for
   fwd+bwd. This compat layer would then thread `softcap` from `flash_attn_func` all the way to
   `attention_aiter_forward/backward_impl` and lift the guard (the Python-side change is already
   fully scoped out).
2. Expose the internal `softcap` of the aiter Triton dense path (currently hardcoded to 0.0), but
   this also needs softcap support in the Triton backward, and the Triton path is currently only
   enabled on the sink branch.
3. Approximate with a Triton epilogue (hand-written `cap*tanh` forward/backward) -- high effort and
   risk, not a minimal change.

### bias status (P0, investigated and fixed: a shape/precision issue, not a kernel dead end)

Additive bias (applied to the pre-softmax logits: `score = q.k^T/sqrt(d) + bias`). The previously
reported "NaN" turned out **not to be a kernel dead end but incorrect shape/precision usage**. On
`rocm/primus:v26.5` (gfx950/MI355X), a small shape (B=2,H=2,S=64,D=64) was swept through
`flash_attn_func(q,k,v, bias=...)` and compared against an fp32 manual additive-bias reference
(`softmax(qk/sqrt(d) + bias) @ v`), with these results:

| bias dtype / shape | Behaviour |
|---|---|
| bf16 `[B,H,Sq,Skv]` / `[1,H,Sq,Skv]` / `[1,1,Sq,Skv]` / `[B,1,Sq,Skv]` (4D) | `RuntimeError: bias shape should be [sq, sk]` (the kernel only accepts 2D) |
| fp32 `[B,H,Sq,Skv]` (4D) | same `RuntimeError` |
| **fp32 `[Sq,Skv]` (2D)** | **NaN output** (this was the real cause of the earlier "NaN": 2D but fp32) |
| **bf16 `[Sq,Skv]` (2D)** | **correct**: forward rel-L2 = 2.1e-3 (< 2e-2); backward dQ/dK/dV/dBias all finite, rel-L2 ~ 2.5e-3 |

**Root cause**: the bias parameter of aiter dense (`aiter.ops.mha._flash_attn_forward` -> underlying
`mha_fwd`/`fmha_v3_fwd`) **only accepts a single `[Sq, Skv]`** matrix (shared across batch/head) and
its dtype must match q (fp16/bf16). The underlying `mha_fwd(..., bias)` and
`mha_bwd(..., dbias, bias)` both have bias/dbias parameters (confirmed from aiter's runtime type
hints), so **both forward and backward are supported**.

**Fix (minimal change, adaptation only on the flex entry side, leaving
`flash_attn_interface.py`/`attention_aiter_impl.py` untouched)**: the entry point gained an explicit
`bias` parameter, and `_validate_and_adapt_bias` adapts the user-provided bias to the `[Sq,Skv]` the
kernel expects (accepting `[Sq,Skv]` or leading-singleton `[1,Sq,Skv]`/`[1,1,Sq,Skv]`; a true
per-head/per-sample bias raises a clear `ValueError`), casts it to q's dtype, aligns the device and
forwards it to `flash_attn_func(bias=...)`.

**Validation (through the flex entry point, end-to-end fwd+bwd, see `bench/_investigate_bias.py` and
`bench/_run_taskB.py`)**: bf16 and fp16 x full and causal, 4 combinations total; forward
rel-L2 in [2.6e-4, 2.3e-3] and dQ rel-L2 in [3.2e-4, 2.7e-3] (all < 2e-2), with
gap(vs no-bias) ~ 0.41-0.49 (confirming the bias really takes effect and is not a silent no-op).

**Known limitation**: only a **single `[Sq,Skv]` bias shared across batch/head** is supported (the
common form for relative position bias / a shared additive mask); per-head/per-sample bias is an
arbitrary score_mod and needs the codegen path (P3).

### varlen / document packing status (P2, supported)

**(Primary deliverable) explicit varlen entry `flex_attention_varlen`**:

```python
from primus_turbo.pytorch.ops.attention import flex_attention_varlen
out = flex_attention_varlen(
    query, key, value,               # THD packed [total_tokens, H, D] (same as Turbo varlen, no transpose needed)
    cu_seqlens_q, cu_seqlens_k,       # int32 [num_seqs+1] prefix sums, on q's device
    max_seqlen_q, max_seqlen_k,       # longest segment length for each (positive int)
    *, causal=False, window_size=(-1, -1), scale=None,
    alibi_slopes=None, dropout_p=0.0, sink=None, softcap=None, return_lse=False,
)
```

- **Direct backend dispatch**: a thin wrapper mapping straight onto `flash_attn_varlen_func` (THD in,
  THD out, no layout conversion). In-document causal uses `causal=True` (standard block-diagonal plus
  in-segment causal, requiring `q_len == k_len` per segment); `window_size=(W,0)` gives a per-segment
  left window.
- **cu_seqlens validation (`_validate_cu_seqlens`)**: int32, 1D, first element 0, monotonically
  non-decreasing, last element == `total` (`query.shape[0]` for q, `key.shape[0]` for k),
  `len(cu_q)==len(cu_k)`, same device as q, `max_seqlen >= longest segment`; `causal=True`
  additionally requires `cu_seqlens_q == cu_seqlens_k` (bottom-right alignment, mismatched segment
  lengths would silently misalign). Anything invalid -> a clear `ValueError`.
- **Reuses the dense validators**: `dropout_p` (`0<=p<1`), `sink` (1D/fp32/len==Hq, head_dim
  constraints) and `alibi_slopes` (1D/fp32/len==Hq) share the same validators as the dense entry
  point; GQA/MQA is natively supported (`Hq % Hkv == 0`, no extra flag needed).
- **Unsupported items raise (same style as dense)**: there is no `score_mod` parameter (an arbitrary
  score_mod never reaches here); `softcap>0` uniformly raises `NotImplementedError` (the varlen
  backward kernel also lacks a softcap parameter, see softcap status above), never silently dropping
  the cap.
- **Cross-build compatibility**: `sink` is only forwarded when the caller actually provides it (this
  build's older varlen kernel has no `sink` parameter; the default `sink=None` is simply not passed,
  which works on both old and new backends); `bias` is not exposed by this entry point and is left
  to the backend default.
- **Measured on GPU (rocm/primus:v26.5, gfx950, total=512, docs=[128,128,256], H=8, D=128)**:
  causal/full/SWA(W=64)/GQA(Hq8/Hkv2)/ALiBi forward rel-L2 in [2.4e-4, 2.3e-3], causal backward
  dQ/dK/dV ~ 2.5e-3 to 2.9e-3 (all < 2e-2); `return_lse` returns `(out, lse)` (lse `[H, total]`);
  `softcap>0`, invalid cu_seqlens and mismatched causal segment lengths all raise correctly.

**(Secondary deliverable) automatic document-mask detection in the dense entry**:

- The `flex_attention` classifier probes `block_mask.mask_mod` on a `min(S,512)` grid; when the
  pattern falls within causal but is neither pure causal nor single-window causal,
  `_detect_document_causal_segments` tries to recognise it as `same_doc(q,kv) & (q>=kv)`: read the
  document boundaries off the sub-diagonal -> recover each segment length -> **rebuild the full
  block-diagonal causal mask and compare bit-exactly**; only on an exact match is it classified as
  `document_causal` (`_classify_block_mask` returns `{"kind":"document_causal","doc_seglens":[...]}`).
- On a hit, the dense entry packs bhsd `[B,H,S,D]` into THD, builds `cu_seqlens` from `doc_seglens`
  (replicated across batch), dispatches through `flash_attn_varlen_func` with `causal=True`, and
  unpacks back to bhsd (`_dispatch_document_varlen`). **It never uses a dense causal that would
  attend across documents.** ALiBi (explicit or detected)/dropout/sink are forwarded along; `bias`
  and `return_lse` raise explicitly on this path (a packed `[Sq,Skv]` bias cannot be aligned, and
  packed LSE does not align with `[B,H,S]` -- use the explicit `flex_attention_varlen` if you need
  LSE/bias).
- **Correctness first, never a misclassification**: routing only happens when `S <= 512` (probing not
  truncated) **and the reconstruction is bit-identical**; `S>512`, windowed, non-square, or any hole
  or deviation returns `None` -> falls back to the existing `NotImplementedError` (status quo
  preserved, never silently wrong).
- **Measured on GPU**: the dense entry with a document `block_mask`, across three cases -- B=1, B=2
  (cu replicated) and with explicit ALiBi -- gives rel-L2 ~ 1.6e-3 to 2.0e-3 (all < 2e-2), matching the
  masked fp32 reference.

**Known limitations**: document detection requires `S <= _MASK_PROBE_LIMIT(512)` (use the explicit
`flex_attention_varlen` for longer sequences); the dense document path does not support
`bias`/`return_lse` (use the explicit varlen entry instead); `softcap>0` is blocked everywhere (see
softcap status).

### P2 (path A: moderate effort)

| Feature | Reason | Path | Prerequisite | Difficulty |
|---|---|---|---|---|
| varlen / document packing | **supported**: new explicit entry `flex_attention_varlen` (THD packing + `cu_seqlens` straight to `flash_attn_varlen_func`); measured on GPU, causal/full/SWA/GQA/ALiBi forward rel-L2 ~ 1.6e-3 to 2.3e-3, backward dQ/dK/dV ~ 2.5e-3 to 2.9e-3 (all < 2e-2) | A | - | done |
| document masking | **supported**: the dense `flex_attention` exactly recognises `same_doc(q,kv) & (q>=kv)` block-diagonal causal -> recovers segment lengths -> builds `cu_seqlens` -> routes through varlen; measured on GPU, B=1/B=2/with ALiBi all rel-L2 ~ 1.6e-3 to 2.0e-3. **No longer capped at the 512 probe grid**: beyond it the boundaries come from `mask_mod` directly and the pattern is verified exactly (chunked) up to `_DOC_EXACT_VERIFY_LIMIT`=16384 | A | depends on the varlen wrapper | done |
| prefixLM (partial) | the current classifier only supports full/causal/single-window causal/document block-diagonal causal | A | add decidable templates and backend mappings | medium |
| broader head_dim / dtype coverage | limited by backend constraints and the dtype guard (currently fp16/bf16 only) | A | backend capability validation (FlyDSL only supports D in {64,128}) | medium |

### P3 (path B: general codegen, hard)

| Feature | Reason | Path | Prerequisite | Difficulty |
|---|---|---|---|---|
| arbitrary score_mod | requires compiling a runtime function into a high-performance kernel | B (codegen + automatic backward) | IR design, operator templates, autograd plan | high |
| arbitrary mask_mod / general block sparsity | requires a general sparse layout and scheduling | B | mask IR + sparse planner + kernel family | high |
| arbitrary score_mod + mask_mod combinations | combinatorial explosion, needs a unified codegen pipeline | B | build on the two items above, then optimise combinations | very high |

`flex_attention` already reserves the `_dispatch_custom(...)` hook as the entry point for path B; it
currently only raises `NotImplementedError` (no real kernel is implemented).

### P4 (not planned for now)

| Feature | Reason | Path | Prerequisite | Difficulty |
|---|---|---|---|---|
| FP8 + arbitrary mods | training stability and quantisation calibration are complex | B (long term) | re-evaluate once P3 matures | very high |
| paged attention | needs KV paging data structures and a scheduling system | separate track (not A/B) | cache management and serving-side protocol | very high |

## Tests

- Pure-logic unit tests (CPU only): `tests/pytorch/ops/test_flex_attention_dispatch_logic.py`
  (**210 cases** total: 95 pre-existing + 44 varlen + 15 document + 32 large-window/cache + 10
  document-beyond-probe + 5 ALiBi-sign self-check + 9 bshd-native entry; zero regressions). Runnable
  on CPU with the backend mocked -- no GPU required. The varlen part covers `_validate_cu_seqlens` (int32/1D/leading 0/monotonic/last
  element/matching segment counts/device/`max_seqlen>=longest segment`/matching causal segment
  lengths, positive and negative cases), `_validate_qkv_varlen` (3D/dtype/head divisibility/head_dim),
  `_validate_window_size`, `_validate_max_seqlen`, and `flex_attention_varlen` end to end (mocked
  backend): THD passthrough with no transpose, causal/window/scale/dropout/alibi/sink forwarding,
  `softcap>0` raising, invalid cu raising before dispatch, `return_lse` returning a tuple, GQA, and
  non-causal cross attention. The document part covers `_detect_document_causal_segments`
  (multi-document recognition; rejection of single causal/SWA/truncated/non-square/holed patterns),
  `_classify_block_mask` returning `document_causal`, and the dense entry routing end-to-end to
  varlen (correct cu_seqlens, B=2 replication, ALiBi forwarding, and raising for
  bias/return_lse/S>512). The large-window/cache part covers `_locate_left_window`
  (W in {256,512,1024,2048,4096} recognised at S=8192; rejection of
  full-causal/non-translation-invariant/holed/non-square), `_classify_block_mask` on large windows
  (including S=8192, `window>=S` classified as causal, and non-standard long masks still raising),
  and the classification/detection cache (same object hits and returns the same object, different
  objects/shapes recompute, non-weakly-referenceable objects skip the cache,
  `_cached_detect_alibi/softcap` hit counts, `clear_classification_cache` resets, and behaviour
  identical to the uncached path).
- The document-beyond-probe part covers `_locate_document_segments` directly (uneven segments; a
  single document, a large sliding window, a holed block-diagonal mask and a non-square shape all
  correctly returning `None`; and the refusal past `_DOC_EXACT_VERIFY_LIMIT`) plus
  `_classify_block_mask` on both truncation shapes (short documents at S=1024, and a first document
  longer than the probe at S=2048) and the dense entry routing S=1024 packing end-to-end to varlen
  with the right `cu_seqlens`.
- The ALiBi-sign part drives `check_alibi_sign_convention` / `assert_alibi_sign_convention` against
  a mocked backend that applies a *known* sign: the `+1` build is detected and accepted, the `-1`
  build is detected and `assert_...` raises, and a backend that ignores ALiBi entirely is reported
  as `sign=None` rather than being assigned the "less wrong" sign.
- The bshd part checks that `flex_attention_bshd` dispatches with the caller's own buffers
  (`data_ptr()` equality on q/k/v and on the returned output -- i.e. the copies really are gone),
  returns `[B,S,H,D]`, agrees numerically with the bhsd entry, forwards the Turbo extension
  arguments, raises the same validation errors, handles `return_lse`, routes a document mask to
  varlen in bshd, and leaves the bhsd entry's own contract untouched.
- The original 95 cases cover: classification of full/causal/SWA/random/banded/head-dependent/
  batch-dependent masks; positive and negative cases for the ALiBi detector; positive and negative
  cases for `_detect_softcap` (recognising cap=20/30/50; rejecting identity/linear/constant/ALiBi/
  hard clamp/alibi+softcap combinations; plus the regression guard that "softcap is not
  misclassified as zero-slope ALiBi"); `choose_backend`/the registry (turbo by default, override
  hitting custom, clear resetting, first match winning, ctx fields including
  `has_dropout/has_sink/has_bias`, parameter validation, matcher exception wrapping, overrides able
  to match on dropout/sink); and the **explicit extension parameters**: the `alibi_slopes` validator
  (1D/length/fp32/non-tensor, positive and negative), the `softcap` normaliser (None/0 disables,
  positive preserved, negative/NaN raises), `_validate_dropout_p` (0/valid values; `>=1`/negative/
  NaN/non-numeric raise), `_validate_explicit_sink` (1D/len==Hq/fp32/equal and power-of-two head_dim,
  positive and negative), `_validate_and_adapt_bias` (`[Sq,Skv]`/leading-singleton broadcast/
  rectangular shapes pass; per-head 4D/per-sample 3D/mismatched last two dims/non-tensor/
  non-floating raise; dtype adapted to q), the `_is_identity_score_mod` probe, and end-to-end
  dispatch (with `flash_attn_func` mocked, running on CPU): explicit slopes forwarded and skipping
  auto-detection, explicit and auto-detected slopes equivalent, combination with causal, non-trivial
  score_mod conflict -> `ValueError`, identity score_mod allowed, invalid length -> `ValueError`,
  explicit `softcap>0` -> `NotImplementedError` (without reaching the backend), `softcap=0/None`
  having no effect, `dropout_p` default/positive forwarding and out-of-range raising, `sink`
  forwarding and invalid shapes raising, `bias` forwarding/adaptation and per-head bias raising, and
  "no extension parameters is identical to the original path (including dropout_p=0/sink=None/
  bias=None)".
- GPU varlen / document validation (in-container):
  `bench/flex_attention_varlen_validation.py` (via `bench/_run_varlen_all.py`, which first overlays
  the workspace `flex_attention.py` onto the installed package, then runs the pure-logic unit tests
  plus this script from /tmp). It covers `flex_attention_varlen` causal/full/SWA/GQA/ALiBi forward +
  causal backward dQ/dK/dV, `return_lse`, and raising on softcap/invalid cu/mismatched segment
  lengths; plus the dense entry's document routing (B=1/B=2/ALiBi) against a masked fp32 reference.
  All measured rel-L2 < 2e-2 (see "varlen / document packing status" above).
  Note: the container has an older installed build whose package `__init__.py` lacks
  `sparse_mla_interface` and whose varlen kernel lacks a `sink` parameter, so the runner
  **only overlays `flex_attention.py`** (not `__init__.py`), and the entry point uses a
  "forward only when provided" strategy for `sink`.
- GPU entry overhead + large-window validation (in-container):
  `bench/bench_flex_entry_overhead.py` (three-way comparison of entry/direct/torch flex) plus
  `bench/_bench_classify_cache.py` (a cold/warm classification-cache micro-benchmark). Measured: the
  **classification cache** takes the cold path (re-probing) from 1.6-4.1ms down to ~0.3-0.4us on a
  warm hit (~5000-12000x); the end-to-end wrapper forward overhead drops from a ~1.5-2.3ms baseline
  to ~0.03-0.27ms (the remainder being the retained transposes). **SWA(W=1024) at S=2048/4096/8192
  goes from the baseline's `NotImplementedError` to passing everywhere**, with entry vs direct
  out rel-L2=0 and dQ ~1e-6 to 1e-5 (far below 2e-2).
- GPU smoke test (in-container): `bench/smoke_flex_attention_turbo.py` covers numerical agreement for
  causal / full / SWA(W=128) / GQA / ALiBi (rel-L2 < 2e-2) and the raising paths; it also adds
  **explicit `alibi_slopes`** (causal, three-way agreement with the manual reference and the
  auto-detected path), **explicit `softcap=30` -> `NotImplementedError`**, **dropout** (`p=0` zero
  regression, `p=0.1` passing forward+backward), **sink** (forwarding byte-for-byte identical to a
  direct `flash_attn_func(sink=...)`, `sink=None` a zero regression), and the raising paths for
  out-of-range dropout and invalid sink shapes.
- GPU bias validation (in-container): `bench/_investigate_bias.py` (shape/precision forensics:
  4D -> RuntimeError, fp32 `[Sq,Skv]` -> NaN, bf16 `[Sq,Skv]` -> correct, also measuring
  dQ/dK/dV/dBias) and `bench/_run_taskB.py` (end-to-end fwd+dQ through the flex entry point,
  bf16/fp16 x full/causal, 4 combinations, all rel-L2 < 2e-2).
- GPU softcap validation (in-container): `bench/softcap_flex_validation.py` verifies that a softcap
  `score_mod` raises explicitly (rather than silently dropping the cap), that the cap is numerically
  material, and that `score_mod=None` causal has no regression. aiter signature forensics scripts:
  `bench/_investigate_softcap.py`, `bench/_investigate_softcap2.py`.
