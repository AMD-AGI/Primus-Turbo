# Flex-attention compat layer — backend capability status

`primus_turbo/pytorch/ops/attention/flex_attention.py` maps a torch-FlexAttention
`block_mask` / `score_mod` onto the fixed Turbo attention kernels. Some things it can
express, the kernels underneath cannot execute. Where that happens the layer **raises**
rather than silently dropping the feature, and the error message points here.

This document records *what* is blocked, *why*, and *what evidence* says so, so that a
reader hitting one of those errors can tell "this build can't" from "you called it wrong",
and so a future reader knows exactly which upstream change unblocks it.

- Build under test: `rocm/primus:v26.5`, torch `2.12.0+rocm7.15.0a20260720`
- Hardware: AMD Instinct MI355X (gfx950, capability `(9, 5)`)
- Everything below is measured on that hardware, not inferred from source reading.
  The full experiment logs live outside this repo in `gpu_runs/GPU_VALIDATION_RESULTS.md`.

## Status at a glance

| Feature | State | Blocker |
|---|---|---|
| causal / sliding-window / full masks | **works** | — |
| document packing (THD, `cu_seqlens`) | **works** | — |
| bshd-native entry | **works** | — |
| GQA / MQA | **works** | — |
| ALiBi via `score_mod` auto-detection | **works** | sign is build-dependent, gated — see below |
| explicit `alibi_slopes=` | **works** | bypasses the detector's conservative limits |
| `dropout_p` | **works** | — |
| attention `sink`, dense entries | **works** | fp16/bf16 accepted, upcast to fp32 internally |
| attention `sink`, varlen entry | **raises** | this build's `flash_attn_varlen_func` has no `sink` parameter |
| additive `bias` | **works, restricted** | aiter dense takes one `[Sq, Skv]` in q's dtype, shared across batch/heads |
| **softcap** (`cap*tanh(score/cap)`) | **raises** | no aiter fwd+bwd *pair* exposes it |
| arbitrary `score_mod` / `mask_mod` | **raises** | needs the unwritten codegen path ("path B") |
| `context_parallel_size > 1` | **raises** | not implemented |
| fp8 attention | **raises** | not implemented |
| `deterministic=True` | **raises** | aiter backward uses fp32 atomics; see "Nondeterminism" |

---

## Softcap — the aiter signature evidence

Cited from: the module docstring, `_detect_softcap`, `flex_attention`'s `softcap` doc,
and the single enablement gate in `flex_attention`.

The claim in the code is that softcap "is blocked at the kernel layer". The precise form
of that claim matters, because a looser version of it ("aiter has no softcap anywhere") is
**false** and would mislead the next person to look. Parameter lists pulled directly from
the installed aiter in the container:

| aiter entry | has softcap? | params |
|---|---|---|
| `mha_fwd` | no | 21 |
| `mha_bwd` | no | 22 |
| `fmha_v3_fwd` | no | 18 |
| `fmha_v3_bwd` | no | 20 |
| `mha_varlen_fwd` | **`logits_soft_cap`** | 29 |
| `mha_varlen_bwd` | no | 27 |
| `fmha_v3_varlen_fwd` | **`logits_soft_cap`** | 28 |
| `fmha_v3_varlen_bwd` | no | 27 |

So the two **varlen forwards do** take `logits_soft_cap`. Every backward, and both dense
forwards, do not. Pairing them up:

```
mha_fwd            fwd=no    mha_bwd             bwd=no   -> NOT USABLE
fmha_v3_fwd        fwd=no    fmha_v3_bwd         bwd=no   -> NOT USABLE
fmha_v3_varlen_fwd fwd=YES   fmha_v3_varlen_bwd  bwd=no   -> NOT USABLE
```

**No fwd+bwd pair can carry the cap.**

This makes the gate *more* necessary, not less. varlen is exactly the path document
packing takes. Someone who notices "varlen forward supports `logits_soft_cap`" and wires
it up gets a **correct forward and a silently wrong backward**: the softcap derivative
carries a `1 - (y/cap)**2` factor, and a backward that never receives `cap` differentiates
as if there were none. Training still converges — to the wrong place, with no error. That
is the failure mode the layer exists to prevent, and the reason the entry raises instead
of degrading.

**To unblock**: upstream aiter must add the parameter to a matching fwd *and* bwd and
recompile. Then delete the single gate in `flex_attention` and thread
`effective_softcap` into `flash_attn_func(softcap=...)`; the `TODO(softcap)` comment marks
the exact spot. The detector, the explicit `softcap=` argument, and their tests are
already in place and need no change.

---

## ALiBi sign convention (build-dependent)

Cited from: `assert_alibi_sign_convention`.

Whether `flash_attn_func(alibi_slopes=s)` adds `+s*(kv - q)` or `-s*(kv - q)` is not
visible from Python. Guessing wrong produces wrong logits *and* wrong gradients with no
error anywhere. The compat layer assumes **`+slope * (kv_idx - q_idx)`**.

`check_alibi_sign_convention` measures it: run the same q/k/v through `alibi_slopes=+s`
and `alibi_slopes=-s`, compare each against an fp32 dense reference that hardcodes
`+s*(kv-q)`, and the smaller error identifies the kernel's real convention.

Measured on this build:

```
   dtype      S    H    D |  sign    plus_err   minus_err     ratio  match
 float16    256    4   64 |   1.0   2.339e-04   1.302e+00    5568.4   True
 float16   1024    8  128 |   1.0   2.374e-04   1.366e+00    5754.0   True
bfloat16    256    4   64 |   1.0   1.864e-03   1.364e+00     731.6   True
bfloat16   2048    8  128 |   1.0   1.903e-03   1.409e+00     740.6   True
```

A 661–5754x discrimination ratio: there is no ambiguous middle ground. The cost of
guessing wrong, measured directly: `|out(+slope) - out(-slope)|` has mean `0.2832` while
`|out|` itself has mean `0.2275` — the error is the size of the signal.

**This is a property of the build, not of the layer.** Call
`assert_alibi_sign_convention()` as a one-line gate in CI or a container smoke test so a
flipped-sign build fails loudly on arrival instead of quietly training something wrong.

---

## Attention bias — the `[Sq, Skv]` restriction

Cited from: `_validate_and_adapt_bias`, and the `bias` entry in `flex_attention`'s docstring.

The aiter dense kernel takes **one 2D bias of shape `[Sq, Skv]` in q's dtype**, added to
the pre-softmax logits and shared across batch and heads. This was pinned empirically, not
read off a signature:

- a 4D / per-head bias raises `RuntimeError: bias shape should be [sq, sk]`
- an fp32 bias yields `NaN`
- only `[Sq, Skv]` in q's dtype is numerically correct in both fwd and bwd (rel-L2 ~2e-3)

The layer therefore accepts `[Sq, Skv]` or a leading-singleton broadcast of it
(`[1, Sq, Skv]`, `[1, 1, Sq, Skv]`), casts to q's dtype, moves to q's device, and rejects a
genuine per-batch / per-head bias with a `ValueError` naming the constraint. A per-head
bias needs the codegen path below.

---

## Attention sink

Dense entries (`flex_attention`, `flex_attention_bshd`) accept a 1D sink of length `Hq` in
any floating dtype and upcast to fp32 internally, because aiter's `mha_fwd(sink_ptr=...)`
requires fp32. The sink kernel path additionally requires `head_dim_qk == head_dim_v` with
a power-of-two head dim (backend constraint).

The **varlen** entry is different, and this is a per-build fact rather than a permanent
limitation: on some builds `flash_attn_varlen_func` has no `sink` parameter at all. The
layer probes the backend signature (`_backend_accepts`) and raises `NotImplementedError`
naming the missing parameter rather than letting a bare
`TypeError: flash_attn_varlen_func() got an unexpected keyword argument 'sink'` escape from
the binding — which would point at the caller instead of at the build.

It raises rather than dropping the sink because sink logits enter the **softmax
denominator**: dropping them changes the output of *every query in the batch*, with no
error. On a build whose varlen backend does take `sink`, the same code threads it through
unchanged. A backend declaring `**kwargs`, or one that cannot be introspected at all (a C
extension), is given the benefit of the doubt and treated as accepting.

---

## Arbitrary `score_mod` / `mask_mod` — the codegen path ("path B")

Cited from: `_dispatch_custom`, the mask classifier, and `_validate_and_adapt_bias`.

The layer recognises a fixed vocabulary of patterns (causal, sliding window, full,
document packing, ALiBi, softcap) and maps them onto Turbo kernels. Anything outside that
vocabulary — a mask with visible positions above the causal diagonal, a per-head bias, a
programmable score transform — has no fixed kernel to land on. `_dispatch_custom` is a
deliberate hard stop so callers never receive a silently wrong result from an unrecognised
modification; it is also where `choose_backend` sends a variant explicitly routed away
from Turbo.

A recognised or explicit softcap is gated *before* this hook, so `softcap > 0` never
reaches it.

**To unblock**: implement the codegen path that compiles an arbitrary `score_mod` /
`mask_mod` into a kernel. Until then the honest answer to "can this run my custom mask" is
no, and the layer says so.

---

## Nondeterminism, and what a numerical diff means here

aiter's backward accumulates with **fp32 atomics**, so two runs of *identical* code can
differ. Measured self-vs-self: the spread is the same magnitude as any cross-path
comparison, which is why `deterministic=True` is rejected rather than approximated.

The practical consequence for anyone validating this layer: **a nonzero diff is not
evidence of a bug until you have run the same comparison against itself.** A CUDA-graph
replay differing from eager by `3.906e-03` looks alarming until the eager-vs-eager control
on the same data also reports `3.906e-03`.

## What has been verified end to end

Under the Primus integration switch `use_turbo_flex_attention`, a 20000-step training run
with the switch ON and OFF produced **bit-identical loss at every single step** (max diff
`0.0`), with both arms learning (5.6987 -> 4.2533). Steady-state overhead measured with
interleaved A/B blocks: **1.064x**.

Sequential timing — running all of one arm and then all of the other — is not a usable
measurement here: it reported `0.770x` and `1.715x` on two runs of the same code, and
bit-identical arms cannot genuinely differ by 70% in either direction. The first arm to
run also pays aiter's one-off JIT kernel builds (~90 s), which is why whole-run averages
must be discarded entirely.
