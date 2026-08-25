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

## Relationship to PyTorch — why there is no BSD-3-Clause banner on this file

The compat layer is a ~2.3k-line module named `flex_attention.py` that exposes a function
named `flex_attention` and advertises itself as a drop-in replacement for
`torch.nn.attention.flex_attention`. That is enough resemblance to make "is this a copy of
torch's implementation?" a fair review question — and the repo has a real answer path for
it: `tools/check_license.py` emits a dual-copyright `Adapted from ...` + SPDX banner for
adapted third-party code (the `primus_turbo/flydsl` tree is the live example). If this file
were derived from PyTorch it would need that treatment with a BSD-3-Clause notice.

It is not derived from PyTorch, and the claim is checked mechanically by
`tools/check_flex_provenance.py` rather than asserted. Three independent measures, run
against the *installed* torch (measured on torch 2.13.0):

| Measure | Clean tree | Threshold |
|---|---|---|
| Shared top-level names, excluding the two interface names | 0 | 0 |
| Max per-function similarity (every local function vs every torch function, comments stripped) | **0.421** | 0.60 |
| 12-token fingerprint overlap over raw text | **0.2233%** (52/23289) | 2% |

The 0.421 top score is a 6-line scalar helper coinciding with an unrelated 6-line torch
helper. The shared fingerprints are entirely unavoidable idiom: `query: torch.Tensor, key:
torch.Tensor`, `query.transpose(1, 2)`, the `query.device != key.device` guard. The two
shared names are `flex_attention` and `create_block_mask` — the interface we deliberately
match, and `create_block_mask` is a passthrough that *calls* torch's.

The measures are complementary, and the thresholds are calibrated against an injection
rather than guessed. Pasting three whole torch functions (`or_masks`, `and_masks`,
`_convert_mask_to_block_mask`) into a copy of the file is caught by all three — but it only
moves the fingerprint overlap to 2.58%, so the original 5% threshold would have missed it.
That is why the limit is 2%.

### What is reused from torch, and what is not

Reused **by import, never by copying**:

* `create_block_mask` — imported at module scope, re-exported through a thin passthrough so
  the returned object is a genuine torch `BlockMask` and stays compatible with code that
  also feeds the same mask to torch's own kernel.
* `BlockMask` — never reimplemented; we consume torch's instances and read `.mask_mod`.

Deliberately **not** reused:

* `create_mask` cannot replace `_probe_mask_grid`. It is built on `torch.vmap`, which
  raises `RuntimeError: ... data-dependent control flow` for scalar-style `mask_mod`
  callables written with Python `and` — verified, not assumed — and the probe helper falls
  back to an element-wise loop for exactly those. `create_mask` also materialises the whole
  `[B, H, Q_LEN, KV_LEN]` mask, whereas `_probe_mask_row` and `_locate_left_window` sample
  single rows so that classifying a long-sequence mask stays O(S log S) instead of O(S²).
  Its default device also resolves to the current accelerator, which would make
  classification require a GPU; the probe path is CPU-only by design, which is what lets
  the dispatch-logic unit tests run without one.
* `and_masks` / `or_masks` / `noop_mask` — not reimplemented here; callers who want them
  should import them from torch directly.

Net: of torch's ten public flex-attention symbols, this module reimplements none. The one
it needs, it imports.

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
`0.0`), with both arms learning (5.6987 -> 4.2533).

Sequential timing — running all of one arm and then all of the other — is not a usable
measurement here: it reported `0.770x` and `1.715x` on two runs of the same code, and
bit-identical arms cannot genuinely differ by 70% in either direction. The first arm to
run also pays aiter's one-off JIT kernel builds (~90 s), which is why whole-run averages
must be discarded entirely.

### Overhead, and why the earlier 1.064x figure was withdrawn

An earlier revision of this document quoted **1.064x** steady-state overhead from
interleaved A/B blocks. **That figure is withdrawn: it does not reproduce, and it was an
artifact of a confound interleaving does not remove.**

Interleaving fixes drift *between* arms. It does nothing about the penalty paid by
whichever arm runs *first* in a fresh container — allocator state, caches, thermal ramp.
The only way to see that penalty is to run the same arm twice. A four-arm
`on off on off` pretrain (llama3.2_1B 16L, seq 8192, 8x MI355X, 500 iterations per arm)
does exactly that, and the A/A pairs are what settle it:

| kind | pair | step-time ratio | 95% CI (paired block bootstrap) |
|---|---|---|---|
| A/B | 1_on / 2_off | 1.0349 | [1.0270, 1.0418] |
| A/B | **3_on / 4_off** | **1.0061** | **[0.9979, 1.0127]** |
| A/A | 1_on / 3_on | 1.0245 | [1.0210, 1.0297] |
| A/A | 2_off / 4_off | 0.9960 | [0.9865, 1.0065] |

`2_off / 4_off` includes 1.0, so past the first run, step-time reproducibility is clean.
`1_on / 3_on` — **the same arm, run twice** — is 1.0245 and excludes 1.0. So most of what
`1_on / 2_off` = 1.0349 looks like is the first-run penalty wearing the costume of an
overhead. The one uncontaminated A/B, `3_on / 4_off`, is 1.0061 with a CI that **contains
1.0**: it does not clear the null.

**Step-level overhead at production shapes is not resolvable from noise; 95% upper bound
~1.3%.**

Per *attention call*, measured on an exclusive node with a byte-identical third arm as an
A/A floor (30 rounds, 40 calls per timing block, 20000-resample paired bootstrap):

| shape | pass | flex/turbo | delta | clears the A/A floor? |
|---|---|---|---|---|
| B2 H8 S1024 D128 | fwd | **1.3110** | +16.74 us | yes |
| B2 H8 S1024 D128 | fwd+bwd | **1.1802** | +49.39 us | yes |
| B2 H16 S2048 D128 | fwd | 1.0067 | +0.52 us | no |
| B2 H16 S2048 D128 | fwd+bwd | 1.0046 | +1.18 us | yes |
| B2 H32 S4096 D128 | fwd | 0.9986 | -0.72 us | no |
| B2 H32 S4096 D128 | fwd+bwd | 1.0009 | +1.39 us | floor itself is biased — unusable |
| B1 H32 S8192 D128 | fwd | 0.9997 | -0.31 us | no |
| B1 H32 S8192 D128 | fwd+bwd | 1.0006 | +1.68 us | no |

The overhead is a **fixed CPU-side cost**, not a proportional one, and it is visible only
while the attention kernel is short enough that the GPU cannot hide it. At S=1024 the
forward kernel is 54 us and the layer's probe/classify/dispatch work is exposed — 1.31x.
At S=2048 the kernel is 77 us, long enough to cover the next call's CPU work, and the
delta collapses to +0.5 us. That is a launch-bound to compute-bound crossover, not an
overhead that shrinks with size.

The `S4096 fwd+bwd` row is the reason the A/A arm exists. Its A/B ratio of 1.0009 looks
reportable until you notice that the **A/A floor's own delta CI is [-2.28, -0.05] us,
excluding zero** — at that shape identical code does not even match itself. Without the
third arm that row would have been written up as a real 0.09% overhead.

That run used a purpose-built 4-layer model, so it proves the path trains but not that it
survives a production stack. The same switch was then A/B'd through Primus's own Megatron
pretrain entry point (llama3.2_1B cut to 4 layers, seq 2048, 20 iterations, mock data, no
checkpoints), the two arms differing only in `use_turbo_flex_attention`. Every iteration's
`lm loss` was bit-identical between the arms (max diff `0.0`, 20/20 iterations), both arms
converged 11.8590 -> 6.2901, and both exited 0.

At production scale (16 layers, seq 8192, 8x MI355X, 500 iterations per arm, ~131M tokens)
the arms are **no longer bit-identical** — and the same self-vs-self rule decides what that
means. The distribution of `|loss_on - loss_off|` over 500 iterations has median `3.76e-06`;
the A/A distribution, from running the *same* arm twice, has median `3.49e-06`. A
permutation test on the difference of medians gives **p = 0.43**: the A/B difference is
indistinguishable from run-to-run nondeterminism. Reading the maxima instead would have
been misleading in both directions — the worst A/B max is `2.759e-03` against an A/A max of
`2.093e-03`, which invites a false alarm, while the worst *relative* disagreement in the
whole experiment belongs to an **A/A** pair (`3.672e-03`), not to any A/B pair. All four
arms converged identically (11.8490 -> 0.0109), produced no nan/inf, and exited 0, at
518 TFLOP/s/GPU and 57.2k tokens/s/GPU.

Both arms dispatch onto the same aiter kernels, which is the intent: with no `score_mod` or
`mask_mod`, the compat layer is supposed to reach the same kernel the direct call reaches.
The arms do take different code paths — the ON arm builds `build_turbo_flex_attention(...)`,
and that path raises outright when the installed Primus-Turbo has no `flex_attention` module,
which is exactly what happened on the first attempt against the released wheel. So `0.0` here
is evidence that the integration preserves shape, scale and causal semantics, not evidence
that the switch did nothing.
