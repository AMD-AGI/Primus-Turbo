# Baseline provenance

## Branch taken: DEVELOP ONE (no matching knowledge-corpus best-practice file)

Step 1 of Op Setup says to look first for a best-practice file under
`knowledge/backends/<op.backend>/<op.type>/` whose `config:` block matches
`op.config` field by field. **No such file exists.** What was actually checked
and observed:

* `op.backend` is `triton`. There is **no `knowledge/backends/triton/`
  directory at all**, and `knowledge/backends/README.md` does not list `triton`
  among the backends it can describe. So the primary lookup path is empty, not
  merely unmatched.
* Every attention recipe in the corpus was read anyway, in case one matched on
  config despite the backend miss. There are five:
  `aiter/attention/fmha_v3_bwd_hd128_bf16`, `flydsl/attention/hd128`,
  `flydsl/attention/hd64`, `hipkittens/attention/gqa_d128`,
  `hipkittens/attention/gqa_d64`. All five are `arch: gfx950`; this job is
  `gfx1250`. Per `knowledge/INDEX.md` precedence ("measured for your arch and
  op" outranks "general technique"), an arch miss is disqualifying on its own.

### `config:` fields that differed, per candidate card

| card | backend | arch | head_dim | dtype | causal | direction | heads_q_per_kv | format |
|---|---|---|---|---|---|---|---|---|
| this job (`op.config`) | triton | gfx1250 | 128 | bf16/fp32-acc | bottom-right | fwd+bwd | 4 | bshd |
| aiter `fmha_v3_bwd_hd128_bf16` | **aiter** | **gfx950** | 128 = | bf16 = | bottom-right = | **bwd only** | (unstated) | (unstated) |
| hipkittens `gqa_d128` | **hipkittens** | **gfx950** | 128 = | bf16 = | **top-left** | fwd+bwd = | 4 = | bshd = |
| hipkittens `gqa_d64` | **hipkittens** | **gfx950** | **64** | bf16 = | **top-left** | fwd+bwd = | 4 = | bshd = |
| flydsl `hd128` | **flydsl** | **gfx950** | 128 = | bf16 = | (unstated) | fwd+bwd = | **requires power of two in [8,256]; job has 4** | **requires thd; job is bshd** |
| flydsl `hd64` | **flydsl** | **gfx950** | **64** | bf16 = | (unstated) | fwd+bwd = | **same refusal** | **same refusal** |

(`=` means the field agrees. Fields only one side states are not compared, per
the brief; those are marked "(unstated)".)

The closest card on `op.config` alone is the aiter one, which agrees on
head_dim, dtype and causal convention but is a different backend, a different
arch, and backward-only. The hipkittens `gqa_d128` card matches the dtype
quartet, head_dim, GQA ratio and memory format exactly but is **top-left**
causal, which is a different masking convention from this job's bottom-right
and therefore a different kernel, not a tuning delta. Both FlyDSL cards
hard-refuse this job's shape on two independent grounds.

So no card matched, and the baseline was developed rather than adopted. Nothing
was "quietly fallen back to": the corpus was searched first, the search came
back empty, and this is the branch the brief prescribes for that outcome.

## What the baseline stands on

Rather than write an attention backward from scratch, the baseline stands on
the seed the job spec itself names in `op.reference.impl`. Per
`knowledge/INDEX.md`, the job's own `job_context/` outranks the corpus, so this
is the strongest available starting point.

* **Source repository:** `https://github.com/AMD-AGI/Primus-Turbo`
* **Host checkout read from:** `/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo`
* **Commit:** `d2f75576`
* **Upstream aiter commit the kernel derives from:** `ffa945f93115d21d6396e9fb16023e833b660764`
* **Licence:** MIT (copied verbatim to `vendor/LICENSE`)
* **Installed package in the image is a *different*, older commit** (`f857e429`);
  the baseline does not use it. See "Pinning" below.

### Files read

* `primus_turbo/triton/attention/attention_kernel.py`
* `primus_turbo/triton/attention/fused_mha_bwd_kernel.py`
* `primus_turbo/triton/attention/FUSED_MHA_PROVENANCE.md`
* `primus_turbo/pytorch/kernels/attention/attention_triton_impl.py`
* `primus_turbo/pytorch/kernels/attention/attention_fused_bwd_impl.py`
* `primus_turbo/pytorch/core/low_precision.py`
* `primus_turbo/pytorch/core/utils.py`

### Files vendored (byte-identical copies, sha256 of the copy under `vendor/`)

| sha256 | path under `vendor/primus_turbo/` |
|---|---|
| `98225ba81917185f59a3ed2780060e5392c5426d8fe16dc4b45fcb35bc63f262` | `triton/attention/attention_kernel.py` |
| `35473f68bcccd859fde61c8675fe29361d156fc26cfbc54865d5d7bc33736655` | `triton/attention/fused_mha_bwd_kernel.py` |
| `05803a83a14e2329f0c97407c6d04efa4889d7cf977a7401f7c42057f2151785` | `pytorch/kernels/attention/attention_triton_impl.py` |
| `36ab2a920c85449d8f6fcb360ef88b1f5525d6694b6c72c760aec861781b132e` | `pytorch/kernels/attention/attention_fused_bwd_impl.py` |
| `b0995781196b494c82e03415eb6d0ad6f4584d9e9e89c4a715a64c7aa6b5b760` | `pytorch/core/low_precision.py` |
| `8a2e4da07e6026481914b515d6eb69bf2e9da6ba9bd9b63f17a06ef764c5b47e` | `pytorch/core/utils.py` |
| `e3b0c442…b7852b855` (empty) | the seven `__init__.py` files that make the tree importable |

`pytorch/core/utils.py` was added after a first run failed with
`ModuleNotFoundError: No module named 'primus_turbo.pytorch.core.utils'` --
`low_precision.py` imports `get_device_compute_capability` from it. It was
vendored rather than stubbed, so that every vendored file stays byte-identical
to the source commit.

### Tuned config deltas carried in the vendored kernel

The vendored `fused_mha_bwd_kernel.py` carries three settings that differ from
the upstream aiter defaults, for the 8192 tile:

| knob | upstream | vendored |
|---|---|---|
| `BLOCK_N1` | 128 | **256** |
| `BLOCK_M2` | 128 | **256** |
| `BLK_SLICE_FACTOR` | 2 | **1** |

`impl.py::fingerprint()` prints the live config at runtime so a later round
cannot silently change it without the change showing up in the benchmark and
validation output.

## Pinning: a deviation from `op.reference.api`, reported not worked around

`op.reference.api` mandates selecting the Triton backend via
`GlobalBackendManager.set_attn_backend(BackendType.TRITON, ...)`. **That API
does not exist in the pinned image**: the module
`primus_turbo.pytorch.core.backend_manager` is absent at the installed commit
`f857e429`. This was verified by import, not inferred.

The baseline therefore pins by vendoring instead: `impl.py` prepends
`baseline/vendor/` to `sys.path` and imports the kernels directly, so **no
dispatcher is reachable at all**. This is a stronger pin than the env-var one
the spec asked for -- there is no code path by which a different backend could
be selected -- but it *is* a deviation from the letter of `op.reference.api`,
and it is recorded here rather than passed over. `impl.py` additionally raises
at import if an already-installed `primus_turbo` was imported first, so the
vendored tree can never be silently shadowed.

## Verification actually performed

* Forward and fused backward both **build and run** on gfx1250. The seed's own
  `FUSED_MHA_PROVENANCE.md` §7 said "Nothing here has been executed"; it has
  now been executed.
* SQNR against `op/eager/` on the job shape `b4_s8192_hq32_hkv8_d128`:
  **out 53.67 dB, dq 52.24 dB, dk 52.31 dB, dv 52.71 dB** -- all above the
  50 dB gate, and matching the numbers recorded in the seed's own provenance
  document, which is strong evidence the vendored kernel is the one that was
  characterised there.

## Known baseline weakness: dead-row dq on Sq > Skv

On the diagnostic shape `sq_gt_skv` (Sq=2048, Skv=1024, bottom-right causal),
the first 1024 query rows are **fully masked** -- their gradient is exactly
zero. The baseline does not zero them. Observed:

* out 53.97 dB, dk 52.10 dB, dv 52.72 dB -- fine.
* **dq -15.95 dB** over the full tensor. Restricted to live rows only, dq is
  **52.05 dB**. The error is localised entirely to the dead rows, which come
  back with `|dq|` up to **5.25** where they should be 0.

This is a real defect in the seed. It is **reported but not gated**, for one
reason: it is unreachable at `Sq == Skv`, and every shape in `op.shape` has
`Sq == Skv`. Gating on it would red every round for a pre-existing condition no
round can cause, fix, or be judged by. `sq_gt_skv` is therefore carried in
`op/ut/shapes.py` as `DIAGNOSTIC_SHAPES` -- measured and printed every run, so
it cannot be forgotten, but excluded from the pass/fail set. The full rationale
is written into `shapes.py` itself.

If a later round changes the masking convention or introduces a path where
Sq != Skv, this becomes live and must be revisited.
