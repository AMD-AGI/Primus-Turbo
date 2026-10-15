# Provenance of the flex-attention Triton kernels

The forward and backward kernels here are adapted from **ROCm/aiter**
(https://github.com/ROCm/aiter, MIT License), path
`aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/`, at commit
`fedccf0af4219a326b82ea1157ac24aaade12f19`.

| Here | AITER |
|---|---|
| `flex_attention_fwd_kernel.py` | `fwd_prefill.py` |
| `flex_attention_bwd_kernel.py` | `bwd.py` |
| `flex_attention_utils.py` | `utils.py` + `common.py` |

`flex_attention_combine_kernel.py` (split-KV reduction) and
`flex_attention_block_mask_kernel.py` (tile classification for `create_block_mask`) are
new. AITER is credited in the top-level `README.md` and `LICENSE`.

At vendoring time the only change was rewriting the AITER package imports to relative
imports. Everything below is a deliberate change made since. The files are also
formatted with this repository's ruff configuration, and the environment variables were
renamed into the `PRIMUS_TURBO_FLEX_ATTENTION_*` namespace (`_AUTOTUNE`,
`_FWD_CONFIG_JSON`, `_DEBUG`) so they do not collide with an installed AITER.

The backward always runs in `bwd.py`'s `"fused"` mode, which writes each dQ/dK/dV tile
from a single program -- the atomics live in the separate `"fused_atomic"` mode, which
is never used -- so the backward is deterministic, verified bitwise-identical across
repeated runs. A split dQ / dK,dV backward was benchmarked first and cost ~1.3-1.8x in
fwd+bwd wall time on MI300, because it recomputes QK^T for every tile pair twice.

## Upstream bug fixed here: `matrix_instr_nonkdim=16`

**Symptom.** `bwd.py`'s causal backward returned a badly wrong `dV` (O(1) absolute error
vs. a PyTorch reference) and a slightly wrong `dK`, while `dQ` was correct. Non-causal
was unaffected.

**Root cause.** All of AITER's tuned backward configs set `matrix_instr_nonkdim=16`,
forcing the 16x16x16 MFMA. On gfx942 the AMD Triton backend then miscompiles the
accumulating `tl.dot(..., acc=...)` in `_bwd_dkdv_inner` when the dot's K dimension is
<= 16: every loop iteration except the last is silently dropped from the dK/dV
accumulators. Only the causal path hits it, because it sweeps its diagonal blocks with
`BLOCK_M1 // BLK_SLICE_FACTOR` (32 // 2 = 16) while the non-causal path uses 32.

**Evidence.** With `BLOCK_N1=64` and a masked block of 16, exactly the last 16 key
positions of each 64-key block had a correct `dV`:

| masked block (`BLOCK_M1/BLK_SLICE_FACTOR`) | `matrix_instr_nonkdim` | correct dV |
|---|---|---|
| 32 | 16 | yes |
| 16 | 16 | **no** (only last 16 keys) |
|  8 | 16 | **no** (only last 8 keys) |
| 16 | 32 | yes |
| 16 | unset | yes |

**Fix.** `_sanitize_nonkdim()` removes `matrix_instr_nonkdim` from any *causal* config
whose masked sub-block (`BLOCK_M1` or `BLOCK_N2`, divided by `BLK_SLICE_FACTOR`) would
be below 32. Non-causal configs keep the hint; sanitizing them too cost ~5% for no
correctness gain. This is what lets causal compose with a window, a score_mod/mask_mod
and asymmetric head dims; causal fwd+bwd is 1.75-2.46x faster than routing it through
AITER's `"split"` backward.

## Second upstream bug fixed here: backward grid does not cover the dQ phase

The fused backward runs two phases off the same program id: dK/dV strides by `BLOCK_N1`,
dQ by `BLOCK_M2`. Upstream sizes the launch grid by `BLOCK_N1` alone, so any config with
`BLOCK_M2 < BLOCK_N1` computes only the first `(seqlen/BLOCK_N1) * BLOCK_M2` rows of dQ.
Every config AITER ships satisfies `BLOCK_M2 >= BLOCK_N1`, so it never fires upstream,
but it blocks otherwise-legal tile shapes. The grid now sizes by
`min(BLOCK_N1, BLOCK_M2)`; both phases guard their own program id, so
over-provisioning is safe.

A measurement caveat: a config sweep that used a constant upstream gradient
(`do = ones`) hid this bug -- `dp - delta` nearly cancels, dQ becomes tiny, and an
absolute error threshold accepts the truncated result. Sweep a backward with a random
upstream gradient and a relative error check.

## Other upstream defects (avoided, not fixed)

- `mode="fused_atomic"` under `causal=True`: the `_bwd_kernel_fused_atomic_causal`
  launch is missing 4 positional arguments.
- The `"split"` backward assumes a single head dim and reads V past its end when
  `head_dim_qk != head_dim_v` (a latent illegal memory access on MI300X).

Neither mode is used.

## Feature changes

- **`score_mod` / `mask_mod` / `score_mod_bwd`** are threaded through `attn_fwd` and
  both fused backward kernels as `tl.constexpr` callables, inlined at compile time.
  `None` compiles them away, so the default path is unchanged.

- **Read-only aux tensors for `score_mod`.** `attn_fwd`, both fused backward kernels
  and their inner loops take up to four extra pointers (`AUX0`..`AUX3`) and a `NUM_AUX`
  constexpr; `apply_score_mod` appends the first `NUM_AUX` of them to the `score_mod`
  call. With `NUM_AUX=0` the call is exactly the previous one.

- **`score_grad_hook`.** `_bwd_dq_inner` (and both fused kernels, which pass it
  through) takes an optional `SCORE_GRAD_HOOK` constexpr and a `SCORE_GRAD` output
  pointer. After forming `ds` it calls the hook with `ds`, the tile's ownership mask
  (`mask_mn`), the indices, `SCORE_GRAD` and the aux pointers, before `score_mod_bwd`.
  Only the dQ sweep calls it, because across that sweep each (q, kv) pair is owned by
  exactly one tile. The mask matters: when the unmasked range's end is not tile-aligned
  its first tile starts inside the causal masked strip, so the two phases share columns
  outside `mask_mn`.

- **dQ window mask in relative-distance form.** `_bwd_dq_inner`'s two-sided window
  mask now uses `rel = kv - q - offset` like `_bwd_dkdv_inner`. The explicit broadcast
  left/right bound tiles turned dQ to NaN on MI300X once the kernel also carried a
  `score_grad_hook`.

- **LSE gradient.** `attention_backward_triton_impl` takes an optional `dlse`. Since
  d(lse)/d(score) is the softmax row, it is folded into the preprocessed `delta`
  (`delta -= dlse`) before the main kernel.

- **Block-sparse iteration.** `attn_fwd` and `bwd_kernel_fused_noncausal` can walk an
  explicit per-`(batch, head, block)` list of block indices (`BLOCK_SPARSE`). The
  forward runs fully-kept blocks with masking disabled, then partial blocks with
  `mask_mod`; the backward walks one combined list per direction. Tile sizes are pinned
  to the block size (the autotuner is bypassed), which is why the sparse path is not a
  drop-in win for plain causal.

- **LDS capping for MLA head dims.** MLA shapes overflow CDNA3's 64 KiB LDS with the
  tuned block size of 128. Both launchers compute a head-dim-aware cap
  (`max_block_for_lds`) and, when it binds, launch the raw JITFunction with a block size
  that fits, bypassing the autotuner (Triton only calls `early_config_prune` when more
  than one config is present, which would miss `PRIMUS_TURBO_FLEX_ATTENTION_AUTOTUNE=0`).
