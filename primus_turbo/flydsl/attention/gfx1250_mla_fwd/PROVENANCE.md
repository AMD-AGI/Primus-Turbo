# gfx1250_mla_fwd -- provenance

The five kernel files are aiter's gfx1250 FlyDSL forward prefill (`aiter/ops/flydsl/kernels/`,
aiter commit 6963ae9d, 2026-09-17):

| file here | aiter source |
|---|---|
| `fmha_fwd_prefill_a16w16_m32x8.py` | `fmha_gfx1250/fmha_fwd_prefill_a16w16_m32x8.py` |
| `fmha_b16_buffer_managers.py` | `fmha_gfx1250/fmha_b16_buffer_managers.py` |
| `kernels_common.py` | `kernels_common.py` (unmodified) |
| `tensor_shim.py` | `tensor_shim.py` |
| `buffer_ops.py` | `buffer_ops.py` (unmodified) |

Import edits only: the files were flattened into one package, so `..x` / `aiter.ops.flydsl.kernels`
imports became `.x` (11 sites, two of them function-local in `tensor_shim.py`), and
`get_lds_capacity_bytes` (aiter `jit/utils/chip_info.py`) was inlined into the kernel file, so
the package depends only on flydsl, torch, numpy and the stdlib.

Kernel changes over aiter, in order (each measured on gfx1250; 4 and 5 at DeepSeek-V3 shapes,
b2 s4096 h128, bf16, causal):

1. Longest-first (LPT) remap of the workgroup id: under causal masking the work per q tile
   grows with its index, so the heaviest tiles are dispatched first (`_lpt_block_id`).
2. Packed exp2 argument: one `v_pk_fma` produces two softmax exponent arguments.
3. Speculative / stale-max softmax disabled (it recomputes many tiles on real data).
4. Softmax scale in fp32: aiter multiplied Q by `softmax_scale` rounded to bf16 (0.1352338 ->
   0.1347656 at the DeepSeek-V3 scale; o 41.5 dB, lse 57 dB vs an fp32 reference). Q now stays
   unscaled, `c = softmax_scale * log2(e)` in fp32 is the multiplier of the exp2 FMA
   `p = exp2(fma(s, c, -m))`, the running max is carried in log2 units, and
   `lse = (m + log2(l)) * ln(2)`. o 54.5 dB, lse 145 dB, the same as aiter's ASM kernel.
5. `RESCALE_THRESHOLD = 0.0`: exact running max (with the deferred-rescale threshold of 8 the
   dominant P weight is no longer exactly 1.0 when rounded to bf16 for the PV GEMM: o 53.2 dB
   instead of 54.5 dB at b2 s4096 h128).
6. Only the `m32x8` kernel is kept (the `m32x2` small-grid and `m16x8` variants carried the same
   bf16-scale issue and are never selected at head dim 192).
