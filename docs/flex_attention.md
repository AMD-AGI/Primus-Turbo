# Flex Attention: differences from PyTorch

`turbo.ops.flex_attention` follows the `torch.nn.attention.flex_attention` interface
(`flex_attention(query, key, value, score_mod=..., block_mask=...)`, `create_block_mask`,
`BlockMask`, `AuxRequest`), backed by Triton kernels for ROCm. This page lists where it
differs. See [examples.md](examples.md) for usage.

## Mods are Triton functions

PyTorch traces Python `score_mod` / `mask_mod` callables with `torch.compile`. Turbo
inlines them into the kernel, so they are `@triton.jit` functions:

```python
@triton.jit
def score_mod(score, b, h, q_idx, kv_idx):          # -> new score
    ...

@triton.jit
def mask_mod(b, h, q_idx, kv_idx):                   # -> bool tile, True = keep
    ...

# A mask reading a tensor: its pointer is appended to the arguments.
@triton.jit
def document_mask(b, h, q_idx, kv_idx, DOC):         # DOC: int32 [B, S] document ids
    q_doc = tl.load(DOC + b * S + q_idx, mask=q_idx < S, other=-1)
    kv_doc = tl.load(DOC + b * S + kv_idx, mask=kv_idx < S, other=-2)
    return (q_doc == kv_doc) & (q_idx >= kv_idx)

block_mask = turbo.ops.create_block_mask(document_mask, B, None, S, S, aux_tensors=[doc_ids])
```

`q_idx` / `kv_idx` are broadcastable index tiles; `b` and `h` are scalars. Consequences:

| | PyTorch | Turbo |
|---|---|---|
| Compilation | needs `torch.compile(flex_attention)`; uncompiled calls fall back to an unfused path that materializes the full score matrix | none needed: every call runs the fused kernels; Triton compiles one variant per distinct mod on first use |
| `score_mod` gradient | auto-differentiated | not traced: pass its VJP as `score_mod_bwd(dscore, score, b, h, q_idx, kv_idx)` (`identity_score_mod_bwd` for any additive mod) |
| Tensors read by `score_mod` | captured by closure | passed explicitly in `aux_tensors` (up to 4, contiguous), appended to the `score_mod` arguments; read-only in the kernel |
| Gradient of such a tensor | scattered with atomics into a buffer of its shape | opt-in: pass the tensor as `score_grad_target` with a `score_grad_hook(dscore, valid, b, h, q_idx, kv_idx, grad, *aux)`, which the dQ sweep calls once per tile to store it; no atomics, so the backward stays deterministic |
| Tensors read by `mask_mod` | captured by closure | passed to `create_block_mask(..., aux_tensors=[...])` / `create_block_mask_varlen(...)` (up to 4, contiguous), appended to the `mask_mod` arguments; the `BlockMask` carries them to the attention kernels. Read-only and not differentiable. `q_idx` / `kv_idx` run past the sequence end in each dimension's last tile, so loads indexed by them need a bounds mask |
| Parameterized mods | closures | build them from generated source so each parameter value is a distinct function (Triton keys compiled kernels on source); `sliding_window_mask` and `make_softcap_score_mod` do this |

## Block masks

| | PyTorch | Turbo |
|---|---|---|
| Building | evaluates `mask_mod` over the dense `[B, H, Q, KV]` grid (memory O(Q x KV) unless the builder is compiled) | a Triton kernel evaluates `mask_mod` per tile and stores one category per tile; the dense mask never exists |
| Reuse | each call builds a new mask; callers cache it themselves | **dense masks without aux tensors are memoized**: a repeated `create_block_mask` call with the same `mask_mod` object, `B`, `H`, lengths, block size and device returns the same `BlockMask`, together with every list it has built since (other block sizes, the backward's lists). Bounded LRU of 64 entries. A mask built with `aux_tensors` depends on their values and is rebuilt on every call |
| `BLOCK_SIZE` | any, may be non-square | square power of two >= 16 |
| Built-in masks | ordinary block masks | `causal_mask`, `sliding_window_mask(left, right)` and `noop_mask` run on the kernel's causal/window path when query and key lengths are equal: no block lists at all |
| Full vs partial blocks | `separate_full_blocks` | always separated in the forward (full tiles run with `mask_mod` compiled out); both backward passes walk one combined list |
| Block size per pass | fixed by the mask | the first call for a given mask and shape times the causal/window path and block sizes 64 / 128 separately for the forward and the backward and reuses the winners (`PRIMUS_TURBO_FLEX_ATTENTION_AUTOTUNE=0` uses static defaults) |
| `BlockMask` API | full | `kv_num_blocks`, `kv_indices`, `full_kv_num_blocks`, `full_kv_indices`, `shape`, `seq_lengths`, `BLOCK_SIZE`, `mask_mod`, `aux_tensors`, `to_dense()`, `sparsity()`, `materialize()`; no `from_kv_blocks`, slicing or `q_*` lists |

Memoization keys on the `mask_mod` object, so a mask defined anew on every call (a fresh
`@triton.jit` function each time) misses the cache; build parameterized masks once, for
example through `functools.lru_cache`.

## Packed sequences (varlen)

PyTorch has no varlen entry point; packed sequences are usually expressed as a document
mask over the concatenated batch. Turbo adds:

- `flex_attention_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, ...)`
  on `[total_tokens, H, D]` tensors. Sequences never attend each other; mods see
  sequence-local `q_idx` / `kv_idx` and the sequence index as `b`.
- `create_block_mask_varlen(mask_mod, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k)`.
  A built-in mask takes the causal/window path when queries and keys share one
  `cu_seqlens` tensor. Custom masks are rebuilt per call (the lists depend on the
  `cu_seqlens` values, so they are not memoized), and the build reads nothing back from
  the device -- index tensors span the full block grid -- so a per-batch rebuild does not
  stall the host.
- `flex_attention_varlen_rel_bias(q, k, v, rel_logits, cu_seqlens, max_seqlen, ...)`:
  causal attention with a relative-position bias table `rel_logits[t, h, q - kv]`
  (Inkling), with gradients for q, k, v and `rel_logits`.

## FlyDSL backend

`kernel_options={"BACKEND": "FLYDSL"}` runs the call on FlyDSL kernels
(`primus_turbo.flydsl.flex_attention`, structured after flash-attention's FA4
`flash_attn/cute`), in place of Triton. `"AUTO"` picks FlyDSL when the call is supported
and Triton otherwise; `PRIMUS_TURBO_FLEX_ATTENTION_BACKEND` sets the default (`TRITON`).
PyTorch's own `kernel_options["BACKEND"]` selects among its backends the same way.

Mods are **dual**: the API still takes the `@triton.jit` function, and the FlyDSL
backend runs its registered twin, a plain Python function over `flydsl.expr` values
with FA4's signatures (one element at a time; branch with `.select`, not `if`):

```python
import flydsl.expr as fx

@triton.jit
def document_mask(b, h, q_idx, kv_idx, DOC):
    ...

def document_mask_flydsl(b, h, q_idx, kv_idx, seqlen_info, aux_tensors):
    docs = fx.slice(aux_tensors[0], (b, None))    # aux tensors keep their shapes
    return (fx.Int32(docs[q_idx]) == fx.Int32(docs[kv_idx])) & (kv_idx <= q_idx)

turbo.ops.dual_mod(document_mask, document_mask_flydsl)
```

`score_mod(score, b, h, q_idx, kv_idx, seqlen_info, aux_tensors)` and
`score_mod_bwd(dscore, score, b, h, q_idx, kv_idx, seqlen_info, aux_tensors)` work the
same way. Indices past a sequence's end are clamped to its last position before a
FlyDSL mod is called, so aux reads stay in bounds without a mask. The built-in masks,
`identity_score_mod_bwd` and `make_softcap_score_mod` have twins already.

| | Triton backend | FlyDSL backend |
|---|---|---|
| Masks | causal/window path or block lists, block size tuned per pass | causal/window path, or the block lists at block size 128 (full blocks skip `mask_mod` in both passes); varlen custom masks run elementwise over every block |
| Backward | dQ sweep plus dK/dV sweep | separate dQ and dK/dV kernels, deterministic, no atomics |
| `score_grad_hook` | supported | supported through a FlyDSL twin that returns `(flat index, value, store?)`; the dQ kernel stores it (registered for `flex_attention_varlen_rel_bias`) |
| Aux tensors | 4 for `score_mod` and 4 for `mask_mod` | 4 in total |
| `num_splits` | causal/window or no mask | the same |

The FlyDSL kernels themselves cover more than this interface exposes: paged KV,
`seqused_q` / `seqused_k`, learnable sinks, and fp8 forward with FA4's arguments
(`FlashAttentionForward`, `FlashAttentionBackward`, `flash_attn_combine`). E4M3 runs in
the GPU's native format (FNUZ on gfx942, OCP on gfx950; the other format is converted
on entry). E5M2 has no MFMA in FlyDSL 0.3.4.1 on either target, so it is dequantized
exactly to bf16 and runs on the bf16 kernel (not with paged KV).

## Other differences

- `return_aux=AuxRequest(lse=True)` is supported; `max_scores` is not.
- `kernel_options` accepts `num_splits` (split the KV loop for short-query /
  long-context shapes; dense inputs with a causal/window or no mask) and `BACKEND`.
- fp16 / bf16 inputs only.
- The backward is deterministic: every dQ / dK / dV tile, and the `score_grad_hook`
  output, is written by a single program.

## Status (November 2026)

Measured on MI300X (gfx942), `rocm/primus:v26.8` (Triton 3.8.0, PyTorch 2.14, FlyDSL
0.3.4.1), one GPU, bf16, head dim 128 unless noted.

### Triton backend: supported and tuned

Everything on this page, forward and backward: built-in masks on the causal/window
path, block-sparse masks (memoized dense lists, sync-free varlen lists), varlen,
`score_mod` / `mask_mod` with aux tensors, `score_grad_hook` (Inkling's relative-position
bias), `num_splits`. Tuned: the first call per mask and shape picks the causal/window
path or block size 64 / 128 separately for the forward and the backward, and the kernels'
own autotune tables were swept on MI300X. Against `torch.compile(flex_attention)`
(`benchmark/ops/training/bench_flex_attention_turbo.py`, 4K-8K tokens, 16/2 heads):

| case | turbo fwd ms (vs torch) | turbo fwd+bwd ms (vs torch) |
|---|---|---|
| causal, 8K | 1.73 (7.0x) | 10.95 (3.0x) |
| sliding window 1K, 8K | 0.67 (4.4x) | 2.72 (2.7x) |
| band 1K (block-sparse), 8K | 0.96 (5.4x) | 5.93 (2.2x) |
| prefix LM (block-sparse), 8K | 1.90 (6.3x) | 11.29 (2.7x) |

### FlyDSL backend: supported, validated, partly tuned

Supported: everything on this page, for bf16 / fp16, forward and deterministic backward. The kernels additionally support paged KV, split-KV
(`num_splits < 1` picks it with FA4's heuristic), learnable sinks and fp8 forward.

Validated with the kernel test suites (forward, mods, block-sparse, paged, split-KV,
fp8, backward) on MI300X (gfx942) and MI355X (gfx950), and through the turbo API by
`tests/pytorch/ops/test_flex_attention_flydsl.py` and
`tests/pytorch/ops/test_flex_attention_flydsl_paged.py` on both.

Tuned per GPU: swizzled LDS tiles, unmasked interior tiles, batched staging loads, a
work-aware GQA split of dK / dV; on gfx950 the 32x32x16 MFMA, LDS transpose reads, K / V
double buffering in the forward and a dK / dV kernel at 2 waves per SIMD. gfx942 and
gfx950 have separate backward kernels (`flash_bwd_gfx942.py`, `flash_bwd_gfx950.py`). The
backward still spends 7 matmuls where 5 suffice. Against the Triton backend on the same
GPU (x = Triton time / FlyDSL time, above 1 is faster; B=2, 16/2 heads, 8K tokens;
Inkling: the full-size layers, 1 document):

| case | MI300X fwd ms | MI300X fwd+bwd ms | MI355X fwd ms | MI355X fwd+bwd ms |
|---|---|---|---|---|
| causal | 2.50 (0.67x) | 13.5 (0.79x) | 1.53 (0.66x) | 7.06 (0.95x) |
| sliding window 1K | 0.99 (0.72x) | 3.85 (0.79x) | 0.49 (0.64x) | 1.77 (0.96x) |
| band 1K | 1.28 (0.76x) | 6.24 (0.96x) | 0.76 (0.70x) | 2.67 (1.54x) |
| prefix LM | 2.76 (0.70x) | 13.8 (0.83x) | 1.77 (0.60x) | 7.18 (1.14x) |
| Inkling global | 2.29 (1.38x) | 12.6 (0.89x) | 1.63 (1.14x) | 7.30 (1.02x) |
| Inkling local | 0.65 (1.27x) | 3.25 (0.76x) | 0.44 (1.02x) | 2.20 (0.76x) |
| Inkling-Small global | 1.22 (1.61x) | 6.88 (0.94x) | 0.93 (1.32x) | 3.89 (1.05x) |
| Inkling-Small local | 0.45 (1.12x) | 2.08 (0.83x) | 0.31 (0.89x) | 1.28 (0.85x) |

Paged decode: `FlashAttentionForward` routes one-token paged calls to
`FlashAttentionDecode` (`flash_decode.py`), modelled on aiter's `paged_attention_ragged`
(vLLM's ll4mi kernel): a CTA per (sequence, partition, KV head) with the GQA group
packed into a 16-row MFMA tile, K loaded straight to registers, V staged in wave-private
LDS (no CTA barriers in P . V), and a parallel FlyDSL reduce over partitions. The
partition is 128 tokens when the grid would fill at most half the CUs, else 256.
It supports `score_mod` / `mask_mod` with aux tensors, sliding windows (the grid covers
only in-window partitions), learnable sinks, softcap, and:

- fp8 KV cache with a bf16 query (`kv_dtype="fp8"`, the GPU's native e4m3, per-tensor
  or per-token `k_scale` / `v_scale` as aiter and SGLang use them), converted to bf16 in
  registers so the scores keep bf16 accuracy. Called on `FlashAttentionDecode` directly
  (the forward's fp8 mode quantizes q too and stays on the prefill kernel).
- Block-sparse lists (FA4's, query block 0): each CTA scans its group's lists into a
  per-block status; partitions no head lists are skipped without reading K / V, and
  `mask_mod` runs only on mask-list blocks.

`benchmark/ops/inference/bench_flex_attention_decode.py`, MiniMax-M2.7 at TP=8 (6 query
heads / 1 KV head, page size 16), MI300X, ms:

| batch x context | flydsl bf16 | aiter bf16 | flydsl fp8 KV | aiter fp8 KV | prefill kernel (before) |
|---|---|---|---|---|---|
| 1 x 32K | 0.015 | 0.019 | 0.014 | 0.019 | 0.155 |
| 8 x 8K | 0.017 | 0.017 | 0.014 | 0.015 | 0.255 |
| 32 x 32K | 0.168 | 0.164 | 0.090 | 0.075 | 5.46 |
| 128 x 32K | 0.671 | 0.653 | 0.395 | 0.365 | 14.3 |

SNR against fp32: FlyDSL ~53 dB in both modes; aiter ~51 dB bf16 and ~25 dB fp8 (it also
rounds q and P to fp8). The fp8 gap at large batch is the fp8 -> bf16 conversion.
Block-sparse with ~1/8 of 128-key blocks listed: 32 x 32K 0.171 -> 0.049 ms,
128 x 8K 0.164 -> 0.054 ms; listing every block costs 1-8 us over dense.

### Known issue: Triton 3.8.0 miscompiles one causal backward config

Triton 3.8.0 generates an out-of-bounds read in the causal backward
(`bwd_kernel_fused_causal` in `primus_turbo/triton/flex_attention/flex_attention_bwd_kernel.py`)
for the autotune config `BLOCK_M1=64, BLOCK_N1=128, BLOCK_M2=128, BLOCK_N2=64,
BLK_SLICE_FACTOR=1` (one of the two extra configs `_extend_bwd_configs` adds) when the
kernel also carries a `score_mod` with aux tensors and a `score_grad_hook`.

- Symptom: `HSA_STATUS_ERROR_MEMORY_FAULT` in `bwd_kernel_fused_causal`, only when the
  autotuner benchmarks that config and the page after the `delta` tensor is unmapped.
  `tests/pytorch/ops/test_flex_attention.py::test_score_grad_hook_returns_captured_tensor_gradient`
  aborts when run after the rest of that file; it passes on its own. When it does not
  fault, dQ can be silently wrong: in the standalone reproducer with ordinary
  allocations, dQ was off by 27x (relative max error) while dK and dV were correct.
- Cause: the masked load `Di = tl.load(Delta + offs_m * stride_delta_m, mask=mask_m)` in
  `_bwd_dq_inner` is issued with lane offsets past the end of `delta`. The source, the
  TTGIR and the LLVM IR are correct; the machine code's wave-uniform offset is wrong
  (1088 where the source computes 600), and Triton's buffer descriptors span 2 GB, so
  the hardware does not stop the read.
- Not a kernel bug: the same source, config and inputs run clean on Triton 3.7.1 and on
  3.8.0 with `AMDGCN_USE_BUFFER_OPS=0`. Skipping any one of `move_up_prologue_loads`,
  `optimize_buffer_op_ptr`, `fold_true_cmpi` or `prepare_if_combining` does not help.
  The exact compiler change is not identified.
- Reproduce deterministically: pin the causal backward to that config and run the test
  with `PYTORCH_NO_HIP_MEMORY_CACHING=1`, allocating each tensor so it ends at a 2 MB
  boundary; a standalone reproducer (no Primus-Turbo build needed) is attached to the
  Triton ticket.
- Fixed in Triton 3.9.0. With the reproducer on MI300X, in `rocm/pytorch` ROCm 7.2.3
  with PyTorch 2.10 so that only Triton changes:

  | Triton | Tensors at allocation ends | Ordinary allocations |
  |---|---|---|
  | 3.8.0 | memory access fault | completes, dQ wrong (27x), dK / dV correct |
  | 3.8.0, `AMDGCN_USE_BUFFER_OPS=0` | correct | not run |
  | 3.9.0 | correct | correct |

  PyPI's Triton 3.9.0 cannot be dropped into `rocm/primus:v26.8`: that image's PyTorch
  (2.14.0+rocm10.1.0) is built against its own ROCm Triton, and loading both aborts with
  a duplicate LLVM option. The fix needs a ROCm Triton 3.9 build in the image.
- Workaround in this repo: on Triton < 3.9, `_drop_causal_faulting_configs` removes that
  config from the causal backward's autotune list (costs 1.02-1.05x on some shapes; the
  non-causal kernel keeps it). On 3.9 and later the list is unchanged.
