# Primus-Turbo: Scope and Supported Operators

- [1. Positioning: a training stack](#1-positioning-a-training-stack)
- [2. Supported operators (PyTorch)](#2-supported-operators-pytorch)
- [3. Supported modules (PyTorch)](#3-supported-modules-pytorch)
- [4. Supported operators (JAX)](#4-supported-operators-jax)
- [5. Appendix: backends per operator](#5-appendix-backends-per-operator)

## 1. Positioning: a training stack

Primus-Turbo is the **training** operator library of the Primus ecosystem on AMD GPUs. Every
operator it ships exists to make an LLM **training step** (forward + backward + communication +
low-precision bookkeeping) faster on MI300X / MI325X / MI350X / MI355X / MI455X.

What that means in practice:

- **Forward and backward.** Every compute operator ships as a `torch.autograd.Function` (or a JAX
  `custom_vjp`) with a backward pass. Forward-only kernels appear only as one dispatch backend of an
  operator that has a backward (for example the Gluon attention forward, picked when no gradient is
  needed).
- **Training-shaped workloads.** Large-M prefill-style shapes, variable-K grouped GEMMs for weight
  gradients, MoE expert parallelism with dispatch/combine backward, context-parallel attention.
- **Training precisions.** BF16 master path plus FP8 (tensorwise / rowwise / blockwise / MXFP8) and
  MXFP4 with the quantize-with-transpose variants backward needs.
- **Training communication.** DeepEP dispatch/combine, MegaMoE comm–compute fusion, Async-TP, and
  ring / all-to-all context parallelism.
- **Acceptance criterion.** A kernel change is accepted when it is correct in forward and backward
  and its gain carries over to an end-to-end training step (see `benchmark/pretrain`).

## 2. Supported operators (PyTorch)

Operators are available as `turbo.ops.<name>` (`import primus_turbo.pytorch as turbo`), except:

- `grouped_mlp_fp8`: `from primus_turbo.pytorch.ops.grouped_mlp_fp8 import grouped_mlp_fp8`
- `fused_mega_moe_fp8_stage1` / `_stage2`: `from primus_turbo.pytorch.ops.moe.fused_mega_moe_fp8 import ...`
- the communication APIs in [2.7](#27-communication-and-overlap), which are listed with their full paths.

Every operator below implements backward unless marked "fwd only".

Each architecture cell gives the operator's [maturity level](./features/maturity.md) there:
**Stable**, **Beta**, or **Exp.** (Experimental); `*` marks a provisional level, whose evidence is
still due; — means not supported or not validated. Operators with a Feature Card link it in the
notes. Shape and layout limits are in the notes too; the kernel backend behind each cell is in
[Appendix: backends per operator](#5-appendix-backends-per-operator).

### 2.1 GEMM

| Op | Precision | MI300X / MI325X<br>gfx942 | MI350X / MI355X<br>gfx950 | MI455X<br>gfx1250 | Notes |
|---|---|:-:|:-:|:-:|---|
| `gemm` | BF16 / FP16 / FP32 | Stable\* | Stable\* | Exp. | |
| `gemm_fp8` | FP8 tensorwise | Stable\* | Stable\* | Exp. | E4M3 / E5M2 |
| | FP8 rowwise | Stable\* | Stable\* | Exp. | |
| | FP8 blockwise | Stable\* | Stable\* | Exp. | |
| | MXFP8 | — | Beta\* | — | |
| `gemm_fp4` | MXFP4 | — | Beta | — | [MXFP4 training](./features/mxfp4.md) |
| `mlp_fp4` | MXFP4 | — | Exp. | — | Dense GLU MLP with fused activation. [MXFP4 training](./features/mxfp4.md) |

### 2.2 Grouped GEMM (MoE experts)

Each grouped GEMM has a forward kernel and a variable-K kernel for the weight gradient.

| Op | Precision | gfx942 | gfx950 | gfx1250 | Notes |
|---|---|:-:|:-:|:-:|---|
| `grouped_gemm` | BF16 / FP16 | Stable\* | Stable\* | Exp. | |
| `grouped_gemm_fp8` | FP8 tensorwise | Stable\* | Stable\* | Exp. | |
| | FP8 rowwise | Stable\* | Stable\* | Exp. | |
| | FP8 blockwise | Stable\* | Stable\* | Exp. | |
| | MXFP8 | — | Beta\* | Exp. | |
| `grouped_gemm_fp4` | MXFP4 | — | Exp. | Exp. | [MXFP4 training](./features/mxfp4.md) |
| `grouped_mlp_fp8` | FP8 tensorwise / MXFP8 | — | Beta\* | — | Expert MLP with SwiGLU / GeGLU fused into the grouped GEMM |
| `grouped_mlp_fp4` | MXFP4 | — | Exp. | — | Same, MXFP4. [MXFP4 training](./features/mxfp4.md) |

### 2.3 Attention

| Op | Precision | gfx942 | gfx950 | gfx1250 | Notes |
|---|---|:-:|:-:|:-:|---|
| `flash_attn_func` | BF16 / FP16 | Stable\* | Stable\* | Exp. | GQA / MQA, causal, sliding window, attention sink |
| `flash_attn_varlen_func` | BF16 / FP16 | Stable\* | Stable\* | — | Packed variable-length sequences |
| `flash_attn_fp8_func` | FP8 | Exp. | Exp. | — | |
| `flash_attn_usp_func`, `flash_attn_varlen_usp_func`, `flash_attn_fp8_usp_func` | BF16 / FP8 | Exp. | Exp. | — | Context parallelism: ring + all-to-all (USP) |
| `sparse_mla_func` | BF16 | Exp. | Exp. | Exp. | DeepSeek sparse MLA |

### 2.4 MoE

| Op | Precision | gfx942 | gfx950 | gfx1250 | Notes |
|---|---|:-:|:-:|:-:|---|
| `fused_group_topk_routing_with_aux_score` | – | Exp. | Exp. | Exp. | Group top-k router with aux-loss scores |
| `tokens_per_expert_to_mask` | – | Exp. | Exp. | Exp. | Index helper, fwd only |
| `moe_permute`, `moe_unpermute` | BF16 / FP8 tensorwise | Exp. | Exp. | Exp. | |
| `moe_dispatch`, `moe_combine` | – | Beta | Beta | Exp. | Expert-parallel token dispatch / combine (intra- and inter-node; gfx1250 intra-node only). See [DeepEP](./features/deep_ep.md) |
| **MegaMoE**<br>`fused_mega_moe` (+ `_stage1` / `_stage2`) | BF16 | — | Beta | Beta | Dispatch + FC1 and FC2 + combine fused, intra-node EP. See [MegaMoE](./features/mega_moe.md) |
| **MegaMoE**<br>`fused_mega_moe_fp8_stage1` / `_stage2` | MXFP8 | — | Exp. | — | Staged API only. See [MegaMoE](./features/mega_moe.md) |

### 2.5 Normalization, activation, position embedding, loss

| Op | Precision | gfx942 | gfx950 | gfx1250 | Notes |
|---|---|:-:|:-:|:-:|---|
| `rmsnorm` | BF16 / FP16 / FP32 | Exp. | Exp. | Exp. | Optional zero-centered gamma |
| `rmsnorm_residual` | BF16 / FP16 / FP32 | Exp. | Exp. | Exp. | Fused residual add |
| `rmsnorm_residual_fp4` | BF16 → MXFP4 | — | Exp. | — | Fused residual add + MXFP4 quantization |
| `swiglu_with_probs`, `geglu_with_probs` | BF16 / FP16 | Exp. | Exp. | Exp. | GLU scaled by routing probability, optional row mask and clamp |
| `fused_qkv_rope` | BF16 | Exp. | Exp. | — | RoPE on packed QKV |
| `fused_qkv_rmsnorm_rope` | BF16 | — | Exp. | — | QK RMSNorm + RoPE on packed QKV |
| `cross_entropy` | BF16 / FP32 | Exp. | Exp. | Exp. | Label smoothing, ignore index, in-place logits. See [Cross Entropy](./design/cross_entropy.md) |

### 2.6 Quantization and low-precision training support

These are fwd-only utilities; the `_with_trans` variants also emit the transposed copy backward needs.

| Op | Precision | gfx942 | gfx950 | gfx1250 | Notes |
|---|---|:-:|:-:|:-:|---|
| `quantize_fp8`, `dequantize_fp8` | FP8 tensorwise / rowwise / blockwise | Exp. | Exp. | Exp. | |
| | MXFP8 | — | Exp. | Exp. | |
| `quantize_fp8_with_trans` | FP8 blockwise | Exp. | Exp. | Exp. | |
| | MXFP8 | — | Exp. | Exp. | |
| `grouped_quantize_fp8` / `_with_trans`, `grouped_dequantize_fp8` | MXFP8 | — | Exp. | Exp. | Per-expert |
| `quantize_fp4`, `quantize_fp4_with_trans`, `dequantize_fp4` | MXFP4 | — | Exp. | Exp. | |
| `grouped_quantize_fp4` / `_with_trans`, `grouped_dequantize_fp4` | MXFP4 | — | Exp. | Exp. | Fuses MoE gather. See [MoE gather quantization](./design/moe_gather_quantization.md) |
| `weight_deosc_update`, `weight_deosc_close` | MXFP4 | Exp. | Exp. | — | Suppresses low-precision weight oscillation during training |
| `weight_deosc_qdq` | MXFP4 | — | Exp. | — | Fused MXFP4 QDQ + tracking |

### 2.7 Communication and overlap

| API | gfx942 | gfx950 | gfx1250 | Notes |
|---|:-:|:-:|:-:|---|
| `turbo.ops.fused_all_gather_matmul`, `turbo.ops.fused_all_gather_scaled_matmul`, `turbo.ops.fused_matmul_reduce_scatter` | Exp. | Exp. | — | Async tensor parallelism; fwd only, the training integration supplies the backward. See [Async-TP](./features/async_tp.md) |
| `primus_turbo.pytorch.dist.FP8AllToAll` | Exp. | Exp. | — | All-to-all on FP8 payloads, with backward |
| `primus_turbo.pytorch.deep_ep.Buffer` | Beta | Beta | Exp. | Intra-node and inter-node EP buffers (inter-node needs rocSHMEM; gfx1250 intra-node only). See [DeepEP](./features/deep_ep.md) |

## 3. Supported modules (PyTorch)

`primus_turbo.pytorch.modules`. Each module has the level of the operator it wraps.

| Module | Wraps |
|---|---|
| `Linear` | `gemm` |
| `Float8Linear` | `gemm_fp8` (tensorwise / rowwise / blockwise / MXFP8) |
| `GroupedLinear` | `grouped_gemm` |
| `TurboAttention` | `flash_attn_func` and the USP variants |
| `RMSNorm` | `rmsnorm` |
| `DeepEPTokenDispatcher` (in `modules.moe`) | `moe_dispatch` / `moe_combine` with permute |

## 4. Supported operators (JAX)

JAX support is Experimental on every architecture. `primus_turbo.jax.lax`:

| Op | Fwd / Bwd |
|---|---|
| `grouped_gemm` | ✅ / ✅ |
| `compute_group_offs` | ✅ / – (index helper) |
| `grouped_gemm_fp8` | ✅ / ✅ |
| `rmsnorm` | ✅ / ✅ |
| `quantize_fp8`, `dequantize_fp8` | ✅ / – |
| `moe.moe_dispatch`, `moe.moe_combine` (DeepEP intra- and inter-node) | ✅ / ✅ |

## 5. Appendix: backends per operator

An operator may register several backends. Pick one in code, with `PRIMUS_TURBO_*_BACKEND`, or with
AutoTune — see [Backend and AutoTune](./examples.md#5-backend-and-autotune). Backends are listed in
registration order, with the arch / format limits from each backend's `can_handle`.

| Op | Backends |
|---|---|
| `gemm` | HIPBLASLT, TRITON |
| `gemm_fp8` | TURBO (MXFP8, gfx950), HIPBLASLT (tensorwise; MXFP8 not on gfx942), CK (tensorwise / rowwise / blockwise; not gfx1250), TRITON (tensorwise / rowwise / blockwise), FLYDSL (tensorwise / MXFP8; gfx950) |
| `gemm_fp4` | AITER (gfx950, NT layout), HIPBLASLT (not gfx942), FLYDSL (gfx950) |
| `mlp_fp4` | FLYDSL (fused GLU + MXFP4 quantization, forward and backward; required, gfx950), HIPBLASLT (remaining GEMMs, default) |
| `grouped_gemm` | CK (not gfx1250), HIPBLASLT, TRITON, FLYDSL (gfx950, gfx1250) |
| `grouped_gemm_fp8` | CK (tensorwise / rowwise; not gfx1250), HIPBLASLT (tensorwise), TRITON (all formats; MXFP8 not on gfx942), FLYDSL (tensorwise / MXFP8; gfx950) |
| `grouped_gemm_fp4` | FLYDSL (gfx950), TRITON (not gfx942) |
| `grouped_mlp_fp8` | FLYDSL (gfx950), TRITON |
| `grouped_mlp_fp4` | FLYDSL (gfx950) |
| `flash_attn_func` | FLYDSL (gfx950: causal, head dim 64/128, sbhd; gfx1250: head dim 128 GQA and DeepSeek-V3 MLA 192/128, the MLA backward on HipKittens kernels with `PRIMUS_TURBO_MLA_BWD_IMPL=hk` where seqlen_q % 384 is 0 or 256), AITER (general fallback; default on gfx942), HIPKITTENS (gfx950, causal, sbhd), GLUON (gfx950, fwd only), TRITON (gfx1250) |
| `flash_attn_varlen_func` | FLYDSL (gfx950), AITER |
| `flash_attn_fp8_func` | TRITON |
| `sparse_mla_func` | FLYDSL (gfx950, gfx1250), TRITON |
| `moe_permute`, `moe_unpermute` | TRITON, TURBO |
| `moe_dispatch`, `moe_combine` | TURBO, DEEP_EP |
| MegaMoE (`fused_mega_moe*`) | FLYDSL |
| `rmsnorm*`, `swiglu_with_probs`, `geglu_with_probs`, `cross_entropy`, router | TRITON (`rmsnorm_residual_fp4`: FLYDSL) |
| `fused_qkv_rope`, `fused_qkv_rmsnorm_rope` | FLYDSL |
| Quantization | TURBO (HIP), TRITON (blockwise), FLYDSL (MXFP8 / MXFP4) |
