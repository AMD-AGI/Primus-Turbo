# Inference Operator Micro-Benchmarks

Per-rank, single-GPU micro-benchmarks for the **forward-only inference** operators
that dominate LLM serving: **MoE** (the FFN of sparse models) and **decode
attention** (the per-step attention of autoregressive decoding).

Each benchmark calls the *exact* production kernel a serving stack
(vLLM / SGLang on ROCm, TensorRT-LLM / FlashInfer on NVIDIA) dispatches for a
given model + precision + parallelism, sweeps a realistic workload grid, and
reports latency / throughput / effective bandwidth (plus an optional SNR
correctness check). Results print as a table and are saved to `.xlsx`.

## Files

| File | Op | Backend | Hardware |
|---|---|---|---|
| `bench_moe_aiter.py` | Fused MoE (FC1+act+FC2) | `aiter.fused_moe` | AMD (MI300 / MI355X) |
| `bench_decode_attn_aiter.py` | Decode attention (GQA / MLA / SWA / DSA) | `aiter` | AMD (MI300 / MI355X) |
| `bench_moe_flashinfer.py` | Fused MoE | FlashInfer `trtllm_*_moe` | NVIDIA (B200 / sm100) |
| `bench_decode_attn_flashinfer.py` | Decode attention | FlashInfer `trtllm_batch_decode_*` | NVIDIA (B200 / sm100) |

The `*_flashinfer.py` files are NVIDIA counterparts that **mirror the structure
and design of the `*_aiter.py` files one-to-one** (same `MODELS`, same helper
functions, same metrics, same CLI, same Excel layout) so a result from one
backend can be compared directly against the other.

## Quick start

```bash
# MoE (AMD): DeepSeek-R1 routed experts, bf16, single rank
python bench_moe_aiter.py --model deepseek-r1 --quant none

# MoE with expert parallelism, imbalanced routing (sweeps all ranks -> min~max)
python bench_moe_aiter.py --model deepseek-r1 --quant fp8_block --ep-size 8

# Decode attention (AMD): MiniMax GQA, vary per-request context lengths
python bench_decode_attn_aiter.py --model minimax-m2.7 --ctx-spread 0.2 --check

# NVIDIA equivalents (on a B200 box)
python bench_moe_flashinfer.py --model deepseek-r1 --quant fp4      # nvfp4 (flagship)
python bench_moe_flashinfer.py --model deepseek-r1 --quant mxfp4    # mxfp4 (vs aiter)
python bench_decode_attn_flashinfer.py --model deepseek-r1 --check
```

Add `--check` to run the SNR correctness gate against an fp32 torch reference.
Add `-o results.xlsx` to control the output path (otherwise auto-named).

---

## Design philosophy (shared by all four files)

1. **Per-rank, single-GPU simulation.** No multi-process launch. We compute one
   rank's *local* shape and feed the matching kernel. Parallelism is modeled by
   sharding the local problem (see below); communication (all-reduce / all-to-all)
   is *not* timed — these are compute-kernel micro-benchmarks.
2. **Clean GPU timing.** `_time_ms()` uses CUDA events with warmup + many
   back-to-back iters, so the number reflects steady-state per-call GPU time and
   excludes host/Python launch overhead. All setup (routing, quantization, weight
   shuffling) happens outside the timed region.
3. **Real model shapes.** `MODELS` carries the routed-expert / attention-head
   shapes of current frontier models, each annotated with its HF link and the
   architectural quirks that pick the kernel path.
4. **Effective-bandwidth accounting.** Decode and small-batch MoE are
   memory-bound, so the headline metric is bytes moved / time. We count only the
   traffic that actually has to cross HBM (activated-expert weights, KV actually
   read), so the BW number is comparable to the GPU's peak.
5. **Correctness decoupled from perf.** `--check` compares against a naive fp32
   reference via **SNR (dB)** — the right gate for quantized kernels (bf16 lands
   very high; fp8 ~30-50 dB; fp4 lower). Off by default to keep sweeps fast.
6. **Robust sweeps.** Every cell is wrapped in `try/except`; an OOM or an
   unsupported shape records an `ERROR` row instead of aborting the whole grid.
7. **Excel output.** A `config` sheet (model dims, GPU, parallelism, seed) plus
   one results sheet per precision / KV dtype.

---

## `bench_moe_aiter.py` — MoE

Benchmarks the routed-expert FFN (`aiter.fused_moe.fused_moe`, the same kernel
vLLM/SGLang call on ROCm). Forward only, SiLU/SwiGLU gating, routed experts only
(no shared expert, no router/dispatch communication).

**Precision (`--quant`):** `none` (bf16), `fp8_block` (per-128×128, DeepSeek
convention), `fp8_token` (per-output-channel), `fp4` (mxfp4: fp4×2 weights +
e8m0 block scales). Activations are quantized dynamically inside the kernel.

**Parallelism (mutually exclusive):**
- `--tp-size N` — **tensor parallel**: shards the FFN intermediate, `local_I = I // N`.
  Every rank holds all experts; load is balanced, so only rank 0 is timed.
- `--ep-size N` — **expert parallel**: shards experts, `local_E = E // N`. Each
  rank owns a global-expert window `[r·local_E, (r+1)·local_E)` selected via an
  `expert_mask`. Without `--balanced`, routing is imbalanced, so **all ranks are
  swept** and Time/TFLOPS/BW are reported as a `min~max` range (the slowest rank
  bounds the real step).

**Key metrics:** TFLOPS from FC1+FC2 over the token-expert pairs in the window;
effective BW from *activated*-expert weight bytes (read once) + bf16 activations.

## `bench_decode_attn_aiter.py` — decode attention

Benchmarks the single-query-step attention kernel SGLang's aiter backend uses,
per attention family. Decode is memory-bound, so the headline is **KV bandwidth**;
the sweep is `batch × context_len`, in both **bf16 and fp8** KV cache.

**Families (auto-selected per model):**
- `gqa` — grouped-query attention, paged KV (`paged_attention_ragged`).
- `mla` — DeepSeek/Kimi absorbed Multi-head Latent Attention, latent KV cache
  (`mla_decode_fwd` asm; triton fallback for fp8 head counts the asm kernel lacks).
- `swa` — sliding-window attention layers with optional attention sinks
  (`unified_attention` triton); asymmetric qk/v dims are padded then sliced back.
- `dsa` — DeepSeek sparse attention: an fp8 paged-MQA-logits **indexer** scores
  the full context and keeps the top-`index_topk` tokens, then absorbed-MLA
  decode runs over just those.

**Parallelism:** `--tp-size` / `--dp-size` give SGLang DP-attention semantics —
per-rank heads = `attn_tp = tp_size // dp_size`. `--ctx-spread S` varies
per-request context lengths uniformly in `[ctx·(1-S), ctx·(1+S)]` to model a
real serving batch. Physical KV pages are deliberately scattered
(`shuffled_pages`) to reflect a shared paging pool's cache-unfriendly gather.

---

## NVIDIA / FlashInfer counterparts

`bench_moe_flashinfer.py` and `bench_decode_attn_flashinfer.py` target Blackwell
(B200, sm100) and call FlashInfer's TensorRT-LLM-gen kernels:

| Family / quant | aiter (AMD) | FlashInfer trtllm (NVIDIA) |
|---|---|---|
| MoE bf16 | `fused_moe` QuantType.No | `trtllm_bf16_moe` |
| MoE fp8 block | `fused_moe` per_128x128 | `trtllm_fp8_block_scale_moe` |
| MoE fp8 per-tensor/token | `fused_moe` per_Token | `trtllm_fp8_per_tensor_scale_moe` |
| MoE fp4 (mxfp4) | `fused_moe` per_1x32 (mxfp4) | `trtllm_fp4_block_scale_moe` (`--quant mxfp4`) |
| MoE fp4 (nvfp4) | — (AMD has no nvfp4) | `trtllm_fp4_block_scale_moe` (`--quant fp4`) |
| Attn GQA | `paged_attention_ragged` | `trtllm_batch_decode_with_kv_cache` |
| Attn MLA | `mla_decode_fwd` | `trtllm_batch_decode_with_kv_cache_mla` |
| Attn SWA | `unified_attention` | `trtllm_batch_decode_with_kv_cache` (`window_left`, `sinks`) |
| Attn DSA | indexer + sparse MLA | `trtllm_batch_decode_with_kv_cache_mla` (`sparse_mla_top_k`) |

**fp4 has two flavors on NVIDIA** (validated on B200 / FlashInfer 0.6.12):
- `--quant fp4` = **nvfp4** (Blackwell flagship): 16-elt blocks, fp8-e4m3 block
  scales, *activations also nvfp4*. No AMD equivalent — use it for the NVIDIA
  perf ceiling, not for cross-vendor comparison.
- `--quant mxfp4` = **OCP micro-scaling**: 32-elt blocks, e8m0 block scales,
  bf16 activations. This is the apples-to-apples match for aiter's `--quant fp4`
  (per_1x32 mxfp4). Many models ship mxfp4 before any nvfp4 checkpoint exists.

> **Note on weight layouts:** the trtllm-gen MoE kernels require intricate,
> kernel-specific weight pre-processing. Two points that bit us and are now
> handled: (1) the kernels apply SwiGLU to the **second** half of `w1`, i.e. they
> expect `[up||gate]`, while the torch reference (and aiter) use `[gate||up]` — so
> `w1` is reordered (`_swap_gate_up`) for the kernel only, keeping `torch_moe_ref`
> identical to the aiter bench. (2) fp4/fp8 block scales must be handed to the
> kernel in an **fp8-e4m3 container** (`.view(torch.float8_e4m3fn)`), and fp8/fp4
> activations are quantized **outside** the timed region (the kernels do not
> accept bf16 activations). The benchmark reuses FlashInfer's own public helpers
> (`reorder_rows_for_gated_act_gemm`, `shuffle_matrix_a`, `shuffle_matrix_sf_a`,
> `fp4_quantize`, `block_scale_interleave`, `convert_to_block_layout`) so the
> layout always matches the kernel.

> **Known kernel-coverage gaps on FlashInfer 0.6.12 / B200:**
> - **MoE fp8_block / fp4 on gpt-oss-120b**: `intermediate=2880` is not a multiple
>   of 128, which the trtllm-gen weight-scale shuffle requires — these cells raise
>   a clean `AssertionError` (sglang pads the intermediate in production; the bench
>   does not, to stay structurally identical to the aiter bench).
> - **Decode SWA on mimo-v2.5-pro (`qk_head_dim=192`)**: trtllm-gen only ships
>   decode kernels for head_dim in {32,64,128,256}; there is no 192 variant, so the
>   cell raises "Missing TRTLLM-GEN kernel". aiter falls back to triton
>   `unified_attention`; there is no NVIDIA fallback.
> - **Decode DSA (sparse MLA)**: implemented against the kernel's documented
>   `(num_seqs, 1, top_k)` page-table contract but **not yet HW-validated** (the
>   test box was released first). Validate before trusting those rows.

---

## Adding a model

Add an entry to the `MODELS` dict in the relevant file with a comment linking the
HF model card and noting any architectural quirk that affects the kernel path.

- **MoE:** `{hidden_size, moe_intermediate_size, n_routed_experts, topk}`.
- **Attention:** a `family` key plus that family's shape fields (see the comment
  block above `MODELS` in `bench_decode_attn_*.py`).

For the FlashInfer MoE benchmark also set the model's routing convention
(`routing_method_type`, and `n_group` / `topk_group` for grouped DeepSeek-V3
routing) — these must match the model or the routing kernel will mis-select.

## Adding a precision / family

1. Add the branch to `quantize_weights` (MoE) or a new `bench_<family>` +
   reference function (attention).
2. Wire it into `benchmark_one` / the CLI `choices`.
3. Update the byte/FLOP accounting so BW and TFLOPS stay comparable.
4. Add an fp32 reference path so `--check` reports a meaningful SNR.

## Conventions for contributors

- Keep the four files structurally parallel — shared helpers (`suppress_output`,
  `compute_snr`, `fp8_dtype`, `_time_ms`) keep the same signature across files.
- Do all setup/quantization/shuffling **outside** `_time_ms`.
- Prefer reusing a backend's official weight-prep helpers over hand-rolling
  layout logic.
- Every new model entry needs a source link and a note on its kernel-path quirk.
