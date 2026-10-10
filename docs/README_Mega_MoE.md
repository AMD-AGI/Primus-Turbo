# Mega MoE

## Overview

**Mega MoE** is a Mixture-of-Experts (MoE) EP intra-node implementation for AMD GPUs, built on
**FlyDSL** and Primus-Turbo. It is **not** a single fully-fused layer kernel; instead it provides
**two communication-computation fused operators** — `dispatch_grouped_gemm` and
`grouped_gemm_combine` — each folding EP intra-node communication and a grouped GEMM into one
FlyDSL kernel so the cross-rank traffic is hidden behind compute.

The design target is intra-node expert parallelism (`gfx950` / MI355X-class devices) where every
rank owns a slice of the experts and tokens are routed directly into a peer rank's memory.

> **Status:** Mega MoE is under active development. The BF16 path is the primary, validated path.

### Key points

- **Two fused operators** — `dispatch_grouped_gemm` (dispatch + L1 grouped GEMM) and
  `grouped_gemm_combine` (L2 grouped GEMM + combine); forward and backward are conjugates, with
  dispatch and combine swapping roles.
- **Comm-compute overlap** — communication overlaps GEMM compute inside each kernel (see
  [Performance](#performance) for measured numbers).
- **Sender-side dedup** — a token is pushed once per destination rank, and its outputs on that
  rank are summed before the single push back (see [Dedup](#4-sender-side-dedup)).
- **Activation recompute** — forward saves only the original `x`; backward recomputes the
  dispatched `x`.
- **No-Sync / CUDA Graph friendly** — no host-side sync points.
- **Python API** — a single autograd op `fused_mega_moe` that takes external routing
  (`topk_idx` / `topk_weights`).

## Core Design

### 1. Fusing communication with computation

The Mega path fuses EP intra-node communication with the grouped GEMM into a single FlyDSL kernel,
yielding two operators: `dispatch_grouped_gemm` and `grouped_gemm_combine`. Both overlap cross-rank
communication with GEMM compute inside the kernel. Let $T_{\text{comm}}$ be the communication time
and $T_{\text{gemm}}$ the GEMM compute time; under perfect overlap the ideal time is
$\max(T_{\text{comm}}, T_{\text{gemm}})$, and the overlap efficiency is defined as:

$$\eta_{\text{overlap}} = \frac{\max(T_{\text{comm}},\, T_{\text{gemm}})}{T_{\text{measured}}}$$

Dedup shrinks $T_{\text{comm}}$ (fewer bytes), so $\eta_{\text{overlap}}$ is not comparable across
versions; the acceptance metric is the absolute $T_{\text{measured}}$ (see [Performance](#performance)).

### 2. Recompute dispatched x in backward to cut activation memory

The original path saves the dispatched `x` in forward for backward use. The Mega path saves only
the original `x` and recomputes the dispatched `x` in backward, reducing forward activation memory.

The key point: this recompute is not a standalone dispatch — it reuses `dispatch_grouped_gemm`, so
the dispatch communication stays hidden behind the grouped GEMM compute. Activation memory is saved
without adding any visible communication overhead in backward.

### 3. No-Sync, CUDA Graph compatible

The Mega path is fully no-sync: it relies on no host-side synchronization points, making it a
natural fit for CUDA Graph capture and training-framework integration. Compared with the
multi-kernel, multi-stage Turbo path, it markedly reduces launch/sync interference and is better
suited for stable reuse across end-to-end training steps.

### 4. Sender-side dedup

With top-k routing a token often selects several experts on the same rank. Dispatch sends its row
to that rank once, and combine sums its expert outputs on that rank before sending one row back (BF16 path).

| Term | Meaning |
| --- | --- |
| route | one (token, top-k position) pair: `route_idx = token_idx * num_topk + topk_position` |
| primary route | among a token's routes to one rank, the one with the smallest expert id (ties broken by top-k position); the only route that pushes `x` and receives the summed output |
| recv token | a (source rank, token) pair as seen by the receiving rank: `recv_token_idx = src_rank * num_max_tokens_per_rank + token_idx` |
| pool row | a row of the receiver's expert-sorted, 256-aligned grouped-GEMM M space |
| fold row | the highest pool row of a recv token on this rank; it sums all pool rows of that recv token and pushes the result |

**Precondition:** the valid expert ids of each token are distinct (`torch.topk` guarantees this;
`-1` marks a dropped route). It is not checked at run time.

**Data flow:** dispatch pushes `x` once per (token, destination rank) into the receiver's
`dispatch_token_buffer[recv_token_idx]`; the L1 GEMM (and the backward NN/TN GEMMs and dW2) gather
their rows through `pool_row_to_recv_token`; the routing weight is applied in the SwiGLU; combine
sums each recv token's L2 rows at its fold row and pushes one row to the primary route; the top-k
reduce sums the primary routes of each token.

**Memory:** the dispatch token buffer holds one row per recv token instead of one per pool row
(`R` ranks, `T` max tokens per rank, `K` top-k, `E` experts per rank, `H` hidden, BF16):

| | rows | DeepSeek-V3, EP8, T = 8192 |
| --- | --- | --- |
| without dedup | `align(R*T*min(K, E) + E*255, 256)` | 532,480 rows × H × 2 B ≈ 7.6 GB |
| with dedup | `R*T` | 65,536 rows × H × 2 B ≈ 0.94 GB |

The L2 output buffer still has one row per pool row.

## Pipeline

The forward layer is the two fused operators with a SwiGLU in between:

```
x ─▶ dispatch_grouped_gemm (L1, NT) ─▶ SwiGLU ─▶ grouped_gemm_combine (L2, NT) ─▶ y
       │  dispatch comm + L1 grouped GEMM          │  L2 grouped GEMM + combine comm
       └─ comm overlapped with GEMM                └─ + topk reduce (sum of primary routes)
```

- **dispatch_grouped_gemm (forward):** push each local token once per destination rank, then run
  the grouped L1 GEMM tile-by-tile on gathered rows, overlapping comm with compute.
- **SwiGLU (forward):** scales each pool row by its routing weight.
- **grouped_gemm_combine (forward):** run the grouped L2 GEMM, sum each token's rows per rank and
  push one row back to the origin rank, then the top-k reduce sums the primary routes per token.

The backward pass is the **conjugate** of the forward: L2 dgrad (NN) + SwiGLUᵀ + dW2 (variable-K)
+ L1 dgrad combine (NN) + dW1 (TN). Dispatch and combine swap roles, and the dispatched `x` is
recomputed by `dispatch_grouped_gemm`.

The gated activation is SiLU-SwiGLU by default (both halves clamped to ±10). The op that owns it
(`fused_mega_moe`, `fused_mega_moe_stage2`, `fused_mega_moe_fp8_stage2`) also takes an
`activation=GLUActivation(...)`, computing `g * sigmoid(alpha * g) * (u + glu_offset)` on the clamped
halves. `GLUActivation.swigluoai()` is MiniMax-M3's `swigluoai` (Megatron's `quick_geglu`): alpha
1.702, offset 1, gate clamped from above only, up clamped to ±7. The spec is a compile-time constant
of the kernels, so it has no runtime cost, and the default reproduces the SiLU kernels bit for bit.

## Performance

### Test Configuration

- **Device:** MI355X (`gfx950`), 8 ranks intra-node (EP8)
- **Model (stage tables):** DeepSeek-V3
- **Shape:** hidden = 7168, intermediate = 2048, experts = 256, top-k = 8, tokens/rank = 8192
- **dtype:** BF16

Measured on MI355X EP8 with the existing `bench_mega_moe.py`: `main` (`origin/main`, no dedup) and
`dedup` run alternately on the same node, at least 5 rounds each; each number is the median over
rounds of the slowest of the 8 ranks' `fused (ms)`. Routing is the script's uniform random top-k.

### dispatch_grouped_gemm

| stage | main (ms) | dedup (ms) | dedup / main |
| --- | --- | --- | --- |
| forward (nt) | 3.663 | 3.201 | 0.874 |
| backward dgrad (nn) | 2.433 | 1.821 | 0.749 |
| backward wgrad dW1 (tn) | 3.655 | 3.077 | 0.842 |

### grouped_gemm_combine

| stage | main (ms) | dedup (ms) | dedup / main |
| --- | --- | --- | --- |
| forward (nt) | 2.496 | 2.191 | 0.878 |
| backward dgrad (nn) | 4.057 | 3.644 | 0.898 |

### Model matrix

dedup / main ratio of the bench CSV Forward / Backward Time (EP8; dispatch and combine columns come
from separate acceptance runs with the same method):

| model | tokens/rank | dispatch fwd | dispatch bwd | combine fwd | combine bwd |
| --- | --- | --- | --- | --- | --- |
| DeepSeek-V2 | 4096 | 0.888 | 0.775 | 0.862 | 0.963 |
| DeepSeek-V2 | 8192 | 0.848 | 0.770 | 0.811 | 0.937 |
| DeepSeek-V3 | 4096 | 0.884 | 0.841 | 0.893 | 0.896 |
| DeepSeek-V3 | 8192 | 0.874 | 0.804 | 0.878 | 0.898 |
| DeepSeek-V3 | 16384 | 0.909 | 0.786 | 0.865 | 0.912 |
| DeepSeek-V4-Pro | 4096 | 0.885 | 0.878 | 0.906 | 0.896 |
| DeepSeek-V4-Pro | 8192 | 0.863 | 0.870 | 0.912 | 0.901 |
| Grok-2 | 4096 | 0.886 | 0.783 | 0.831 | 0.580 |
| Grok-2 | 8192 | 0.843 | 0.764 | 0.762 | 0.534 |
| Kimi-K2 | 4096 | 0.910 | 0.866 | 0.890 | 0.892 |
| Kimi-K2 | 8192 | 0.902 | 0.834 | 0.892 | 0.893 |
| MiniMax-M3 | 4096 | 0.902 | 0.853 | 0.910 | 0.909 |
| MiniMax-M3 | 8192 | 0.893 | 0.842 | 0.897 | 0.904 |
| Mixtral-8x22B | 4096 | 0.901 | 0.767 | 0.811 | 0.542 |
| Mixtral-8x22B | 8192 | 0.819 | 0.739 | 0.811 | 0.598 |
| Mixtral-8x7B | 4096 | 0.886 | 0.754 | 0.803 | 0.540 |
| Mixtral-8x7B | 8192 | 0.869 | 0.724 | 0.799 | 0.579 |
| Qwen3-235B-A22B | 4096 | 0.919 | 0.845 | 0.901 | 0.917 |
| Qwen3-235B-A22B | 8192 | 0.912 | 0.840 | 0.899 | 0.903 |
| Qwen3-30B-A3B | 4096 | 0.883 | 0.835 | 0.834 | 0.947 |
| Qwen3-30B-A3B | 8192 | 0.885 | 0.786 | 0.815 | 0.907 |

Geometric mean over tokens/rank ∈ {4096, 8192} (80 ratios): **0.832**; max 0.963, min 0.534.

### Reproduce

A single benchmark script covers both fused operators, selected with `--mode`. Each compares the
fused path against the serial baseline — the same work measured as a separate GEMM-only leg and a
separate communication-only leg — over 8 ranks, and reports both `speedup (vs serial)` and the
roofline ratio $\max(T_{\text{comm}}, T_{\text{gemm}}) / T_{\text{measured}}$. For the tables above,
run the same command in an `origin/main` export and in this tree and compare the `fused (ms)`
columns. Run from the repo root:

```bash
export PYTORCH_ROCM_ARCH=gfx950

# fused BF16 dispatch + grouped GEMM
python benchmark/ops/training/bench_mega_moe.py --mode dispatch_grouped_gemm --models DeepSeek-V3 --num-processes 8

# fused BF16 grouped GEMM + combine
python benchmark/ops/training/bench_mega_moe.py --mode grouped_gemm_combine --models DeepSeek-V3 --num-processes 8
```

## Implementation Map

| Component | File |
| --- | --- |
| Autograd op | `primus_turbo/pytorch/ops/moe/fused_mega_moe.py` |
| Forward / backward custom ops | `primus_turbo/pytorch/kernels/fused_mega_moe/` |
| Dispatch + grouped GEMM kernel | `primus_turbo/flydsl/mega/bf16/dispatch_grouped_gemm_bf16_kernel.py` |
| Grouped GEMM + combine kernel | `primus_turbo/flydsl/mega/bf16/grouped_gemm_combine_bf16_kernel.py` |
| Dispatch prologue (routing and dedup tables, `DispatchHandle`) | `primus_turbo/flydsl/mega/bf16/dispatch_prologue_kernel.py` |
| SwiGLU fwd/bwd | `primus_turbo/flydsl/utils/swiglu_kernel.py` |
| SwiGLU + MXFP8 quant fwd/bwd | `primus_turbo/flydsl/mega/fp8/swiglu_mxfp8_kernel.py` |
| Activation spec (`GLUActivation`) | `primus_turbo/flydsl/utils/glu_activation.py` |
| Cross-rank tiles (dispatch/combine/reduce) | `primus_turbo/flydsl/mega/bf16/ep_intranode.py` |
| Symmetric buffer and workspace layout | `primus_turbo/flydsl/mega/bf16/symm_buffer.py` |
| Grid sync, XGMI barrier, flag wait, epoch bump | `primus_turbo/flydsl/mega/bf16/barrier.py` |
| Shared BF16 GEMM tiles (incl. row gather) | `primus_turbo/flydsl/gemm/gemm_bf16_kernel.py`, `primus_turbo/flydsl/grouped_gemm/grouped_gemm_bf16_kernel.py` |

## Acknowledgements

- [**Triton-distributed**](https://github.com/ByteDance-Seed/Triton-distributed) (ByteDance-Seed,
  MIT License) — Mega MoE's comm-compute overlapping design (symmetric-memory push, signal/wait
  synchronization, fusing intra-node EP communication into the GEMM kernel) references
  Triton-distributed's overlapping-kernel approach.
- [**DeepGEMM**](https://github.com/deepseek-ai/DeepGEMM) (DeepSeek, MIT License) — Mega MoE's
  cross-rank barrier and symmetric-buffer layout follow DeepGEMM's design; see the file headers of
  `primus_turbo/flydsl/mega/bf16/barrier.py` and `primus_turbo/flydsl/mega/bf16/symm_buffer.py` for details.
