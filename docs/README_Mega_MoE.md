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
- **Comm-compute overlap** — communication overlaps GEMM compute, reaching 85%+ of the ideal
  roofline, 90%+ in some cases.
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

In practice $\eta_{\text{overlap}}$ reaches **85%+**, with most cases at or above **90%** — i.e.
$T_{\text{measured}}$ exceeds the ideal time by only ~**0.3–0.5 ms**.

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

## Pipeline

The forward layer is the two fused operators with a SwiGLU in between:

```
x ─▶ dispatch_grouped_gemm (L1, NT) ─▶ SwiGLU ─▶ grouped_gemm_combine (L2, NT) ─▶ y
       │  dispatch comm + L1 grouped GEMM          │  L2 grouped GEMM + combine comm
       └─ comm overlapped with GEMM                └─ + topk reduce (weighted scatter-add)
```

- **dispatch_grouped_gemm (forward):** scatter local tokens into the destination rank, then run
  the grouped L1 GEMM tile-by-tile, overlapping comm with compute.
- **grouped_gemm_combine (forward):** run the grouped L2 GEMM, push outputs back to origin ranks,
  then the top-k reduce weights and sums the `num_topk` contributions per token.

The backward pass is the **conjugate** of the forward: L2 dgrad (NN) + SwiGLUᵀ + dW2 (variable-K)
+ L1 dgrad combine (NN) + dW1 (TN). Dispatch and combine swap roles, and the dispatched `x` is
recomputed by `dispatch_grouped_gemm`.

## Performance

### Test Configuration

- **Device:** MI355X (`gfx950`), 8 ranks intra-node (EP8)
- **Model:** DeepSeek-V3
- **Shape:** hidden = 7168, intermediate = 2048, experts = 256, top-k = 8, tokens/rank = 8192
- **dtype:** BF16
- **Overlap efficiency:** $\eta_{\text{overlap}} = \max(T_{\text{comm}}, T_{\text{gemm}}) / T_{\text{measured}}$

### dispatch_grouped_gemm

| stage | $T_{\text{comm}}$ (ms) | $T_{\text{gemm}}$ (ms) | $T_{\text{measured}}$ (ms) | $\eta_{\text{overlap}}$ |
| --- | --- | --- | --- | --- |
| forward (nt) | 2.23 | 3.26 | 3.56 | 91.4% |
| backward dgrad (nn) | 2.23 | 1.63 | 2.38 | 93.7% |
| backward wgrad dW1 (tn) | 2.23 | 3.28 | 3.76 | 87.3% |

### grouped_gemm_combine

| stage | $T_{\text{comm}}$ (ms) | $T_{\text{gemm}}$ (ms) | $T_{\text{measured}}$ (ms) | $\eta_{\text{overlap}}$ |
| --- | --- | --- | --- | --- |
| forward (nt) | 2.32 | 1.80 | 2.57 | 90.0% |
| backward dgrad (nn) | 2.89 | 3.47 | 3.90 | 89.0% |

### Reproduce

A single benchmark script covers both fused operators, selected with `--mode`; each compares the
fused path against the Primus-Turbo (DeepEP) baseline over 8 ranks. Run from the repo root:

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
| Dispatch + grouped GEMM kernel | `primus_turbo/flydsl/mega/dispatch_grouped_gemm_bf16_kernel.py` |
| Grouped GEMM + combine kernel | `primus_turbo/flydsl/mega/grouped_gemm_combine_bf16_kernel.py` |
| Dispatch prologue (routing tables) | `primus_turbo/flydsl/mega/dispatch_prologue_kernel.py` |
| SwiGLU fwd/bwd | `primus_turbo/flydsl/mega/swiglu_kernel.py` |
| Cross-rank tiles (dispatch/combine/reduce) | `primus_turbo/flydsl/mega/ep_intranode.py` |
