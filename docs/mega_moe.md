# MegaMoE

## Overview

**MegaMoE** is a Mixture-of-Experts (MoE) EP intra-node implementation for AMD GPUs, built on
**FlyDSL** and Primus-Turbo. It is **not** a single fully-fused layer kernel; instead it provides
**two communication-computation fused operators** — `dispatch_grouped_gemm` and
`grouped_gemm_combine` — each folding EP intra-node communication and a grouped GEMM into one
FlyDSL kernel so the cross-rank traffic is hidden behind compute.

The design target is intra-node expert parallelism (`gfx950` / MI355X-class devices) where every
rank owns a slice of the experts and tokens are routed directly into a peer rank's memory.

> **Status:** MegaMoE is under active development. The BF16 path is the primary, validated path.

### Key points

- **Two fused operators** — `dispatch_grouped_gemm` (dispatch + L1 grouped GEMM) and
  `grouped_gemm_combine` (L2 grouped GEMM + combine); forward and backward are conjugates, with
  dispatch and combine swapping roles.
- **Comm-compute overlap** — communication overlaps GEMM compute, reaching 76%+ of the ideal
  roofline, 90%+ in some cases.
- **Activation recompute** — forward saves only the original `x`; backward recomputes the
  dispatched `x`.
- **Buffers sized to the routing** — one host sync per layer reads back how many rows were
  dispatched to this rank, and every per-layer buffer is sized to them (not CUDA Graph capturable).
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

In practice $\eta_{\text{overlap}}$ reaches **76%+**, with most cases at or above **84%** — i.e.
$T_{\text{measured}}$ exceeds the ideal time by only ~**0.13–0.51 ms**.

### 2. Recompute dispatched x in backward to cut activation memory

The original path saves the dispatched `x` in forward for backward use. The Mega path saves only
the original `x` and recomputes the dispatched `x` in backward, reducing forward activation memory.

The key point: this recompute is not a standalone dispatch — it reuses `dispatch_grouped_gemm`, so
the dispatch communication stays hidden behind the grouped GEMM compute. Activation memory is saved
without adding any visible communication overhead in backward.

### 3. Buffers sized to the dispatched rows

The dispatch prologue counts the rows each rank receives, and the host reads this rank's count back
once per layer (a small D2H, like DeepEP's notify). The L1 output, the SwiGLU tensors, the L2 output
staged for the combine push and the handle's row maps are all allocated at that count, not at the
worst-case pool capacity. Only the symmetric buffer -- the dispatch and combine token buffers, sized
by the token count -- is allocated ahead of time, and the pool capacity stays the worst case, so no
routing overflows it. On a DeepSeek-V3 4-layer proxy at EP8 the peak memory drops from 152.9 to
122.3 GB, and the symmetric buffer is 1.8 GiB instead of 8.9 GiB.

The sync is the price: the op is not CUDA Graph capturable.

The MXFP8 path (`fused_mega_moe_fp8_stage1` / `_stage2`) trades that for memory: after its routing
prologue each layer reads the rows dispatched to every rank back to the host once (a small D2H,
overlapped with the token quantization) and allocates every per-layer tensor at the rows this rank
actually received. Only the symmetric receive pool and combine buffer, shared by all layers, are
sized ahead of time: the pool holds the worst-case routing, and every row-indexed tensor is
addressed in 64-bit, so a rank can receive all of it however far past 2 / 4 GiB that takes its
tensors. The MXFP8 op is not CUDA-graph capturable in any case.

## Pipeline

The forward layer is the two fused operators with a SwiGLU in between:

```
x ─▶ dispatch_grouped_gemm (L1, NT) ─▶ SwiGLU ─▶ grouped_gemm_combine (L2, NT) ─▶ y
       │  dispatch comm + L1 grouped GEMM          │  L2 grouped GEMM + combine comm
       └─ comm overlapped with GEMM                └─ + topk reduce (sum of per-rank rows)
```

- **dispatch_grouped_gemm (forward):** send each local token once to every destination rank it
  routes to, then run the grouped L1 GEMM tile-by-tile, overlapping comm with compute.
- **grouped_gemm_combine (forward):** run the grouped L2 GEMM, sum each token's outputs on a rank
  and push one row back to its origin rank, then the top-k reduce sums the per-rank rows per token.

The backward pass is the **conjugate** of the forward: L2 dgrad (NN) + SwiGLUᵀ + dW2 (variable-K)
+ L1 dgrad combine (NN) + dW1 (TN). Dispatch and combine swap roles, and the dispatched `x` is
recomputed by `dispatch_grouped_gemm`.

The gated activation is SiLU-SwiGLU by default (both halves clamped to ±10). The op that owns it
(`fused_mega_moe`, `fused_mega_moe_stage2`, `fused_mega_moe_fp8_stage2`) also takes an
`activation=GLUActivation(...)`, computing `g * sigmoid(alpha * g) * (u + glu_offset)` on the clamped
halves. `GLUActivation.swigluoai()` is MiniMax-M3's `swigluoai` (Megatron's `quick_geglu`): alpha
1.702, offset 1, gate clamped from above only, up clamped to ±7. The spec is a compile-time constant
of the kernels, so it has no runtime cost, and the default reproduces the SiLU kernels bit for bit.

The staged ops (`fused_mega_moe_stage2`, `fused_mega_moe_fp8_stage2`) also take per-expert biases,
`fc1_bias` [E_local, 2I] and `fc2_bias` [E_local, H], with Megatron's TEGroupedMLP semantics:
`y = sum_k p_k (act(x @ w1ᵀ + fc1_bias) @ w2ᵀ + fc2_bias)`, the fc1 bias added before the activation's
clamp (GPT-OSS). In BF16 the fc1 bias rides the SwiGLU kernels and the fc2 bias the combine's fold
sum; their gradients are two-phase pool reductions in a fixed order. In MXFP8 the fc1 bias rides the
fp8 SwiGLU kernels (the backward emits the per-32-row column sums its gradient is made of), and the
fc2 bias is added after the reduce, on the token's own rank, from the EP-gathered bias.

BF16 takes a hidden / intermediate that leaves a 64- or 128-column tail of the 256-wide GEMM tile
(GPT-OSS's 2880). The MXFP8 kernels need hidden % 1024 == 0 and intermediate % 256 == 0; other
shapes run zero-padded up to that grid (2880 runs as 3072), which is exact, and aligned shapes are
untouched.

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
| forward (nt) | 1.53 | 2.90 | 3.20 | 90.5% |
| backward dgrad (nn) | 1.53 | 1.35 | 1.82 | 84.3% |
| backward wgrad dW1 (tn) | 1.53 | 2.82 | 3.08 | 91.5% |

### grouped_gemm_combine

| stage | $T_{\text{comm}}$ (ms) | $T_{\text{gemm}}$ (ms) | $T_{\text{measured}}$ (ms) | $\eta_{\text{overlap}}$ |
| --- | --- | --- | --- | --- |
| forward (nt) | 1.48 | 1.68 | 2.19 | 76.7% |
| backward dgrad (nn) | 1.48 | 3.49 | 3.62 | 96.5% |

### Reproduce

A single benchmark script covers both fused operators, selected with `--mode`. Each compares the
fused path against the serial baseline — the same work measured as a separate GEMM-only leg and a
separate communication-only leg — over 8 ranks, and reports both `speedup (vs serial)` and the
roofline ratio $\max(T_{\text{comm}}, T_{\text{gemm}}) / T_{\text{measured}}$ used in the tables
above. Run from the repo root:

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
| SwiGLU fwd/bwd | `primus_turbo/flydsl/utils/swiglu_kernel.py` |
| SwiGLU + MXFP8 quant fwd/bwd | `primus_turbo/flydsl/mega/fp8/swiglu_mxfp8_kernel.py` |
| Activation spec (`GLUActivation`) | `primus_turbo/flydsl/utils/glu_activation.py` |
| Cross-rank tiles (dispatch/combine/reduce) | `primus_turbo/flydsl/mega/ep_intranode.py` |

## Acknowledgements

- [**Triton-distributed**](https://github.com/ByteDance-Seed/Triton-distributed) (ByteDance-Seed,
  MIT License) — MegaMoE's comm-compute overlapping design (symmetric-memory push, signal/wait
  synchronization, fusing intra-node EP communication into the GEMM kernel) references
  Triton-distributed's overlapping-kernel approach.
- [**DeepGEMM**](https://github.com/deepseek-ai/DeepGEMM) (DeepSeek, MIT License) — MegaMoE's
  cross-rank barrier and symmetric-buffer layout follow DeepGEMM's design; see the file headers of
  `primus_turbo/flydsl/mega/barrier.py` and `primus_turbo/flydsl/mega/symm_buffer.py` for details.
