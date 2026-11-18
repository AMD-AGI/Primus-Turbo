# MegaMoE

## Overview

**MegaMoE** is a Mixture-of-Experts (MoE) EP intra-node implementation for AMD GPUs, built on
**FlyDSL** and Primus-Turbo. It is **not** a single fully-fused layer kernel; instead it provides
**two communication-computation fused operators** — `dispatch_grouped_gemm` and
`grouped_gemm_combine` — each folding EP intra-node communication and a grouped GEMM into one
FlyDSL kernel so the cross-rank traffic is hidden behind compute.

The design target is intra-node expert parallelism, where every rank owns a slice of the experts and
tokens are routed directly into a peer rank's memory. It runs on MI355X (`gfx950`) in BF16 and
MXFP8, and on MI455X (`gfx1250`) in BF16 and MXFP8 with its own kernels (see
[gfx1250 (MI455X)](#gfx1250-mi455x)). Usage, support and limitations: [MegaMoE](../features/mega_moe.md).

### Key points

- **Two fused operators** — `dispatch_grouped_gemm` (dispatch + L1 grouped GEMM) and
  `grouped_gemm_combine` (L2 grouped GEMM + combine); forward and backward are conjugates, with
  dispatch and combine swapping roles.
- **Comm-compute overlap** — communication overlaps GEMM compute, reaching 76%+ of the ideal
  roofline, 90%+ in some cases.
- **Activation recompute (BF16)** — forward saves only the original `x`; backward re-dispatches it.
- **Buffers sized to the routing** — one host sync per layer reads back how many rows were
  dispatched to this rank, and every per-layer buffer is sized to them (not CUDA Graph capturable).
- **Python API** — the autograd op `fused_mega_moe`, and the staged pair `fused_mega_moe_stage1` /
  `fused_mega_moe_stage2` (MXFP8: `fused_mega_moe_fp8_stage1` / `_stage2`), split at the FC1
  output so each owns one weight; all take external routing (`topk_idx` / `topk_weights`).

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
without adding any visible communication overhead in backward. On gfx1250 the re-dispatch for dW1
runs as peer copies on a side stream during the L1 dgrad (see [gfx1250 (MI455X)](#gfx1250-mi455x)).

This is the BF16 path. The MXFP8 dW1 contracts over the column-wise MXFP8 copy of the dispatched
`x` that the forward already produces, so it keeps that copy instead of re-dispatching.

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

BF16 on gfx950 takes a hidden / intermediate that is a multiple of 64 (the K depth of a tile), at
least 128 (two k-tiles), and leaves a 64- or 128-column tail of the 256-wide GEMM tile, or none
(GPT-OSS's 2880). The MXFP8 kernels need hidden % 1024 == 0 and intermediate % 256 == 0; other
shapes run zero-padded up to that grid (2880 runs as 3072), which is exact, and aligned shapes are
untouched.

The BF16 staged ops also take `fuse_wgrad_accum_pattern="megatron"`, the switch the grouped GEMM
ops have. On gfx1250 the expert wgrads then add dW1 / dW2 straight into the weights' bf16
`main_grad` (one rounding, no dW buffer, no separate add), and the op returns a dummy gradient with
`grad_added_to_main_grad` set, so Megatron's DDP hook skips its own add. Weights the kernel cannot
serve (a `main_grad` that is not a contiguous bf16 tensor of the weight's shape, or an output shape
that is not a whole number of the wgrad's output tiles) keep the ordinary gradient. This follows
Megatron DDP's contract, which clears `grad_added_to_main_grad` once per step; Megatron-FSDP
clears it before every backward and may overwrite or swap `main_grad`, so do not pass the pattern
under FSDP. On gfx950 the pattern is accepted and has no effect: the weights keep the ordinary
gradient.

## gfx1250 (MI455X)

The gfx950 kernels do not lower on gfx1250 (their direct-to-LDS loads have no gfx1250 form), so
MI455X runs its own BF16 kernels behind the same ops and contracts: `fused_mega_moe` and the staged
pair, the activation spec, the expert biases and the dispatched-row sizing all behave as above. The
six GEMM stages of one step:

| Stage | Kernel | What it does |
| --- | --- | --- |
| L1 forward | dispatch + GEMM, NT | dispatch `x`, FC1 (gate-up) GEMM |
| L2 forward | GEMM + combine, NT | FC2 (down) GEMM, combine, top-k reduce |
| L2 dgrad | dispatch + GEMM, NN | dispatch `dy`, FC2 dgrad |
| dW2 | variable-K GEMM | reads the dispatched `dy` in place, no communication of its own |
| L1 dgrad | GEMM + combine, NN | FC1 dgrad, combine, top-k reduce with the routing-weight gradient |
| dW1 | variable-K GEMM | reads `x` re-dispatched during the L1 dgrad |

- **Weights as stored:** the forward GEMMs read `w1` / `w2` in NT and the dgrads in NN, so no
  weight is transposed.
- **dW2:** the dispatched `dy` stays in the symmetric buffer until the next dispatch, and the
  variable-K GEMM reads it in pool order through the pool's row table instead of gathering it.
- **dW1:** `x` is needed again at every rank. Peer copies on a side stream push it into a second
  receive buffer while the L1 dgrad + combine runs, instead of a dispatch exposed after it; the
  variable-K GEMM then reads it through the same row table.
- **Shapes:** NT weights, hidden and intermediate multiples of 64 and at least 128 (the tiles take K
  in k-tiles of 128, or 64 when 128 does not divide it, at least two).
- **`main_grad`:** with `fuse_wgrad_accum_pattern="megatron"` the variable-K GEMMs add into the
  bf16 `main_grad` directly, for weights whose output shape is a whole number of their output tiles.

Validated at EP1 / EP2 / EP4 ([#582](https://github.com/AMD-AGI/Primus-Turbo/pull/582)).
Performance: [MI455X](#mi455x-gfx1250).

## Performance

### MI355X (gfx950) test configuration

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

End-to-end results on MI355X (MXFP8, DeepSeek-V3 proxy, [#566](https://github.com/AMD-AGI/Primus-Turbo/pull/566)):
[MegaMoE § End-to-end impact](../features/mega_moe.md#end-to-end-impact).

### MI455X (gfx1250)

From [#582](https://github.com/AMD-AGI/Primus-Turbo/pull/582), 4 × MI455X (EP4), Primus-Turbo
`68821ec6`, BF16.

One MoE layer: hidden 7168, intermediate 2048, 32 experts per rank, top-8. The baseline is
Megatron's all-to-all token dispatcher with Primus-Turbo's `moe_permute` / `moe_unpermute` and
grouped GEMM. The outputs and gradients of the two paths agree to an SNR of 47 dB or more.

| Tokens per rank, routing | Baseline fwd+bwd | MegaMoE fwd+bwd | Speedup, fwd+bwd / fwd |
| --- | --- | --- | --- |
| 4096, random | 9.42 ms | 6.69 ms | 1.41× / 1.70× |
| 8192, random | 16.62 ms | 11.78 ms | 1.41× / 1.66× |
| 8192, skewed | 38.81 ms | 27.85 ms | 1.39× / 1.60× |

End to end: DeepSeek-V3 proxy with 4 MoE layers, 128 experts, top-8, sequence 4096, MBS 2, GBS 64,
mock data, against Primus's default MoE path.

| Routing | TFLOP/s/GPU, baseline / MegaMoE | Change | Peak memory, baseline / MegaMoE |
| --- | --- | --- | --- |
| Even (Primus's default) | 1218.2 / 1338.8 | +9.9% | 170.9 / 155.0 GB |
| Uniform | 1170.0 / 1292.2 | +10.4% | 179.7 / 155.3 GB |
| Real routing (50 steps) | 986.7 / 1285.3 | +30.3% | 179.6 / 155.4 GB |

Per-stage view (gemm_only / comm_only / fused / roofline for each of the six GEMM stages, then the
whole step through `fused_mega_moe`):

```bash
python benchmark/ops/training/bench_mega_moe_gfx1250.py --num-processes 4
```

## Implementation Map

| Component | File |
| --- | --- |
| Autograd ops, BF16 | `primus_turbo/pytorch/ops/moe/fused_mega_moe.py` |
| Autograd ops, MXFP8 | `primus_turbo/pytorch/ops/moe/fused_mega_moe_fp8.py` |
| Forward / backward custom ops and stage compositions | `primus_turbo/pytorch/kernels/fused_mega_moe/` |
| gfx1250 stage compositions, shape check | `primus_turbo/pytorch/kernels/fused_mega_moe/fused_mega_moe_gfx1250_impl.py` |
| gfx1250 MXFP8 stage compositions, contract check | `primus_turbo/pytorch/kernels/fused_mega_moe/fused_mega_moe_gfx1250_fp8_impl.py` |
| MXFP8 weight prep and its per-step cache | `primus_turbo/pytorch/kernels/fused_mega_moe/mega_moe_fp8_weights.py` |
| Expert-bias gradients | `primus_turbo/pytorch/kernels/fused_mega_moe/mega_moe_bias_impl.py` |
| Symmetric memory (HIP IPC) | `primus_turbo/pytorch/core/symm_mem.py` |
| gfx950 BF16: dispatch + grouped GEMM, grouped GEMM + combine | `primus_turbo/flydsl/mega/bf16/dispatch_grouped_gemm_bf16_kernel.py`, `grouped_gemm_combine_bf16_kernel.py` |
| gfx950 BF16: dispatch prologue (routing tables) | `primus_turbo/flydsl/mega/bf16/dispatch_prologue_kernel.py` |
| gfx950 BF16: cross-rank tiles (dispatch / combine / reduce) | `primus_turbo/flydsl/mega/bf16/ep_intranode.py` |
| gfx950 BF16: symmetric buffer, barrier | `primus_turbo/flydsl/mega/bf16/symm_buffer.py`, `barrier.py` |
| gfx950 SwiGLU fwd / bwd | `primus_turbo/flydsl/utils/swiglu_kernel.py` |
| gfx950 MXFP8: dispatch + grouped GEMM, grouped GEMM + combine | `primus_turbo/flydsl/mega/fp8/dispatch_grouped_gemm_mxfp8_kernel.py`, `grouped_gemm_combine_fp8_kernel.py` |
| gfx950 MXFP8: dispatch prologue, combine CU autotune | `primus_turbo/flydsl/mega/fp8/dispatch_prologue.py`, `combine_autotune.py` |
| gfx950 MXFP8: SwiGLU + MXFP8 quant fwd / bwd | `primus_turbo/flydsl/mega/fp8/swiglu_mxfp8_kernel.py` |
| gfx950 MXFP8: symmetric buffer, barrier | `primus_turbo/flydsl/mega/fp8/symm_buffer.py`, `barrier.py` |
| gfx1250: dispatch + GEMM, GEMM + combine | `primus_turbo/flydsl/mega/gfx1250/bf16/dispatch_gemm_kernel.py`, `gemm_combine_kernel.py` |
| gfx1250: variable-K wgrad, `x` re-dispatch | `primus_turbo/flydsl/mega/gfx1250/bf16/variable_k.py`, `x_redispatch_kernel.py` |
| gfx1250: dispatch prologue, cross-rank tiles, SwiGLU | `primus_turbo/flydsl/mega/gfx1250/dispatch_prologue_kernel.py`, `ep_intranode.py`, `swiglu_kernel.py` |
| gfx1250: symmetric buffer, barrier | `primus_turbo/flydsl/mega/gfx1250/symm_buffer.py`, `barrier.py` |
| Activation spec (`GLUActivation`) | `primus_turbo/flydsl/utils/glu_activation.py` |

## Acknowledgements

- [**Triton-distributed**](https://github.com/ByteDance-Seed/Triton-distributed) (ByteDance-Seed,
  MIT License) — MegaMoE's comm-compute overlapping design (symmetric-memory push, signal/wait
  synchronization, fusing intra-node EP communication into the GEMM kernel) references
  Triton-distributed's overlapping-kernel approach.
- [**DeepGEMM**](https://github.com/deepseek-ai/DeepGEMM) (DeepSeek, MIT License) — MegaMoE's
  cross-rank barrier and symmetric-buffer layout follow DeepGEMM's design (`barrier.py` and
  `symm_buffer.py` under `primus_turbo/flydsl/mega/bf16/`, `fp8/` and `gfx1250/`).
- [**FlyDSL**](https://github.com/ROCm/FlyDSL) (Apache License 2.0) — the barrier and
  symmetric-buffer files above are adapted from FlyDSL and stay under the Apache License 2.0 (see
  their file headers and `LICENSE-APACHE`).
