# MegaMoE

## Summary

MegaMoE runs the expert part of an MoE layer under intranode expert parallelism (EP) as two fused
kernels: `dispatch_grouped_gemm` (token dispatch + FC1 grouped GEMM) and `grouped_gemm_combine`
(FC2 grouped GEMM + combine). Each kernel overlaps the cross-rank traffic with its GEMM, instead of
running a dispatch, a grouped GEMM and a combine one after another. The backward pass is the
conjugate of the forward, with dispatch and combine swapping roles. In BF16 the backward
re-dispatches the input tokens instead of saving the dispatched ones, which keeps less activation
memory; MXFP8 keeps the dispatched tokens in MXFP8 for its dW1. It is for training: every op has a
backward.

Routing is computed outside: the ops take `topk_idx` and `topk_weights`. The symmetric buffer the
ranks exchange tokens through is allocated and cached by the ops; the caller does not allocate it.

## Support matrix

| | gfx942 (MI300X / MI325X) | gfx950 (MI350X / MI355X) | gfx1250 (MI455X) |
|---|:-:|:-:|:-:|
| BF16: `fused_mega_moe`, `fused_mega_moe_stage1` / `_stage2` | — | ✅ (tested at EP8) | ✅ (tested at EP1 / EP2 / EP4) |
| MXFP8: `fused_mega_moe_fp8_stage1` / `_stage2` | — | ✅ (tested at EP8) | ✅ (tested at EP1) |
| `fuse_wgrad_accum_pattern="megatron"` (staged ops) | — | ignored | ✅ (BF16 and MXFP8) |

Intranode EP only: the GPUs of one host. On MI455X, a Helios rack connects its 72 GPUs in one
scale-up domain, so expert parallelism seldom has to leave the rack; MegaMoE covers only the GPUs of
one host there today, because its symmetric buffer is shared through HIP IPC, which does not cross
hosts. The kernels' peer-address table already holds 72 ranks, a whole rack.

Frameworks: PyTorch. See [Supported Operators § MoE](../supported_ops.md#24-moe).

## Usage

Install Primus-Turbo as in the [Quick Start](../../README.md#-quick-start).

Inputs, per rank: `x` [T, H] bf16; `topk_idx` [T, K], global expert ids (cast to int64);
`topk_weights` [T, K] (cast to fp32); `w1` [E_local, 2I, H] bf16, the gate rows then the up rows;
`w2` [E_local, H, I] bf16. This rank owns experts `[rank * E_local, (rank + 1) * E_local)`.

- BF16, one autograd op for the whole expert part (also `turbo.ops.fused_mega_moe`):

  ```python
  from primus_turbo.pytorch.ops.moe import fused_mega_moe

  y = fused_mega_moe(group, x, topk_idx, topk_weights, w1, w2)  # [T, H]
  ```

- BF16, two autograd ops, one per weight, so a framework can map them to Megatron's
  `experts.linear_fc1` / `experts.linear_fc2` and keep DDP's per-weight gradient overlap:

  ```python
  from primus_turbo.pytorch.ops.moe import fused_mega_moe_stage1, fused_mega_moe_stage2

  l1_out, dispatch_weights, handle = fused_mega_moe_stage1(x, topk_idx, topk_weights, w1, group)
  y = fused_mega_moe_stage2(l1_out, dispatch_weights, handle, topk_idx, topk_weights, w2, group)
  ```

- MXFP8 (gfx950, gfx1250), staged only. Pass the bf16 weights; their MXFP8 copies are cached inside, and
  the caller must invalidate the cache once per optimizer step:

  ```python
  from primus_turbo.pytorch.kernels.fused_mega_moe import advance_weight_generation
  from primus_turbo.pytorch.ops.moe.fused_mega_moe_fp8 import (
      fused_mega_moe_fp8_stage1,
      fused_mega_moe_fp8_stage2,
  )

  l1, dispatch_weights, handle, state = fused_mega_moe_fp8_stage1(x, topk_idx, topk_weights, w1, group)
  y = fused_mega_moe_fp8_stage2(l1, dispatch_weights, handle, state, topk_idx, topk_weights, w2, group)
  ...
  optimizer.step()
  advance_weight_generation()  # without it the experts keep training on their step-0 weights
  ```

Full example: [Examples § MegaMoE](../examples.md#14-megamoe).

## Configuration

### Activation

The gated activation between the two GEMMs defaults to SiLU-SwiGLU, with both halves clamped to
±10. Other gated activations are passed as `activation=GLUActivation(...)`, which computes
`g * sigmoid(alpha * g) * (u + glu_offset)` on the clamped gate / up halves; for example
`GLUActivation.swigluoai()` (GPT-OSS). Only the ops that own the activation take it:
`fused_mega_moe`, `fused_mega_moe_stage2` and `fused_mega_moe_fp8_stage2`.

### Expert biases

`fused_mega_moe_stage2` and `fused_mega_moe_fp8_stage2` take `fc1_bias` [E_local, 2I] and
`fc2_bias` [E_local, H], with the semantics of Megatron's TEGroupedMLP:
`y = sum_k p_k (act(x @ w1ᵀ + fc1_bias) @ w2ᵀ + fc2_bias)`, the fc1 bias added before the
activation's clamp (GPT-OSS).

### Gradient accumulation into `main_grad`

`fused_mega_moe_stage1` / `fused_mega_moe_stage2` take `fuse_wgrad_accum_pattern="megatron"`. On
MI455X the expert wgrads are then added straight into the weights' bf16 `main_grad` (one rounding,
no separate dW buffer), and the op returns a dummy gradient with `grad_added_to_main_grad` set, so
Megatron DDP's hook skips its own add. A weight keeps the ordinary gradient when its `main_grad` is
not a contiguous bf16 tensor of the weight's shape, or when the weight's shape is not a whole number
of the wgrad's output tiles. The MXFP8 staged ops take the option the same way on MI455X. On MI355X
the option is accepted and has no effect.

### Buffers

The symmetric buffer is sized from the shape of the call: EP size, number of experts, tokens per
rank, top-k, hidden and intermediate. One buffer is kept per process. A call with another shape
frees it and allocates a new one; BF16 reallocates on any change, MXFP8 only when it needs a larger
one. The allocation is collective over the EP group.

The per-layer tensors are allocated at the rows each rank actually receives: each layer's forward
reads that count back to the host once.

### Environment variables

MXFP8 combine tuning (`PT_MEGA_FP8_COMBINE_*`, `PT_MEGA_FP8_L1_COMBINE_CU`,
`PT_MEGA_FP8_L2_COMBINE_CU`): see [Environment variables § Mega MoE](../env.md#mega-moe-flydsl-fp8-combine).

## Performance

### MI355X: overlap inside each fused kernel

8 × MI355X (EP8), DeepSeek-V3 expert shape: hidden 7168, intermediate 2048, 256 experts, top-8,
8192 tokens per rank, BF16. T_comm and T_gemm are the communication and the GEMM run alone;
overlap efficiency is max(T_comm, T_gemm) / T_measured.

| Kernel, stage | T_comm (ms) | T_gemm (ms) | T_measured (ms) | Overlap efficiency |
|---|---|---|---|---|
| dispatch_grouped_gemm, forward | 1.53 | 2.90 | 3.20 | 90.5% |
| dispatch_grouped_gemm, dgrad | 1.53 | 1.35 | 1.82 | 84.3% |
| dispatch_grouped_gemm, wgrad dW1 | 1.53 | 2.82 | 3.08 | 91.5% |
| grouped_gemm_combine, forward | 1.48 | 1.68 | 2.19 | 76.7% |
| grouped_gemm_combine, dgrad | 1.48 | 3.49 | 3.62 | 96.5% |

- Source: [MegaMoE design § Performance](../design/mega_moe.md#performance).
- Reproduce: `python benchmark/ops/training/bench_mega_moe.py --mode dispatch_grouped_gemm --models DeepSeek-V3 --num-processes 8`,
  and the same with `--mode grouped_gemm_combine`.

### MI455X: one MoE layer against the all-to-all path

4 × MI455X (EP4). Hidden 7168, intermediate 2048, 32 experts per rank, top-8, BF16. The baseline
is Megatron's all-to-all token dispatcher with Primus-Turbo's `moe_permute` / `moe_unpermute` and
grouped GEMM.

| Tokens per rank, routing | Baseline fwd+bwd | MegaMoE fwd+bwd | Speedup, fwd+bwd / fwd |
|---|---|---|---|
| 4096, random | 9.42 ms | 6.69 ms | 1.41× / 1.70× |
| 8192, random | 16.62 ms | 11.78 ms | 1.41× / 1.66× |
| 8192, skewed | 38.81 ms | 27.85 ms | 1.39× / 1.60× |

The outputs and gradients of the two paths agree to an SNR of 47 dB or more.

- Source: [#582](https://github.com/AMD-AGI/Primus-Turbo/pull/582), Primus-Turbo `68821ec6`.
- Per-stage times on MI455X: `python benchmark/ops/training/bench_mega_moe_gfx1250.py --num-processes 4`.

Other benchmarks: [Benchmarks § MegaMoE](../../benchmark/README.md#megamoe).

## End-to-end impact

Primus-LM's Megatron backend replaces the MoE experts with MegaMoE when `use_turbo_mega_moe` is
set (off by default; on in the MI455X DeepSeek-V3 BF16 recipe), and `turbo_mega_moe_precision`
picks BF16 or MXFP8. With MXFP8 it calls `advance_weight_generation()` after every optimizer step.

### MI355X, MXFP8

DeepSeek-V3 4-layer proxy, 8 × MI355X, EP8, MBS 2, GBS 512, sequence 4096, 20 iterations. The
baseline is the MXFP8 MoE path with DeepEP.

| | ms / iteration | TFLOP/s/GPU | Peak memory | LM loss at step 20 |
|---|---|---|---|---|
| Baseline (DeepEP) | 9229.5 | 869.2 | 141.3 GB | 8.9262 |
| MegaMoE | 7091.2 | 1131.3 | 149.3 GB | 8.9170 |

MegaMoE is 1.30× faster and peaks 8.0 GB higher. Source:
[#566](https://github.com/AMD-AGI/Primus-Turbo/pull/566).

On the same proxy in BF16, sizing the per-layer buffers to the dispatched rows lowered the peak
memory from 152.9 to 122.3 GB ([design § 3](../design/mega_moe.md#3-buffers-sized-to-the-dispatched-rows)).

### MI455X, BF16

DeepSeek-V3 proxy with 4 MoE layers, 128 experts, top-8, sequence 4096, MBS 2, GBS 64, mock data,
4 × MI455X (EP4). The baseline is Primus's default MoE path.

| Routing | TFLOP/s/GPU, baseline / MegaMoE | Change | Peak memory, baseline / MegaMoE |
|---|---|---|---|
| Even (Primus's default) | 1218.2 / 1338.8 | +9.9% | 170.9 / 155.0 GB |
| Uniform | 1170.0 / 1292.2 | +10.4% | 179.7 / 155.3 GB |
| Real routing (50 steps) | 986.7 / 1285.3 | +30.3% | 179.6 / 155.4 GB |

Source: [#582](https://github.com/AMD-AGI/Primus-Turbo/pull/582), Primus-Turbo `68821ec6`.

## Known limitations

- Intranode EP only. For a larger EP use [DeepEP](./deep_ep.md) or the all-to-all token
  dispatcher with the grouped GEMM.
- Not capturable by CUDA graphs: each layer's forward reads the number of rows dispatched to this
  rank back to the host once (a small device-to-host copy) and allocates its buffers at that count.
- Every rank of the EP group passes the same number of tokens in a call: the buffer layout is derived
  from the token count, and the buffer is allocated collectively.
- Shapes:
  - BF16 on MI355X: hidden and intermediate are multiples of 64, at least 128, and leave a remainder
    of 0, 64 or 128 modulo 256 (the GEMM tile is 256 wide and takes a 64- or 128-column tail, so
    GPT-OSS's 2880 runs).
  - BF16 on MI455X: NT weights, hidden and intermediate multiples of 64 and at least 128.
  - MXFP8 on MI355X: the kernels need hidden % 1024 == 0 and intermediate % 256 == 0. Other shapes
    run zero-padded up to that grid (2880 runs as 3072), which is exact but does the padded work.
  - MXFP8 on MI455X: hidden % 512 == 0 and intermediate % 128 == 0, not padded; SiLU-SwiGLU given
    explicitly as the activation, no expert biases, at most 256 experts per rank.
  - MI455X, BF16 and MXFP8: the token count of a call, rounded up to 128, times top-k times hidden
    stays below 2^31 per rank (37,376 tokens for DeepSeek-V3's top-8 at hidden 7168); a larger call
    raises `ValueError`.
- MXFP8: staged ops only; `advance_weight_generation()` must be called once per
  optimizer step, and a forward can be backpropagated only once (no `retain_graph`).
- `fuse_wgrad_accum_pattern="megatron"` follows Megatron DDP's `main_grad` contract, which clears
  `grad_added_to_main_grad` once per step. Do not use it under Megatron-FSDP, which clears the flag
  before every backward and may overwrite or swap `main_grad`.

## Further reading

- Design: [MegaMoE](../design/mega_moe.md)
- Tests (`tests/pytorch/ops/`): `test_fused_mega_moe.py`, `test_fused_mega_moe_fp8.py`,
  `test_mega_moe_activation.py`, `test_mega_fp8_combine_autotune.py`, and the MI455X tests
  `test_mega_bf16_gfx1250_*.py`
- Benchmarks: `benchmark/ops/training/bench_mega_moe.py` (MI355X),
  `benchmark/ops/training/bench_mega_moe_gfx1250.py` (MI455X)
