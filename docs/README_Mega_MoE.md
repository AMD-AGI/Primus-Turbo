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

In practice $T_{\text{measured}}$ exceeds the ideal time by only ~**0.18–0.43 ms**, which puts
$\eta_{\text{overlap}}$ at **81–95%** across the five stages. Note that $\eta$ is a ratio, not a
wall-clock figure: as the slower leg gets faster the bar it sets rises, so a stage can get faster
in absolute terms and still report a lower $\eta$ (see the comparison table below).

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
- **Measured on:** `feat/mega-moe-dedup-dispatch-combine` @ `19fe104` (combine sub-segment tickets
  + lead-GEMM grid rotation), 2026-08-19
- All latencies are the **max over the 8 ranks**; every number below is one `--iters 30` run.

### dispatch_grouped_gemm

| stage | $T_{\text{comm}}$ (ms) | XGMI | $T_{\text{gemm}}$ (ms) | $T_{\text{measured}}$ (ms) | TFLOP/s | $\eta_{\text{overlap}}$ | speedup vs serial |
| --- | --- | --- | --- | --- | --- | --- | --- |
| forward (nt) | 1.570 | 346.6 GB/s | 3.148 | 3.382 | 1203.9 | 93.1% | 1.39× |
| backward dgrad (nn) | 1.564 | 348.0 GB/s | 1.606 | 1.933 | 1053.4 | 83.1% | 1.64× |
| backward wgrad dW1 (tn) | 1.573 | 345.9 GB/s | 3.337 | 3.770 | 1080.0 | 88.5% | 1.30× |

### grouped_gemm_combine

| stage | $T_{\text{comm}}$ (ms) | XGMI | $T_{\text{gemm}}$ (ms) | $T_{\text{measured}}$ (ms) | TFLOP/s | $\eta_{\text{overlap}}$ | speedup vs serial |
| --- | --- | --- | --- | --- | --- | --- | --- |
| forward (nt) | 1.457 | 373.6 GB/s | 1.732 | 2.143 | 950.2 | 80.8% | 1.49× |
| backward dgrad (nn) | 1.517 | 358.8 GB/s | 3.520 | 3.695 | 1102.1 | 95.3% | 1.36× |

### Compared with the previous measurement

Previous = the figures this section carried before, taken prior to the dispatch push-store rework
(`5950a03` → `02e3012`) and the combine sub-segment/lead-GEMM change.

| operator | stage | $T_{\text{measured}}$ prev → now (ms) | Δ | $T_{\text{comm}}$ prev → now | $T_{\text{gemm}}$ prev → now | $\eta$ prev → now |
| --- | --- | --- | --- | --- | --- | --- |
| dispatch | forward (nt) | 3.56 → **3.38** | **−5.0%** | 2.23 → 1.57 | 3.26 → 3.15 | 91.4% → 93.1% |
| dispatch | backward dgrad (nn) | 2.38 → **1.93** | **−18.8%** | 2.23 → 1.56 | 1.63 → 1.61 | 93.7% → 83.1% |
| dispatch | backward wgrad dW1 (tn) | 3.76 → **3.77** | +0.3% | 2.23 → 1.57 | 3.28 → 3.34 | 87.3% → 88.5% |
| combine | forward (nt) | 2.57 → **2.14** | **−16.6%** | 2.32 → 1.46 | 1.80 → 1.73 | 90.0% → 80.8% |
| combine | backward dgrad (nn) | 3.90 → **3.70** | **−5.3%** | 2.89 → 1.52 | 3.47 → 3.52 | 89.0% → 95.3% |
| **all 5 stages** | **sum** | **16.17 → 14.92** | **−1.25 ms / −7.7%** | 11.90 → 7.68 (−35.5%) | 13.44 → 13.34 (−0.7%) | 90.0% → 89.4% |

The sum row adds the five stages as measured; it is not a full layer time (dW2 and the SwiGLU legs
are not part of either fused operator). The aggregate $\eta$ is $\sum\max(T_{\text{comm}},
T_{\text{gemm}})\,/\,\sum T_{\text{measured}}$, not an average of the per-stage ratios.

**All five stages are now GEMM-bound.** Previously two of them (dispatch dgrad, combine fwd) had
comm as the slower leg; after the push-store rework and the dedup'd combine, $T_{\text{comm}}$ is
below $T_{\text{gemm}}$ everywhere, which is why $\sum\max(\cdot)$ now equals $\sum T_{\text{gemm}}$
exactly. Further gains have to come from the GEMM leg, not from the link.

Two things to read carefully in the comparison table:

- **$\eta$ fell on two stages that got substantially faster.** On dispatch dgrad and combine forward
  the comm leg used to be the slower one, so it set the roofline; now the GEMM leg is slower and the
  bar is much lower (dispatch dgrad 2.23 → 1.61 ms, combine fwd 2.32 → 1.73 ms). Measured against
  that stricter bar the ratio drops even though wall time fell 19% and 17%. The absolute gap to the
  roofline is the honest metric: 0.30/0.15/0.48/0.25/0.43 ms before, 0.23/0.33/0.43/0.41/0.18 ms now.
- **The combine comm leg changed definition.** `combine_only` previously pushed ~1.5× the bytes of
  `dispatch_only` because the standalone baseline did not dedup; it now runs the same sender-side
  dedup as the fused path, so 2.32/2.89 → 1.46/1.52 ms is part real speedup, part a corrected
  baseline. The dispatch comm drop (2.23 → 1.57 ms) has no such caveat — it is the push-store
  rework alone.

`bench_mega_moe.py` no longer measures the dense single-weight GEMM roofline leg (`dense_gemm` /
`grouped/dense`); it was reference-only and cost a full extra timing pass per stage.

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

### End-to-end training (DeepSeek-V3 pretrain)

A/B measured with Primus + Megatron-LM on one node, toggling `use_turbo_mega_moe`. The baseline
leg has since been re-measured with the MoE dense path switched to the **legacy grouped GEMM** and
with **gradient-reduce / param-gather overlap enabled**; both baseline variants are listed below.

#### Environment

| item | value |
| --- | --- |
| Device | 8 × MI355X (`gfx950`), single node, 288 GB HBM each |
| Framework | Primus `main` @ `16472c01` + Megatron-LM |
| Primus-Turbo | `feat/mega-moe-dedup-dispatch-combine` @ `02e3012` |
| Base config | `examples/megatron/configs/MI355X/deepseek_v3-BF16-pretrain.yaml` |
| Date | 2026-08-18 |

#### Model / run configuration

| item | value |
| --- | --- |
| Model | DeepSeek-V3 proxy: `num_layers=4` + `mtp_num_layers=1` |
| Hidden / dense FFN / MoE FFN | 7168 / 18432 / 2048 |
| Attention | MLA, 128 heads, `q_lora_rank=1536`, `kv_lora_rank=512` |
| MoE | 256 experts, top-k 8, `moe_layer_freq=1`, no shared expert |
| Parallelism | TP1 / PP1 / CP1 / **EP8**, distributed optimizer |
| Batch | `seq_length=4096`, `micro_batch_size=2`, `global_batch_size=1024`, `train_iters=15` |
| dtype | BF16 (`fp8=None`) |
| Recompute | disabled (`recompute_granularity=null`) |
| Dispatcher | `alltoall`, `moe_grouped_gemm=True`, `moe_permute_fusion=True` |
| Routing | `moe_router_force_load_balancing_type=uniform` (forced uniform, for perf measurement) |
| Turbo ops on | `use_turbo_deepep`, `use_turbo_rms_norm` |
| Turbo ops off | `use_turbo_attention`, `use_turbo_gemm`, `use_turbo_autotune`, `enable_turbo_attention_float8` |

Two settings differ between the current and the earlier measurement, and are called out per row in
the result table:

| item | earlier | current |
| --- | --- | --- |
| Dense-path grouped GEMM | `use_turbo_grouped_gemm=True`, `moe_use_legacy_grouped_gemm=False` | `use_turbo_grouped_gemm=False`, **`moe_use_legacy_grouped_gemm=True`** |
| Grad/param comm overlap | `overlap_grad_reduce=False`, `overlap_param_gather=False` | **`overlap_grad_reduce=True`, `overlap_param_gather=True`** |

`use_turbo_grouped_gemm=True` together with `moe_use_legacy_grouped_gemm=True` is not a usable
combination — the run aborts before the first iteration.

`overlap_grad_reduce` / `overlap_param_gather` were measured both on and off on the legacy
grouped-GEMM path: **the two flags make no meaningful difference to throughput**, so the baseline
gain below comes from the legacy grouped GEMM alone.

#### Result

Steady-state average over iterations 3–15 (iterations 1–2 are warm-up and excluded by the
throughput extension):

| variant | grouped GEMM | overlap | time/iter (ms) | TFLOP/s/GPU | tokens/s/GPU | tokens/s (8 GPU) | peak mem |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Mega MoE** (`use_turbo_mega_moe=True`) | turbo | off | **17218** | **931.8** | **30449** | **243.6k** | 190.5 GB |
| baseline, current | legacy | on | 19030 | 843.1 | 27551 | 220.4k | 166.3 GB |
| baseline, earlier | turbo | off | 19606 | 818.3 | 26741 | 213.9k | 166.8 GB |

Mega MoE vs the current baseline: **1.105×**; vs the earlier baseline: 1.139×. Switching the dense
path to the legacy grouped GEMM lifts the baseline by 3.0% (818.3 → 843.1 TFLOP/s/GPU).

Best single iteration on the Mega path: 17171 ms/iter, **934.3 TFLOP/s/GPU**, **30533 tokens/s/GPU**.

The Mega figure is config-insensitive on these two flags: re-run with legacy grouped GEMM and
overlap enabled it lands at 931.8 TFLOP/s again (average over iterations 3–10), which is expected
— `use_turbo_mega_moe` replaces the MoE path wholesale, so the dense grouped-GEMM selection only
moves the baseline.

> **Note:** this A/B is a throughput measurement only. In the Mega run the grad norm stays at
> ~4.2e5–5.3e5 and `lm loss` is flat at 1.2015e1 across all 15 iterations (clipped by
> `clip_grad=1.0`), while the baseline runs show grad norm 1.4–25.6 and `lm loss` 11.90 → 9.87.
> Numerics are not yet aligned between the two paths.

#### Reproduce

```bash
# in the Primus repo
export EXP=examples/megatron/configs/MI355X/deepseek_v3-BF16-pretrain.yaml
USE_TURBO_MEGA_MOE=True  bash test_mega_moe.sh   # Mega MoE
USE_TURBO_MEGA_MOE=False bash test_mega_moe.sh   # baseline
```

`test_mega_moe.sh` carries the current settings:

```
--use_turbo_grouped_gemm False --moe_use_legacy_grouped_gemm True \
--overlap_grad_reduce True --overlap_param_gather True
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

## Acknowledgements

- [**Triton-distributed**](https://github.com/ByteDance-Seed/Triton-distributed) (ByteDance-Seed,
  MIT License) — Mega MoE's comm-compute overlapping design (symmetric-memory push, signal/wait
  synchronization, fusing intra-node EP communication into the GEMM kernel) references
  Triton-distributed's overlapping-kernel approach.
- [**DeepGEMM**](https://github.com/deepseek-ai/DeepGEMM) (DeepSeek, MIT License) — Mega MoE's
  cross-rank barrier and symmetric-buffer layout follow DeepGEMM's design; see the file headers of
  `primus_turbo/flydsl/mega/barrier.py` and `primus_turbo/flydsl/mega/symm_buffer.py` for details.
