---
name: verify-performance
description: Primus-Turbo performance verification — run single-operator and suite benchmarks, read the latency/TFLOPS metrics, source real-model shapes, and derive a combined training-step metric. Use when measuring latency or throughput of a Primus-Turbo operator.
---

# Verifying Performance

Measure latency and throughput. Benchmarks live in `benchmark/ops/`; shared shapes and helpers in `benchmark/ops/config.py`. Read the hub [`../SKILL.md`](../SKILL.md) first.

## Single-operator benchmark

```bash
cd benchmark/ops
python bench_gemm_turbo.py --dtype bf16
python bench_gemm_turbo.py --dtype fp8 --granularity blockwise   # tensorwise|rowwise|blockwise|mxfp8
python bench_gemm_turbo.py --dtype bf16 -o result.csv

# Force a backend to compare (env beats autotune beats default)
PRIMUS_TURBO_GEMM_BACKEND=CK python bench_gemm_turbo.py --dtype fp8 --granularity blockwise
PRIMUS_TURBO_AUTO_TUNE=1     python bench_gemm_turbo.py --dtype fp8 --granularity blockwise
```

Other ops follow the same `bench_<op>_turbo.py` naming (`bench_attention_turbo.py`, `bench_grouped_gemm_turbo.py`), with `*_torch.py` / `*_te.py` baselines and `bench_deepep_intranode.py`, `bench_symmetric_memory.py`.

## What it measures

Each `bench_*_turbo.py` runs the correctness check first, then profiles with `torch.utils.benchmark.Timer` (20 warmup + 100 timed iters) and writes a CSV row:

| Column | Meaning |
|--------|---------|
| `Check` | correctness gate — `PASS` / `FAIL` / `ERROR` (FAIL invalidates the timing) |
| `Forward Time (ms)` / `Forward TFLOPS` | `2*M*N*K / time / 1e12` |
| `Backward Time (ms)` / `Backward TFLOPS` | `2 * forward FLOPs / time / 1e12` |

The in-bench check uses SNR for fp8/fp4 (`check_gemm_correctness_by_snr`; thresholds 25/20/10 — canonical gate doc is [`../verify-accuracy/SKILL.md`](../verify-accuracy/SKILL.md)) and `allclose` for bf16. A `FAIL` row means the number is not trustworthy — fix correctness first.

## Shapes from real model configs

`benchmark/ops/config.py` derives shapes from real models, so numbers reflect training workloads:
- Dense GEMM: 4 shapes/model (attn QKV, attn out, MLP gate+up, MLP down) × `MBS ∈ {1,2,4}` (Llama-2, Llama-3.1, Qwen2.5, Mistral).
- Grouped GEMM / MoE: `MoEModelConfigs` (DeepSeek-V3/V2, Qwen3-MoE, Mixtral, Kimi-K2, ...) with `gen_grouped_gemm_group_lens(b, m, balance=...)` (`balance=True` → uniform, `balance=False` → skewed) for both token distributions.
- Attention: `gen_attention_test_cases()` (dedup over dense + MoE/MLA heads).

Add shapes for a new op by extending `config.py`, not by hardcoding in the bench script.

## Batch suite

```bash
python benchmark/ops/run_suite.py -d output/                 # all tasks (benchmark_suite.yaml)
python benchmark/ops/run_suite.py -d output/ -g gemm_fp8     # one group
python benchmark/ops/run_suite.py -d output/ -n 4            # 4 GPUs
```

Aggregate with `benchmark/ops/summarize_results.py`.

## Combined training-step metric

For ops optimized as a forward+backward pair, compare the **combined step**, not each direction alone:

```
Combined Step Time (ms) = Forward Time (ms) + Backward Time (ms)
Combined Step TFLOPS    = 6 * M * N * K / (Combined Step Time * 1e-3) / 1e12
```

Use the **geometric mean** of `Combined Step TFLOPS` across shapes as the single comparison score; keep forward/backward TFLOPS as diagnostics.

## End-to-end

For module/model-level impact (a real training step), use `benchmark/pretrain/pytorch/` (entry `pretrain_main.py`, model defs in `models/turbo_llama.py`) — confirm a microbench gain transfers to training.

## Going further

Spot-checking perf is this file. Driving a kernel toward the hardware limit (profiling-guided, accept/rollback rounds) is the optimization loop — hand off via [`../optimize-handoff/SKILL.md`](../optimize-handoff/SKILL.md). Runtime profiling evidence: [`../run_profile/tool-rocprof/SKILL.md`](../run_profile/tool-rocprof/SKILL.md).

## Checklist

```
- [ ] Check=PASS before trusting any timing
- [ ] Shapes from config.py (real model shapes), incl. skew for MoE/grouped
- [ ] Backends compared via PRIMUS_TURBO_*_BACKEND when relevant
- [ ] Combined-step geomean used for fwd+bwd comparison
- [ ] CSV saved (-o / suite output dir) for before/after comparison
```
