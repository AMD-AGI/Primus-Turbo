---
name: primus-turbo-develop
description: Build, test, benchmark, backend system, and code structure for the Primus-Turbo project. A reference guide for daily development and operator optimization.
---

# Primus-Turbo Development Guide

## Build

### Prerequisites

- ROCm installed (`ROCM_HOME` defaults to `/opt/rocm`)
- Git submodules initialized (automatically checked during build; can also be done manually with `git submodule sync && git submodule update --init --recursive`)

### Installation

```bash
# Standard install
pip install -r requirements.txt
GPU_ARCHS=gfx942 pip install --no-build-isolation .

# Developer mode install (editable install, recommended for daily development)
pip install -r requirements.txt
GPU_ARCHS=gfx942 pip install --no-build-isolation -e . -v
```

**`pip install -r requirements.txt` must be run before install** because it pins critical dependency versions such as Triton and PyTorch. Skipping it is a common source of environment drift that breaks tests and benchmarks.

`--no-build-isolation` is required; otherwise, the build environment will lack already-installed dependencies (e.g., triton, torch).

**Difference between the two**: `pip install .` copies the code into site-packages — source changes have no effect. `pip install -e .` is editable mode — Python code (including Triton kernels) takes effect immediately after modification, with no reinstall needed. C++ extensions require compilation in both modes.

```bash
# Other architectures
GPU_ARCHS=gfx950 pip install --no-build-isolation -e . -v          # MI350X (gfx950)
GPU_ARCHS="gfx942;gfx950" pip install --no-build-isolation -e . -v # Multiple architectures
GPU_ARCHS=native pip install --no-build-isolation -e . -v           # Auto-detect current GPU
```

### Key Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `GPU_ARCHS` | Target GPU architecture (`gfx942`, `gfx950`, `native`; semicolon-separated for multiple) | Auto-detect current GPU |
| `ROCM_HOME` | ROCm installation path | `/opt/rocm` |
| `MAX_JOBS` | Parallel compilation threads | `64` |
| `PRIMUS_TURBO_FRAMEWORK` | Which frontends to build (`PYTORCH`, `JAX`; semicolon-separated) | `PYTORCH` |

### Build Artifacts

The build is split into three layers, decoupling the kernel library from frontend bindings:

| Artifact | Source | Description |
|----------|--------|-------------|
| `libprimus_turbo_kernels.so` | All C++/HIP code under `csrc/kernels/` | Shared kernel library containing all backend implementations (CK, hipBLASLt, turbo, etc.), frontend-agnostic |
| `primus_turbo.pytorch._C` | `csrc/pytorch/` | PyTorch bindings, linked against `libprimus_turbo_kernels` |
| `primus_turbo.jax._C` | `csrc/jax/` | JAX bindings (requires `PRIMUS_TURBO_FRAMEWORK=JAX`), also linked against `libprimus_turbo_kernels` |

Install dependencies before building: `pip install -r requirements.txt`.

The build process also automatically installs `amd-aiter` and `origami` (pinned commits; see `setup.py`).

---

## Testing

### Common Commands

```bash
# GEMM tests
pytest tests/pytorch/ops/test_gemm.py -v       # BF16/FP16/FP32
pytest tests/pytorch/ops/test_gemm_fp8.py -v    # FP8 all granularities
pytest tests/pytorch/ops/test_gemm_fp4.py -v    # FP4

# Filter by granularity (-k matches keywords in test function names)
pytest tests/pytorch/ops/test_gemm_fp8.py -v -k "blockwise"
pytest tests/pytorch/ops/test_gemm_fp8.py -v -k "tensorwise"

# Determinism tests
pytest tests/pytorch/ops/test_gemm.py -v -k "deterministic"

# Grouped GEMM
pytest tests/pytorch/ops/test_grouped_gemm.py -v
pytest tests/pytorch/ops/test_grouped_gemm_fp8.py -v
```

### Correctness Criteria

| Type | Method | Threshold |
|------|--------|-----------|
| BF16 / FP16 | `torch.allclose` | `rtol=1e-2, atol=1e-2` |
| FP32 | `torch.allclose` | `rtol=1e-4, atol=1e-4` |
| FP8 E4M3 | SNR (signal-to-noise ratio) | >= 25 dB |
| FP8 E5M2 | SNR | >= 20 dB |
| FP4 | SNR | >= 10 dB |
| Determinism | Bitwise comparison over 10 runs | `rtol=0, atol=0` |

### Shape Coverage in Tests

BF16 GEMM: `m in {1,16,128,256,512,1024,2048}` x `n in {1,16,129,512,1024,2048,4096}` x `k in {1,16,127,255,512,1024,2048}` x `layout in {TN,NN,NT}`

FP8 blockwise: smaller m/n/k grid, `block_size=128`, layout NT/NN, backends TRITON/CK.

### Running Tests for a Specific Backend

Backends are parameterized via `@pytest.mark.parametrize("backend", [...])` in the test files, generating independent test cases for each backend. Use `-k` to match backend names in the test ID:

```bash
# Run only Triton backend tests
pytest tests/pytorch/ops/test_gemm_fp8.py -v -k "TRITON"

# Run only CK backend blockwise tests
pytest tests/pytorch/ops/test_gemm_fp8.py -v -k "blockwise and CK"

# Run only BF16 GEMM Triton backend tests
pytest tests/pytorch/ops/test_gemm.py -v -k "TRITON"
```

Note: `-k` uses pytest expression syntax, matching strings within the test ID; `and` means both must be present.

---

## Benchmark

### Single-Operator Benchmark

```bash
cd benchmark/ops

# BF16 GEMM
python bench_gemm_turbo.py --dtype bf16

# FP8 GEMM (granularity options: tensorwise / rowwise / blockwise / mxfp8)
python bench_gemm_turbo.py --dtype fp8 --granularity tensorwise

# Compare specific backends
PRIMUS_TURBO_GEMM_BACKEND=TRITON python bench_gemm_turbo.py --dtype fp8 --granularity blockwise
PRIMUS_TURBO_GEMM_BACKEND=CK    python bench_gemm_turbo.py --dtype fp8 --granularity blockwise

# Enable autotune
PRIMUS_TURBO_AUTO_TUNE=1 python bench_gemm_turbo.py --dtype fp8 --granularity blockwise

# Output to a specific CSV
python bench_gemm_turbo.py --dtype bf16 -o result.csv
```

### CLI Arguments

| Argument | Options | Description |
|----------|---------|-------------|
| `--dtype` | `bf16`, `fp8`, `fp4` | Data type |
| `--granularity` | `tensorwise`, `rowwise`, `blockwise`, `mxfp8`, `mxfp4` | Quantization granularity for FP8/FP4 |
| `-o` / `--output` | filename | Output CSV path |

### Benchmark Shape Source

Benchmarks use real model configurations to generate GEMM shapes. Each model produces 4 shape groups (attn QKV, attn out, MLP gate+up, MLP down), then scales the M dimension by `MBS in {1,2,4}`.

Covered models: Llama-2-7B, Llama-2-70B, Llama-3.1-8B, Llama-3.1-405B, Qwen2.5-7B, Qwen2.5-72B, Mistral-7B.

### Output Metrics

| Column | Meaning |
|--------|---------|
| Forward Time (ms) | Average forward pass latency (100 iterations, 20 warmup) |
| Forward TFLOPS | `2*M*N*K / time / 1e12` |
| Backward Time (ms) | Average backward pass latency |
| Backward TFLOPS | `2 * forward FLOPs / time / 1e12` |
| Check | Correctness PASS/FAIL |

### Suite Batch Run

```bash
# Run all benchmarks
python3 benchmark/ops/run_suite.py -d output/

# Run a specific group only
python3 benchmark/ops/run_suite.py -d output/ -g gemm_fp8

# Specify GPU count
python3 benchmark/ops/run_suite.py -d output/ -n 4
```

Suite configuration is in `benchmark/ops/benchmark_suite.yaml`. Each task includes label, group, script, args, env, and output.

---

## Backend System

### Core Components

Two core classes in `primus_turbo/pytorch/core/backend.py`:

**`GlobalBackendManager`**: Global backend manager that selects backends by operator type x precision type.

**`AutoKernelDispatcher`**: Base dispatcher class for each operator, supporting autotune, default backend, and fallback.

### Backend Priority (highest to lowest)

1. Programmatic setting (`GlobalBackendManager.set_gemm_backend(...)`)
2. Environment variable (`PRIMUS_TURBO_GEMM_BACKEND`)
3. Autotune (`PRIMUS_TURBO_AUTO_TUNE=1`)
4. In-code default
5. Fallback: tries all registered backends in order

### Environment Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `PRIMUS_TURBO_GEMM_BACKEND` | Select GEMM backend | `TRITON`, `CK`, `HIPBLASLT` |
| `PRIMUS_TURBO_GROUPED_GEMM_BACKEND` | Select grouped GEMM backend | Same as above |
| `PRIMUS_TURBO_AUTO_TUNE` | Enable autotune | `1` |

Per-precision backend selection: `PRIMUS_TURBO_GEMM_BACKEND="fp8:CK,other:TRITON"`

### Available GEMM Backends

| Backend | BF16 | FP8 tensor/row/block | FP8 MX | Notes |
|---------|------|----------------------|--------|-------|
| `HIPBLASLT` | Y | Y (tensorwise) | — | hipBLASLt library, typically the default |
| `TRITON` | Y | Y (all) | — | Triton kernel, tunable |
| `CK` | — | Y (all) | — | Composable Kernel |
| `TURBO` | — | — | Y (gfx950) | In-house MX-series kernels |

### Resetting Backend State

```python
from primus_turbo.pytorch.core.backend import GlobalBackendManager
GlobalBackendManager.reset()  # Clear all backend settings and autotune cache
```

---

## Code Structure Quick Reference

```
primus_turbo/
├── triton/                      # Triton kernel implementations
│   ├── gemm/                    #   gemm_kernel.py (BF16), gemm_fp8_kernel.py (FP8)
│   ├── grouped_gemm/            #   Grouped GEMM variants
│   ├── attention/               #   Attention kernels
│   ├── activation/              #   swiglu, geglu
│   ├── moe/                     #   MoE routing, permutation
│   ├── quantization/            #   blockwise/mxfp quantization
│   └── utils/                   #   Triton helpers
├── pytorch/
│   ├── core/                    #   backend.py, stream.py, low_precision.py
│   ├── ops/                     #   Python API (gemm.py, gemm_fp8.py, ...)
│   ├── kernels/                 #   Dispatchers (gemm_impl.py, gemm_fp8_impl.py, ...)
│   └── modules/                 #   High-level modules
csrc/
├── kernels/                     # -> libprimus_turbo_kernels.so (shared kernel library)
│   ├── gemm/                    #   CK / hipBLASLt / turbo backends + arch specializations
│   └── ...                      #   grouped_gemm/, attention/, quantization/, etc.
├── pytorch/                     # -> primus_turbo.pytorch._C (PyTorch bindings, links the .so above)
├── jax/                         # -> primus_turbo.jax._C (JAX bindings, links the .so above)
└── include/                     # Public headers (gemm.h, float8.h, etc.)
tests/pytorch/ops/               # Operator tests
benchmark/ops/                   # Operator benchmarks
```

### Key GEMM-Related Files

| Layer | BF16 | FP8 |
|-------|------|-----|
| Triton kernel | `primus_turbo/triton/gemm/gemm_kernel.py` | `primus_turbo/triton/gemm/gemm_fp8_kernel.py` |
| Dispatcher / backend registration | `primus_turbo/pytorch/kernels/gemm/gemm_impl.py` | `primus_turbo/pytorch/kernels/gemm/gemm_fp8_impl.py` |
| Python API | `primus_turbo/pytorch/ops/gemm.py` | `primus_turbo/pytorch/ops/gemm_fp8.py` |
| Tests | `tests/pytorch/ops/test_gemm.py` | `tests/pytorch/ops/test_gemm_fp8.py` |
| Benchmark | `benchmark/ops/bench_gemm_turbo.py` | Same as left (`--dtype fp8`) |

---

## Operator Optimization

This section is the agent's **first stop** when receiving an operator optimization request.

### Environment Verification

Before optimization, check the local install with `pip show primus_turbo`:

- Not installed: follow [Build](#build) and do a full editable install
- Installed but not editable, or not pointing at this repository: reinstall in editable mode
- Installed and correct: still confirm the pinned dependencies from `requirements.txt` are consistent with the local environment

### Process Overview

1. **Verify the environment** (see [Environment Verification](#environment-verification))
2. **Read `kernel-optimize/SKILL.md`'s "Prerequisite Information"** to understand what the optimization framework needs from the project
3. **Collect project information per the requirement checklist** (from sections in this file)
   - Kernel source file path: look up in [Code Structure](#code-structure-quick-reference) and [Key GEMM-Related Files](#key-gemm-related-files)
   - Focused test command: assemble from [Testing](#testing)
   - Focused benchmark command: assemble from [Benchmark](#benchmark)
   - Quick validation script template: use [Quick Validation](#quick-validation)
   - Benchmark output format and available performance metrics: look up in [Output Metrics](#output-metrics)
   - Scoring rules: from [Optimization Scoring](#optimization-scoring) below
   - `execution_mode` recommendation and rebuild requirements: from [Optimization Environment](#optimization-environment) below
4. **Hand off to `kernel-optimize/SKILL.md`** with the above information; it autonomously completes the optimization loop and outputs a report
5. **Acceptance**: After optimization is complete, return to the project perspective for final confirmation
   - Run **full tests** in the main repo (not just focused tests) to ensure no regressions: `pytest tests/pytorch/ops/ -v`
   - Review the optimization report: whether performance improvement meets targets, whether changes are reasonable, whether there are residual risks
   - Confirm code is correctly committed to the main repo with a clean diff

### Optimization Environment

The agent decides whether to use `repo-mode` (modify directly in the repository) or `workspace-mode` (set up a minimal development environment, then integrate back after optimization) based on task characteristics. Here are Primus-Turbo's recommendations:

- **Triton kernel**: Typically `repo-mode`. Under editable install, changes take effect immediately with no rebuild needed, enabling fast iteration.
- **HIP / C++ kernel**: Depends on the situation. For small scope changes focused on parameter tuning, `repo-mode` works (incremental rebuild: `GPU_ARCHS=<arch> pip install --no-build-isolation -e . -v`). For extensive trial-and-error or writing a new kernel from scratch, `workspace-mode` may be more appropriate to avoid running the main repo's full build pipeline each time.

The setup specification for `workspace-mode` is defined in `kernel-optimize/SKILL.md`.

### Optimization Scoring

Performance metrics depend on the operator type and are determined by the benchmark output columns. Common metrics:

| Operator Type | Typical Metrics |
|--------------|-----------------|
| Compute-bound (forward only) | `Forward TFLOPS` |
| Compute-bound (forward + backward training path) | `Combined Step TFLOPS` (derived), with `Forward TFLOPS` / `Backward TFLOPS` retained as diagnostics |
| Memory-bound (elementwise, quantization, etc.) | `Forward GB/s`, `Backward GB/s` |

The user confirms which metrics to optimize (`primary_metric`) at the start of optimization.

For GEMM campaigns where the user wants both forward and backward optimized together, derive:
- `Combined Step Time (ms) = Forward Time (ms) + Backward Time (ms)`
- `Combined Step TFLOPS = 6 * M * N * K / (Combined Step Time (ms) * 1e-3) / 1e12`

Use the **geometric mean** of `Combined Step TFLOPS` across all target shapes as the `aggregate score` for accept/rollback. Keep `Forward TFLOPS` and `Backward TFLOPS` in the per-round report as diagnostics, but do not require them to improve independently if the combined training step metric improves.

If the campaign is forward-only or backward-only, use the corresponding direct metric (`Forward TFLOPS`, `Backward TFLOPS`, etc.) as the `aggregate score`.

Any shape with `Check = FAIL` results in immediate rejection of that candidate.

For benchmark output format, shape sources, and measurement methodology, see the [Benchmark](#benchmark) section above.

### Quick Validation

Each VALIDATE round defaults to quick validation (representative shape subset) to accelerate iteration. BASELINE and final acceptance use full validation.

**Representative shape selection criteria**: During the BASELINE phase, the agent selects 3-5 representative shapes from the full benchmark results, covering small/medium/large scales. Selection criteria:
- Select only shapes whose correctness check passed in the full benchmark
- Include at least 1 small shape (prone to exposing launch overhead issues)
- Include at least 1 large shape (prone to exposing memory / compute bottlenecks)
- Prefer shapes with large performance variance (more sensitive, can quickly detect regressions)

The selected shape list is recorded in the `representative_shapes` field of `manifest.yaml`.

**Implementation pattern**: During PREPARE_ENVIRONMENT, generate `quick_test_bench.py` in the campaign directory while the project API context is still fresh. Leave `SHAPES` empty at first, then fill it after BASELINE chooses `representative_shapes`.

**Measurement-consistency contract**: BASELINE (round-1) and every subsequent VALIDATE read the same authoritative CSV produced by this script via `--summary-csv`. The CSV schema MUST match the canonical Primus-Turbo columns so `mcp__turbo__parse_bench_csv` parses round-1 and round-N identically. The mandatory columns are:

```
label,B,M,N,K,Check,Forward TFLOPS,Forward TFLOPS_stddev,Backward TFLOPS,Backward TFLOPS_stddev,Forward Time (ms),Backward Time (ms),out_snr,da_snr,db_snr
```

`Check` uses the strings `PASS` / `FAIL`; `*_stddev` values are absolute stddev in the metric's own unit (the scorer converts to %).

The FP8 blockwise GEMM template below is the default reference. Adapt it for other operators as needed — but keep the `--summary-csv` writer and column schema verbatim:

```python
"""Quick correctness + benchmark for representative shapes.
Auto-generated during PREPARE_ENVIRONMENT.

Usage:
    python quick_test_bench.py [--repeats N] [--iters-per-repeat M]
        [--summary-csv PATH] [--csv PATH]

`--summary-csv` is the authoritative per-round CSV consumed by BASELINE
and every VALIDATE via `mcp__turbo__parse_bench_csv`. Its schema MUST
match the canonical Primus-Turbo columns so round-1 and round-N are
directly comparable. See the project skill's "Measurement-consistency
contract" for the column list.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
import sys
import time
from typing import Any, Dict, List

os.environ.setdefault("PRIMUS_TURBO_GEMM_BACKEND", "<target_backend>")

import torch  # noqa: E402

import primus_turbo.pytorch as turbo  # noqa: E402
from primus_turbo.pytorch.core.low_precision import (  # noqa: E402
    Float8QuantConfig,
    Format,
    ScalingGranularity,
)


SHAPES: List[Dict[str, Any]] = [
    # Fill from representative_shapes in manifest after BASELINE
]

CONFIG = Float8QuantConfig(
    format=Format.E4M3,
    granularity=ScalingGranularity.BLOCKWISE,
    block_size=128,
)
SNR_THRESHOLD = 25.0

# Canonical CSV header used by BASELINE and every VALIDATE. Do NOT
# reorder or rename — mcp__turbo__parse_bench_csv reads these columns
# verbatim and any drift silently disables the per-shape regression
# gate between rounds.
SUMMARY_CSV_HEADER: List[str] = [
    "label", "B", "M", "N", "K", "Check",
    "Forward TFLOPS", "Forward TFLOPS_stddev",
    "Backward TFLOPS", "Backward TFLOPS_stddev",
    "Forward Time (ms)", "Backward Time (ms)",
    "out_snr", "da_snr", "db_snr",
]


def compute_snr(ref: torch.Tensor, test: torch.Tensor) -> float:
    noise = test.float() - ref.float()
    signal_power = ref.float().norm() ** 2
    noise_power = noise.norm() ** 2
    if noise_power.item() == 0.0:
        return float("inf")
    return float(10 * torch.log10(signal_power / noise_power).item())


def _timed_ms(fn, iters: int) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def run_one(shape: Dict[str, Any], repeats: int, iters_per_repeat: int) -> Dict[str, Any]:
    torch.manual_seed(0)  # stable inputs across rounds
    M = int(shape["M"]); N = int(shape["N"]); K = int(shape["K"])
    dtype = shape.get("dtype", torch.bfloat16)
    label = str(shape.get("label", f"M{M}_N{N}_K{K}"))
    device = "cuda"

    a = torch.randn(M, K, dtype=dtype, device=device, requires_grad=True)
    b = torch.randn(N, K, dtype=dtype, device=device, requires_grad=True)

    out = turbo.ops.gemm_fp8(a, b, trans_b=True, config=CONFIG)
    grad_out = torch.randn_like(out)
    with torch.no_grad():
        ref = a.detach().float() @ b.detach().float().T
    out_snr = compute_snr(ref.to(dtype), out.detach())
    a_ref = a.detach().clone().requires_grad_(); b_ref = b.detach().clone().requires_grad_()
    (a_ref.float() @ b_ref.float().T).backward(grad_out.float())
    out.backward(grad_out, retain_graph=True)
    da_snr = compute_snr(a_ref.grad, a.grad); db_snr = compute_snr(b_ref.grad, b.grad)
    a.grad = None; b.grad = None
    correct = all(s > SNR_THRESHOLD for s in (out_snr, da_snr, db_snr))

    fwd_fn = lambda: turbo.ops.gemm_fp8(a, b, trans_b=True, config=CONFIG)
    out_for_bwd = fwd_fn()
    bwd_fn = lambda: out_for_bwd.backward(grad_out, retain_graph=True)

    fwd_ms_samples, bwd_ms_samples = [], []
    total_repeats = max(2, repeats + 1)  # always have >=1 warm-up + >=1 timed
    for r in range(total_repeats):
        fwd_ms = _timed_ms(fwd_fn, iters_per_repeat)
        bwd_ms = _timed_ms(bwd_fn, iters_per_repeat)
        if r == 0:
            continue  # warm-up
        fwd_ms_samples.append(fwd_ms); bwd_ms_samples.append(bwd_ms)

    fwd_flops = 2.0 * M * N * K; bwd_flops = 2.0 * fwd_flops
    fwd_tflops_samples = [fwd_flops / (t * 1e-3) / 1e12 for t in fwd_ms_samples]
    bwd_tflops_samples = [bwd_flops / (t * 1e-3) / 1e12 for t in bwd_ms_samples]
    def mean_std(xs):
        m = statistics.fmean(xs); s = statistics.pstdev(xs) if len(xs) > 1 else 0.0
        return m, s
    fwd_ms_mean, _ = mean_std(fwd_ms_samples)
    bwd_ms_mean, _ = mean_std(bwd_ms_samples)
    fwd_tflops_mean, fwd_tflops_std = mean_std(fwd_tflops_samples)
    bwd_tflops_mean, bwd_tflops_std = mean_std(bwd_tflops_samples)
    return {
        "label": label, "B": 1, "M": M, "N": N, "K": K,
        "Check": "PASS" if correct else "FAIL",
        "Forward TFLOPS": fwd_tflops_mean,
        "Forward TFLOPS_stddev": fwd_tflops_std,
        "Backward TFLOPS": bwd_tflops_mean,
        "Backward TFLOPS_stddev": bwd_tflops_std,
        "Forward Time (ms)": fwd_ms_mean,
        "Backward Time (ms)": bwd_ms_mean,
        "out_snr": out_snr, "da_snr": da_snr, "db_snr": db_snr,
    }


def _failed_row(shape: Dict[str, Any], reason: str) -> Dict[str, Any]:
    """Emit a canonical-schema FAIL row so BASELINE still sees the shape."""
    return {
        "label": str(shape.get("label", "?")),
        "B": int(shape.get("B", 1)), "M": int(shape.get("M", 0)),
        "N": int(shape.get("N", 0)), "K": int(shape.get("K", 0)),
        "Check": "FAIL",
        "Forward TFLOPS": float("nan"), "Forward TFLOPS_stddev": float("nan"),
        "Backward TFLOPS": float("nan"), "Backward TFLOPS_stddev": float("nan"),
        "Forward Time (ms)": float("nan"), "Backward Time (ms)": float("nan"),
        "out_snr": float("nan"), "da_snr": float("nan"), "db_snr": float("nan"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--iters-per-repeat", type=int, default=50)
    ap.add_argument("--summary-csv", type=str, default=None)
    ap.add_argument("--csv", type=str, default=None)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA / ROCm device required.", file=sys.stderr); return 2

    rows: List[Dict[str, Any]] = []
    for s in SHAPES:
        try:
            rows.append(run_one(s, args.repeats, args.iters_per_repeat))
        except Exception as e:  # noqa: BLE001
            print(f"{s.get('label', str(s))}: ERROR {e}", file=sys.stderr)
            rows.append(_failed_row(s, str(e)))

    print(f"{'label':<22} {'Check':<5} {'Fwd TFLOPS':>12} {'Bwd TFLOPS':>12}")
    for r in rows:
        print(f"{r['label']:<22} {r['Check']:<5} "
              f"{r['Forward TFLOPS']:>12.2f} {r['Backward TFLOPS']:>12.2f}")

    pass_fwd = [r["Forward TFLOPS"] for r in rows if r["Check"] == "PASS"]
    pass_bwd = [r["Backward TFLOPS"] for r in rows if r["Check"] == "PASS"]
    def geomean(xs):
        xs = [x for x in xs if x > 0 and not math.isnan(x)]
        return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else 0.0
    print(f"Geomean Fwd TFLOPS: {geomean(pass_fwd):.2f}")
    print(f"Geomean Bwd TFLOPS: {geomean(pass_bwd):.2f}")

    if args.summary_csv:
        with open(args.summary_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=SUMMARY_CSV_HEADER)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in SUMMARY_CSV_HEADER})
        print(f"Summary CSV: {args.summary_csv}")

    all_pass = all(r["Check"] == "PASS" for r in rows)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
```

Record the script invocation in the manifest as `quick_command: "python <campaign_dir>/quick_test_bench.py"`. BASELINE and every VALIDATE round append `--summary-csv <round_dir>/artifacts/benchmark.csv` at call time.

### Demo: Optimizing blockwise FP8 GEMM Triton on MI300X

The following uses blockwise FP8 GEMM Triton on MI300X as an example.

**1. Read `kernel-optimize/SKILL.md`'s prerequisite information**

Learn that the optimization framework needs: kernel source file, focused test, focused benchmark, quick validation script template, benchmark output format, scoring rules, execution_mode recommendation, and rebuild requirements.

**2. Collect project information per the requirement checklist**

Gather from sections in this file:
- Kernel source file: `primus_turbo/triton/gemm/gemm_fp8_kernel.py` (see [Key GEMM-Related Files](#key-gemm-related-files))
- Focused test: `pytest tests/pytorch/ops/test_gemm_fp8.py -v -k "blockwise and TRITON"` (see [Testing](#testing))
- Focused benchmark: `PRIMUS_TURBO_GEMM_BACKEND=TRITON python benchmark/ops/bench_gemm_turbo.py --dtype fp8 --granularity blockwise` (see [Benchmark](#benchmark))
- Quick validation template: generate `quick_test_bench.py` during PREPARE_ENVIRONMENT, then fill representative shapes after BASELINE (see [Quick Validation](#quick-validation))
- Available metrics: `Forward TFLOPS`, `Backward TFLOPS`, `Check` as correctness gate (see [Optimization Scoring](#optimization-scoring))
- Environment recommendation: Triton → `repo-mode`, no rebuild needed (see [Optimization Environment](#optimization-environment))

**3. Hand off to `kernel-optimize/SKILL.md`**

With the above information, it completes DEFINE_TARGET (including target confirmation with user), PREPARE_ENVIRONMENT, the optimization loop, and outputs a report.

**4. Acceptance**

Return to the project perspective for final confirmation:
- Run full tests in the main repo (not just focused tests): `pytest tests/pytorch/ops/ -v`
- Review the optimization report: whether performance improvement meets targets, whether changes are reasonable
- Confirm code is correctly committed (if `git_commit=true`) with a clean diff

---

## Related Skills

- **Operator performance optimization**: `kernel-optimize/SKILL.md` (optimization loop, workflow, Triton/HIP strategies)

## Additional References

> All paths below are relative to the repository root.

- `README.md`: Quick Start, installation, testing, packaging, Minimal Example
- `docs/examples.md`: API usage for each operator (GEMM, Attention, Grouped GEMM, FP8/FP4, Backend/AutoTune)
- `docs/install_dependencies.md`: rocSHMEM (DeepEP dependency) installation guide
- `benchmark/README.md`: DeepEP benchmark instructions
- `CONTRIBUTING.md`: Branch naming and commit conventions
