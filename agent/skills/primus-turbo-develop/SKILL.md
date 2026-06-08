---
name: primus-turbo-develop
description: Develop, debug, and validate Primus-Turbo operators and modules on AMD GPUs. Covers the layered architecture (ops / kernels-dispatcher / Triton / HIP-CK csrc / modules), how to add or change a feature end-to-end, accuracy verification (SNR, tolerances, reference implementations), performance benchmarking, the backend dispatch system, and build/test/bench commands. Use for any Primus-Turbo development task (GEMM, Attention, GroupedGEMM, MoE, quantization, normalization, activation) and for accuracy or performance validation.
---

# Primus-Turbo Development Guide

Primus-Turbo is a high-performance training-operator library for AMD GPUs (gfx942 / gfx950). This skill is the entry point for **developing a feature** and **validating its accuracy and performance**.

## Pick Your Task

| Goal | Start here |
|------|-----------|
| Add or change an operator / module end-to-end | [develop-feature/SKILL.md](develop-feature/SKILL.md) |
| Verify numerical correctness | [verify-accuracy/SKILL.md](verify-accuracy/SKILL.md) |
| Measure latency / throughput | [verify-performance/SKILL.md](verify-performance/SKILL.md) |
| Drive an autonomous kernel-optimization campaign | [optimize-handoff/SKILL.md](optimize-handoff/SKILL.md) → `kernel-optimize/SKILL.md` |
| Profile a hot kernel with runtime evidence | [run_profile/tool-rocprof/SKILL.md](run_profile/tool-rocprof/SKILL.md) |

The rest of this file is the shared reference (build, architecture, code map, backend system) those documents rely on. Read it once, then jump to the task-specific file. (`run_profile/` is a category folder for profiling tools and currently holds the single `tool-rocprof/` skill.)

## Build & Iterate

### Install

```bash
# Always install pinned deps first (Triton / PyTorch versions are critical).
pip install -r requirements.txt

# Editable install (recommended for development).
GPU_ARCHS=gfx942 pip install --no-build-isolation -e . -v
```

- `GPU_ARCHS`: `gfx942` (MI300X/MI325X), `gfx950` (MI350X/MI355X), `native` (auto-detect), or `"gfx942;gfx950"`.
- `--no-build-isolation` is required so the build sees already-installed `torch` / `triton`.
- `pip install .` copies into site-packages (source edits have no effect); `pip install -e .` is editable.
- The build also auto-installs pinned `amd-aiter` and `origami` (see `setup.py`).

### What needs a rebuild

| You changed | Rebuild needed? |
|-------------|-----------------|
| Python / Triton (`primus_turbo/**.py`) | No (editable install picks it up immediately) |
| C++ / HIP (`csrc/**`) or op schema in `bindings_pytorch.cpp` | Yes: re-run the editable install command |

Arch-specialized sources are filtered by filename suffix: `*_gfx942.{cu,hip}` and `*_gfx950.{cu,hip}` compile only when that arch is in `GPU_ARCHS` (see `setup.py: filter_files_by_arch`).

### Build artifacts (3 layers, kernel lib decoupled from frontends)

| Artifact | Source | Notes |
|----------|--------|-------|
| `libprimus_turbo_kernels.so` | `csrc/kernels/` | All HIP/CK/hipBLASLt/turbo kernels, frontend-agnostic |
| `primus_turbo.pytorch._C` | `csrc/pytorch/` | PyTorch bindings, links the `.so` above |
| `primus_turbo.jax._C` | `csrc/jax/` | JAX bindings (`PRIMUS_TURBO_FRAMEWORK=JAX`) |

### Key environment variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `GPU_ARCHS` | Target arch(s) | auto-detect |
| `ROCM_HOME` | ROCm path | `/opt/rocm` |
| `MAX_JOBS` | Parallel compile jobs | `64` |
| `PRIMUS_TURBO_FRAMEWORK` | `PYTORCH` / `JAX` (`;`-separated) | `PYTORCH` |
| `PRIMUS_TURBO_LOG_LEVEL` | Logger level | `WARNING` |

Verify a working editable install with `pip show primus_turbo` (look for the `Editable project location` field) or `python -c "import primus_turbo; print(primus_turbo.__file__)"` (it should point into this source tree, not site-packages); reinstall in editable mode if not.

## Architecture: the layered design

Every operator is a vertical slice through these layers. Knowing which layer to touch is the core of fast development.

```
modules/   nn.Module wrappers (Linear, GroupedLinear, Attention, ...)
   │ calls
ops/       Python API + torch.autograd.Function (forward/backward)        ← user-facing
   │ calls
kernels/   AutoKernelDispatcher + KernelBackend  (selects a backend)      ← multi-backend ops
   │ dispatches to
triton/    Triton kernels (Python, no rebuild)        csrc/  HIP/CK/hipBLASLt kernels (rebuild)
                                                          │ bound via TORCH_LIBRARY in csrc/pytorch/
                                                          └→ torch.ops.primus_turbo_cpp_extension.*
```

Three op wiring patterns exist (see [develop-feature/SKILL.md](develop-feature/SKILL.md) for the canonical file to copy for each):
- **Multi-backend** (`gemm`, `gemm_fp8`, `grouped_gemm`): `ops` → `kernels` dispatcher → Triton **and/or** csrc.
- **Direct C++** (`rmsnorm`): `ops` autograd Function → `torch.ops.primus_turbo_cpp_extension.*` (no dispatcher).
- **Direct Triton** (`swiglu_with_probs`): `ops` autograd Function → a Triton-backed helper in `kernels/` (no dispatcher).

## Code Map

> Column roots differ: `ops/` = `primus_turbo/pytorch/ops/`, `kernels/` = `primus_turbo/pytorch/kernels/`, `triton/` = `primus_turbo/triton/` (top level, **not** under `pytorch/`); `csrc/kernels/`, `tests/`, `benchmark/` are repo-root. The user API lives in `ops/`; the `kernels/` layer holds the dispatcher or the backend impl (`*_impl.py`).

| Operator family | API (`ops/`) | `kernels/` (dispatcher/impl) | Triton (`triton/`) | C++/HIP (`csrc/kernels/`) | Tests | Bench |
|-----------------|--------------|--------------------------|--------------------|----------------------------|-------|-------|
| GEMM (bf16/fp16/fp32) | `gemm.py` | `gemm/gemm_impl.py` | `gemm/gemm_kernel.py` | `gemm/` (hipBLASLt) | `ops/test_gemm.py` | `bench_gemm_turbo.py` |
| GEMM FP8 | `gemm_fp8.py` | `gemm/gemm_fp8_impl.py` | `gemm/gemm_fp8_kernel.py` | `gemm/ck`, `gemm/turbo` | `ops/test_gemm_fp8.py` | `bench_gemm_turbo.py --dtype fp8` |
| GEMM FP4 | `gemm_fp4.py` | `gemm/gemm_fp4_impl.py` | — | `gemm/` (hipBLASLt fp4) | `ops/test_gemm_fp4.py` | `bench_gemm_turbo.py --dtype fp4` |
| Grouped GEMM | `grouped_gemm.py`, `grouped_gemm_fp8.py` | `grouped_gemm/` | `grouped_gemm/` | `grouped_gemm/ck` | `ops/test_grouped_gemm*.py` | `bench_grouped_gemm_turbo.py` |
| Attention | `attention/` | `attention/` (aiter/triton/turbo) | `attention/attention_kernel.py` | `attention/turbo` | `ops/test_attention*.py` | `bench_attention_turbo.py` |
| Quantization | `quantization.py` | `quantization/` | `quantization/` | `quantization/` | `ops/test_quantization.py` | `accuracy/eval_sf_accuracy.py` |
| Activation (swiglu/geglu) | `activation.py` | `activation/` | `activation/` | — | `ops/test_activation.py` | — |
| Normalization (rmsnorm) | `normalization.py` | — (direct C++) | — | `normalization/` | `ops/test_normalization.py` | — |
| MoE permute / router / dispatch | `moe/` | `moe/`, `moe_permute` | `moe/` | `moe_permute/`, `deep_ep/` | `ops/test_moe_permute.py`, `test_fused_moe_router.py` | `bench_deepep_intranode.py` |
| Async-TP (gemm+comm) | `async_tp.py` | `async_tp/` | `async_tp/` | — | `ops/test_fused_*.py` | — |

C++ binding plumbing (shared by all csrc ops): declarations in `csrc/pytorch/extensions.h`, schema + `CUDA` + `Meta` registration in `csrc/pytorch/bindings_pytorch.cpp`.

## Backend System

Two classes in `primus_turbo/pytorch/core/backend.py`:
- **`GlobalBackendManager`** — global selection by operator × precision.
- **`AutoKernelDispatcher`** — per-operator base with autotune, default, and fallback.

Selection priority (high → low): code setter → env var → autotune → in-code default → fallback (try all `can_handle`).

| `BackendType` | Used by | Notes |
|---------------|---------|-------|
| `HIPBLASLT` | GEMM (bf16, fp8 tensorwise) | default for dense GEMM |
| `TRITON` | GEMM, GroupedGEMM, Attention, ... | tunable, no rebuild |
| `CK` | GEMM/GroupedGEMM FP8 (row/block) | Composable Kernel |
| `TURBO` | MXFP8/MXFP4 GEMM, Attention | in-house (gfx950) |
| `AITER` | Attention | default attention |
| `DEEP_EP` | MoE dispatch/combine | needs DeepEP install |
| `FLYDSL` *(planned)* | GEMM / Attention (authoring DSL) | upcoming AMD tile-DSL backend; **not yet a `BackendType` member** |

> `FLYDSL` is on the roadmap — a Python-embedded tile DSL lowering through MLIR → ROCDL. Its tuning knowledge already lives in the kernel-optimize knowledge base at [`knowledge/backend/flydsl/`](../kernel-optimize/knowledge/backend/flydsl/overview.md), but it is **not yet registered in `BackendType`**, so `GlobalBackendManager.set_*_backend(BackendType.FLYDSL)` is not available today.

```python
from primus_turbo.pytorch.core.backend import BackendType, GlobalBackendManager
GlobalBackendManager.set_gemm_backend(BackendType.CK)   # force
GlobalBackendManager.set_auto_tune(True)                # or PRIMUS_TURBO_AUTO_TUNE=1
GlobalBackendManager.reset()                            # clear settings + autotune cache
```

Env vars (per-precision form `"fp8:CK,other:TRITON"` supported): `PRIMUS_TURBO_GEMM_BACKEND`, `PRIMUS_TURBO_GROUPED_GEMM_BACKEND`, `PRIMUS_TURBO_ATTENTION_BACKEND`, `PRIMUS_TURBO_MOE_DISPATCH_COMBINE_BACKEND`, `PRIMUS_TURBO_AUTO_TUNE`.

## Test & Bench Quick Reference

```bash
# Correctness (single-GPU, parallel; each xdist worker pins one GPU via conftest.py)
pytest tests/pytorch/ -n 8
pytest tests/pytorch/ops/test_gemm_fp8.py -v -k "blockwise and TRITON"   # filter op+backend
pytest tests/pytorch/ -n 8 --deterministic-only                          # bitwise-determinism suite
pytest tests/pytorch/ --dist-only                                        # multi-GPU tests

# Performance
python benchmark/ops/bench_gemm_turbo.py --dtype fp8 --granularity blockwise
python benchmark/ops/run_suite.py -d output/ -g gemm_fp8                  # batch suite
```

Correctness gates: bf16/fp16 `rtol=atol=1e-2`, fp32 `1e-4` (allclose tolerances from `get_tolerances` in `tests/pytorch/test_utils.py`); FP8 SNR ≥ 25 dB (E4M3) / 20 dB (E5M2), FP4 SNR ≥ 10 dB (SNR via `compute_snr`; thresholds are hardcoded in the test/bench files, **not** in `get_tolerances`); determinism `rtol=atol=0`. Details and patterns: [verify-accuracy/SKILL.md](verify-accuracy/SKILL.md).

## Additional References

- `README.md` — quick start, install, packaging
- `docs/examples.md` — per-operator API usage (GEMM, Attention, GroupedGEMM, FP8/FP4, Backend/AutoTune)
- `benchmark/README.md`, `benchmark/accuracy/README.md` — DeepEP bench, cross-platform accuracy
- `CONTRIBUTING.md` — branch naming and commit conventions
- Kernel optimization loop: `kernel-optimize/SKILL.md` (drive via [optimize-handoff/SKILL.md](optimize-handoff/SKILL.md))
