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

**`pip install -r requirements.txt` must be run before install** — it pins critical dependency versions (triton, torch, etc.). Skipping it causes version mismatches that break tests and benchmarks.

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

## 算子优化

收到算子优化需求时，本节是 agent 的**第一站**。

### 环境验证（优化前必做）

开始优化前，用 `pip show primus_turbo` 检查安装状态：
- **未安装**：按 [Build](#build) 节执行完整安装（`pip install -r requirements.txt` → editable install）
- **已安装但非 editable 或路径不是当前仓库**：同上，重新安装
- **已安装且正确**：仍需确认 `requirements.txt` 中的关键依赖（如 triton）版本一致

### 流程总览

1. **环境验证**（见 [环境验证](#环境验证优化前必做)）
2. **读 `kernel-optimize/SKILL.md` 的「前置信息需求」**，了解优化框架需要项目提供什么
3. **按需求清单收集项目信息**（从本文件各节获取）
   - kernel 源文件路径：从 [Code Structure](#code-structure-quick-reference) 和 [Key GEMM-Related Files](#key-gemm-related-files) 查到
   - focused test 命令 + quick test 命令：从 [Testing](#testing) 和 [Quick 验证](#quick-验证) 组装
   - focused benchmark 命令 + quick benchmark 命令：从 [Benchmark](#benchmark) 和 [Quick 验证](#quick-验证) 组装
   - benchmark 输出格式和可用性能指标：从 [Output Metrics](#output-metrics) 查到
   - 评分规则：从下方 [优化评分](#优化评分) 获取
   - `execution_mode` 建议和 rebuild 要求：从下方 [优化环境](#优化环境) 获取
4. **交给 `kernel-optimize/SKILL.md`**，带着上述信息，由其自主完成优化闭环并输出报告
5. **验收**：优化完成后，回到项目视角做最终确认
   - 在主仓跑**完整测试**（不只是 focused test），确保无回归：`pytest tests/pytorch/ops/ -v`
   - 审查优化报告：性能提升是否达标、改动是否合理、是否有遗留风险
   - 确认代码已正确 commit 到主仓，diff 干净

### 优化环境

由 agent 根据任务特点判断使用 `repo-mode`（直接在仓库改）还是 `workspace-mode`（搭建最小开发环境，优化后再集成回来）。以下是 Primus-Turbo 的参考建议：

- **Triton kernel**：通常 `repo-mode`。Editable install 下改完即生效，无需 rebuild，迭代很快。
- **HIP / C++ kernel**：视情况。改动面小、调参为主时可 `repo-mode`（增量 rebuild：`GPU_ARCHS=<arch> pip install --no-build-isolation -e . -v`）；若需要大规模试错或从头写新 kernel，`workspace-mode` 可能更合适，避免每次都跑主仓的完整构建链路。

`workspace-mode` 的搭建规范由 `kernel-optimize/SKILL.md` 定义。

### 优化评分

性能指标取决于算子类型，由 benchmark 输出列决定。常见指标：

| 算子类型 | 典型指标 |
|---------|---------|
| Compute-bound（GEMM 等） | `Forward TFLOPS`、`Backward TFLOPS` |
| Memory-bound（elementwise、quantization 等） | `Forward GB/s`、`Backward GB/s` |

用户在优化开始时确认需要优化哪些指标（`primary_metric`）。优化闭环中以所有目标 shape 的指定指标 **几何平均**作为 `aggregate score`，任一 shape `Check = FAIL` 则该候选直接拒绝。

Benchmark 的输出格式、shape 来源、测量方式见上方 [Benchmark](#benchmark) 节。

### Quick 验证

每轮 VALIDATE 默认使用 quick 验证（代表性 shape subset），加速迭代。BASELINE 和最终验收使用 full 验证。

**代表性 shape 选取原则**：agent 在 BASELINE 阶段从 full benchmark 结果中选取 3-5 个代表性 shape，覆盖小/中/大规模。选取标准：
- 至少包含 1 个小 shape（容易暴露 launch overhead 问题）
- 至少包含 1 个大 shape（容易暴露 memory / compute 瓶颈）
- 优先选择性能差异大的 shape（更敏感，能快速检测退步）

选出的 shape 列表记录在 `manifest.yaml` 的 `representative_shapes` 字段中。

**Quick test**：使用 focused test 命令加 `--maxfail=3` 快速失败，侧重快速拦截明显问题。

**Quick benchmark**：如果 benchmark 脚本支持 shape 过滤参数，则只跑代表性 shape；如果不支持，跑 full benchmark 后从结果中提取代表性 shape 的数据做快速判断。

### Demo：优化 blockwise FP8 GEMM Triton on MI300X

以下以 blockwise FP8 GEMM Triton on MI300X 为例。

**1. 读 `kernel-optimize/SKILL.md` 的前置信息需求**

得知优化框架需要：kernel 源文件、focused test/quick test、focused benchmark/quick benchmark、benchmark 输出格式、评分规则、execution_mode 建议、rebuild 要求。

**2. 按需求清单收集项目信息**

从本文件各节获取：
- Kernel 源文件：`primus_turbo/triton/gemm/gemm_fp8_kernel.py`（查 [Key GEMM-Related Files](#key-gemm-related-files)）
- Focused test：`pytest tests/pytorch/ops/test_gemm_fp8.py -v -k "blockwise and TRITON"`（查 [Testing](#testing)）
- Quick test：同上 + `--maxfail=3`（查 [Quick 验证](#quick-验证)）
- Focused benchmark：`PRIMUS_TURBO_GEMM_BACKEND=TRITON python benchmark/ops/bench_gemm_turbo.py --dtype fp8 --granularity blockwise`（查 [Benchmark](#benchmark)）
- Quick benchmark：同上，从 full 结果中提取代表性 shape 数据（查 [Quick 验证](#quick-验证)）
- 可用指标：`Forward TFLOPS`、`Backward TFLOPS`，`Check` 为正确性门控（查 [优化评分](#优化评分)）
- 环境建议：Triton → `repo-mode`，无需 rebuild（查 [优化环境](#优化环境)）

**3. 交给 `kernel-optimize/SKILL.md`**

带着以上信息，由其完成 DEFINE_TARGET（含向用户确认目标）、PREPARE_ENVIRONMENT、优化闭环并输出报告。

**4. 验收**

回到项目视角做最终确认：
- 在主仓跑完整测试（不只是 focused test）：`pytest tests/pytorch/ops/ -v`
- 审查优化报告：性能提升是否达标、改动是否合理
- 确认代码已正确 commit（若 `git_commit=true`），diff 干净

---

## 相关 Skills

- **算子性能优化**: `kernel-optimize/SKILL.md`（优化闭环、工作流、Triton/HIP 策略）

## Additional References

> All paths below are relative to the repository root.

- `README.md`: Quick Start, installation, testing, packaging, Minimal Example
- `docs/examples.md`: API usage for each operator (GEMM, Attention, Grouped GEMM, FP8/FP4, Backend/AutoTune)
- `docs/install_dependencies.md`: rocSHMEM (DeepEP dependency) installation guide
- `benchmark/README.md`: DeepEP benchmark instructions
- `CONTRIBUTING.md`: Branch naming and commit conventions
