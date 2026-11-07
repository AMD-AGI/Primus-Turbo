# Unit-test coverage and remaining validation

This document records the historical PyTorch/JAX run of the coverage work and its final GEMM supplement on 2026-11-07. The source was based on upstream `0f76da5628517b4af2204e404fd98c09267afcb0` plus the module changes described below. The measurements apply to that combined source snapshot, not to each module branch in isolation or to subsequent review fixes. Publish this document after integrating the associated code changes.

The reviewed snapshot consists of the following module heads on that upstream base:

| Module | Reviewed commit |
|---|---|
| Core | `dfad4851403b504f266924be6daa4ee8f52348bb` |
| Quantization | `c0f8467516c31645ad0eef80e4b418082ee4f3d4` |
| GEMM / MLP | `a2f65a5b0218268b4d2268a5454f1201d354b2c7` |
| Attention | `92f8b170ec7f8a58170e31a38ee5eee0bd1e3731` |
| Communication | `4eff83c5be0871028965017b41914442e6f2e85c` |
| MoE | `e78fe525a73928abf29f2ea7f05c111708303baf` |
| JAX | `1a712a3fb81fc4e23ffbb435cf162ffdcb5b3005` |

The changes below describe this historical snapshot. Review fixes add accumulation scratch protection for per-operation autotuning, separate ordinary/accumulating cache entries, hardware gates, environment restoration and stronger numerical references. They require a fresh GPU run before updating any coverage figure in this document. The separate removal of FP8AllToAll also changes the measured source and denominator.

## Results and scope

The six CI entry points and independent reruns of cases skipped after a preceding failure produced **26,504 passed, 5 failed, and 18,979 skipped** unique test results. The final tensorwise FP8 GEMM supplement added **32 passed, 0 failed, and 0 skipped**, giving **26,536 passed, 5 failed, and 18,979 skipped** across the combined evidence. No collected test remained unobserved or incompletely executed in the full-run audit.

The combined Python line coverage is **90.30% (10,422/11,542)** and branch coverage is **80.24% (2,165/2,698)**. The CI summary additionally excludes `__init__.py`, giving **90.26% (10,317/11,430)** line coverage. The existing coverage exclusions in `pyproject.toml` are unchanged. Python coverage does not measure C++/HIP device code or the excluded Triton/FlyDSL/Gluon implementations.

The merged data consists of 24 raw inputs from the full run and one raw input from the final GEMM supplement. Earlier coverage campaigns and diagnostic probes are excluded. The final supplement was a targeted run, not a second full-suite execution.

| Module | Covered / executable lines | Line coverage | Branch coverage |
|---|---:|---:|---:|
| Attention / USP | 1,863 / 2,177 | 85.58% | 69.57% |
| Communication / async TP | 366 / 417 | 87.77% | 72.08% |
| Core | 1,197 / 1,291 | 92.72% | 84.79% |
| DeepEP | 492 / 802 | 61.35% | 59.09% |
| GEMM / MLP | 937 / 937 | 100.00% | 98.96% |
| Grouped GEMM / MLP | 1,876 / 1,876 | 100.00% | 97.19% |
| JAX (excluding DeepEP) | 974 / 987 | 98.68% | 95.68% |
| MoE / MegaMoE | 1,421 / 1,642 | 86.54% | 69.88% |
| Other | 879 / 996 | 88.25% | 78.45% |
| Quantization | 417 / 417 | 100.00% | 100.00% |

The full package has not reached the 95% line target: with this denominator, at least **543 additional lines** must execute to reach 10,965 covered lines. A module's 100% line coverage does not imply complete branch, numerical, concurrency, or hardware validation. In particular, tensorwise `gemm_fp8.py` has 216/216 lines covered, while the path that reuses an existing activation transpose in rowwise FP8 GEMM remains a branch-coverage gap in this snapshot.

## Changes by module

- **Core and test infrastructure:** accept `BackendChoice` in backend resolution, including per-operation autotune and pinned backend precedence. Test environment/cache semantics, logging, dependency errors, HIP/stream diagnostics, gradient ownership, LRU behavior, dispatcher failure/capture behavior, and RMSNorm numerical gradients. Skip modules named for gfx1250 before importing them on gfx950, including their host-only cases; these modules load JIT launches inspected by the process-wide mutable-global regression. Shared numerical fixtures restore the backend settings they override and scope caller-thread backward to individual tests.
- **Quantization:** normalize MXFP4 public row payloads to contiguous raw bytes while preserving the FP4 dtype. Supply `MXFP4_BLOCK_SIZE` in the gather-equivalence test. Cover quantized tensor layouts and cached weights, padding, argument validation, fake outputs, HIP fallbacks, and gathered activations.
- **GEMM and MLP:** make dense GEMM output gradients contiguous before backend execution; distinguish ordinary and accumulating FP8 GEMM autotune keys; provide a scaling recipe when rebuilding a grouped MXFP8 MLP activation transpose. Test dense/grouped BF16, FP8 and FP4 gradients, partial training, cached inputs, variable-K layouts, fake outputs, backend eligibility, and `main_grad` accumulation.
- **Attention:** require complete 64-head groups for gfx950 FlyDSL sparse MLA, allowing 32-head inputs to fall back to Triton. Cover dense/ragged gradients, optional sinks and outputs, FP8, stable ring merging, fake outputs, and real two-GPU context parallel attention.
- **Communication and DeepEP:** test symmetric-memory failure cleanup and real two-GPU IPC, caller-owned async TP output buffers, FP8 AllToAll splitting and constant-cotangent backward checks, and DeepEP configuration/event contracts.
- **MoE and MegaMoE:** keep staged BF16/FP8 MegaMoE outside Dynamo tracing with eager custom-autograd boundaries. Test router/permute partial gradients and EP1 MegaMoE forward/backward against a local reference, including compiled callers.
- **JAX:** support the public `jax.enable_x64` context manager with the legacy experimental import as a fallback. Cover grouped GEMM vmap/VJP, int64 offset preservation, batching, public errors, setup, runtime lifecycle, and MoE abstract shapes.

## Execution environment and remaining failures

The run used eight MI350X/gfx950 GPUs on one Ruby node, `rocm/primus:v26.8` for PyTorch, and `rocm/jax-training:maxtext-v26.8` for JAX. It exercised the single-GPU, deterministic, distributed, JAX single-GPU, JAX in-process, and JAX per-process CI entry points.

Five existing `test_fused_mega_moe.py` cases failed the 40 dB input-gradient SNR threshold: two GPT-OSS cases measured about 34.5 dB, and three SwiGLU-OAI cases measured about 26.46 dB. These failures remain in the results; the tolerances were not relaxed. The new EP1 MegaMoE tests do not resolve or replace those distributed failures. The full suite is not green.

The JAX in-process entry point skipped all 4,583 collected cases: 4,580 single-GPU cases and three existing in-process DeepEP tests disabled for known deadlocks. The separate JAX per-process entry point passed four tests. No gfx942/gfx1250 numerical execution or cross-node RDMA validation is claimed.

Host tests that mock driver, communication, or backend boundaries verify contracts only. FakeTensor/Meta and JAX trace-only tests verify metadata, shape or lowering behavior. The MegaMoE compile tests exercise an `aot_eager` caller around eager stages; they do not establish fullgraph capture or Inductor performance support. Existing Stream runtime concurrency limitations remain.

## Coverage acquisition

PyTorch can execute Python custom-autograd callbacks on native worker threads that coverage.py does not trace. A backward line can therefore be missing even when a numerical test executes successfully. Selected supplementary tests use `torch.autograd.set_multithreading_enabled(False)` to trace callbacks on the caller thread, with scoped restoration.

The final GEMM suite tests both native-worker and caller-thread modes across NN/TN layouts, E4M3/HYBRID formats, contiguous/strided output gradients, and regular/accumulating weight gradients. It checks real GPU outputs and both operand gradients against an independent FP32 reference, including the identity and contents of a pre-existing `main_grad` buffer. This exercises tensorwise FP8 backward layout preparation and both ordinary and accumulating weight-gradient paths without replacing numerical kernels.

The normally collected `tests/pytorch/ops/test_fp8_alltoall_small.py` also avoids pytest's default exclusion of a directory named `dist`. The older `tests/pytorch/dist/` suite has not been moved or implicitly enabled.

## Reproduction and evidence

Use the built PyTorch and JAX environments and the six entry points in [ci.yaml](../.github/workflows/ci.yaml), retaining its subprocess coverage configuration, timeout and distributed test selection. Collect fresh coverage for the exact combined source revision and preserve logs/JUnit even when a group fails. Run dependent module PRs together when reproducing these combined measurements.

The final GEMM supplement can be run in the built PyTorch environment with:

```bash
python -m pytest tests/pytorch/ops/test_gemm_fp8_non_nt_backward.py -v \
  --cov=primus_turbo --cov-branch --cov-report=term-missing
```

The original run artifacts are not distributed with this repository. The figures above summarize archived logs and coverage data and cannot be independently recalculated from this document alone. A reproducible new run must retain the exact combined source revision, environment/image metadata, commands, pytest logs, JUnit results, raw coverage files and the combined JSON/XML/HTML reports, including failures and skips.

## Review validation boundaries

The review regressions use CPU arithmetic in place of device backend launches to verify actual wrapper, cache and tuning control flow, including cold accumulation caches, per-operation/global autotuning, pinned backends, capture guards and failure fallback. They do not validate GPU kernels or performance. JAX fixture restoration is checked with absent and pre-existing launch-mode variables, including exceptional teardown.

The updated GPU tests add nonzero RoPE and RoPE-gradient checks, distinguishable AllToAll cotangents, a pure PyTorch FP32 MLP reference, and dual FP4 quantization/dequantization/shuffle comparisons at non-256-aligned widths. Their numerical results are not included in the historical measurements above. Cache-versus-rebuild and fused-versus-unfused MLP checks establish consistency; only the separate FP32 references test independent arithmetic.

The contiguous MXFP4 row payload remains part of the public quantization contract because dequantization and shuffle consumers require compact storage. This can add a copy at widths such as 1408 and 2880. FlyDSL accepts compact payloads, but the performance cost versus pitched rows has not been measured.

## Next validation priorities

1. Resolve the five MegaMoE numerical failures with independent forward and gradient references.
2. Cover the remaining DeepEP and MoE paths on supported hardware, including real multi-rank lifecycle and cross-node communication where required.
3. Exercise remaining Attention/USP optional branches, partial gradients, layouts and concurrency on their supported backends.
4. Complete Core and async TP runtime/error-path checks without substituting mock results for hardware execution.
5. Validate architecture-specific paths on gfx942/gfx1250 and restore JAX in-process tests after their deadlock issue is resolved.

Recompute the coverage denominator after any production change. Reaching 95% requires comparable source and coverage configuration, explicit reporting of failures/skips, and separate accounting for line coverage and successful numerical validation.
