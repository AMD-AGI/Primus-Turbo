---
name: verify-accuracy
description: Primus-Turbo accuracy verification — compare an operator against a higher-precision reference for forward and backward, with the right gate (allclose for bf16/fp16/fp32, SNR for fp8/fp4) and FP8 encoding awareness. Use when validating numerical correctness of a Primus-Turbo operator.
---

# Verifying Accuracy

Prove an operator is numerically correct against a higher-precision PyTorch reference, for **both forward and backward**. Read the hub [`../SKILL.md`](../SKILL.md) first.

## The pattern

Copy the structure of an existing test rather than writing one from scratch:

| Your case | Copy from |
|-----------|-----------|
| Dense fwd+bwd, parametrized shapes/dtypes/layouts | `tests/pytorch/ops/test_gemm.py` |
| Low-precision (SNR gate, backend parametrize) | `tests/pytorch/ops/test_gemm_fp8.py` |
| Determinism (bitwise) | `test_gemm.py::test_gemm_deterministic` |
| Attention / MoE / grouped | `tests/pytorch/ops/test_attention.py`, `test_moe_permute.py`, `test_grouped_gemm.py` |

Rules those tests follow: seed with `torch.manual_seed(42)`; clone a `_ref` input with its own `requires_grad_()` so grads stay independent; normalize sensitive inputs (`a = a / a.abs().max()`); reset `a.grad = None` between reuses; `pytest.skip` unsupported combos instead of failing.

## Reference implementations (`tests/pytorch/ref/`)

Put ground-truth math here, computed in `float()`/fp32, then compare the lower-precision Turbo output against it. Existing references to reuse or mirror: `gemm_ref.py` (incl. `grouped_gemm_ref`), `attention_ref.py` (uses `torch.nn.functional.scaled_dot_product_attention` under `sdpa_kernel`), `moe_ref.py`, `permuatation_ref.py`, `quantization_ref.py`, `deep_ep_ref.py`.

## Tolerances and metrics

`get_tolerances(dtype)` in `tests/pytorch/test_utils.py` returns **allclose tolerances only**: fp32 `1e-4`, fp16/bf16 `1e-2`, and intentionally-loose fp8 `1e-1` / fp4 `0.5`. Use it with `torch.testing.assert_close` for fp32/fp16/bf16. **The fp8/fp4 tolerances are too loose to be a real gate** — low precision gates on SNR instead (element-wise tolerance is meaningless after quantization).

| Type | Real gate | Threshold | Defined in |
|------|-----------|-----------|------------|
| FP32 | `assert_close(**get_tolerances)` | `rtol=atol=1e-4` | `test_utils.py` |
| FP16 / BF16 | `assert_close(**get_tolerances)` | `rtol=atol=1e-2` | `test_utils.py` |
| FP8 E4M3 | SNR (dB) | `≥ 25` | test/bench constant (e.g. `check_gemm_correctness_by_snr`, `SNR_THRESHOLD`) |
| FP8 E5M2 | SNR (dB) | `≥ 20` | same |
| FP4 (E2M1) | SNR (dB) | `≥ 10` | same |
| Determinism | bitwise over N runs | `rtol=atol=0` | test body |

The SNR thresholds (25 / 20 / 10) are **not** returned by `get_tolerances`; they are hardcoded in the test/bench files. `compute_snr(ref, actual)` is the primary low-precision gate (arg order is reference-first); `relative_error`, `mean_squared_error`, `max_abs_error`, `cosine_similarity`, `symmetric_similarity_diff` are for diagnosis.

## FP8 / FP4: state the encoding

Encoding is arch-dependent and chosen in `core/low_precision.py`: `gfx942` → **FNUZ**, `gfx950` → **OCP** (MXFP8/MXFP4 require gfx950). Build the config with `Float8QuantConfig` / `Float4QuantConfig`; gate with `check_fp8_support()` / `check_mxfp8_support()` / `check_mxfp4_support()` before running. Separate conversion errors from scale errors from kernel-math errors — never label a failure only as "fp8".

## Running tests

```bash
pytest tests/pytorch/ops/test_gemm_fp8.py -v -k "blockwise and CK"   # focused: op + backend
pytest tests/pytorch/ -n 8                                            # single-GPU suite (1 GPU/worker)
pytest tests/pytorch/ -n 8 --deterministic-only                      # bitwise-determinism suite
pytest tests/pytorch/ --dist-only                                    # multi-GPU suite
```

`tests/conftest.py` pins each xdist worker to one GPU and defines the `deterministic` / `multigpu` markers (those suites are skipped in the normal run). Backends are parametrized inside test files, so `-k "TRITON"` / `-k "CK"` selects one; force a backend in-test with `GlobalBackendManager.set_gemm_backend(...)` and `reset()` in teardown.

## Cross-platform numerical accuracy

For "does AMD match CPU / NVIDIA on this math?" (not pass/fail): `benchmark/accuracy/eval_gemm_accuracy.py`, `eval_sf_accuracy.py` (GPU-vs-CPU, write `.xlsx`). GPU-vs-GPU is a 3-step dump/transfer/compare flow — see `benchmark/accuracy/README.md`.

## Checklist

```
- [ ] Reference impl in tests/pytorch/ref/ (fp32 / PyTorch-native)
- [ ] Forward AND backward compared
- [ ] Parametrized over representative shapes, dtypes, layouts, backends
- [ ] Correct gate (allclose for bf16/fp16/fp32, SNR for fp8/fp4)
- [ ] FP8/FP4 encoding (FNUZ vs OCP) and arch support asserted
- [ ] Determinism test if bit-reproducibility is required
- [ ] Focused test green, then `pytest tests/pytorch/ -n 8` green
```
