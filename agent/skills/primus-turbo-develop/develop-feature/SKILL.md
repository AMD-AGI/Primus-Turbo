---
name: develop-feature
description: Primus-Turbo feature development workflow — the layered architecture (ops / kernels-dispatcher / Triton / HIP-CK csrc / modules), how to wire a new operator end-to-end, which layer to touch, and which existing file to copy. Use when adding or changing a Primus-Turbo operator or module on AMD GPUs.
---

# Developing a Feature in Primus-Turbo

Read the hub [`../SKILL.md`](../SKILL.md) first for the architecture diagram and code map. This file describes the workflow; instead of templates it points to the canonical file to copy for each layer.

## Decide which layers to touch

```
1. Existing op close to mine?  → copy its vertical slice and adapt (fastest path).
2. Language?    Triton → Python only, NO rebuild.   HIP/CK → C++ in csrc/, rebuild.
3. Backends?    one → call the kernel directly.     several → add a kernels/ dispatcher.
4. Autograd?    implement forward AND backward in a torch.autograd.Function.
5. Surface?     expose a function in ops/, optionally a nn.Module in modules/.
```

Minimum viable op = **`ops/` Function + one kernel**. Add the dispatcher and module only when justified.

## Layer 1 — `ops/` (Python API + autograd)

A `torch.autograd.Function` (forward/backward) wrapped by a thin user function, exported from `primus_turbo/pytorch/ops/__init__.py`. Copy the closest existing op:

| Your case | Copy from |
|-----------|-----------|
| Simple 2-input op | `ops/gemm.py` |
| Multi-backend with quant variants | `ops/gemm_fp8.py` |
| Grouped / variable-shape | `ops/grouped_gemm.py` |
| Direct C++ op (no dispatcher) | `ops/normalization.py` |
| Direct Triton op (no dispatcher) | `ops/activation.py` |

Conventions visible in those files: validate dims early; infer `out_dtype` with `torch.result_type` when `None`; carry forward intermediates via `ctx.save_for_backward`; return one grad per forward arg (`None` for non-tensors); `grad_out = grad_out.contiguous()` before feeding kernels that need contiguity. For FP8/FP4, the `config.granularity` selects which `autograd.Function` variant runs (see `ops/gemm_fp8.py`).

## Layer 2 — `kernels/` dispatcher (multi-backend ops only)

Add a dispatcher when more than one backend can serve the op. Copy `kernels/gemm/gemm_impl.py`, which shows the whole pattern:

- one `KernelBackend` subclass per backend, each with `can_handle` + `execute`;
- an `AutoKernelDispatcher` subclass declaring `_backends` (`BackendType → BackendEntry`) and `make_key`;
- the `*_impl` entry wrapped as `@torch.library.custom_op(..., device_types="cuda")` with a `@*_impl.register_fake` twin, calling `Dispatcher.dispatch(default_backend, user_backend, **kwargs)`.

Rules: `can_handle` must **return `False`**, never raise, for unsupported inputs (the dispatcher uses it for fallback). `make_key` must capture everything that changes the best kernel (shapes, dtypes, layout, key scalars) but never `id(tensor)`. `register_fake` must mirror the real output shape/dtype. To add a new backend env var / global setter, extend `core/backend.py` and `common/constants.py` following the `_gemm_backend` / `ENV_GEMM_BACKEND` pattern.

## Layer 3a — Triton kernel (`triton/`, no rebuild)

Put the `@triton.jit` kernel under `primus_turbo/triton/<area>/`. Bind it in the kernels-layer launcher with `torch._library.wrap_triton` so it composes with `torch.compile`. Reference: `kernels/attention/attention_triton_impl.py`. Pushing autotune space / MFMA variant / tiles / pipelining toward the hardware limit is an optimization campaign — see [`../optimize-handoff/SKILL.md`](../optimize-handoff/SKILL.md).

## Layer 3b — C++ / HIP kernel (`csrc/`, rebuild required)

Four edits to expose a HIP/CK kernel to PyTorch — trace any GEMM op end-to-end as the reference:

1. **Kernel** under `csrc/kernels/<area>/` → compiles into `libprimus_turbo_kernels.so`. Arch-specialized files use the `*_gfx942.{cu,hip}` / `*_gfx950.{cu,hip}` suffix (filtered by `GPU_ARCHS` in `setup.py`).
2. **Declare** the host function and its `_meta` twin in `csrc/pytorch/extensions.h`.
3. **Register** in `csrc/pytorch/bindings_pytorch.cpp`: schema under `TORCH_LIBRARY`, real impl under `TORCH_LIBRARY_IMPL(..., CUDA)`, shape inference under `TORCH_LIBRARY_IMPL(..., Meta)`.
4. **Call** from Python via `torch.ops.primus_turbo_cpp_extension.<name>` (see `ops/normalization.py`).

Rebuild after: `GPU_ARCHS=<arch> pip install --no-build-isolation -e . -v`. Public headers: `csrc/include/primus_turbo/`. CK template / pipeline tuning toward the hardware limit is an optimization campaign — see [`../optimize-handoff/SKILL.md`](../optimize-handoff/SKILL.md).

## Layer 4 — `modules/` (optional nn.Module)

Wrap the op for model code (hold `nn.Parameter`s, reshape, call the op). Reference: `modules/linear.py`, `modules/grouped_linear.py`.

## After implementing

```
- [ ] Correctness first  → ../verify-accuracy/SKILL.md   (add ref in tests/pytorch/ref/, test in tests/pytorch/ops/)
- [ ] Then performance    → ../verify-performance/SKILL.md (add/extend benchmark/ops/bench_*.py)
- [ ] Focused test green, then `pytest tests/pytorch/ -n 8`
- [ ] Rebuilt if a C++ schema changed
- [ ] Push toward HW limit → ../optimize-handoff/SKILL.md
```
