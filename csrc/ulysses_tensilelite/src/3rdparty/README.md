# Vendored files

| File | Upstream | Commit |
|---|---|---|
| `SdmaQueue.hpp` | `projects/hipblaslt/tensilelite/client/include/SdmaQueue.hpp` in ROCm/rocm-libraries | `15ff8cb10eda74fd0f1eb4f982c6e8b625d5d066` |
| `../barrier.hip` kernel `boundaryBarrierKernel` | `projects/hipblaslt/tensilelite/client/include/A2ABoundaryBarrier.hpp`, deleted upstream by `66f81c65c88` | `66f81c65c88^` |

`SdmaQueue.hpp` is header-only and not part of `libtensilelite-host.so`.
Re-copy it whenever the upstream file changes; do not edit it here.
The barrier kernel has no live upstream; keep its atomics (release add, acquire load, relaxed
subtract, system scope) as they are unless the fused kernels' synchronization changes.

Copied into Primus-Turbo from attn-a2a-demo 7a7343ef47 (`csrc/third_party/`).
Reformatted by Turbo's clang-format (pre-commit); otherwise unchanged.
