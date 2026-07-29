###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Offline autotune driver for the grouped GEMM family.

    python -m primus_turbo.tuning.offline_tune_grouped_gemm [--shapes s.json] [--gpus N]

``shapes.json``: ``{"gmnk": [[g, m, n, k], ...]}`` where ``g`` is the number of experts one
rank owns (``num_experts / EP``) and ``m`` the rows *per expert*
(``bs * seq * topk / num_experts``), so ``a`` is ``[g * m, k]`` and ``b`` is ``[g, n, k]``.
Omit for one smoke-test shape.

Note the tune-cache key records the *total* row count ``g * m``, so for a lookup to hit
at runtime ``g * m`` must equal that workload's ``sum(group_lens)``.

Each shape is swept with load-balanced groups; the dispatchers load-balance internally
for profiling anyway, so this matches what they measure.

To add a precision: write its ``_jobs_*`` builder and add a row to ``_PRECISIONS``.
Sweeping, perf annotation, sharding and the CLI live in ``_driver``.
"""

import itertools

import torch

from primus_turbo.common.logger import logger
from primus_turbo.pytorch.core.low_precision import (
    Float4QuantConfig,
    check_mxfp4_support,
)
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_fp4_impl import (
    GroupedGEMMFP4KernelDispatcher,
    GroupedGEMMFP4VariableKKernelDispatcher,
)
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_fp8_impl import (
    GroupedGEMMFP8KernelDispatcher,
    GroupedGEMMFP8VariableKKernelDispatcher,
)
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_impl import (
    GroupedGEMMKernelDispatcher,
    GroupedGEMMVariableKKernelDispatcher,
)
from primus_turbo.pytorch.ops import grouped_gemm, grouped_gemm_fp4, grouped_gemm_fp8
from primus_turbo.tuning._driver import (
    DTYPES,
    FP8_FORMATS,
    FP8_GRANULARITIES,
    Family,
    fp8_config,
    logical_k,
    main,
)

# --- What gets swept ---------------------------------------------------------

_DEFAULT_GMNK = [(8, 2048, 4096, 4096)]

# --- One job list per precision ----------------------------------------------


def _fwd_bwd(op, dtype, device, **op_kwargs):
    """Build a job running one grouped gemm fwd + bwd, so the grad gemms get tuned too.

    ``b`` is ``[g, n, k]`` (trans_b=True), the layout MoE weights are stored in.
    """

    def run_one(g, m, n, k):
        group_lens = torch.full((g,), m, dtype=torch.int64, device=device)
        a = torch.randn(g * m, k, dtype=dtype, device=device, requires_grad=True)
        b = torch.randn(g, n, k, dtype=dtype, device=device, requires_grad=True)
        out = op(a, b, group_lens, trans_b=True, **op_kwargs)
        out.backward(torch.randn_like(out))

    return run_one


def _jobs_grouped_gemm(device):
    """bf16/fp16 grouped: one job per dtype."""
    return [(str(dtype), _fwd_bwd(grouped_gemm, dtype, device)) for dtype in DTYPES]


def _jobs_grouped_gemm_fp8(device):
    """fp8 grouped: dtype x format x granularity."""
    return [
        (
            f"{dtype} {fmt.name}/{gran.name}",
            _fwd_bwd(grouped_gemm_fp8, dtype, device, config=fp8_config(fmt, gran)),
        )
        for dtype, fmt, gran in itertools.product(DTYPES, FP8_FORMATS, FP8_GRANULARITIES)
    ]


def _jobs_grouped_gemm_fp4(device):
    """fp4 grouped: one job per dtype; format and granularity are fixed by Float4QuantConfig."""
    supported, reason = check_mxfp4_support()
    if not supported:
        logger.info(f"[grouped_gemm_fp4] skipped: {reason}")
        return []
    return [
        (str(dtype), _fwd_bwd(grouped_gemm_fp4, dtype, device, config=Float4QuantConfig()))
        for dtype in DTYPES
    ]


def _counts_fwd(key):
    """`[m, k] @ [g, n, k] -> [m, n]`: the group multiplicity is on the weights."""
    g, m, n, k, a_dtype, b_dtype, out_dtype = key[:7]
    flops = 2 * m * n * logical_k(k, a_dtype)
    moved = m * k * a_dtype.itemsize + g * n * k * b_dtype.itemsize + m * n * out_dtype.itemsize
    return flops, moved


def _counts_vk(key):
    """`[k, m] @ [k, n] -> [g, m, n]` (wgrad): both operands are 2D, the output is grouped."""
    g, m, n, k, a_dtype, b_dtype, out_dtype = key[:7]
    flops = 2 * m * n * logical_k(k, a_dtype)
    moved = m * k * a_dtype.itemsize + n * k * b_dtype.itemsize + g * m * n * out_dtype.itemsize
    return flops, moved


# Forward/dgrad and variable-K (wgrad) are separate dispatchers with separate assets, but a
# single fwd+bwd sweep fills both, so each precision lists the pair.
FAMILY = Family(
    module="primus_turbo.tuning.offline_tune_grouped_gemm",
    shapes_key="gmnk",
    default_shapes=_DEFAULT_GMNK,
    precisions=(
        (
            _jobs_grouped_gemm,
            (
                ("grouped_gemm", GroupedGEMMKernelDispatcher, _counts_fwd),
                ("grouped_gemm_vk", GroupedGEMMVariableKKernelDispatcher, _counts_vk),
            ),
        ),
        (
            _jobs_grouped_gemm_fp8,
            (
                ("grouped_gemm_fp8", GroupedGEMMFP8KernelDispatcher, _counts_fwd),
                ("grouped_gemm_fp8_vk", GroupedGEMMFP8VariableKKernelDispatcher, _counts_vk),
            ),
        ),
        (
            _jobs_grouped_gemm_fp4,
            (
                ("grouped_gemm_fp4", GroupedGEMMFP4KernelDispatcher, _counts_fwd),
                ("grouped_gemm_fp4_vk", GroupedGEMMFP4VariableKKernelDispatcher, _counts_vk),
            ),
        ),
    ),
)


if __name__ == "__main__":
    main(FAMILY)
