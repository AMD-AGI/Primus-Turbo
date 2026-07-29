###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Offline autotune driver for the dense GEMM family.

    python -m primus_turbo.tuning.offline_tune_gemm [--shapes s.json] [--gpus N]

``shapes.json``: ``{"mnk": [[m, n, k], ...]}``. Omit for one smoke-test shape.
``--gpus N`` shards the shapes over N single-GPU workers and merges their output.

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
from primus_turbo.pytorch.kernels.gemm.gemm_fp4_impl import GEMMFP4KernelDispatcher
from primus_turbo.pytorch.kernels.gemm.gemm_fp8_impl import GEMMFP8KernelDispatcher
from primus_turbo.pytorch.kernels.gemm.gemm_impl import GEMMKernelDispatcher
from primus_turbo.pytorch.ops import gemm, gemm_fp4, gemm_fp8
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

_DEFAULT_MNK = [(4096, 4096, 4096)]


# --- One job list per precision ----------------------------------------------


def _nt_fwd_bwd(op, dtype, device, **op_kwargs):
    """Build a job running one NT gemm fwd + bwd, so the two grad gemms get tuned too."""

    def run_one(m, n, k):
        # NT: a is [m, k], b is [n, k]; both need grad.
        a = torch.randn(m, k, dtype=dtype, device=device, requires_grad=True)
        b = torch.randn(n, k, dtype=dtype, device=device, requires_grad=True)
        out = op(a, b, False, True, dtype, **op_kwargs)
        out.backward(torch.randn_like(out))

    return run_one


def _jobs_gemm(device):
    """bf16/fp16 dense: one job per dtype."""
    return [(str(dtype), _nt_fwd_bwd(gemm, dtype, device)) for dtype in DTYPES]


def _jobs_gemm_fp8(device):
    """fp8: dtype x format x granularity."""
    return [
        (
            f"{dtype} {fmt.name}/{gran.name}",
            _nt_fwd_bwd(gemm_fp8, dtype, device, config=fp8_config(fmt, gran)),
        )
        for dtype, fmt, gran in itertools.product(DTYPES, FP8_FORMATS, FP8_GRANULARITIES)
    ]


def _jobs_gemm_fp4(device):
    """fp4: dtype x preshuffle. Format and granularity are fixed by Float4QuantConfig."""
    supported, reason = check_mxfp4_support()
    if not supported:
        logger.info(f"[gemm_fp4] skipped: {reason}")
        return []
    return [
        (
            f"{dtype} preshuffle={preshuffle}",
            _nt_fwd_bwd(gemm_fp4, dtype, device, config=Float4QuantConfig(use_preshuffle=preshuffle)),
        )
        for dtype, preshuffle in itertools.product(DTYPES, (False, True))
    ]


def _counts(key):
    """`[m, k] @ [n, k] -> [m, n]`; the key leads with the extents."""
    m, n, k, a_dtype, b_dtype, out_dtype = key[:6]
    flops = 2 * m * n * logical_k(k, a_dtype)
    moved = m * k * a_dtype.itemsize + n * k * b_dtype.itemsize + m * n * out_dtype.itemsize
    return flops, moved


FAMILY = Family(
    module="primus_turbo.tuning.offline_tune_gemm",
    shapes_key="mnk",
    default_shapes=_DEFAULT_MNK,
    precisions=(
        (_jobs_gemm, (("gemm", GEMMKernelDispatcher, _counts),)),
        (_jobs_gemm_fp8, (("gemm_fp8", GEMMFP8KernelDispatcher, _counts),)),
        (_jobs_gemm_fp4, (("gemm_fp4", GEMMFP4KernelDispatcher, _counts),)),
    ),
)


if __name__ == "__main__":
    main(FAMILY)
