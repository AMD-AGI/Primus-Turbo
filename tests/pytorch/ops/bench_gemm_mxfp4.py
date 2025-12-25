###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import pytest
import torch

from primus_turbo.pytorch.core.backend import BackendType, GlobalBackendManager
from primus_turbo.pytorch.core.low_precision import (
    Float4QuantConfig,
    Format,
    MXScalingRecipe,
    ScaleDtype,
    ScalingGranularity,
)
from primus_turbo.pytorch.ops.gemm_fp4 import gemm_fp4

torch.manual_seed(42)

PROBLEM_SIZES = [
    [16384, 22016, 4096],
    [16384, 4096, 22016],
    [22016, 4096, 16384],
    [16384, 12288, 4096],
    [16384, 4096, 12288],
    [16384, 4096, 11008],
    [16384, 11008, 4096],
    [12288, 4096, 16384],
    [4096, 11008, 16384],
    [16384, 4096, 4096],
    [4096, 4096, 16384],
    [28672, 57344, 8192],
    [28672, 8192, 57344],
    [57344, 8192, 28672],
    [28672, 8192, 28672],
    [28672, 28672, 8192],
    [28672, 8192, 8192],
    [8192, 28672, 28672],
    [28672, 10240, 8192],
    [28672, 8192, 10240],
    [10240, 8192, 28672],
    [8192, 8192, 28672],
    [16384, 28672, 4096],
    [28672, 4096, 16384],
    [16384, 4096, 28672],
    [16384, 14336, 4096],
    [16384, 4096, 14336],
    [4096, 14336, 16384],
    [16384, 4096, 4096],
    [16384, 6144, 4096],
    [16384, 4096, 6144],
    [6144, 4096, 16384],
    [4096, 4096, 16384],
]


@pytest.mark.parametrize("layout", ["NT"])
@pytest.mark.parametrize(
    "format",
    [
        Format.E2M1_X2,
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        torch.bfloat16,
    ],
)
@pytest.mark.parametrize("granularity", [ScalingGranularity.MX_BLOCKWISE])
@pytest.mark.parametrize("backend", [BackendType.HIPBLASLT])
def test_gemm_mxfp4(layout, format, dtype, granularity, backend):
    from primus_turbo.pytorch.core.low_precision import check_mxfp4_support

    # Skip unit test on gfx942.
    mxfp4_supported, reason = check_mxfp4_support()
    if not mxfp4_supported:
        pytest.skip(reason)

    # Set backend and auto_tune config
    GlobalBackendManager.set_gemm_backend(backend)

    device = "cuda:0"

    trans_a = layout[0] == "T"
    trans_b = layout[1] == "T"

    m, n, k = PROBLEM_SIZES[0]

    a_shape = (m, k) if not trans_a else (k, m)
    b_shape = (k, n) if not trans_b else (n, k)

    a = torch.randn(a_shape, dtype=dtype, device=device, requires_grad=True)
    b = torch.randn(b_shape, dtype=dtype, device=device, requires_grad=True)

    config = Float4QuantConfig(
        granularity=granularity, format=format, block_size=32, scale_dtype=ScaleDtype.E8M0
    )
    config.scaling_recipe["a_fwd"] = MXScalingRecipe(use_2d_block=False, use_sr=False)
    config.scaling_recipe["b_fwd"] = MXScalingRecipe(use_2d_block=True, use_sr=False)
    config.scaling_recipe["grad_bwd"] = MXScalingRecipe(use_2d_block=False, use_sr=True)
    config.scaling_recipe["a_bwd"] = MXScalingRecipe(use_2d_block=False, use_sr=False)
    config.scaling_recipe["b_bwd"] = MXScalingRecipe(use_2d_block=True, use_sr=True)

    for _ in range(0, 10):
        c = gemm_fp4(a, b, trans_a, trans_b, dtype, config)
        c.backward(torch.ones_like(c))

    torch.cuda.synchronize()

    prof = torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./profile"),
        record_shapes=True,
        profile_memory=False,
        with_stack=True,
    )

    prof.start()
    for _ in range(0, 10):
        for problem_size in PROBLEM_SIZES:
            m, n, k = problem_size
            a_shape = (m, k) if not trans_a else (k, m)
            b_shape = (k, n) if not trans_b else (n, k)

            a = torch.randn(a_shape, dtype=dtype, device=device, requires_grad=True)
            b = torch.randn(b_shape, dtype=dtype, device=device, requires_grad=True)

            a.detach().clone().requires_grad_()
            b.detach().clone().requires_grad_()

            c = gemm_fp4(a, b, trans_a, trans_b, dtype, config)
            c.backward(torch.ones_like(c))
        prof.step()
    torch.cuda.synchronize()

    prof.stop()

    # Reset config and caches
    GlobalBackendManager.reset()
