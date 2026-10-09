###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Compare gathered MXFP4 payloads/scales with an explicit BF16 permutation."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("Requires a gfx950 GPU and built Turbo extensions", allow_module_level=True)

from primus_turbo.flydsl.quantization import mxfp4_quant_kernel
from primus_turbo.pytorch.core.low_precision import ScalingGranularity, ScalingRecipe, float4_e2m1fn_x2
from primus_turbo.pytorch.ops.quantization import grouped_quantize_fp4_with_trans


@pytest.mark.parametrize("use_sr", [False, True])
@pytest.mark.parametrize("zero_probs", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_gather_matches_materialized_quantization(use_sr, zero_probs, dtype):
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    if (props.major, props.minor) != (9, 5):
        pytest.skip("Gather fusion is performance-qualified on gfx950")
    torch.manual_seed(9555)
    source = torch.randn(257, 2880, dtype=dtype, device="cuda")
    mapping = torch.randint(0, 257, (1028,), dtype=torch.int32, device="cuda")
    probabilities = torch.ones(1028, device="cuda")
    probabilities[::7] = 0
    probabilities[1::11] = -0.0
    probabilities = probabilities if zero_probs else None
    explicit = source.index_select(0, mapping.long())
    if probabilities is not None:
        explicit[probabilities == 0] = 0
    lens = torch.tensor([257, 0, 514, 257], device="cuda", dtype=torch.int64)
    offsets = torch.cat((lens.new_zeros(1), lens.cumsum(0)))
    results = []
    counters = []
    saved_counter = mxfp4_quant_kernel._SR_COUNTER[0]
    try:
        for value, kwargs in [
            (explicit, {}),
            (source, dict(dest2src=mapping, permuted_probs=probabilities, total_m=1028)),
        ]:
            mxfp4_quant_kernel._SR_COUNTER[0] = 9555
            results.append(
                grouped_quantize_fp4_with_trans(
                    value,
                    float4_e2m1fn_x2,
                    ScalingGranularity.MX_BLOCKWISE,
                    lens,
                    offsets,
                    scaling_recipe=ScalingRecipe(use_sr=use_sr),
                    scaling_recipe_for_trans=ScalingRecipe(use_sr=use_sr, use_rht=True),
                    scale_rounding_mode=2,
                    **kwargs,
                )
            )
            counters.append(mxfp4_quant_kernel._SR_COUNTER[0])
        assert counters[0] == counters[1]
        for index, (expected, actual) in enumerate(zip(*results)):
            if index in (0, 1, 2, 3):
                expected, actual = expected.view(torch.uint8), actual.view(torch.uint8)
            if index in (2, 3):
                used = int(results[0][7][-1].item())
                divisor = 2 if index == 2 else 32
                expected, actual = expected[:, : used // divisor], actual[:, : used // divisor]
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        mxfp4_quant_kernel._SR_COUNTER[0] = saved_counter
