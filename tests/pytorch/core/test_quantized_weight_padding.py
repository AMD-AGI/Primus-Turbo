###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Padded cached weights dequantize to their logical 2D/3D dimensions."""

import pytest
import torch

from primus_turbo.pytorch.core.low_precision import ScalingGranularity, float8_e4m3
from primus_turbo.pytorch.core.quantized_tensor import QuantizedTensor
from tests.pytorch.test_utils import compute_snr


@pytest.mark.parametrize("shape", [(79, 137), (2, 79, 137)])
def test_dequantized_weight_discards_both_padded_dimensions(shape):
    torch.manual_seed(822)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    quant = QuantizedTensor.quantize(
        x, float8_e4m3, ScalingGranularity.TENSORWISE, pad_align_last=128, pad_align_penultimate=128
    )
    assert quant.shape == x.shape
    assert quant.qdata.shape[-2:] == (128, 256)
    restored = quant.dequantize()
    assert restored.shape == x.shape
    assert restored.is_contiguous()
    assert compute_snr(x.float(), restored.float()) > 30
