###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.
###############################################################################

import pytest
import torch

from primus_turbo.pytorch.core.low_precision import Float8QuantConfig, Format, ScalingGranularity
from primus_turbo.pytorch.dist import FP8AllToAll


@pytest.mark.parametrize(
    "config, error, message",
    [
        (Float8QuantConfig(format=Format.E5M2), ValueError, "only support E4M3 and HYBRID"),
        (Float8QuantConfig(granularity=ScalingGranularity.ROWWISE), AssertionError, None),
    ],
)
def test_invalid_config_fails_before_collective(config, error, message):
    # A sentinel group is never touched because validation precedes communication.
    with pytest.raises(error, match=message):
        FP8AllToAll.apply(object(), torch.ones(2, 4), None, None, False, False, config)


def test_missing_group_is_rejected():
    with pytest.raises(AssertionError, match="group should not be None"):
        FP8AllToAll.apply(None, torch.ones(2, 4), None, None, False, False, Float8QuantConfig())
