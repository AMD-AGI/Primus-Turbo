###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Shared stub-setup helpers for CPU-only unit tests.

Import this at the top of any test file that needs to import primus_turbo
modules in an environment without a built C extension or GPU.

Usage:
    from tests.conftest_stubs import apply_cpu_stubs
    apply_cpu_stubs()
"""

import sys
from unittest.mock import MagicMock


def apply_cpu_stubs():
    """Inject all stubs necessary to import primus_turbo in a CPU-only env.

    Must be called BEFORE any primus_turbo import in the test module.
    """
    import torch

    _MOCKS = [
        "primus_turbo.pytorch._C",
        "primus_turbo.pytorch._C.runtime",
        "primus_turbo.pytorch._C.deep_ep",
        "aiter",
        "aiter.ops",
        "aiter.ops.mha",
        "aiter.ops.triton",
        "aiter.ops.triton.attention",
        "aiter.ops.triton.attention.mha",
        "aiter.ops.triton.attention.mha_onekernel_bwd",
        "deep_ep",
        "origami",
    ]
    for name in _MOCKS:
        if name not in sys.modules:
            sys.modules[name] = MagicMock()

    # Simulate a ROCm gfx942 device (compute capability 9.4) for module-level
    # checks in low_precision.py and utils.py.
    if not torch.cuda.is_available():
        _props = MagicMock()
        _props.major = 9
        _props.minor = 4
        torch.cuda.get_device_properties = lambda d: _props
        torch.cuda.current_device = lambda: 0
