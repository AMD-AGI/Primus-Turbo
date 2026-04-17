###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
conftest.py for tests/pytorch/ops

Installs CPU stubs for the _C extension and optional GPU-dependent modules
so that pure-Python unit tests in this directory can be collected and run
without a GPU or a compiled C extension.

This conftest only activates the stubs when the C extension is absent;
on a real GPU build the module is already available and nothing is patched.
"""

import sys
from unittest.mock import MagicMock


def pytest_configure(config):
    try:
        import primus_turbo.pytorch._C  # noqa: F401 – already built, nothing to do
        return
    except ImportError:
        pass

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

    if not torch.cuda.is_available():
        _props = MagicMock()
        _props.major = 9
        _props.minor = 4
        torch.cuda.get_device_properties = lambda d: _props
        torch.cuda.current_device = lambda: 0
