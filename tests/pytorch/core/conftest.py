###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Pytest conftest for tests/pytorch/core.

Stubs out all compiled C extensions and GPU-touching imports so that
pure-Python unit tests in this directory run without a GPU or ROCm build.
Must run before any primus_turbo module is imported, which pytest guarantees
by loading conftest.py before collecting test files.
"""

import sys
import unittest.mock as _mock


def _stub(name: str):
    if name not in sys.modules:
        sys.modules[name] = _mock.MagicMock()


# Logger (required by backend.py)
_stub("primus_turbo.common")
_stub("primus_turbo.common.logger")
sys.modules["primus_turbo.common.logger"].logger = _mock.MagicMock()

# Compiled C extension and sub-modules
_stub("primus_turbo.pytorch._C")
_stub("primus_turbo.pytorch._C.runtime")

# In-tree DeepEP (requires ROCm build)
_stub("primus_turbo.pytorch.deep_ep")

# Modules that trigger GPU / device calls during import
_stub("primus_turbo.pytorch.modules")
_stub("primus_turbo.pytorch.ops")
_stub("primus_turbo.pytorch.core.stream")
_stub("primus_turbo.pytorch.core.symm_mem")
_stub("primus_turbo.pytorch.core.low_precision")
_stub("primus_turbo.pytorch.core.utils")
