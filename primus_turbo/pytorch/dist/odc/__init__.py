###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""ODC (on-demand communication) symmetric-memory primitives.

:mod:`.runtime_config` must be populated before :mod:`.primitives` is first
imported (the primitives select their backend at import time), so this package
deliberately imports nothing.
"""
