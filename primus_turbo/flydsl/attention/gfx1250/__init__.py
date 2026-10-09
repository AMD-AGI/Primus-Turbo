###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 FlyDSL Project Contributors
#
# Adapted from FlyDSL (https://github.com/ROCm/FlyDSL)
# Modified by the Primus-Turbo team.
#
# This file is distributed under the Apache License 2.0 (see LICENSE-APACHE),
# not the MIT license that covers the rest of Primus-Turbo (see LICENSE).
###############################################################################

"""FlyDSL flash attention for gfx1250 (MI455X): forward and backward, bf16, head_dim 128.

Import only on gfx1250; the kernels are wave32 WMMA code. The entry points live in
``interface``. The kernels need a flydsl that satisfies ``flydsl_version.FLYDSL_REQUIREMENT``
(``>=0.3.4.1,<0.3.5``): their modules raise ImportError under any other flydsl. This package
module imports nothing, so ``flydsl_version`` can be asked first.
"""
