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

"""FlyDSL flash-attention backward for gfx1250 (MI455X), DeepSeek-V3 MLA head dims (192 / 128).

``impl`` loads its sibling modules by path under directory-unique names, so importing this
package never touches ``sys.path`` or binds a generic top-level module name.
"""

from .impl import flydsl_attn_bwd

__all__ = ["flydsl_attn_bwd"]
