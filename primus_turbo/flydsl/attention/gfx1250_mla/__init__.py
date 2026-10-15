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

"""FlyDSL flash attention for gfx1250 (MI455X) at the DeepSeek-V3 MLA head dims (qk 192, v 128)."""

import hashlib
import pathlib

_PKG = pathlib.Path(__file__).resolve().parent
_UTILS = _PKG.parents[1] / "utils"
# FlyDSL's disk-cache key misses code reached via nested helpers or other directories; launchers read this.
SOURCE_DIGEST = hashlib.sha256(
    b"".join(
        p.read_bytes() for p in (*sorted(_PKG.glob("*.py")), _UTILS / "buffer_ops.py", _UTILS / "prims.py")
    )
).hexdigest()
