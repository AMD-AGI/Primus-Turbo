###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Put the Primus-Turbo checkout this ladder lives in (or $HKL_TURBO_DIR) first on sys.path, so the FlyDSL
kernels the MLA rungs call (k_delta, the baseline) are this checkout's."""

import os
import sys
from pathlib import Path

ROOT = os.environ.get("HKL_TURBO_DIR") or str(Path(__file__).resolve().parents[4])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
