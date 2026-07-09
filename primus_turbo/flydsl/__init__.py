###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import os
import tempfile

# Per-PID cache dir: flydsl autotune's single-JSON cache races across EP ranks.
os.environ["FLYDSL_AUTOTUNE_CACHE_DIR"] = os.path.join(
    os.environ.get("FLYDSL_AUTOTUNE_CACHE_DIR", os.path.join(tempfile.gettempdir(), "flydsl_autotune")),
    f"pid_{os.getpid()}",
)
