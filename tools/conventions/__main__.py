###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""``python tools/conventions`` (from anywhere) and ``python -m tools.conventions`` (from the root)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.conventions.core import main  # noqa: E402

sys.exit(main())
